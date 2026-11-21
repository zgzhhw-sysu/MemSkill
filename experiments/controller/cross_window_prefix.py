"""Evaluate the last evidence-bearing span after replaying its full chronological training prefix."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn.functional import normalize

from experiments.controller import measured_cost
from experiments.controller.factorial_rollout import measure_cell
from experiments.controller.late_pair_policy import logistic, read
from experiments.controller.prefix_return import budget_envelope, predicted_offsets
from experiments.controller.rank_return import rank_offsets
from experiments.controller.source_compression import (
    file_hash,
    measure_source,
    register,
)
from experiments.designer.designer_pilot import prefix
from src.evaluation_cache import history_embeddings

PLAN = Path(__file__).with_name("protocols") / "cross_window_prefix.json"


def background(sid, n):
    return (
        hashlib.sha256(f"memskill-cross-window-v1/{sid}/{n}".encode()).digest()[0] & 1
    )


def endpoint_cohort(data, pilot_training):
    groups = defaultdict(list)
    for sid, convo in data.items():
        turns = {}
        for key, events in convo["conversation"].items():
            if key.startswith("session_") and key[8:].isdigit():
                for t in events:
                    if t["dia_id"] in turns:
                        raise ValueError("Ambiguous source turn")
                    turns[t["dia_id"]] = int(key[8:])
        for i, q in enumerate(convo["qa"]):
            evidence = set(q.get("evidence") or ())
            if int(q["category"]) == 5 or not evidence or not evidence <= turns.keys():
                continue
            positions = sorted({turns[t] for t in evidence})
            if (positions[0] - 1) // 2 != (positions[-1] - 1) // 2:
                groups[sid, positions[-1]].append(
                    {
                        "original_index": i,
                        "type": int(q["category"]),
                        "sessions": positions,
                    }
                )
    return [
        {
            "dialogue": sid,
            "end": n,
            "background": [background(sid, i) for i in range(1, n)],
            "held": sid not in pilot_training,
            "questions": groups[sid, n],
        }
        for sid, n in sorted(groups, key=lambda v: (v[1], v[0] in pilot_training, v[0]))
    ]


def focused_conversation(conversation, assigned):
    c = prefix(conversation, assigned["end"])
    by_index = {q["original_index"]: q for q in c["qa"]}
    c["qa"] = [by_index[q["original_index"]] for q in assigned["questions"]]
    if (
        len(
            [
                k
                for k in c["conversation"]
                if k.startswith("session_") and k[8:].isdigit()
            ]
        )
        != assigned["end"]
    ):
        raise ValueError("Full prefix is missing intervening sessions")
    return c


def compare_prefixes(cells, events, expected):
    left, right = (v[0] for v in cells)
    if [r["use"] for r in left] != [*expected, 0] or [r["use"] for r in right] != [
        *expected,
        1,
    ]:
        raise ValueError("Unexpected executed choices")
    if (
        left[:-1] != right[:-1]
        or events[0][:-1] != events[1][:-1]
        or len(events[0]) != len(left)
        or len(events[1]) != len(right)
    ):
        raise ValueError("Earlier memory trajectory differs before focal action")
    for k in ("base_logit", "state", "plain", "grounded"):
        if left[-1][k] != right[-1][k]:
            raise ValueError("Focal pre-action state differs")
    for k in ("source", "source_turns", "retrieved", "refs", "span_id"):
        if events[0][-1][k] != events[1][-1][k]:
            raise ValueError("Focal source or retrieval differs")
    return left[-1]


def probabilities(decision, previous, features, terminal, pilots):
    base = decision["base_logit"]
    w = torch.tensor(terminal, dtype=torch.float64)
    delta = float(w[0] + w[1:] @ normalize(w.new_tensor(decision["state"]), dim=0))
    p = {
        "native": logistic(base),
        "terminal": logistic(base + delta),
        "temporal": 0.0,
        "activity": 1.0,
    }
    for label, func in (("strength", predicted_offsets), ("rank", rank_offsets)):
        for k, offsets in func(pilots[label], features).items():
            if k == "prefix_strength":
                continue
            key = label + k[len("prefix") :]
            p[key] = logistic(base + offsets[1 + previous])
    return {**p, **{k + "_greedy": float(v >= 0.5) for k, v in p.items()}}


def summarize(reports, complete=False):
    rows = [(r, q) for r in reports for q in r["questions"]]
    if not rows:
        return {"endpoints": 0, "questions": 0}

    def group(items):
        return {
            "questions": len(items),
            "mean_abs_change_from_native": {
                a: mean(
                    abs(r["probabilities"][a] - r["probabilities"]["native"])
                    for r, q in items
                )
                for a in ("strength", "rank")
            },
            "last_writes_cover_all_evidence_sessions": {
                a: mean(
                    set(q["sessions"]).issubset(q["last_writer_sessions"][a])
                    for _, q in items
                )
                for a in ("temporal", "activity")
            },
            "f1": {
                k: mean(q["focal"][k] for _, q in items) for k in items[0][1]["focal"]
            },
        }

    out = {"endpoints": len(reports), "all": group(rows)}
    for name, key in (
        ("dialogue", lambda r, q: r["dialogue"]),
        ("type", lambda r, q: q["type"]),
        ("evidence_gap", lambda r, q: q["sessions"][-1] - q["sessions"][0]),
        ("prefix_length", lambda r, q: r["end"]),
    ):
        groups = defaultdict(list)
        for r, q in rows:
            groups[str(key(r, q))].append((r, q))
        out[name] = {k: group(v) for k, v in groups.items()}
    out["budget"] = {
        actor: budget_envelope(
            (
                len(r["questions"]),
                r["probabilities"]["native"],
                r["probabilities"][actor],
                mean(q["activity"] - q["temporal"] for q in r["questions"]),
            )
            for r in reports
        )
        for actor in ("strength", "rank")
    }
    if complete and len(out["dialogue"]) >= 2:
        histories = list(out["dialogue"].values())
        count = np.array([h["questions"] for h in histories])
        draws = np.array(list(product(range(len(histories)), repeat=len(histories))))
        intervals = {}
        for actor, controls in (
            ("strength", ("native", "terminal", "strength_rate", "strength_max")),
            ("rank", ("native", "terminal", "rank_rate", "rank_context", "strength")),
        ):
            for control in controls:
                differences = np.array(
                    [h["f1"][actor] - h["f1"][control] for h in histories]
                )
                intervals[f"{actor}-{control}"] = np.quantile(
                    (count * differences)[draws].sum(1) / count[draws].sum(1),
                    [0.025, 0.975],
                ).tolist()
        out["paired_dialogue_bootstrap"] = intervals
    return out


def prepare(study, corpus, locked, train):
    dest = locked.parent
    out = dest / "cross-window-prefix"
    out.mkdir(exist_ok=True)
    protocol = read(study / "protocol.json")
    data = {c["sample_id"]: c for c in train}
    if (
        set(data) != set(protocol["train"])
        or len(data) != 6
        or set(data) & set(protocol["test"] + protocol["validation"])
        or protocol["input_sha256"] != file_hash(corpus)
        or protocol["rule"] != 2
        or not protocol["equal_sources"]
    ):
        raise ValueError("Different training population or rule")
    archive = {
        k: dest / n / "pilot/forecast.json"
        for k, n in (("strength", "prefix-return"), ("rank", "rank-return"))
    }
    parents = {k: read(p) for k, p in archive.items()}
    if any(
        m["inputs"]["history"] != file_hash(locked) or m["stage"] != "pilot"
        for m in parents.values()
    ):
        raise ValueError("Different pilot actors")
    pilots = {k: v["fits"]["pilot"] for k, v in parents.items()}
    train_ids = set(pilots["strength"]["gate"]["dialogues"])
    if train_ids != set(pilots["rank"]["gate"]["dialogues"]) or len(train_ids) != 2:
        raise ValueError("Mismatched pilot training histories")
    assigned = endpoint_cohort(data, train_ids)
    local_cohort = dest / "cohort.json"
    if read(locked)["inputs"]["cohort"] != file_hash(local_cohort):
        raise ValueError("Local-window comparison population changed")
    local_ids = {
        (r["dialogue"], q)
        for r in read(local_cohort)["assignments"]
        for q in r["questions"]
    }
    new_ids = [
        (r["dialogue"], q["original_index"]) for r in assigned for q in r["questions"]
    ]
    if len(set(new_ids)) != len(new_ids) or local_ids.intersection(new_ids):
        raise ValueError(
            "Cross-window questions must be unique and disjoint from the local panel"
        )
    register(out / "plan.json", read(PLAN))
    register(out / "main-plan.json", read(PLAN.with_name("cross_window_main.json")))
    versions = [s["version"] for s in protocol["skills"]]
    manifest = {
        "plan": file_hash(PLAN),
        "baseline": file_hash(locked),
        "protocol": file_hash(study / "protocol.json"),
        "pilots": {k: file_hash(p) for k, p in archive.items()},
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
        "corpus": protocol["input_sha256"],
        "skills": versions,
        "endpoints": assigned,
    }
    register(out / "manifest.json", manifest)
    return data, assigned, pilots, protocol, versions


def read_origins(spans, queries):
    writers = {
        (c["memory_id"], c["version"]): sp["span_id"] + 1
        for sp in spans
        for c in sp["changes"]
        if c["after"] is not None
    }
    result = {}
    for q in queries:
        j = int(str(q["query_id"]).rsplit("_", 1)[-1])
        if j in result:
            raise ValueError("Repeated question ID in read trajectory")
        result[j] = [writers[tuple(ref)] for ref in q["refs"]]
    return result


def collect_trajectories(trainer, convo, root, w, versions):
    with history_embeddings():
        cells, events, origins = [], [], []
        for action in (0, 1):
            bits = tuple([*w["background"], action])
            cells.append(measure_cell(trainer, convo, root, bits, versions))
            folder = root / "".join(map(str, bits))
            (trace,) = (folder / "traces").glob("*.jsonl")
            raw = [json.loads(line) for line in trace.read_bytes().splitlines()]
            spans = [r for r in raw if r["kind"] == "span"]
            events.append(spans)
            origins.append(read_origins(spans, [r for r in raw if r["kind"] == "qa"]))
    return cells, events, origins


def trajectory_hashes(folder):
    (trace,) = (folder / "traces").glob("*.jsonl")
    return {
        str(p.relative_to(folder)): file_hash(p)
        for p in (
            trace,
            folder / "decisions.json",
            folder / "scores.json",
            folder / "usage.jsonl",
        )
    }


def run(trainer, study, locked, train, specs, limit):
    dest = locked.parent
    out = dest / "cross-window-prefix"
    data, assigned, pilots, protocol, versions = prepare(
        study, Path(trainer.args.data_file), locked, train
    )
    original = read(locked)
    if [s["version"] for s in specs] != versions:
        raise ValueError("Loaded writer skills differ from the original protocol")
    files = [dest.parent / "source-compression/features.json", dest / "features.json"]
    if [file_hash(p) for p in files] != [
        original["inputs"][k] for k in ("early_features", "features")
    ]:
        raise ValueError("Source feature inputs changed")
    features = {
        (r["dialogue"], r["session"]): r for p in files for r in read(p)["rows"]
    }
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    fingerprint = file_hash(out / "manifest.json")
    reports = []
    for w in assigned[:limit]:
        sid, n = w["dialogue"], w["end"]
        root = out / f"{sid}_to{n:02d}"
        root.mkdir(exist_ok=True)
        with (root / "measure.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = root / "pair.json"
            branches = [
                root / "".join(map(str, [*w["background"], action]))
                for action in (0, 1)
            ]
            if not result.exists():
                c = focused_conversation(data[sid], w)
                cells, events, origins = collect_trajectories(
                    trainer, c, root, w, versions
                )
                dec = compare_prefixes(cells, events, w["background"])
                source = events[0][-1]["source"]
                f = features.get((sid, n))
                if f is None:
                    f = measure_source(source, protocol["retriever"])
                if f["source_sha256"] != hashlib.sha256(source.encode()).hexdigest():
                    raise ValueError("Different source at the focal state")
                with torch.device(trainer.args.device):
                    probs = probabilities(
                        dec,
                        w["background"][-1],
                        f["features"],
                        original["terminal"][sid],
                        pilots,
                    )
                questions = []
                for i, q in enumerate(w["questions"]):
                    off, on = [v[2][i] for v in cells]
                    questions.append(
                        dict(
                            q,
                            last_writer_sessions={
                                name: v[i]
                                for name, v in zip(
                                    ("temporal", "activity"), origins, strict=True
                                )
                            },
                            temporal=off,
                            activity=on,
                            focal={k: off + p * (on - off) for k, p in probs.items()},
                        )
                    )
                register(
                    result,
                    {
                        "manifest": fingerprint,
                        "dialogue": sid,
                        "end": n,
                        "held": w["held"],
                        "features": f["features"],
                        "source_sha256": f["source_sha256"],
                        "probabilities": probs,
                        "cost": {
                            b.name: measured_cost(b, trace, len(questions))
                            for b, trace in zip(branches, events, strict=True)
                        },
                        "trajectories": {
                            b.name: trajectory_hashes(b) for b in branches
                        },
                        "questions": questions,
                    },
                )
            report = read(result)
            if any(
                trajectory_hashes(b) != report["trajectories"][b.name] for b in branches
            ):
                raise ValueError("Archived focal branch changed")
            if report["manifest"] != fingerprint or [
                (q["original_index"], q["type"], q["sessions"])
                for q in report["questions"]
            ] != [
                (q["original_index"], q["type"], q["sessions"]) for q in w["questions"]
            ]:
                raise ValueError("Cached cross-window outcomes differ")
            reports.append(report)
        print(
            sid,
            n,
            len(report["questions"]),
            "cross-window questions verified",
            flush=True,
        )
    if reports:
        summary = {
            "manifest": fingerprint,
            "all_histories": summarize(reports, len(reports) == len(assigned)),
            "held_histories": summarize(
                [r for r in reports if r["held"]], len(reports) == len(assigned)
            ),
            "complete": len(reports) == len(assigned),
        }
        register(out / f"through-{len(reports):03d}.json", summary)
        if summary["complete"]:
            register(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register the cross-window training cohort before generation"
    )
    for label in ("study", "corpus", "locked"):
        parser.add_argument(label, type=Path)
    args = parser.parse_args()
    subjects = set(read(args.study / "protocol.json")["train"])
    train = [c for c in read(args.corpus) if c["sample_id"] in subjects]
    _, rows, _, _, _ = prepare(args.study, args.corpus, args.locked, train)
    print(
        len(rows),
        "full-prefix intervention states;",
        sum(len(r["questions"]) for r in rows),
        "cross-window questions",
    )
