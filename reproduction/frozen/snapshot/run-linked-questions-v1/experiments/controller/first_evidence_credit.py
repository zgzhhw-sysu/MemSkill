"""Measure delayed returns after changing the skill used at the first supporting session."""

import argparse
import fcntl
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.audit_cross_window import verified_decision
from experiments.controller.credit_rank_allocation import (
    ARMS,
    data_for,
    permutation,
    probabilities,
)
from experiments.controller.cross_window_prefix import (
    background,
    focused_conversation,
    measured_cost,
    read_origins,
    trajectory_hashes,
)
from experiments.controller.factorial_rollout import measure_cell, read
from experiments.controller.root_write_repairs import seed_controls
from experiments.controller.source_compression import file_hash, register
from experiments.controller.turn_key_credit import paired_interval
from src.credit_assignment import _digest, recovered
from src.evaluation_cache import history_embeddings

PLAN = Path(__file__).with_name("protocols") / "first_evidence_credit.json"
BACKEND = "251-hf-llama3-bf16-pre16-duty22-gpu7-qwen2-shared-b2-duty30"


def cohort(endpoints):
    by_history = defaultdict(list)
    ids = set()
    for end in endpoints:
        sid, n = end["dialogue"], end["end"]
        groups = defaultdict(list)
        if end["background"] != [background(sid, i) for i in range(1, n)]:
            raise ValueError("Original background is inconsistent")
        for q in end["questions"]:
            sessions = q["sessions"]
            if (
                sessions != sorted(set(sessions))
                or sessions[-1] != n
                or (sessions[0] - 1) // 2 >= (n - 1) // 2
                or (sid, q["original_index"]) in ids
            ):
                raise ValueError("Invalid or duplicated cross-session question")
            ids.add((sid, q["original_index"]))
            groups[sessions[0]].append(q)
        for first, qs in sorted(groups.items()):
            endpoint = f"{sid}_to{n:02d}"
            by_history[sid].append(
                {
                    "name": f"{endpoint}_from{first:02d}",
                    "endpoint": endpoint,
                    "dialogue": sid,
                    "first": first,
                    "end": n,
                    "questions": qs,
                }
            )
    return [
        ws[i]
        for i in range(max(map(len, by_history.values()), default=0))
        for _, ws in sorted(by_history.items())
        if i < len(ws)
    ]


def first_write(out, group):
    return (
        out
        / "first-writes"
        / f"{group['dialogue']}_s{group['first']:02d}"
        / "response.json"
    )


def paths(sid, first, end):
    base = [background(sid, i) for i in range(1, end + 1)]
    changed = base.copy()
    changed[first - 1] = 1 - base[first - 1]
    return base, changed


def register_trial(base):
    late = base / "late-pairs"
    cross = late / "cross-window-prefix"
    rank = base / "source-compression/capped-source-credit"
    parent = rank / "policy/joint-mask/fit.json"
    numeric = rank / "rank-allocation/joint-mask/fit.json"
    manifest = cross / "manifest.json"
    spec = read(manifest)
    groups = cohort(spec["endpoints"])
    if (
        len(groups),
        len(spec["endpoints"]),
        sum(len(g["questions"]) for g in groups),
        len({g["dialogue"] for g in groups}),
    ) != (165, 99, 185, 6):
        raise ValueError("Wrong fixed training population")
    if (
        spec["backend"] != BACKEND
        or read(numeric)["parent"] != file_hash(parent)
        or spec["baseline"] != file_hash(late / "frozen.json")
    ):
        raise ValueError("Different encoder, learned policy or terminal probabilities")
    out = late / "first-evidence"
    out.mkdir(exist_ok=True)
    identity = {
        "plan": file_hash(PLAN),
        "manifest": file_hash(manifest),
        "parent": file_hash(parent),
        "rank": file_hash(numeric),
        "baseline": spec["baseline"],
        "groups": groups,
    }
    path = out / "registration.json"
    if not path.exists():
        cutoff = time.time_ns()
        prior = {
            str(p.relative_to(cross)): file_hash(p)
            for p in cross.glob("conv-*_to*/*/scores.json")
            if p.stat().st_mtime_ns <= cutoff
        }
        if any(out.glob("conv-*/*/scores.json")):
            raise ValueError(
                "Earlier intervention already measured before registration"
            )
        register(path, dict(identity, cutoff_ns=cutoff, existing=prior))
    locked = read(path)
    if any(locked[k] != v for k, v in identity.items()) or any(
        file_hash(cross / k) != v for k, v in locked["existing"].items()
    ):
        raise ValueError("Training panel or policy changed")
    return path


def events(cell):
    (p,) = (cell / "traces").glob("*.jsonl")
    trace = [json.loads(line) for line in p.read_bytes().splitlines()]
    spans = [e for e in trace if e["kind"] == "span"]
    return spans, read_origins(spans, [e for e in trace if e["kind"] == "qa"])


def verify_switch(old, new, old_dec, new_dec, first, versions):
    t = first - 1
    if (
        old[:t] != new[:t]
        or old_dec[:t] != new_dec[:t]
        or any(
            old_dec[t][k] != new_dec[t][k]
            for k in ("state", "base_logit", "plain", "grounded")
        )
        or any(
            old[t][k] != new[t][k]
            for k in ("source", "source_turns", "retrieved", "refs", "span_id")
        )
    ):
        raise ValueError(
            "The intervention did not begin from the identical physical state"
        )
    for i, (before, after, dec, alt) in enumerate(
        zip(old, new, old_dec, new_dec, strict=True)
    ):
        chosen = [s["version"] for s in after["selected_skills"]]
        factual = [s["version"] for s in before["selected_skills"]]
        if (
            any(before[k] != after[k] for k in ("source", "span_id", "source_turns"))
            or dec["plain"] != alt["plain"]
            or dec["grounded"] != alt["grounded"]
        ):
            raise ValueError("Source or ordered action set changed")
        key = "grounded" if alt["use"] else "plain"
        if (
            len(chosen) != 3
            or len(set(chosen)) != 3
            or [versions[k] for k in alt[key]] != chosen
            or chosen[:2] != factual[:2]
            or (factual[-1] != chosen[-1]) != (i == t)
            or (dec["use"] != alt["use"]) != (i == t)
        ):
            raise ValueError("More than the selected first-evidence skill differs")


def forecast(rank, sid, decs, first, terminal):
    archived = read(rank / "policy/joint-mask/fit.json")
    numeric = read(rank / "rank-allocation/joint-mask/fit.json")
    donors = [r for r in permutation(archived["observations"]) if r["dialogue"] != sid]
    result = {}
    for origin, i in (("first", first - 1), ("last", len(decs) - 1)):
        stage = int(i > 0)
        fold = numeric["folds"][f"{sid}/{stage}"]
        if set(fold["inner_histories"]) != {r["dialogue"] for r in donors}:
            raise ValueError("Selection rule was fitted on different histories")
        model = data_for(donors, stage, "cuda"), fold["rules"]
        result[origin] = probabilities(decs[i], model, terminal)
    return result


def pair_value(low, high, ps):
    return {k: low + p * (high - low) for k, p in ps.items()}


def freeze_control(base, registration, g, template, versions):
    late = base / "late-pairs"
    cross = late / "cross-window-prefix"
    out = registration.parent
    sid, n, first = g["dialogue"], g["end"], g["first"]
    src = cross / g["endpoint"]
    ref = read(src / "pair.json")
    if ref["manifest"] != file_hash(cross / "manifest.json"):
        raise ValueError("Different parent endpoint")
    verified_decision(src, template, ref, versions)
    factual, other = paths(sid, first, n)
    bits = "".join(map(str, factual))
    cell = src / bits
    decisions = read(cell / "decisions.json")
    if [int(r["use"]) for r in decisions] != factual:
        raise ValueError("Wrong full background")
    prediction = forecast(
        base / "source-compression/capped-source-credit",
        sid,
        decisions,
        first,
        read(late / "frozen.json")["terminal"][sid],
    )
    if any(
        abs(prediction["last"][k] - ref["probabilities"][k]) > 1e-10
        for k in ("native", "terminal")
    ):
        raise ValueError(
            "Native or terminal policy does not reproduce at the last evidence span"
        )
    dest = out / g["name"]
    dest.mkdir(exist_ok=True)
    fp = dest / "forecast.json"
    alt_cell = dest / "".join(map(str, other))
    identity = {
        "registration": file_hash(registration),
        "parent_pair": file_hash(src / "pair.json"),
        "first_state": _digest(decisions[first - 1]),
        "last_state": _digest(decisions[-1]),
        "background": factual,
        "ordered_common": [r["plain"][:2] for r in decisions],
    }
    with (dest / "forecast.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not fp.exists():
            if alt_cell.exists():
                raise ValueError("Switched trajectory predates its forecast")
            register(fp, dict(identity, probabilities=prediction))
        saved = read(fp)
    if any(saved[k] != v for k, v in identity.items()) or any(
        not math.isclose(
            saved["probabilities"][position][k], p, abs_tol=1e-10, rel_tol=0
        )
        for position, ps in prediction.items()
        for k, p in ps.items()
    ):
        raise ValueError("Could not reproduce the previously frozen probabilities")
    prediction = saved["probabilities"]
    return prediction, ref, cell, decisions


def forecast_available(base):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use one leased GPU for numeric forecasts")
    torch.cuda.set_per_process_memory_fraction(0.06)
    reg = register_trial(base)
    cross = base / "late-pairs/cross-window-prefix"
    manifest = read(cross / "manifest.json")
    templates = {f"{w['dialogue']}_to{w['end']:02d}": w for w in manifest["endpoints"]}
    for g in read(reg)["groups"]:
        if (cross / g["endpoint"] / "pair.json").exists() and not (
            reg.parent / g["name"] / "forecast.json"
        ).is_file():
            freeze_control(base, reg, g, templates[g["endpoint"]], manifest["skills"])


def one(trainer, base, registration, g, template, data, versions):
    prediction, ref, cell, decisions = freeze_control(
        base, registration, g, template, versions
    )
    out = registration.parent
    cross = out.parent / "cross-window-prefix"
    src = cell.parent
    sid, n, first = g["dialogue"], g["end"], g["first"]
    factual, other = paths(sid, first, n)
    bits = cell.name
    dest = out / g["name"]
    fp = dest / "forecast.json"
    alt_cell = dest / "".join(map(str, other))
    spans, origins = events(cell)
    cache = out / "responses"
    for a in (0, 1):
        seed_controls(
            src / ("".join(map(str, factual[:-1])) + str(a)), cross / "responses", cache
        )
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    c = focused_conversation(data[sid], g)
    control = read(cell / "scores.json")["answers"]
    qa_map = {q["original_index"]: i for i, q in enumerate(template["questions"])}
    if [r["qa_index"] for r in control] != list(range(len(qa_map))):
        raise ValueError("Different factual reader order")
    prev = trainer.controller.common_schedule
    try:
        trainer.controller.common_schedule = [d["plain"][:2] for d in decisions]
        with history_embeddings():
            measured, _, _ = measure_cell(trainer, c, dest, tuple(other), versions)
    finally:
        trainer.controller.common_schedule = prev
    if (alt_cell / "scores.json").stat().st_mtime_ns < fp.stat().st_mtime_ns:
        raise ValueError("Switched scores predate the frozen probabilities")
    changed, new_origins = events(alt_cell)
    verify_switch(spans, changed, decisions, measured, first, versions)
    cached = read(first_write(out, g))
    usage = [
        v
        for line in (alt_cell / "usage.jsonl").read_bytes().splitlines()
        if (v := json.loads(line))["kind"] == "executor"
    ]
    actual = usage[first - 1]
    if (
        actual["cache_key"] != cached["key"]
        or actual["response_sha256"] != cached["response_sha256"]
        or not actual["cache_hit"]
        or [actual["input"], actual["output"]] != cached["tokens"]
    ):
        raise ValueError(
            "Changed earliest action failed to reproduce the premeasured response"
        )
    answers = read(alt_cell / "scores.json")["answers"]
    details = []
    for i, (q, qa, a) in enumerate(zip(g["questions"], c["qa"], answers, strict=True)):
        idx = q["original_index"]
        j = qa_map[idx]
        original = control[j]
        rq = ref["questions"][j]
        truth = trainer.evaluator.get_ground_truth(qa)
        if (
            not set(qa["evidence"]).intersection(spans[first - 1]["source_turns"])
            or not set(qa["evidence"]).intersection(spans[-1]["source_turns"])
            or qa["original_index"] != idx
            or a["qa_index"] != i
            or any(rq[k] != q[k] for k in ("original_index", "type", "sessions"))
        ):
            raise ValueError("Wrong evidence session or question")
        arms = []
        for v in (original, a):
            score = trainer.evaluator.compute_f1(v["prediction"], truth, qa)
            if not math.isclose(score, v["f1"], abs_tol=1e-12) or min(v["tokens"]) <= 0:
                raise ValueError("Physical reader score did not reproduce")
            arms.append(
                {
                    "answer": v["prediction"],
                    "f1": score,
                    "ok": True,
                    "tokens": v["tokens"],
                }
            )
        if arms[0]["f1"] != rq[("temporal", "activity")[factual[-1]]]:
            raise ValueError("Wrong physical factual branch")
        low, high = arms if factual[first - 1] == 0 else arms[::-1]
        strict = int(recovered(low, high, truth)) - int(recovered(high, low, truth))
        details.append(
            dict(
                q,
                first_answers={"temporal": low, "activity": high},
                factual_f1=arms[0]["f1"],
                last_observed={k: rq[k] for k in ("temporal", "activity")},
                first_f1=pair_value(low["f1"], high["f1"], prediction["first"]),
                last_f1=pair_value(rq["temporal"], rq["activity"], prediction["last"]),
                writer_sessions={"background": origins[j], "switched": new_origins[i]},
                strict_recovery_sign=strict,
                additional_credit=None,
            )
        )
    register(
        dest / "pair.json",
        {
            "forecast": file_hash(fp),
            "name": g["name"],
            "dialogue": sid,
            "end": n,
            "first": first,
            "first_write": file_hash(first_write(out, g)),
            "changed": trajectory_hashes(alt_cell),
            "control": ref["trajectories"][bits],
            "questions": details,
            "cost": measured_cost(alt_cell, changed, len(details)),
            "probabilities": prediction,
        },
    )


def stats(groups):
    if not groups:
        return {"groups": 0, "questions": 0}
    rows = [(g, r) for g in groups for r in g["questions"]]

    def aggregate(part):
        values = {
            pos: {k: mean(r[pos + "_f1"][k] for _, r in part) for k in ARMS}
            for pos in ("first", "last")
        }
        return {
            "questions": len(part),
            "f1": values,
            "background": mean(r["factual_f1"] for _, r in part),
            "first_changed_f1": sum(
                r["first_answers"]["temporal"]["f1"]
                != r["first_answers"]["activity"]["f1"]
                for _, r in part
            ),
            "strict_candidates": sum(bool(r["strict_recovery_sign"]) for _, r in part),
            "first_minus_last_increment": (
                values["first"]["credit"] - values["first"]["native"]
            )
            - (values["last"]["credit"] - values["last"]["native"]),
            "first_abs_probability_shift": mean(
                abs(
                    g["probabilities"]["first"]["credit"]
                    - g["probabilities"]["first"]["native"]
                )
                for g, _ in part
            ),
        }

    histories = {
        sid: aggregate([(g, r) for g, r in rows if g["dialogue"] == sid])
        for sid in sorted({g["dialogue"] for g in groups})
    }
    result = dict(
        aggregate(rows),
        groups=len(groups),
        by_history=histories,
        by_type={
            str(t): aggregate([(g, r) for g, r in rows if r["type"] == t])
            for t in sorted({r["type"] for _, r in rows})
        },
        evidence_gap=[
            min(g["end"] - g["first"] for g in groups),
            max(g["end"] - g["first"] for g in groups),
        ],
    )
    if len(histories) >= 2:
        result["first_minus_last_increment_95pct"] = paired_interval(
            [v["questions"] for v in histories.values()],
            [v["first_minus_last_increment"] for v in histories.values()],
        )
    return result


def measure(trainer, study, reg, train, specs):
    base = reg.parents[2]
    if reg != register_trial(base) or os.environ.get("MEMSKILL_BACKEND_ID") != BACKEND:
        raise ValueError("Wrong registered trial or physical backend")
    cross = base / "late-pairs/cross-window-prefix"
    manifest = read(cross / "manifest.json")
    versions = [s["version"] for s in specs]
    if versions != manifest["skills"] or manifest["protocol"] != file_hash(
        study / "protocol.json"
    ):
        raise ValueError("Checkpoint or skill bank changed")
    locked = read(reg)
    out = reg.parent
    template = {f"{w['dialogue']}_to{w['end']:02d}": w for w in manifest["endpoints"]}
    data = {c["sample_id"]: c for c in train}
    with (out / "measure.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for g in locked["groups"]:
            if (
                (cross / g["endpoint"] / "pair.json").exists()
                and first_write(out, g).is_file()
                and not (out / g["name"] / "pair.json").exists()
            ):
                one(trainer, base, reg, g, template[g["endpoint"]], data, versions)
                break
        reports = []
        for g in locked["groups"]:
            d = out / g["name"]
            p = d / "pair.json"
            if p.exists():
                v = read(p)
                if v["forecast"] != file_hash(d / "forecast.json") or [
                    q["original_index"] for q in v["questions"]
                ] != [q["original_index"] for q in g["questions"]]:
                    raise ValueError("Different first-evidence outcomes")
                reports.append(v)
        old = {k.split("/")[0] for k in locked["existing"]}
        fresh = [r for r in reports if r["name"].split("_from")[0] not in old]
        report = {
            "registration": file_hash(reg),
            "files": {
                r["name"]: file_hash(out / r["name"] / "pair.json") for r in reports
            },
            "all": stats(reports),
            "new_controls_and_switch": stats(fresh),
        }
        register(out / f"through-{len(reports):03d}.json", report)
        if len(reports) == len(locked["groups"]):
            register(out / "summary.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("--forecasts", action="store_true")
    args = p.parse_args()
    if args.forecasts:
        forecast_available(args.base)
    else:
        print(register_trial(args.base))
