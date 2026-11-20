"""Apply the locked passage score to a chronological multi-session endpoint."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.controller.audit_cross_window import verified_decision
from experiments.controller.evidence_transport_data import sources
from experiments.controller.factorial_rollout import read
from experiments.controller.late_credit_kernel import support
from experiments.controller.passage_advantage import summarize_panel
from experiments.controller.passage_evidence_actor import (
    ARMS,
    case_scores,
    gpu,
    score_sessions,
)
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import (
    anchor_logits,
    cases,
    completed,
    limited,
    match_all,
    offsets,
)
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "passage_evidence_prefix.json"


def extend(base, study, corpus, stage):
    src = base / "source-compression/witness-passages"
    trial = src / "evidence-advantage"
    local = trial / stage / "forecast.json"
    if not local.exists():
        return None
    directory = trial / stage / "long-prefix"
    directory.mkdir(exist_ok=True)
    dest = directory / "forecast.json"
    manifest = base / "late-pairs/cross-window-prefix/manifest.json"
    encoder = trial / "prefix-sources/complete.json"
    identity = {
        "policy": file_hash(local),
        "manifest": file_hash(manifest),
        "source_extension": file_hash(encoder),
        "plan": file_hash(PLAN),
    }
    match_file = directory / "case-matches.json"
    if dest.exists() and match_file.exists():
        if read(dest)["inputs"] != identity:
            raise ValueError("Changed long-prefix forecast")
        return dest
    parent = read(src / stage / "forecast.json")
    known = read(local)
    inv = read(src / "registration.json")
    if (
        known["inputs"]["parent"] != file_hash(src / stage / "forecast.json")
        or known["inputs"]["ranks"] != file_hash(src / "registration.json")
        or inv["inputs"]["corpus"] != file_hash(corpus)
    ):
        raise ValueError("Changed passage actor inputs")
    turns = sources(src.parent / "evidence-transport", "cuda")
    more = sources(trial / "prefix-sources", "cuda")
    if (
        turns.keys() & more.keys()
        or {k: v[3] for k, v in more.items()} != read(encoder)["files"]
    ):
        raise ValueError("Prefix source files overlap or changed")
    turns = dict(sorted((turns | more).items()))
    targets = {f"{v['dialogue']}_s{v['end']}" for v in read(manifest)["endpoints"]}
    if len(targets) != 99 or not targets <= turns.keys():
        raise ValueError("Missing full-prefix target")
    with np.load(src / "passages.npz", allow_pickle=False) as arr:
        if str(arr["texts"]) != digest(inv["texts"]) or read(src / "encoder.json")[
            "vectors"
        ] != file_hash(src / "passages.npz"):
            raise ValueError("Passage vectors changed")
        bank = torch.tensor(arr["vectors"], device="cuda", dtype=torch.float64)
    rec = inv["records"]
    active = cases(rec, completed(src, rec) if stage == "witness" else None)
    training = set(read(study / "protocol.json")["train"])
    data = {v["sample_id"]: v for v in read(corpus) if v["sample_id"] in training}
    old = dict(parent["offsets"])
    packets = match_all(active, bank, turns)
    for i, name in enumerate(turns):
        if name not in old:
            with limited():
                old[name] = offsets(active, packets, turns, name.rsplit("_s", 1)[0], i)[
                    0
                ]
    gated, inventory = case_scores(active, rec, bank, turns, data)
    with limited():
        updated, detail = score_sessions(turns, gated, old)
    if inventory != known["inventory"] or any(
        abs(val - updated[n][k]) > 1e-9
        for n, z in known["offsets"].items()
        for k, val in z.items()
    ):
        raise ValueError("Full-prefix extension failed to reproduce the local policy")
    register(
        dest,
        {
            "inputs": identity,
            "sources": {k: turns[k][2] for k in sorted(targets)},
            "offsets": {k: updated[k] for k in sorted(targets)},
        },
    )
    register(
        match_file,
        {
            "forecast": file_hash(dest),
            "matches": {n: detail.get(n, []) for n in sorted(targets)},
        },
    )
    return dest


def evaluate(base, path):
    late = base / "late-pairs"
    root = late / "cross-window-prefix"
    manifest = root / "manifest.json"
    reg = read(manifest)
    specs = reg["endpoints"]
    frozen = read(path)
    arms = tuple(frozen.get("arms", ARMS))
    split = read(late / "selection-labels/split.json")["windows"]
    local = {
        (name.rsplit("_s", 1)[0], i)
        for group in split.values()
        for name, ids in group.items()
        for i in ids
    }
    population = [
        (w["dialogue"], q["original_index"]) for w in specs for q in w["questions"]
    ]
    if (
        len(specs) != 99
        or len(local) != 466
        or len(set(population)) != len(population)
        or len(population) != 185
        or local.intersection(population)
        or reg["baseline"] != file_hash(late / "frozen.json")
    ):
        raise ValueError("Long-prefix split mismatch")
    cutoff = path.stat().st_mtime_ns
    branches = {
        str(f.relative_to(root)): file_hash(f)
        for f in root.glob("conv-*_to*/*/scores.json")
        if set(f.parent.name) <= set("01") and f.stat().st_mtime_ns <= cutoff
    }
    register(
        path.parent / "exposure.json",
        {"forecast": file_hash(path), "branches": branches, "cutoff_ns": cutoff},
    )
    old = {n.split("/")[0] for n in branches}
    weights = read(late / "frozen.json")["terminal"]
    raw = read(
        base / "source-compression/capped-source-credit/policy/joint-repaired/fit.json"
    )["observations"]
    models = {sid: support(raw, sid, 1, "cuda")[0] for sid in weights}
    panels = []
    for w in specs:
        sid = w["dialogue"]
        n = w["end"]
        name = f"{sid}_to{n:02d}"
        session = f"{sid}_s{n}"
        folder = root / name
        marker = folder / "pair.json"
        if not marker.exists():
            continue
        r = read(marker)
        if (
            r["manifest"] != frozen["inputs"]["manifest"]
            or [
                {k: q[k] for k in ("original_index", "type", "sessions")}
                for q in r["questions"]
            ]
            != w["questions"]
        ):
            raise ValueError("Different paired prefix endpoints")
        d = verified_decision(folder, w, r, reg["skills"])
        first = "".join(map(str, [*w["background"], 0]))
        (trace,) = (folder / first / "traces").glob("*.jsonl")
        spans = [
            e
            for l in trace.read_bytes().splitlines()
            if (e := json.loads(l))["kind"] == "span"
        ]
        if digest(spans[-1]["source"]) != frozen["sources"][session]:
            raise ValueError("Policy saw a different source")
        with limited():
            z, term, anchor = anchor_logits(d, models[sid], weights[sid])
            logits = {k: anchor + v for k, v in frozen["offsets"][session].items()}
            logits.update(native=z, terminal=term, raw=anchor)
            p = {k: float(logits[k].sigmoid()) for k in arms}
        if any(
            abs(p[k] - r["probabilities"][k]) > 1e-10 for k in ("native", "terminal")
        ):
            raise ValueError("Native or terminal prefix propensity mismatch")
        ends = []
        for i, a in enumerate(("temporal", "activity")):
            bits = "".join(map(str, [*w["background"], i]))
            scored = read(folder / bits / "scores.json")["answers"]
            if [v["qa_index"] for v in scored] != list(range(len(w["questions"]))) or [
                v["f1"] for v in scored
            ] != [q[a] for q in r["questions"]]:
                raise ValueError("Prefix answers differ")
            ends.append(bits)
        rows = []
        endpoints = []
        for q in r["questions"]:
            lo, hi = q["temporal"], q["activity"]
            endpoints.append([lo, hi])
            vals = {k: lo + t * (hi - lo) for k, t in p.items()}
            if any(
                not math.isclose(vals[k], q["focal"][k], abs_tol=1e-10)
                for k in ("native", "terminal")
            ):
                raise ValueError("Prefix F1 changed")
            rows.append(
                {
                    "index": q["original_index"],
                    "type": q["type"],
                    "sessions": q["sessions"],
                    "f1": vals,
                }
            )
        advantage = sum(hi - lo for lo, hi in endpoints) / len(endpoints)
        costs = {
            b: {
                op: sum(
                    r["cost"][b]["model"][op][k]
                    for k in ("input_tokens", "output_tokens")
                )
                for op in ("reader", "executor")
            }
            for b in ends
        }
        panel = {
            "inputs": {"pair": file_hash(marker), "forecast": file_hash(path)},
            "window": name,
            "dialogue": sid,
            "end": n,
            "probabilities": {n: p},
            "focal_probabilities": p,
            "pivot": 1,
            "base": d["base_logit"],
            "endpoints": endpoints,
            "rows": rows,
            "expected_tokens": {
                k: {
                    op: (1 - t) * costs[ends[0]][op] + t * costs[ends[1]][op]
                    for op in ("reader", "executor")
                }
                for k, t in p.items()
            },
            "alignment": {
                f"{k}-{c}": [[p[k] - p[c], advantage]]
                for k in ("credit", "parent")
                for c in ("native", "raw")
            },
        }
        register(path.parent / (name + ".json"), panel)
        panels.append(panel)
    if panels:
        strict = [w for w in panels if w["window"] not in old]
        with limited():
            result = {
                "forecast": file_hash(path),
                "all": summarize_panel(panels, False, arms),
                "strict": summarize_panel(strict, False, arms),
            }
        register(path.parent / f"through-{len(panels):02d}.json", result)
        if len(panels) == 99:
            register(path.parent / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "study", "corpus"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    gpu()
    for stage in ("group", "witness"):
        path = extend(args.base, args.study, args.corpus, stage)
        if path:
            evaluate(args.base, path)
            print(path.parent, flush=True)
