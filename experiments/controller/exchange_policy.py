"""Predict span effects from measured natural exchanges in other training histories."""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.controller.evidence_transport_data import sources
from experiments.controller.exchange_questions import population, verify_question
from experiments.controller.exchange_status import panel_state, resolved, stage_jobs
from experiments.controller.factorial_rollout import read
from experiments.controller.passage_advantage import evaluate
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.passage_evidence_prefix import evaluate as evaluate_prefix
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited

ARMS = (
    "credit",
    "raw",
    "aux_raw",
    "joint",
    "reassigned",
    "parent",
    "native",
    "terminal",
)


def collect(reg, jobs):
    out = reg.parent
    by_session = {}
    receipts = {}
    counts = Counter()
    for job in jobs:
        name = job["id"] + ".json"
        q = out / "questions" / name
        row = read(q)
        verify_question(reg, job, row)
        session = f"{job['dialogue']}_s{job['session']}"
        entry = by_session.setdefault(
            session,
            {
                "history": job["dialogue"],
                "pivot": job["pivot"],
                "attempts": 0,
                "raw": 0.0,
                "credit": 0.0,
            },
        )
        entry["attempts"] += 1
        counts[row["status"]] += 1
        receipts[job["id"]] = {"question": file_hash(q)}
        if row["status"] == "accepted":
            p = out / "prepared" / name
            f = out / "outcomes" / name
            obs = read(f)
            if (
                obs["question"] != file_hash(q)
                or obs["inputs"] != file_hash(p)
                or read(p)["inputs"] != file_hash(q)
            ):
                raise ValueError("Missing physical provenance for accepted item")
            receipts[job["id"]]["outcome"] = file_hash(f)
            counts[obs["status"]] += 1
            counts["nonzero_credit"] += int(obs["credit"] != 0.0)
            counts["nonzero_raw"] += int(obs["raw"] != 0.0)
            entry["raw"] += obs["raw"]
            entry["credit"] += obs["credit"]
    for entry in by_session.values():
        for signal in ("raw", "credit"):
            entry[signal] /= entry["attempts"]
    return {"sessions": by_session, "counts": dict(counts), "receipts": receipts}


def predict(target, stage, donors, vectors):
    history = target.rsplit("_s", 1)[0]
    rows = sorted(
        k for k, v in donors.items() if v["history"] != history and v["pivot"] == stage
    )
    if not rows:
        return dict.fromkeys(("credit", "aux_raw", "joint", "reassigned"), 0.0)
    mass = Counter(donors[k]["history"] for k in rows)
    ref = vectors[target]
    similarities = []
    for k in rows:
        score = ref @ vectors[k].T
        similarities.append((score.amax(dim=0).mean() + score.amax(dim=1).mean()) / 2)
    group = ref.new_tensor([mass[donors[k]["history"]] for k in rows])
    weights = (torch.stack(similarities) / 0.05 - group.log()).softmax(dim=0)
    raw = ref.new_tensor([donors[k]["raw"] for k in rows])
    credit = ref.new_tensor([donors[k]["credit"] for k in rows])

    def adjust(v):
        return float((weights.dot(v) / 0.05).clamp(-2.0, 2.0))

    return {
        "credit": adjust(credit),
        "aux_raw": adjust(raw),
        "joint": adjust(raw) + adjust(credit),
        "reassigned": adjust(torch.roll(credit, 1)),
    }


def freeze(base, reg, stage, jobs):
    out = reg.parent / "policy" / stage
    local = out / "forecast.json"
    long = out / "long-prefix/forecast.json"
    groups = base / "source-compression"
    evidence = groups / "witness-passages/evidence-advantage"
    old = evidence / "group/forecast.json"
    prefix = evidence / "group/long-prefix/forecast.json"
    long.parent.mkdir(parents=True, exist_ok=True)
    fit = out / "fit.json"
    if not fit.exists():
        register(fit, {"registration": file_hash(reg), **collect(reg, jobs)})
    a, b = read(old), read(prefix)
    hashes = {
        "registration": file_hash(reg),
        "fit": file_hash(fit),
        "parent": file_hash(old),
        "long_parent": file_hash(prefix),
        "manifest": b["inputs"]["manifest"],
    }
    if local.exists() and long.exists():
        if read(local)["inputs"] != hashes or read(long)["inputs"] != {
            **hashes,
            "local": file_hash(local),
        }:
            raise ValueError("Numeric forecast inputs changed")
        return local, long
    known = sources(groups / "evidence-transport", "cuda")
    more = sources(evidence / "prefix-sources", "cuda")
    if (
        known.keys() & more.keys()
        or a["sources"]
        != {k: {"text": v[2], "vectors": v[3]} for k, v in known.items()}
        or {k: v[3] for k, v in more.items()}
        != read(evidence / "prefix-sources/complete.json")["files"]
    ):
        raise ValueError("Changed frozen source vectors")
    all_sources = known | more
    if (len(a["sources"]), len(b["sources"])) != (128, 99) or any(
        all_sources[k][2] != v for k, v in b["sources"].items()
    ):
        raise ValueError("Missing local or long-prefix sources")
    tensors = {k: F.normalize(v[1].double()) for k, v in all_sources.items()}
    donors = read(fit)["sessions"]
    stages = defaultdict(dict)
    extra = {}
    for k in sorted(all_sources):
        with limited():
            for j in (0, 1):
                offsets = predict(k, j, donors, tensors)
                if k in a["sources"]:
                    stages[str(j)][k] = {**offsets, "parent": a["offsets"][k]["parent"]}
                if j == 1 and k in b["sources"]:
                    extra[k] = {**offsets, "parent": b["offsets"][k]["parent"]}
    register(
        local,
        {"inputs": hashes, "sources": a["sources"], "by_stage": stages, "arms": ARMS},
    )
    register(
        long,
        {
            "inputs": {**hashes, "local": file_hash(local)},
            "sources": b["sources"],
            "offsets": extra,
            "arms": ARMS,
        },
    )
    return local, long


def run(base, study, corpus):
    reg = population(base, study, corpus)
    out = reg.parent
    panels = panel_state(out)
    for stage, jobs in stage_jobs(read(reg)["jobs"]).items():
        if not all(resolved(out, j) for j in jobs):
            continue
        folder = out / "policy" / stage
        current = folder / "scored-state.json"
        if current.is_file() and read(current) == panels:
            continue
        local, prefix = freeze(base, reg, stage, jobs)
        evaluate(
            base,
            study,
            corpus,
            local,
            final_stage="full",
            gate_controls=("raw", "aux_raw", "reassigned"),
        )
        evaluate_prefix(base, prefix)
        pending = folder / ".scored-state.tmp"

        pending.write_text(json.dumps(panels))
        os.replace(pending, current)
        print(stage, read(folder / "fit.json")["counts"], flush=True)
    full = out / "policy/full"
    if all(
        (full / n).is_file()
        for n in ("decision.json", "donor/summary.json", "long-prefix/summary.json")
    ):
        (out / "policy.complete").touch()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    gpu()
    run(a.base, a.study, a.corpus)
