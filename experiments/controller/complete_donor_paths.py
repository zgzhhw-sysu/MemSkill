"""Physically measure complementary writer trajectories in the randomized donor block only."""

import fcntl
import json
import math
import os
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller.factorial_rollout import (
    CELLS,
    load_cell,
    measure_cell,
    read,
)
from experiments.controller.late_pair_trial import paths
from experiments.controller.paired_grounding import window
from experiments.controller.retrieval_competition import budgets, mass
from experiments.controller.root_write_repairs import seed_controls
from experiments.controller.source_compression import file_hash, register
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "donor_factorial_completion.json"


def complement(pivot, bg):
    observed = paths(pivot, bg)
    return observed, [bits for bits in CELLS if bits not in observed]


def returns(cases, probabilities, assigned, pivot, bg, i):
    focal = probabilities[0 if pivot == 0 else 1 + bg]
    low, high = [cases[bits][i] for bits in assigned]
    result = {}
    for arm, p in focal.items():
        both = sum(mass(b, probabilities, arm) * cases[b][i] for b in CELLS)
        ht = 2 * sum(mass(b, probabilities, arm) * cases[b][i] for b in assigned)
        result[arm] = {"exact": both, "ht": ht, "focal": low + p * (high - low)}
    return result


def decision_context(span, decision, expected_source, bit):
    skills = span["selected_skills"]
    action = ("capture_temporal_context", "capture_activity_details")[bit]
    if (
        digest(span["source"]) != expected_source
        or len(skills) != 3
        or skills[-1]["name"] != action
    ):
        raise ValueError(
            "The actual source or ordered three-skill intervention differs from the forecast"
        )
    return digest(
        [
            decision["state"],
            decision["base_logit"],
            decision["plain"],
            decision["grounded"],
            span["retrieved"],
            span["refs"],
            skills[:-1],
        ]
    )


def score(trainer, convo, original, extra, w, pred, weights, versions):
    observed, missing = complement(w["pivot"], w["background"])
    pair = read(original / "pair.json")
    path_hash = {}
    cases = {}
    ps = {}
    prefixes = {}
    contexts = {}
    usage = {}
    if [r["original_index"] for r in pair["questions"]] != w["questions"] or pair[
        "paths"
    ] != [list(b) for b in observed]:
        raise ValueError("Wrong donor question order or assignment")
    for bits in CELLS:
        a, b = bits
        k = f"{a}{b}"
        cell = (extra if bits in missing else original) / k
        ds, prefix, y = load_cell(trainer, convo, cell, bits, versions)
        if prefixes.setdefault(a, prefix) != prefix:
            raise ValueError("Writer prefix changed before terminal intervention")
        (trace,) = (cell / "traces").glob("*.jsonl")
        spans = [
            e
            for e in map(json.loads, trace.read_bytes().splitlines())
            if e["kind"] == "span"
        ]
        for j, d, span, src, bit in zip(
            (0, 1 + a), ds, spans, pred["sources"], bits, strict=True
        ):
            context = decision_context(span, d, src, bit)
            if contexts.setdefault(j, context) != context:
                raise ValueError(
                    "Factual predecision context differs between matched paths"
                )
            z = torch.tensor(
                d["base_logit"], device=trainer.device, dtype=torch.float64
            )
            vec = z.new_tensor(d["state"])
            wt = z.new_tensor(weights)
            probs = {
                key: float((z + v).sigmoid()) for key, v in pred["offsets"][j].items()
            }
            probs.update(
                native=float(z.sigmoid()),
                terminal=float(
                    (z + wt[0] + wt[1:] @ F.normalize(vec, dim=0)).sigmoid()
                ),
            )
            if ps.setdefault(j, probs) != probs:
                raise ValueError("The same decision context has different propensities")
        path_hash[k] = {
            f.name: file_hash(f)
            for f in (
                cell / "scores.json",
                cell / "decisions.json",
                cell / "usage.jsonl",
                trace,
            )
        }
        usage[k] = budgets(cell.parent, k)
        cases[bits] = [y[i] for i in range(len(w["questions"]))]
    rows = []
    for i, r in enumerate(pair["questions"]):
        metrics = returns(cases, ps, observed, w["pivot"], w["background"], i)
        if [cases[k][i] for k in observed] != [r["temporal"], r["activity"]] or any(
            not math.isclose(metrics[arm][m], r[m][arm], abs_tol=1e-10)
            for arm in ("native", "terminal")
            for m in ("focal", "ht")
        ):
            raise ValueError(
                "Original randomized estimates or raw answers failed reproduction"
            )
        rows.append(
            {"index": r["original_index"], "type": r["type"], "returns": metrics}
        )
    return {
        "window": original.name,
        "dialogue": w["dialogue"],
        "probabilities": ps,
        "contexts": contexts,
        "paths": path_hash,
        "costs": usage,
        "rows": rows,
    }


def run(trainer, locked, data, manifest, limit):
    late = locked.parent
    base = late.parent
    cohort = read(late / "cohort.json")["assignments"]
    donors = [w for w in cohort if w["start"] <= 11]
    split = read(late / "selection-labels/split.json")["windows"]["donor"]
    if (
        len(donors) != 12
        or sum(len(w["questions"]) for w in donors) != 88
        or {f"{w['dialogue']}_s{w['start']}": w["questions"] for w in donors} != split
    ):
        raise ValueError("Different randomized donor population")
    dest = late / "donor-factorial"
    dest.mkdir(exist_ok=True)
    register(
        dest / "manifest.json",
        {
            "plan": file_hash(PLAN),
            "randomized": file_hash(late / "outcomes/manifest.json"),
            "split": file_hash(late / "selection-labels/split.json"),
            "windows": donors,
        },
    )
    cache = dest / "responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    fitted = read(locked)["terminal"]
    completed = []
    added = 0
    for w in donors:
        sid = w["dialogue"]
        start = w["start"]
        name = f"{sid}_s{start}"
        original = late / "outcomes" / name
        extra = dest / name
        forecast = (
            base
            / "source-compression/evidence-transport/pilot/predictions"
            / (name + ".json")
        )
        if not (original / "pair.json").exists() or not forecast.exists():
            continue
        extra.mkdir(exist_ok=True)
        inputs = {
            "pair": file_hash(original / "pair.json"),
            "forecast": file_hash(forecast),
        }
        observed, missing = complement(w["pivot"], w["background"])
        with (extra / "measure.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = extra / "complete.json"
            if not result.exists():
                convo = window(data[sid], start, 2)
                if [q["original_index"] for q in convo["qa"]] != w["questions"]:
                    raise ValueError("Incorrect donor questions")
                register(extra / "registration.json", inputs)
                for a, b in observed:
                    seed_controls(
                        original / f"{a}{b}", late / "outcomes/responses", cache
                    )
                for bits in missing:
                    cell = extra / "".join(map(str, bits))
                    if cell.exists():
                        continue
                    if added >= limit:
                        return
                    measure_cell(
                        trainer, convo, extra, bits, manifest["skill_versions"]
                    )
                    added += 1
                r = score(
                    trainer,
                    convo,
                    original,
                    extra,
                    w,
                    read(forecast),
                    fitted[sid],
                    manifest["skill_versions"],
                )
                register(result, dict(r, inputs=inputs))
            report = read(result)
            if report["inputs"] != inputs:
                raise ValueError("Changed assigned pair or frozen decision")
            completed.append(report)
        rows = [r for g in completed for r in g["rows"]]
        means = {
            k: {
                m: mean(r["returns"][k][m] for r in rows)
                for m in ("exact", "ht", "focal")
            }
            for k in rows[0]["returns"]
        }
        summary = {
            "windows": [g["window"] for g in completed],
            "questions": len(rows),
            "mean": means,
        }
        # Archive membership as well as count when original random pairs finish out of order.
        register(
            dest / f"through-{len(completed):02d}-{digest(summary)[:16]}.json", summary
        )
        if len(completed) == 12:
            register(dest / "summary.json", summary)
