"""Calibrate span credit against native choices on inner held dialogues and lock a later training panel."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
from statistics import mean, stdev
import time

import torch

from experiments.controller.audit_source_compression import check_forecasts, panel
from experiments.controller.conditional_credit import read
from experiments.controller.paired_grounding import window
from experiments.controller.paired_policy import _shift_for_mean_probability
from experiments.controller.source_compression import file_hash, fit_actor, measure_source, register
from src.data_processing.locomo import LoCoMoProcessor
from src.rag_utils import init_context_model, set_retriever_flash_attn


STEPS = (0.0, 0.25, 0.5, 1.0, 2.0)
PLAN = Path(__file__).with_name("protocols") / "late_pairs.json"


def assignments(data):
    rows = []
    for sid, c in sorted(data.items()):
        n = sum(k.startswith("session_") and not k.endswith("_date_time") for k in c["conversation"])
        rng = random.Random("memskill-late-pairs-v1/" + sid)
        count = 0
        for start in range(9, n, 2):
            qa = window(c, start, 2)["qa"]
            if not qa:
                continue
            if count % 4 == 0:
                block = [(0, 0), (0, 1), (1, 0), (1, 1)]
                rng.shuffle(block)
            pivot, background = block[count % 4]
            rows.append(
                dict(
                    dialogue=sid,
                    start=start,
                    pivot=pivot,
                    background=background,
                    questions=[q["original_index"] for q in qa],
                    types=[int(q["category"]) for q in qa],
                )
            )
            count += 1
    return sorted(rows, key=lambda r: (r["start"], r["dialogue"]))


def cohort(study, corpus, dest):
    protocol = read(study / "protocol.json")
    if (
        file_hash(corpus) != protocol["input_sha256"]
        or not protocol.get("equal_sources")
        or not protocol.get("deployment_writer")
        or protocol["rule"] != 2
    ):
        raise ValueError("Unexpected historical training protocol")
    data = {c["sample_id"]: c for c in read(corpus) if c["sample_id"] in protocol["train"]}
    if len(data) != 6 or set(data) & set(protocol["validation"] + protocol["test"]):
        raise ValueError("Expected six disjoint training histories")
    dest.mkdir(exist_ok=True, parents=True)
    plan = dest / "plan.json"
    register(plan, read(PLAN))
    if file_hash(plan) != file_hash(PLAN):
        raise ValueError("The registered plan changed")
    assigned = assignments(data)
    register(
        dest / "cohort.json",
        {"corpus": file_hash(corpus), "plan": file_hash(plan), "assignments": assigned},
    )
    return protocol, data, assigned


def features(study, corpus, dest):
    protocol, data, assigned = cohort(study, corpus, dest)
    output = dest / "features.json"
    identity = {"cohort": file_hash(dest / "cohort.json"), "retriever": protocol["retriever"]}
    if output.exists():
        if read(output)["inputs"] != identity:
            raise ValueError("Different source registration")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a GPU lease for Qwen")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.76 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise RuntimeError("Insufficient headroom for Qwen")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, model = init_context_model(protocol["retriever"])
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("Source encoder offloaded from CUDA")
    proc = LoCoMoProcessor(chunk_mode="full-session")
    rows = []
    for w in assigned:
        sid = w["dialogue"]
        c = data[sid]["conversation"]
        for n in (w["start"], w["start"] + 1):
            k = f"session_{n}"
            (src,) = proc._extract_session_chunks(c[k], c[k + "_date_time"])
            rows.append(dict(dialogue=sid, session=n, **measure_source(src, protocol["retriever"])))
    register(
        output,
        {"inputs": identity, "rows": rows, "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20},
    )


def actor(rows):
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float64)
    mu, scale, weights = fit_actor(x, x.new_tensor([r["signed"] for r in rows]))
    base = x.new_tensor([r["base_logit"] for r in rows])
    slope = weights[1:]
    direction = (((x - mu) / scale) @ slope).clamp(-2.0, 2.0)
    shifts = {str(v): _shift_for_mean_probability(base + v * direction, -v * direction) for v in STEPS}
    for v in STEPS:
        if not math.isclose(
            float((base + v * direction + shifts[str(v)]).sigmoid().mean()),
            float(base.sigmoid().mean()),
            abs_tol=1e-10,
        ):
            raise ValueError("Calibration changed the fitting states' mean native probability")
    return dict(center=mu.tolist(), scale=scale.tolist(), slope=slope.tolist(), shifts=shifts)


def correction(fit, xs, step):
    z = sum(
        (x - m) / s * w for x, m, s, w in zip(xs, fit["center"], fit["scale"], fit["slope"], strict=True)
    )
    return step * max(-2.0, min(2.0, z)) + fit["shifts"][str(step)]


def logistic(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def select_step(gains):
    reports = {}
    best = 0.0
    criterion = 0.0
    for step in STEPS:
        v = gains[step]
        if len(v) != 5:
            raise ValueError("Step selection needs exactly five inner held dialogues")
        robust = mean(v) - stdev(v) / math.sqrt(len(v))
        reports[str(step)] = {"dialogue_gains": v, "mean_minus_se": robust}
        if sum(x > 1e-12 for x in v) >= 3 and robust > criterion + 1e-12:
            best, criterion = step, robust
    return best, reports


def measured_gain(rows, model, step):
    effects = []
    for r in rows:
        if r["scores"]:
            contrast = mean(q["activity"] - q["temporal"] for q in r["scores"])
            base = r["base_logit"]
            effects.append(
                (logistic(base + correction(model, r["features"], step)) - logistic(base)) * contrast
            )
    return mean(effects)


def calibrated_models(training):
    ids = sorted({r["dialogue"] for r in training})
    cache = {}

    def fit(excluded):
        key = tuple(sorted(excluded))
        if key not in cache:
            then = time.monotonic()
            cache[key] = actor([r for r in training if r["dialogue"] not in key])
            if torch.empty(0).is_cuda:
                torch.cuda.synchronize()
                time.sleep((time.monotonic() - then) * (1 / 0.3 - 1))
        return cache[key]

    models = {}
    for held in ids:
        inner = [d for d in ids if d != held]
        gains = defaultdict(list)
        for d in inner:
            f = fit((d, held))
            observed = [r for r in training if r["dialogue"] == d]
            for step in STEPS:
                gains[step].append(measured_gain(observed, f, step))
        step, reports = select_step(gains)
        models[held] = dict(fit((held,)), step=step, inner=inner, calibration=reports)
    return models


def freeze(previous, current, corpus, source, dest):
    _, _, assigned = cohort(current, corpus, dest)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("Fit policies on a single CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    comp = source.parent / "write-components/frozen.json"
    inputs = {
        name: file_hash(p)
        for name, p in {
            "cohort": dest / "cohort.json",
            "features": dest / "features.json",
            "parent": source / "frozen.json",
            "early_features": source / "features.json",
            "components": comp,
        }.items()
    }
    locked = dest / "frozen.json"
    if locked.exists():
        if read(locked)["inputs"] != inputs:
            raise ValueError("Frozen inputs changed")
        return
    if any((dest / "outcomes").glob("*/*/scores.json")):
        raise RuntimeError("A later training path was scored before coefficients were locked")
    parent = read(source / "frozen.json")
    components = read(comp)
    if (
        parent["inputs"]["features.json"] != inputs["early_features"]
        or components["inputs"]["parent"] != inputs["parent"]
        or read(dest / "features.json")["inputs"]["cohort"] != inputs["cohort"]
    ):
        raise ValueError("Mismatched parent studies")
    early = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    late = read(dest / "features.json")["rows"]
    if len(late) != 2 * len(assigned) or {(r["dialogue"], r["session"]) for r in late} != {
        (w["dialogue"], s) for w in assigned for s in (w["start"], w["start"] + 1)
    }:
        raise ValueError("Incomplete later source features")
    with torch.device("cuda"):
        training, receipts, backend = panel(previous, corpus, early, (1, 3), supplement=True)
        if receipts != parent["prior_hashes"] or backend != parent["backend"]:
            raise ValueError("Original paired training outcomes changed")
        check_forecasts(early, parent)
        models = calibrated_models(training)
    forecasts = []
    for r in [*early.values(), *late]:
        sid = r["dialogue"]
        feat = r["features"]
        m = models[sid]
        entry = {
            "dialogue": sid,
            "session": r["session"],
            "source_sha256": r["source_sha256"],
            "residual": correction(m, feat, m["step"]),
            "unit_residual": correction(m, feat, 1.0),
        }
        for k, f in (
            ("source", parent["fits"][sid]),
            ("component", components["models"]["filtered"]["fits"][sid]),
        ):
            entry[k] = logistic(
                f["actor"][0]
                + sum(
                    (x - mu) / s * w
                    for x, mu, s, w in zip(feat, f["center"], f["scale"], f["actor"][1:], strict=True)
                )
            )
        forecasts.append(entry)
    register(
        locked,
        {
            "inputs": inputs,
            "models": models,
            "forecasts": forecasts,
            "training_evidence": receipts,
            "terminal": {k: v["terminal"] for k, v in parent["fits"].items()},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("cohort", "features", "freeze"))
    for name in ("previous", "current", "corpus", "source", "dest"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if a.phase == "freeze":
        freeze(a.previous, a.current, a.corpus, a.source, a.dest)
    elif a.phase == "features":
        features(a.current, a.corpus, a.dest)
    else:
        cohort(a.current, a.corpus, a.dest)
