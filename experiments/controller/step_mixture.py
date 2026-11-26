"""Bag the inner-dialogue step choice before measuring any later paired outcomes."""

import argparse
from itertools import product
import math
from pathlib import Path

import torch

from experiments.controller.late_pair_policy import STEPS, actor, correction, logistic, measured_gain, read
from experiments.controller.audit_source_compression import panel
from experiments.controller.source_compression import file_hash, register


def resampling_weights(calibration):
    gains = torch.tensor([calibration[str(s)]["dialogue_gains"] for s in STEPS], dtype=torch.float64)
    if (
        gains.ndim != 2
        or gains.shape[0] != len(STEPS)
        or gains.shape[1] not in (5, 6)
        or not torch.isfinite(gains).all()
    ):
        raise ValueError("Expected paired gains for five steps on five or six held dialogues")
    n = gains.shape[1]
    draws = torch.tensor(list(product(range(n), repeat=n)), dtype=torch.long)
    utilities = gains[:, draws].mean(dim=2)
    tied = utilities >= utilities.amax(dim=0) - 1e-12
    winners = tied.to(torch.int64).argmax(dim=0)
    return (torch.bincount(winners, minlength=len(STEPS)).double() / len(draws)).tolist()


def propensity(base, weights, offsets):
    if (
        len(weights) != len(STEPS)
        or any(x < 0 for x in weights)
        or not math.isclose(sum(weights), 1.0, abs_tol=1e-7)
    ):
        raise ValueError("Invalid step mixture")
    return sum(w * logistic(base + v) for w, v in zip(weights, offsets, strict=True)) / sum(weights)


def freeze(source, dest):
    root = dest / "step-mixture"
    plan = Path(__file__).with_name("protocols") / "late_step_mixture.json"
    if not (root / "plan.json").exists():
        root.mkdir(exist_ok=True)
        register(root / "plan.json", read(plan))
    if file_hash(root / "plan.json") != file_hash(plan):
        raise ValueError("Different step-mixture specification")
    locked = root / "forecast.json"
    inputs = {"plan": file_hash(root / "plan.json"), "base": file_hash(dest / "frozen.json")}
    if locked.exists():
        if read(locked)["inputs"] != inputs:
            raise ValueError("Previously registered mixture differs")
        return
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("Use a single CUDA lease for step selection")
    torch.cuda.set_per_process_memory_fraction(0.06)
    base = read(dest / "frozen.json")
    if base["inputs"]["features"] != file_hash(dest / "features.json") or base["inputs"][
        "early_features"
    ] != file_hash(source / "features.json"):
        raise ValueError("Source features changed")
    features = read(source / "features.json")["rows"] + read(dest / "features.json")["rows"]
    with torch.device("cuda"):
        weights = {k: resampling_weights(f["calibration"]) for k, f in base["models"].items()}
    rows = [
        {
            "dialogue": r["dialogue"],
            "session": r["session"],
            "offsets": [correction(base["models"][r["dialogue"]], r["features"], s) for s in STEPS],
        }
        for r in features
    ]
    if any((dest / "outcomes").glob("*/*/scores.json")):
        raise RuntimeError("Later path already measured before mixture freeze")
    register(locked, {"inputs": inputs, "weights": weights, "rows": rows})


def fit_for_deployment(previous, corpus, source, dest):
    root = dest / "step-mixture"
    plan = Path(__file__).with_name("protocols") / "late_full_fit.json"
    rule = read(root / "forecast.json")
    base = read(dest / "frozen.json")
    if rule["inputs"]["base"] != file_hash(dest / "frozen.json"):
        raise ValueError("Fold weights and registered source models differ")
    early = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Fit deployment weights on a CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    with torch.device("cuda"):
        training, receipts, _ = panel(previous, corpus, early, (1, 3), supplement=True)
        if receipts != base["training_evidence"]:
            raise ValueError("Training episodes differ from the cross-fitted experiment")
        fits = base["models"]
        ids = sorted(fits)
        gains = {
            str(s): {
                "dialogue_gains": [
                    measured_gain([r for r in training if r["dialogue"] == d], fits[d], s) for d in ids
                ]
            }
            for s in STEPS
        }
        full = actor(training)
        weights = resampling_weights(gains)
        for d, fit in fits.items():
            if set(fit["inner"]) != set(ids) - {d}:
                raise ValueError("An inner evaluation dialogue was used to fit its source model")
    record = {
        "inputs": {
            "specification": file_hash(plan),
            "crossfits": file_hash(root / "forecast.json"),
            "corpus": file_hash(corpus),
        },
        "training_dialogues": ids,
        "training_spans": len(training),
        "measured_links": sum(len(r["scores"]) for r in training),
        "inner_gains": gains,
        "weights": weights,
        "actor": full,
    }
    register(root / "deployment.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--full-fit", type=Path, nargs=2, metavar=("PREVIOUS", "CORPUS"))
    a = p.parse_args()
    if a.full_fit:
        fit_for_deployment(*a.full_fit, a.source, a.dest)
    else:
        freeze(a.source, a.dest)
