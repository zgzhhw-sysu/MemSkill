"""Audit logical token cost, propensity changes and the contribution of each policy stage."""

import argparse
from pathlib import Path
from statistics import mean

from experiments.controller.factorial_rollout import read
from experiments.controller.retrieval_competition import ARMS, mass
from experiments.controller.source_compression import file_hash, register


def diagnostics(windows):
    if not windows:
        return {"windows": 0, "questions": 0}
    total = sum(len(w["rows"]) for w in windows)
    costs = {
        arm: {"writer_tokens_per_window": 0.0, "reader_tokens_per_question": 0.0}
        for arm in ARMS
    }
    probs = {j: [] for j in (0, 1, 2)}
    for window in windows:
        states = {int(k): v for k, v in window["probabilities"].items()}
        for j, p in states.items():
            probs[j].append(p)
        partial = "focal_probabilities" in window
        for bits, usage in window["tokens"].items():
            for arm in ARMS:
                weight = mass(bits, states, arm) * (2 if partial else 1)
                costs[arm]["writer_tokens_per_window"] += (
                    weight * usage["executor"] / len(windows)
                )
                costs[arm]["reader_tokens_per_question"] += (
                    weight * usage["reader"] / total
                )
    metric = "ht" if "focal_probabilities" in windows[0] else "f1"
    values = {
        arm: mean(r[metric][arm] for w in windows for r in w["rows"]) for arm in ARMS
    }
    root = values["root_only"] - values["native"]
    last = values["last_only"] - values["native"]
    return {
        "windows": len(windows),
        "questions": total,
        "costs": costs,
        "stage_probabilities": {
            j: {arm: mean(p[arm] for p in states) for arm in ARMS}
            for j, states in probs.items()
            if states
        },
        "stage_metric": metric,
        "root_only_gain": root,
        "last_only_gain": last,
        "interaction": values["credit"] - values["native"] - root - last,
        "increment_over_raw_terminal_supervision": values["credit"]
        - values["return_only"],
    }


def audit(dest):
    fit = read(dest / "fit.json")
    calibration = read(dest / "calibration.json")
    if calibration["fit"] != file_hash(dest / "fit.json"):
        raise ValueError("Mismatched calibration")
    # The initial collector writes its numeric cutoff and never uses later paths for fitting.
    prior = {key.split("/")[0] for key in read(dest / "preexisting.json")["scores"]}
    reports = [read(path) for path in sorted((dest / "pairs").glob("*.json"))]
    through = dest / f"through-{len(reports):02d}.json"
    if not through.exists():
        return
    base = read(through)
    fresh = [w for w in reports if w["window"] not in prior]
    if base["fit"] != calibration["fit"] or any(
        w["inputs"]["fit"] != base["fit"] for w in reports
    ):
        raise ValueError("Paired inputs changed")
    group = base["prospective"]
    eligible = False
    if (
        group["windows"]
        and len(reports) == 52
        and group["questions"] >= 0.9 * base["cohort_questions"]
        and len(group["by_dialogue"]) == 6
    ):
        before = group["f1"]
        ht = group["ht"]
        eligible = (
            all(before["credit"] >= before[k] + 0.02 for k in ("native", "terminal"))
            and all(before["credit"] > before[k] for k in ("rate", "return_only"))
            and sum(
                x["f1"]["credit"] > x["f1"]["rate"]
                for x in group["by_dialogue"].values()
            )
            >= 4
            and all(ht["credit"] > ht[k] for k in ("native", "terminal"))
        )
    result = {
        "input": file_hash(through),
        "training_dialogues": len(fit["models"]["deployment"]["credit"]["donors"]),
        "calibration": diagnostics(calibration["windows"]),
        "all": diagnostics(reports),
        "prospective": diagnostics(fresh),
        "eligible_for_independent_development": eligible,
    }
    register(dest / f"diagnostics-{len(reports):02d}.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dest", type=Path)
    audit(**vars(p.parse_args()))
