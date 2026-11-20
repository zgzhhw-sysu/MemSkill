"""Randomized paired interventions for later spans in the six LoCoMo training histories."""

from collections import defaultdict
import fcntl
import hashlib
from itertools import product
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller.factorial_rollout import measure_cell, path_mass
from experiments.controller.late_pair_policy import assignments, logistic, read
from experiments.controller.paired_grounding import window
from experiments.controller.source_compression import file_hash, register
from experiments.controller.step_mixture import propensity


def paths(pivot, background):
    if pivot not in (0, 1) or background not in (0, 1):
        raise ValueError("Invalid paired randomization")
    return [(a, background) if pivot == 0 else (background, a) for a in (0, 1)]


def path_policies(decisions, forecast, terminal_weights, bits):
    probabilities, masses = {}, {}
    w = torch.as_tensor(terminal_weights, dtype=torch.float64)
    for i, (dec, pred) in enumerate(zip(decisions, forecast, strict=True)):
        alternate = forecast[1 - i]
        base = dec["base_logit"]
        off = float(w[0] + w[1:] @ F.normalize(w.new_tensor(dec["state"]), dim=0))
        current = {
            "native": logistic(base),
            "terminal": logistic(base + off),
            "source": pred["source"],
            "component": pred["component"],
            "calibrated": logistic(base + pred["residual"]),
            "unit": logistic(base + pred["unit_residual"]),
            "permuted": logistic(base + alternate["residual"]),
            "unit_permuted": logistic(base + alternate["unit_residual"]),
            "temporal": 0.0,
            "activity": 1.0,
        }
        if "prefix_offsets" in pred:
            node = 0 if i == 0 else 1 + bits[0]
            current.update({key: logistic(base + values[node]) for key, values in pred["prefix_offsets"].items()})
        if "offsets" in pred:
            current["mixture"] = propensity(base, pred["weights"], pred["offsets"])
            current["mixture_permuted"] = propensity(base, pred["weights"], alternate["offsets"])
        for k, p in list(current.items()):
            current[k + "_greedy"] = float(p >= 0.5) if k not in ("source", "component") else float(p > 0.5)
        for k, p in current.items():
            probabilities.setdefault(k, []).append(p)
    for k, probs in probabilities.items():
        masses[k] = path_mass(bits, *probs)
    # Both within-window permutations, including the identity, preserve the two proposed corrections.
    permutations = [("permuted", "calibrated"), ("unit_permuted", "unit")]
    if "mixture" in probabilities:
        permutations.append(("mixture_permuted", "mixture"))
    for key, reference in permutations:
        for suffix in ("", "_greedy"):
            arm = key + suffix
            original = reference + suffix
            masses[arm] = (masses[arm] + masses[original]) / 2
            probabilities[arm] = [
                (a + b) / 2 for a, b in zip(probabilities[arm], probabilities[original], strict=True)
            ]
    return probabilities, masses


def summarize_pair(w, cells, preds, weights):
    chosen = paths(w["pivot"], w["background"])
    if set(cells) != set(chosen):
        raise ValueError("Measured paths do not match the registered randomization")

    def state(d):
        return {k: d[k] for k in ("state", "base_logit", "plain", "grounded")}

    off, on = [cells[k] for k in chosen]
    j = w["pivot"]
    if (
        state(off[0][0]) != state(on[0][0])
        or state(off[0][j]) != state(on[0][j])
        or (j == 1 and off[1] != on[1])
        or set(off[2]) != set(on[2])
        or set(off[2]) != set(range(len(w["questions"])))
    ):
        raise ValueError("Observed decisions, factual prefix or question population differ")
    endpoints = [path_policies(cells[k][0], preds, weights, k) for k in chosen]
    focus = {k: p[j] for k, p in endpoints[0][0].items()}
    if any(not math.isclose(p, endpoints[1][0][k][j], abs_tol=1e-12) for k, p in focus.items()):
        raise ValueError("The focal propensity depends on its unobserved action")
    first_mass, second_mass = [e[1] for e in endpoints]
    mass = {k: 2 * (m + second_mass[k]) for k, m in first_mass.items()}
    rows = []
    for i, (orig, typ) in enumerate(zip(w["questions"], w["types"], strict=True)):
        low, high = off[2][i], on[2][i]
        rows.append(
            {
                "original_index": orig,
                "type": typ,
                "temporal": low,
                "activity": high,
                "focal": {k: low + p * (high - low) for k, p in focus.items()},
                "ht": {k: 2 * (first_mass[k] * low + second_mass[k] * high) for k in focus},
            }
        )
    return {
        "dialogue": w["dialogue"],
        "start": w["start"],
        "pivot": j,
        "background": w["background"],
        "paths": [list(k) for k in chosen],
        "focal_probabilities": focus,
        "sampling_mass": mass,
        "questions": rows,
    }


def aggregate(reports, *, complete=False):
    if not reports:
        raise ValueError("No completed randomized windows")
    allrows = [(w, r) for w in reports for r in w["questions"]]

    def summary(group):
        n = len(group)
        arms = group[0][1]["ht"]
        out = {}
        for k in arms:
            mass = sum(w["sampling_mass"][k] for w, r in group)
            numerator = sum(r["ht"][k] for w, r in group)
            out[k] = {
                "focal": mean(r["focal"][k] for w, r in group),
                "ht": numerator / n,
                "mass": mass / n,
                "self_normalized": numerator / mass if mass > 0 else None,
            }
        return {"questions": n, "arms": out}

    families = {
        "dialogue": lambda w, r: w["dialogue"],
        "type": lambda w, r: str(r["type"]),
        "pivot": lambda w, r: str(w["pivot"]),
    }
    result = {"windows": len(reports), "complete": complete, "all": summary(allrows)}
    for k, grouping in families.items():
        groups = defaultdict(list)
        for w, r in allrows:
            groups[grouping(w, r)].append((w, r))
        result[k] = {g: summary(v) for g, v in groups.items()}
    if len(result["dialogue"]) == 6:
        groups = list(result["dialogue"].values())
        counts = np.array([g["questions"] for g in groups])
        draws = np.array(list(product(range(6), repeat=6)))
        intervals = {}
        targets = ["calibrated", "calibrated_greedy", "unit"]
        if "mixture" in groups[0]["arms"]:
            targets.append("mixture")
        for target in targets:
            for benchmark in ("native", "terminal", "permuted", "unit", "source"):
                for metric in ("focal", "ht"):
                    sums = np.array(
                        [
                            g["questions"] * (g["arms"][target][metric] - g["arms"][benchmark][metric])
                            for g in groups
                        ]
                    )
                    intervals[f"{target}-{benchmark}-{metric}"] = np.quantile(
                        sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
                    ).tolist()
        result["dialogue_bootstrap"] = intervals
    if complete and len(result["dialogue"]) == 6:
        arms = result["all"]["arms"]
        local = arms["calibrated"]["focal"]
        positive = sum(
            g["arms"]["calibrated"]["focal"] > g["arms"]["permuted"]["focal"] + 1e-12
            for g in result["dialogue"].values()
        )
        result["positive_dialogues_vs_permuted"] = positive
        result["eligible_for_development"] = (
            positive >= 4
            and all(local >= arms[b]["focal"] + 0.02 for b in ("native", "terminal"))
            and all(local > arms[b]["focal"] + 1e-12 for b in ("permuted", "unit", "temporal", "activity"))
            and all(arms["calibrated"]["ht"] > arms[b]["ht"] for b in ("native", "terminal", "permuted"))
        )
    if complete and len(result["dialogue"]) == 6 and "mixture" in result["all"]["arms"]:
        arms = result["all"]["arms"]
        local = arms["mixture"]["focal"]
        positive = sum(
            g["arms"]["mixture"]["focal"] > g["arms"]["mixture_permuted"]["focal"] + 1e-12
            for g in result["dialogue"].values()
        )
        result["mixture_positive_dialogues_vs_permuted"] = positive
        result["mixture_eligible_for_development"] = (
            positive >= 4
            and all(local >= arms[b]["focal"] + 0.02 for b in ("native", "terminal"))
            and all(
                local > arms[b]["focal"] + 1e-12
                for b in ("mixture_permuted", "unit", "calibrated", "temporal", "activity")
            )
            and all(arms["mixture"]["ht"] > arms[b]["ht"] for b in ("native", "terminal"))
        )
    return result


def run(trainer, study, locked, train, specs, limit=None, *, complete_donors=False):
    dest = locked.parent
    cohort = read(dest / "cohort.json")
    frozen = read(locked)
    if (
        frozen["inputs"]["cohort"] != file_hash(dest / "cohort.json")
        or cohort["corpus"] != file_hash(Path(trainer.args.data_file))
        or frozen["inputs"]["features"] != file_hash(dest / "features.json")
        or cohort["plan"] != file_hash(dest / "plan.json")
    ):
        raise ValueError("Different training cohort, features, or corpus")
    data = {c["sample_id"]: c for c in train}
    if assignments(data) != cohort["assignments"]:
        raise ValueError("Eligible questions or randomized assignments changed")
    root = dest / "outcomes"
    root.mkdir(exist_ok=True)
    manifest = {
        "forecast": file_hash(locked),
        "protocol": file_hash(study / "protocol.json"),
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
        "skill_versions": [s["version"] for s in specs],
    }
    register(root / "manifest.json", manifest)
    if complete_donors:
        from experiments.controller.complete_donor_paths import run as complete
        return complete(trainer, locked, data, manifest, limit or 1)
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(root / "responses")
    forecast = {(r["dialogue"], r["session"]): r for r in frozen["forecasts"]}
    reports = []
    for w in cohort["assignments"][:limit]:
        sid, start = w["dialogue"], w["start"]
        convo = window(data[sid], start, 2)
        preds = [forecast[sid, s] for s in (start, start + 1)]
        for i, pred in enumerate(preds, start):
            key = f"session_{i}"
            (src,) = trainer.data_processor._extract_session_chunks(
                convo["conversation"][key], convo["conversation"][key + "_date_time"]
            )
            if hashlib.sha256(src.encode()).hexdigest() != pred["source_sha256"]:
                raise ValueError("Policy forecast refers to different pre-action source")
        episode = root / f"{sid}_s{start}"
        episode.mkdir(exist_ok=True)
        with (episode / "measure.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cells = {
                k: measure_cell(trainer, convo, episode, k, manifest["skill_versions"])
                for k in paths(w["pivot"], w["background"])
            }
            with torch.device(trainer.args.device):
                report = summarize_pair(w, cells, preds, frozen["terminal"][sid])
            register(episode / "pair.json", report)
        reports.append(report)
        print(f"{sid} {start} paired with {len(report['questions'])} questions", flush=True)
    if len(reports) == len(cohort["assignments"]):
        register(root / "summary.json", {"manifest": manifest, **aggregate(reports, complete=True)})
