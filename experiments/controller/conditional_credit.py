"""Learn dependence between successive skill choices while preserving both marginal probabilities."""

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.controller.audit_source_compression import check_forecasts, panel
from experiments.controller.factorial_rollout import CELLS, load_cell, record_scored_paths, summarize_window
from experiments.controller.granularity_placement import load_panel
from experiments.controller.paired_grounding import window
from experiments.controller.source_compression import file_hash, register
from src.data_processing.locomo import LoCoMoProcessor
from src.eval import get_evaluator


def read(path):
    return json.loads(path.read_bytes())


def coupled_paths(p, q, strength):
    """P(second=1|first=a) implements the same joint law without using future source at step one."""
    if not (0 <= p <= 1 and 0 <= q <= 1 and -1 <= strength <= 1):
        raise ValueError("Invalid probability or coupling strength")
    independent = [(1 - p) * (1 - q), (1 - p) * q, p * (1 - q), p * q]
    delta = min(independent) * strength
    mass = [v + (delta if a == b else -delta) for (a, b), v in zip(CELLS, independent, strict=True)]
    if (
        min(mass) < 0
        or not math.isclose(mass[2] + mass[3], p, abs_tol=1e-12)
        or not math.isclose(mass[1] + mass[3], q, abs_tol=1e-12)
    ):
        raise ValueError("Coupling changed a marginal action probability")
    return {
        "delta": delta,
        "mass": mass,
        "conditional_second": [mass[1] / (1 - p) if p < 1 else q, mass[3] / p if p > 0 else q],
    }


def fit_moderation(training):
    effects = []
    for r in training:
        other = [a["final_effect"] for a in training if a["dialogue"] != r["dialogue"]]
        baseline = mean(other)
        effects.append(2 * (1 if r["first_activity"] else -1) * (r["final_effect"] - baseline))
    x = torch.tensor([r["features"] for r in training], dtype=torch.float64)
    y = x.new_tensor(effects)
    center, scale = x.mean(0), x.std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    z = (x - center) / scale
    ridge = torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * len(training)
    slopes = torch.linalg.solve(z.T @ z + ridge, z.T @ (y - y.mean()))
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "slopes": slopes.tolist(),
        "intercept": float(y.mean()),
        "rms": float(y.square().mean().sqrt()),
    }


def predict_strength(fit, features):
    x = torch.as_tensor(features, dtype=torch.float64)
    z = (x - x.new_tensor(fit["center"])) / x.new_tensor(fit["scale"])
    h = fit["intercept"] + float(z @ x.new_tensor(fit["slopes"]))
    return h / (abs(h) + fit["rms"]) if fit["rms"] else 0.0


def fit_windows(rows, features, probs):
    indexed = {(r["dialogue"], r["session"]): r for r in rows}
    samples = []
    for (sid, start), f in sorted(features.items()):
        if start % 2 == 0:
            continue
        diff = [
            abs(x - y) for x, y in zip(f["features"], features[sid, start + 1]["features"], strict=True)
        ]
        entry = {
            "dialogue": sid,
            "start": start,
            "features": diff,
            "p": probs[sid, start],
            "q": probs[sid, start + 1],
        }
        if start in (1, 3):
            first, second = indexed[sid, start], indexed[sid, start + 1]
            if first["prob"] != 0.5 or second["prob"] != 0.5:
                raise ValueError("Moderation training requires fair randomized exploration")
            entry.update(first_activity=first["use"], final_effect=second["signed"])
        samples.append(entry)
    fits, forecasts = {}, []
    for held in sorted({r["dialogue"] for r in samples}):
        started = time.monotonic()
        train = [r for r in samples if r["dialogue"] != held and r["start"] in (1, 3)]
        model = fit_moderation(train)
        fits[held] = dict(model, training_windows=[[r["dialogue"], r["start"]] for r in train])
        for r in samples:
            if r["dialogue"] == held:
                rho = predict_strength(model, r["features"])
                path = coupled_paths(r["p"], r["q"], rho)
                forecasts.append(
                    {k: r[k] for k in ("dialogue", "start", "features", "p", "q")}
                    | {"strength": rho, **path}
                )
        if torch.empty(0).is_cuda:
            torch.cuda.synchronize()
            time.sleep((time.monotonic() - started) * (1 / 0.35 - 1))
    return fits, forecasts


def check_coupling(locked, features, parent_probs):
    keys = {(sid, n) for sid, n in features if n % 2}
    rows = {(r["dialogue"], r["start"]): r for r in locked["forecasts"]}
    if len(rows) != len(locked["forecasts"]) or rows.keys() != keys:
        raise ValueError("Missing or duplicate coupling forecasts")
    for (sid, n), r in rows.items():
        train = [[other, i] for other, i in sorted(keys) if other != sid and i in (1, 3)]
        fit = locked["fits"][sid]
        diff = [
            abs(x - y)
            for x, y in zip(features[sid, n]["features"], features[sid, n + 1]["features"], strict=True)
        ]
        p, q = (parent_probs[sid, k] for k in (n, n + 1))
        if fit["training_windows"] != train or diff != r["features"] or (p, q) != (r["p"], r["q"]):
            raise ValueError("Wrong training split or references for coupling forecasts")
        rho = predict_strength(fit, diff)
        path = coupled_paths(p, q, rho)
        if (
            not math.isclose(rho, r["strength"], abs_tol=1e-12)
            or not math.isclose(path["delta"], r["delta"], abs_tol=1e-12)
            or any(
                not math.isclose(a, b, abs_tol=1e-12)
                for k in ("mass", "conditional_second")
                for a, b in zip(path[k], r[k], strict=True)
            )
        ):
            raise ValueError("Coupling forecast does not reproduce")


def orientation_support(forecasts):
    """A constant orientation within a dialogue is identical to its sign permutation for every F1 table."""
    groups = defaultdict(list)
    for r in forecasts:
        groups[r["dialogue"]].append(r)
    possible = sum(
        len({r["strength"] >= 0 for r in group}) > 1 and any(r["delta"] != 0 for r in group)
        for group in groups.values()
    )
    return {
        "windows": len(forecasts),
        "positive_deltas": sum(r["delta"] > 0 for r in forecasts),
        "negative_deltas": sum(r["delta"] < 0 for r in forecasts),
        "max_dialogues_beating_shuffled_sign": possible,
        "can_beat_both_fixed_signs": any(r["delta"] > 0 for r in forecasts)
        and any(r["delta"] < 0 for r in forecasts),
    }


def score_questions(windows, predictions, *, partial_cohort=False):
    by_key = {(r["dialogue"], r["start"]): r for r in predictions}
    if len(by_key) != len(predictions) or len({(w["dialogue"], w["start"]) for w in windows}) != len(
        windows
    ):
        raise ValueError("Duplicate window")
    groups = defaultdict(list)
    for w in windows:
        groups[w["dialogue"]].append(by_key[w["dialogue"], w["start"]])
    rows, windows_report = [], []
    for w in windows:
        sid, start = w["dialogue"], w["start"]
        r = by_key[sid, start]
        absolute = abs(r["delta"])
        sign_mean = mean(1 if v["strength"] >= 0 else -1 for v in groups[sid])
        interaction = []
        for orig, sample in w["questions"].items():
            values = [sample["f1"][f"{a}{b}"] for a, b in CELLS]
            v00, v01, v10, v11 = values
            contrast = v11 - v01 - v10 + v00
            interaction.append(contrast)
            independent = sum(
                v * m for v, m in zip(values, coupled_paths(r["p"], r["q"], 0.0)["mass"], strict=True)
            )
            scores = {
                "independent": independent,
                "conditional": independent + r["delta"] * contrast,
                "favor_same": independent + absolute * contrast,
                "favor_different": independent - absolute * contrast,
                "shuffled_sign": independent + absolute * sign_mean * contrast,
            }
            if not math.isclose(
                scores["conditional"],
                sum(v * m for v, m in zip(values, r["mass"], strict=True)),
                abs_tol=1e-12,
            ):
                raise ValueError("Path weighting disagrees with the mixed difference")
            if "controls" in sample:
                if not math.isclose(sample["controls"]["credit"], independent, abs_tol=1e-10):
                    raise ValueError("Original frozen independent return did not reproduce")
                scores.update({k: v for k, v in sample["controls"].items() if k != "credit"})
            rows.append(
                {
                    "dialogue": sid,
                    "start": start,
                    "question": orig,
                    "type": str(sample["question"]["category"]),
                    "scores": scores,
                }
            )
        windows_report.append(
            {
                "dialogue": sid,
                "start": start,
                "questions": len(interaction),
                "interaction": mean(interaction),
                "delta": r["delta"],
                "conditional_second": r["conditional_second"],
                "best_gain_at_same_magnitude": absolute * abs(mean(interaction)),
            }
        )

    def averages(group):
        return {k: mean(r["scores"][k] for r in group) for k in group[0]["scores"]}

    overall = averages(rows)
    dialogues = {sid: averages([r for r in rows if r["dialogue"] == sid]) for sid in groups}
    types = {}
    for cat in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == cat]
        types[cat] = {"questions": len(group), "means": averages(group)}
    # Exact enumeration of the empirical cluster bootstrap over the six whole dialogues.
    ordered = sorted(groups)
    if not (1 <= len(ordered) <= 6) or (not partial_cohort and len(ordered) != 6):
        raise ValueError("Expected six held training dialogues")
    counts = np.array([sum(r["dialogue"] == sid for r in rows) for sid in ordered])
    draws = np.array(list(itertools.product(range(len(ordered)), repeat=len(ordered))))
    intervals = {}
    for baseline in ("independent", "favor_same", "favor_different", "shuffled_sign"):
        if len(ordered) == 1:
            intervals[baseline] = None
            continue
        totals = counts * [dialogues[sid]["conditional"] - dialogues[sid][baseline] for sid in ordered]
        means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
        intervals[baseline] = np.quantile(means, [0.025, 0.975]).tolist()
    gains = ("independent", "favor_same", "favor_different", "shuffled_sign")
    above = all(overall["conditional"] > overall[k] + 1e-12 for k in gains)
    positive = sum(v["conditional"] > v["shuffled_sign"] + 1e-12 for v in dialogues.values())
    return {
        "questions": len(rows),
        "windows": windows_report,
        "aggregate": overall,
        "per_dialogue": dialogues,
        "per_type": types,
        "cluster_bootstrap_difference": intervals,
        "positive_dialogues_vs_shuffled_sign": positive,
        "placement_criterion": above and positive >= 4,
        "promote_to_development": above
        and positive >= 4
        and all(
            k in overall and overall["conditional"] >= overall[k] + 0.02 for k in ("native", "terminal")
        ),
        "rows": rows,
    }


def writer_outcomes(study, corpus, source, frozen, candidate=None, *, completed_only=False):
    protocol = read(study / "protocol.json")
    archive = source / "factorial"
    summary = None if completed_only else read(archive / "summary.json")
    manifest = read(archive / "manifest.json")
    if (
        manifest["forecast"] != file_hash(source / "frozen.json")
        or manifest["protocol"] != file_hash(study / "protocol.json")
        or manifest["windows"] != [[sid, start] for sid in protocol["train"] for start in (5, 7)]
    ):
        raise ValueError("Wrong four-path experiment")
    trainer = SimpleNamespace(
        data_processor=LoCoMoProcessor(chunk_mode="full-session"),
        evaluator=get_evaluator("locomo", SimpleNamespace()),
    )
    versions = [s["version"] for s in protocol["skills"]]
    data = {
        c["sample_id"]: c for c in json.loads(corpus.read_bytes()) if c["sample_id"] in protocol["train"]
    }
    probs = {(r["dialogue"], r["session"]): r["p_activity"] for r in frozen["forecasts"]}
    candidate_probs = (
        {(r["dialogue"], r["session"]): r["p_activity"] for r in candidate["forecasts"]}
        if candidate
        else {}
    )
    windows = []
    for sid, start in manifest["windows"]:
        convo = window(data[sid], start, 2)
        root = archive / f"{sid}_s{start}"
        if completed_only and not (root / "measurement.json").exists():
            continue
        ids = [q["original_index"] for q in convo["qa"]]
        cells = {k: load_cell(trainer, convo, root / f"{k[0]}{k[1]}", k, versions) for k in CELLS}
        rep = summarize_window(cells, [probs[sid, start], probs[sid, start + 1]], frozen["fits"][sid], ids)
        if rep != read(root / "measurement.json"):
            raise ValueError(f"Four-path measurement did not reproduce at {root}")
        if candidate:
            rep = summarize_window(
                cells, [candidate_probs[sid, n] for n in (start, start + 1)], candidate["fits"][sid], ids
            )
        questions = {
            orig: {
                "question": convo["qa"][i],
                "f1": {f"{a}{b}": cells[a, b][2][i] for a, b in CELLS},
                "controls": rep["per_question"][i]["f1"],
            }
            for i, orig in enumerate(ids)
        }
        windows.append({"dialogue": sid, "start": start, "questions": questions})
    if summary and (
        len(windows) != summary["windows"]
        or sum(len(w["questions"]) for w in windows) != summary["questions"]
    ):
        raise ValueError("Incomplete measured factorial outcomes")
    return windows


def run(phase, previous, current, corpus, proxy, source):
    dest = source / "conditional-credit"
    parent = read(source / "frozen.json")
    features = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    probs = {(r["dialogue"], r["session"]): r["p_activity"] for r in parent["forecasts"]}
    if (
        parent["inputs"]["features.json"] != file_hash(source / "features.json")
        or len(features) != 48
        or file_hash(corpus) != read(source / "features.json")["corpus"]
    ):
        raise ValueError("Unexpected source features or corpus")
    inputs = {"plan": file_hash(dest / "plan.json"), "parent": file_hash(source / "frozen.json")}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    with torch.device("cuda"):
        check_forecasts(features, parent)
        if phase == "freeze":
            rows, hashes, backend = panel(previous, corpus, features, (1, 3), supplement=True)
            if hashes != parent["prior_hashes"] or backend != parent["backend"]:
                raise ValueError("Training interventions changed")
            fits, preds = fit_windows(rows, features, probs)
            report = {"inputs": inputs, "fits": fits, "forecasts": preds}
            locked = dest / "frozen.json"
            if not locked.exists() and any((source / "factorial").glob("*/measurement.json")):
                raise RuntimeError("A later full window was already scored before coupling registration")
            check_coupling(report, features, probs)
            register(locked, report)
            record_scored_paths(source / "factorial", locked)
            register(
                dest / "orientation.json",
                {
                    "forecast_sha256": file_hash(locked),
                    "historical": orientation_support([r for r in preds if r["start"] in (1, 3)]),
                    "prospective": orientation_support([r for r in preds if r["start"] in (5, 7)]),
                },
            )
            old = load_panel(previous, corpus, proxy, include_questions=True)
            register(dest / "historical.json", score_questions(old, preds))
        else:
            locked = read(dest / "frozen.json")
            if locked["inputs"] != inputs:
                raise ValueError("Changed conditional policy registration")
            check_coupling(locked, features, probs)
            windows = writer_outcomes(current, corpus, source, parent)
            result = score_questions(windows, locked["forecasts"])
            register(
                dest / "prospective.json", {"forecast_sha256": file_hash(dest / "frozen.json"), **result}
            )
            early = record_scored_paths(source / "factorial", dest / "frozen.json")
            fresh = [w for w in windows if f"{w['dialogue']}_s{w['start']}" not in early]
            register(
                dest / "strictly-later.json",
                {
                    "forecast": file_hash(dest / "frozen.json"),
                    "windows_with_any_early_scores": sorted(early),
                    "result": score_questions(fresh, locked["forecasts"], partial_cohort=True)
                    if fresh
                    else None,
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    for name in ("previous", "current", "corpus", "proxy", "source"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
