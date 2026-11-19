"""Estimate two-step stochastic policy returns from complete, randomized training trajectories."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from scipy.special import expit, logit

from experiments.controller.audit_paired import probability
from experiments.controller.paired_grounding import window
from experiments.controller.paired_policy import (fit_anchored_credit, fit_constrained_placement, fit_memory_novelty, fit_policy,
                                                  match_training_probability, novelty_logit_shift)
from experiments.controller.resume_status import load_episode
from src.eval import get_evaluator


def checked_trajectory(directory, convo, evaluator, deployment_writer):
    ids = [q["original_index"] for q in convo["qa"]]
    rows = load_episode(directory, convo["sample_id"], 2, ids, deployment_writer=deployment_writer)
    trace, = (directory / "traces").glob("*.jsonl")
    events = list(map(json.loads, trace.read_text().splitlines()))
    for row, span in zip(rows, (v for v in events if v["kind"] == "span"), strict=True):
        coin = span["sampling_noise"]["uniform"]
        if not 0 <= coin < 1 or row["prob"] != 0.5 or row["use"] != (coin < 0.5):
            raise ValueError(f"Original fair-coin action disagrees with the trace at {directory}")
    answers = [v for v in events if v["kind"] == "qa"]
    expected = {f'{convo["sample_id"]}_{i}': q for i, q in enumerate(convo["qa"])}
    if len(answers) != len(expected) or not answers or {a["query_id"] for a in answers} != expected.keys():
        raise ValueError(f"Missing or repeated end-of-window answers at {directory}")
    scores = []
    for a in answers:
        q = expected[a["query_id"]]
        score = evaluator.compute_f1(a["prediction"], evaluator.get_ground_truth(q), q)
        if not a["success"] or a["question"] != q["question"] or min(a["tokens"]) <= 0 or not math.isclose(score, a["f1"]):
            raise ValueError(f"Invalid answer in {directory}: {a['query_id']}")
        scores.append(score)
    reward = mean(scores)
    if any(not math.isclose(r["terminal"], reward) for r in rows):
        raise ValueError(f"Training return differs from the mean F1 of all questions at {directory}")
    return {"rows": rows, "questions": len(scores), "return": reward}


def policy_value(episodes, arm):
    """Question-weighted trajectory importance sampling with an out-of-fold control variate."""
    total = sum(e["questions"] for e in episodes)
    masses, weighted, adjusted = [], 0., 0.
    for e in episodes:
        probabilities = e["probabilities"][arm]
        # Only the probabilities at the actual, visited prefix states enter the likelihood ratio.
        ratio = math.prod(2 * (p if used else 1 - p) for p, used in zip(probabilities, e["uses"], strict=True))
        mass = e["questions"] * ratio
        masses.append(mass)
        weighted += mass * e["return"]
        adjusted += e["questions"] * (e["baseline"] + ratio * (e["return"] - e["baseline"]))
    mass_sum = sum(masses)
    return {"baseline_adjusted_f1": adjusted / total, "ordinary_is_f1": weighted / total,
            "self_normalized_f1": weighted / mass_sum if mass_sum else None,
            "trajectory_ess_with_question_weights": mass_sum**2 / sum(x*x for x in masses) if mass_sum else 0.,
            "mean_probability_on_logged_states": mean(p for e in episodes for p in e["probabilities"][arm])}


def contrasts(episodes):
    values = {arm: policy_value(episodes, arm) for arm in episodes[0]["probabilities"]}
    compared = {}
    comparators = {"credit": ("terminal", "credit_rate"), "anchor": ("terminal", "anchor_rate"),
                   "novelty": ("anchor", "anchor_rate")}
    if "constrained" in values:
        comparators["constrained"] = ("anchor", "anchor_rate")
    for arm, controls in comparators.items():
        result = {}
        for baseline in controls:
            result[baseline] = {}
            for k in ("baseline_adjusted_f1", "ordinary_is_f1", "self_normalized_f1"):
                a, b = values[arm][k], values[baseline][k]
                result[baseline][k] = a - b if a is not None and b is not None else None
            result[baseline]["greedy_switches_on_logged_states"] = sum(
                (p >= .5) != (b >= .5) for e in episodes
                for p, b in zip(e["probabilities"][arm], e["probabilities"][baseline], strict=True))
        compared[arm] = result
    return {"windows": len(episodes), "questions": sum(e["questions"] for e in episodes),
            "sampled_actions": dict(Counter("".join(str(int(v)) for v in e["uses"]) for e in episodes)),
            "values": values, "differences": compared}


def audit(study, locomo, *, progress=False, constrained_candidate=False):
    protocol = json.loads((study / "protocol.json").read_text())
    starts = [1, 3, 5, 7]
    if protocol["train_sessions"] != [[s, s+1] for s in starts]:
        raise ValueError("This forward evaluation uses the four paired windows through session 8")
    raw = locomo.read_bytes()
    train = set(protocol["train"])
    if (hashlib.sha256(raw).hexdigest() != protocol["input_sha256"] or len(train) != 6
            or train.intersection(protocol["validation"] + protocol["test"])):
        raise ValueError("Source corpus or frozen training split mismatch")
    corpus = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in train}
    evaluator = get_evaluator("locomo", SimpleNamespace())
    windows, pending = {}, []
    for start in starts:
        for sid in protocol["train"]:
            directory = study / "train" / f"{sid}_s{start}"
            if progress and start == 7 and not directory.exists():
                pending.append(directory.name)
                continue
            windows[sid, start] = checked_trajectory(directory, window(corpus[sid], start, 2),
                                                     evaluator, protocol.get("deployment_writer", False))
    episodes = []
    for (held, start), target in windows.items():
        if start < 5:
            continue
        training = [v for (sid, earlier), v in windows.items() if sid != held and earlier < start]
        fit = [row for v in training for row in v["rows"]]
        terminal = fit_policy(fit, 0)
        credit = fit_policy(fit, 1)
        anchor = fit_anchored_credit(fit, terminal)
        weights = {"terminal": terminal, "credit": credit, "anchor": anchor,
                   "credit_rate": match_training_probability(fit, terminal, credit),
                   "anchor_rate": match_training_probability(fit, terminal, anchor)}
        if constrained_candidate:
            weights["constrained"], _ = fit_constrained_placement(fit, weights["anchor_rate"], anchor)
        rows = target["rows"]
        probabilities = {arm: [probability(row, w.numpy()) for row in rows] for arm, w in weights.items()}
        novelty = fit_memory_novelty(fit, anchor)
        probabilities["novelty"] = [float(expit(logit(p) + novelty_logit_shift(row["state"], novelty)))
                                    for p, row in zip(probabilities["anchor"], rows, strict=True)]
        probabilities["logging"] = [.5, .5]
        baseline = sum(v["return"] * v["questions"] for v in training) / sum(v["questions"] for v in training)
        episodes.append({"dialogue": held, "start": start, "fit_windows": len(training), "novelty_fit": novelty, "baseline": baseline,
                         "questions": target["questions"], "return": target["return"], "uses": [r["use"] for r in rows],
                         "probabilities": probabilities})
    by_dialogue = {sid: contrasts([e for e in episodes if e["dialogue"] == sid]) for sid in protocol["train"]}
    return {"scope": "forward two-step randomized training windows, fitted on earlier windows from other dialogues",
            "pending_windows": pending, "by_dialogue": by_dialogue, "all": contrasts(episodes),
            "by_start": {str(s): contrasts(group) for s in (5, 7) if (group := [e for e in episodes if e["start"] == s])},
            "episodes": episodes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("locomo", type=Path)
    parser.add_argument("--progress", action="store_true", help="Include only atomically completed session 7–8 windows")
    parser.add_argument("--constrained-candidate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.study, args.locomo, progress=args.progress, constrained_candidate=args.constrained_candidate), indent=2))
