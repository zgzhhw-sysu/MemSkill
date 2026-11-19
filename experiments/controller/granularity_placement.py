"""Measure source-only span placement on four real reader outcomes per training question."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from src.eval import get_evaluator


def source_features(turns):
    lengths = [len(turn["text"].split()) for turn in turns]
    questions = ["?" in turn["text"] for turn in turns]
    pairs = range(0, len(turns) - 1, 2)
    return [mean(math.log1p(n) for n in lengths), mean(n <= 5 for n in lengths), mean(questions),
            mean(questions[i] and lengths[i + 1] <= 12 for i in pairs) if pairs else 0.]


def load_panel(study, data_file, run, *, completed_only=False):
    protocol = json.loads((study / "protocol.json").read_text())
    raw = data_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["input_sha256"]:
        raise ValueError("Wrong training corpus")
    train = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in protocol["train"]}
    if len(train) != 6 or set(train) != set(protocol["train"]) or set(train) & set(protocol["validation"] + protocol["test"]):
        raise ValueError("Expected the six independent training dialogues")
    jobs = defaultdict(list)
    for job in json.loads((run / "granularity-train-prompts.json").read_text()):
        jobs[job["dialogue"], job["start"]].append(job)
    if jobs.keys() != set(itertools.product(train, [1, 3])):
        raise ValueError("Incomplete training window manifest")
    ev = get_evaluator("locomo", SimpleNamespace())
    windows = []
    for (sid, start), prompts in sorted(jobs.items()):
        if completed_only and any(not (run / "granularity-predictions" / (t["hash"] + ".json")).is_file()
                                  for t in prompts):
            continue
        c = train[sid]
        sessions = [c["conversation"][f"session_{i}"] for i in (start, start + 1)]
        turns = {turn["dia_id"] for session in sessions for turn in session}
        questions = [q for _, q in ev.filter_qa_list(c["qa"]) if q.get("evidence") and set(q["evidence"]) <= turns]
        scores = {key: {} for key in ("00", "01", "10", "11")}
        for task in prompts:
            qi = task["question_index"]
            key = f'{task["first_pair"]}{task["second_pair"]}'
            h = hashlib.sha256(task["prompt"].encode()).hexdigest()
            q = questions[qi]
            if task["question"] != q["question"] or task["hash"] != h or qi in scores[key]:
                raise ValueError("Mismatched source-only question, prompt hash or repeated treatment cell")
            reply = json.loads((run / "granularity-predictions" / (h + ".json")).read_text())
            cost = reply["usage"]
            if reply["hash"] != h or cost["prompt_tokens"] <= 0 or not 0 < cost["completion_tokens"] <= 32:
                raise ValueError("Missing measured reader response")
            scores[key][qi] = ev.compute_f1(reply["prediction"], ev.get_ground_truth(q), q)
        if not questions or any(set(values) != set(range(len(questions))) for values in scores.values()):
            raise ValueError("Missing measured cell in the two-by-two training intervention")
        windows.append({"dialogue": sid, "start": start, "n": len(questions),
                        "state": [source_features(session) for session in sessions],
                        "f1": {key: mean(values.values()) for key, values in scores.items()}})
    return windows


def fit_ranker(states, gains, sizes):
    """Use per-question weighting and a fixed ridge penalty on paired span contrasts."""
    difference = states[:, 0] - states[:, 1]
    weights = torch.zeros(states.shape[-1], dtype=states.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=100, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = (sizes * gains.abs() * F.softplus(-gains.sign() * (difference @ weights))).sum() / sizes.sum()
        loss = loss + .01 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    weights = weights.detach()
    # Fix otherwise unidentified intercept using training states only.
    logits = states @ weights
    low, high = -float(logits.max()), -float(logits.min())
    for _ in range(50):
        mid = (low + high) / 2
        if (logits + mid).sigmoid().mean() < .5:
            low = mid
        else:
            high = mid
    return weights, (low + high) / 2


def measured_return(logits, scores):
    p, q = logits.sigmoid().tolist()
    first = (logits[0] - logits[1]).sigmoid().item()
    return {"probabilities": [p, q], "first_given_one": first,
            "given_one_f1": first * scores["10"] + (1 - first) * scores["01"],
            "unconditional_f1": ((1-p)*(1-q)*scores["00"] + p*(1-q)*scores["10"]
                                 + (1-p)*q*scores["01"] + p*q*scores["11"])}


def compare(windows):
    histories = {w["dialogue"] for w in windows}
    locations = {(w["dialogue"], w["start"]) for w in windows}
    if len(histories) != 6 or len(windows) != 12 or locations != set(itertools.product(histories, [1, 3])):
        raise ValueError("A placement screen requires all twelve training windows")
    panels = []
    for held in sorted(histories):
        fit = [w for w in windows if w["dialogue"] != held]
        states = torch.tensor([w["state"] for w in fit], dtype=torch.float64)
        center = states.mean(dim=(0, 1))
        std = states.std(dim=(0, 1), unbiased=False)
        scale = torch.where(std > 1e-9, std, 1.)
        gains = states.new_tensor([w["f1"]["10"] - w["f1"]["01"] for w in fit])
        sizes = states.new_tensor([w["n"] for w in fit])
        slope, bias = fit_ranker((states - center) / scale, gains, sizes)
        slots = states.new_tensor([[1.], [-1.]])
        slot_slope, slot_bias = fit_ranker(slots.expand(len(fit), -1, -1), gains, sizes)
        for w in windows:
            if w["dialogue"] != held:
                continue
            x = (states.new_tensor(w["state"]) - center) / scale
            scores = w["f1"]
            logits = {"source": x @ slope + bias, "slot": slots @ slot_slope + slot_bias,
                      "uniform": states.new_zeros(2)}
            panels.append({"dialogue": held, "start": w["start"], "n": w["n"], "cells": scores,
                           "arms": {a: measured_return(z, scores) for a, z in logits.items()}})
    def summarize(group):
        return {a: {k: sum(r["n"] * r["arms"][a][k] for r in group) / sum(r["n"] for r in group)
                    for k in ("given_one_f1", "unconditional_f1")} for a in ("source", "slot", "uniform")}
    total = summarize(panels)
    folds = {sid: summarize([r for r in panels if r["dialogue"] == sid]) for sid in sorted({r["dialogue"] for r in panels})}
    positive = sum(d["source"]["given_one_f1"] > d["uniform"]["given_one_f1"] for d in folds.values())
    advance = (total["source"]["given_one_f1"] >= total["uniform"]["given_one_f1"] + .02 and positive >= 4
               and total["source"]["given_one_f1"] > total["slot"]["given_one_f1"]
               and total["source"]["unconditional_f1"] > max(total[a]["unconditional_f1"] for a in ("uniform", "slot")))
    return {"scope": "source-only four-cell reader interventions in held-out training dialogues",
            "questions": sum(r["n"] for r in panels), "windows": len(panels), "folds": folds, "aggregate": total,
            "positive_conditional_folds_vs_uniform": positive, "advance_to_full_writer_probes": advance, "panels": panels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("locomo", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--progress", action="store_true", help="Show only fully measured windows without fitting a policy")
    args = parser.parse_args()
    panel = load_panel(args.study, args.locomo, args.run, completed_only=args.progress)
    report = ([{k: v for k, v in w.items() if k != "state"} for w in panel] if args.progress else compare(panel))
    print(json.dumps(report, indent=2))
