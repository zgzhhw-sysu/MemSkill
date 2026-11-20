"""Fit a training-only placement policy on measured two-arm reader F1, leaving out whole dialogues."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.controller.granularity_placement import fit_ranker
from experiments.controller.resume_status import load_episode
from src.eval import get_evaluator
from src.evaluation_cache import digest, read_response


def load_outcomes(study, directory):
    protocol = json.loads((study / "protocol.json").read_bytes())
    plan = json.loads((directory / "jobs.json").read_bytes())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    summary = json.loads((directory / "summary.json").read_bytes())
    if (digest(plan) != manifest["plan"] or plan["windows"] != manifest["windows"]
            or len(plan["jobs"]) != manifest["questions"] or len(plan["jobs"]) != summary["questions"]
            or len(set(protocol["train"])) != 6 or set(protocol["train"]) & set(protocol["validation"] + protocol["test"])):
        raise ValueError("Expected a completed, disjoint training panel with matching prompts")
    cache = directory / "responses"
    receipts = set()
    for path in cache.glob("*/*.json"):
        row = read_response(cache, path.stem)["payload"]
        receipts.add((row["prompt_sha256"], row["response"], tuple(row["tokens"])))
    evaluator = get_evaluator("locomo", SimpleNamespace())
    scores = defaultdict(dict)
    matched = 0
    for job in plan["jobs"]:
        episode, i, q = job["episode"], job["original_index"], job["question"]
        row = json.loads((directory / f"{episode}-q{i}.json").read_bytes())
        if (i in scores[episode] or row["original_index"] != i or row["episode"] != episode
                or q["original_index"] != i or set(row["scores"]) != {"activity", "temporal"}
                or row["original_retrieval_reproduced"] != job["original_retrieval_reproduced"]):
            raise ValueError("Different or repeated original question")
        matched += row["original_retrieval_reproduced"]
        cell = {}
        for arm, result in row["scores"].items():
            h = hashlib.sha256(job["prompts"][arm].encode()).hexdigest()
            cell[arm] = float(evaluator.compute_f1(result["answer"], evaluator.get_ground_truth(q), q))
            if (h != result["prompt_sha256"] or (h, result["answer"], tuple(result["tokens"])) not in receipts
                    or not math.isclose(cell[arm], result["f1"], abs_tol=1e-12)):
                raise ValueError("Reader F1, prompt or response differs from the measured request")
        scores[episode][i] = cell
    if (matched != summary["original_retrieval_matches"]
            or any(not math.isclose(mean(cell[a] for group in scores.values() for cell in group.values()),
                                    summary["mean_f1"][a], abs_tol=1e-12) for a in ("activity", "temporal"))):
        raise ValueError("Recomputed scores or original retrieval count disagree with the completed report")
    windows = defaultdict(list)
    for a, b in protocol["train_sessions"]:
        for sid in protocol["train"]:
            name = f"{sid}_s{a}"
            episode = study / "train" / name
            trace, = (episode / "traces").glob("*.jsonl")
            if hashlib.sha256(trace.read_bytes()).hexdigest() != plan["windows"][name]["trace"]:
                raise ValueError("Memory intervention and pre-action state have different factual traces")
            rows = load_episode(episode, sid, b-a+1, deployment_writer=protocol.get("deployment_writer", False))
            cells = scores.pop(name)
            if set(cells) != set(json.loads((episode / "questions.json").read_bytes())):
                raise ValueError("Reader results omit original training questions")
            windows[sid].append({"start": a, "state": rows[-1]["state"], "n": len(cells),
                                 **{arm: mean(r[arm] for r in cells.values()) for arm in ("activity", "temporal")}})
    if scores or any([w["start"] for w in pair] != [1, 3] for pair in windows.values()):
        raise ValueError("The screen requires the two prespecified windows in each training dialogue")
    return windows


def crossfit(windows, device):
    rows = []
    for held, test in sorted(windows.items()):
        fit = [pair for sid, pair in windows.items() if sid != held]
        states = F.normalize(torch.tensor([[w["state"] for w in pair] for pair in fit], dtype=torch.float64, device=device), dim=-1)
        center = states.mean(dim=(0, 1))
        centered = states - center
        scale = centered.square().sum(dim=-1).mean().sqrt().clamp_min(1e-9)
        sizes = states.new_tensor([sum(w["n"] for w in pair) for pair in fit])
        gains = states.new_tensor([pair[0]["n"] * (pair[0]["temporal"] - pair[0]["activity"])
                                   - pair[1]["n"] * (pair[1]["temporal"] - pair[1]["activity"]) for pair in fit]) / sizes
        weights, bias = fit_ranker(centered / scale, gains, sizes)
        features = F.normalize(states.new_tensor([w["state"] for w in test]), dim=-1)
        probs = (((features-center) / scale) @ weights + bias).sigmoid().tolist()
        rate = ((centered / scale) @ weights + bias).sigmoid().mean().item()
        greedy_rate = mean(p >= .5 for p in probs)
        for w, p in zip(test, probs, strict=True):
            gain = w["temporal"] - w["activity"]
            rows.append({"dialogue": held, "start": w["start"], "questions": w["n"], "temporal_probability": p,
                         "f1": {"activity": w["activity"], "temporal": w["temporal"],
                                "credit": w["activity"] + p*gain, "training_rate": w["activity"] + rate*gain,
                                "shuffled_within_dialogue": w["activity"] + mean(probs)*gain,
                                "greedy": w["temporal"] if p >= .5 else w["activity"],
                                "greedy_shuffled_within_dialogue": w["activity"] + greedy_rate*gain,
                                "best_per_window": max(w["activity"], w["temporal"])}})
    def average(group):
        return {a: sum(r["questions"]*r["f1"][a] for r in group)/sum(r["questions"] for r in group) for a in rows[0]["f1"]}
    aggregate = average(rows)
    folds = {sid: average([r for r in rows if r["dialogue"] == sid]) for sid in windows}
    controls = ("activity", "temporal", "training_rate", "shuffled_within_dialogue")
    positive = sum(v["credit"] > v["shuffled_within_dialogue"] + 1e-9 for v in folds.values())
    return {"scope": "one final-span source insertion in a fixed factual writer bank, training dialogues only",
            "questions": sum(r["questions"] for r in rows), "aggregate": aggregate, "folds": folds,
            "positive_placement_folds": positive, "rows": rows,
            "advance_to_full_controller": aggregate["credit"] > max(aggregate[a] for a in controls) + 1e-9
            and aggregate["greedy"] > max(aggregate[a] for a in ("activity", "temporal", "greedy_shuffled_within_dialogue")) + 1e-9
            and aggregate["best_per_window"] >= max(aggregate[a] for a in ("activity", "temporal")) + .02 and positive >= 4
            and 0 < sum(r["temporal_probability"] >= .5 for r in rows) < len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("reader_directory", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(crossfit(load_outcomes(args.study, args.reader_directory), args.device), indent=2))
