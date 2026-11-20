"""Freeze evidence-trained span probabilities before measuring their transfer to reader F1."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import torch

from experiments.controller.audit_budget_policy import load_outcomes
from experiments.controller.granularity_placement import fit_preference
from experiments.controller.skill_source_budget import retained_turns
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest
from src.memory_bank import get_retriever_embedding_dim
from src.rag_utils import get_embeddings, set_retriever_flash_attn


def turn_contrast(turns, selected, processor, retriever, device):
    """Common turns cancel in the difference between two means of the same size."""
    temporal, activity = set(selected["temporal"]), set(selected["activity"])
    different = temporal ^ activity
    if not different:
        return torch.zeros(get_retriever_embedding_dim(retriever), device=device, dtype=torch.float64)
    items = [turn for turn in turns if turn["dia_id"] in different]
    vectors = get_embeddings(retriever, [processor._format_turn(turn) for turn in items], "context", batch_size=2)
    vectors = torch.as_tensor(vectors, device=device, dtype=torch.float64)
    signs = vectors.new_tensor([1. if turn["dia_id"] in temporal else -1. for turn in items])
    return signs @ vectors / len(temporal)


def dialogue_fits(windows):
    forecasts, fits = [], {}
    for held in sorted({r["dialogue"] for r in windows}):
        train = [r for r in windows if r["dialogue"] != held]
        x = torch.stack([r["contrast"] for r in train])
        gains = x.new_tensor([r["temporal"] - r["activity"] for r in train])
        nonidentical = x.square().sum(dim=-1) > 0
        scale = x[nonidentical].square().sum(dim=-1).mean().sqrt() if nonidentical.any() else x.new_tensor(1.)
        active = gains != 0
        if (active & ~nonidentical).any():
            raise ValueError("Identical candidate embeddings have different evidence rewards")
        weights = (fit_preference(x[active] / scale, gains[active].sign(), gains[active].abs())
                   if active.any() else x.new_zeros(x.shape[1]))
        rate = float(((x / scale) @ weights).sigmoid().mean())
        fits[held] = {"training_dialogues": sorted({r["dialogue"] for r in train}), "training_spans": len(train),
                      "nonzero_labels": int(active.sum()), "scale": float(scale), "weights": weights.tolist(),
                      "training_temporal_rate": rate}
        for r in windows:
            if r["dialogue"] == held:
                logit = float((r["contrast"] / scale) @ weights)
                forecasts.append({"dialogue": held, "session": r["session"], "logit": logit,
                                  "p": float(weights.new_tensor(logit).sigmoid()), "training_rate": rate})
    return forecasts, fits


def compare(rows, denominator):
    groups = defaultdict(list)
    for r in rows:
        groups[r["dialogue"]].append(r)
    total = defaultdict(float)
    per_dialogue = {}
    for sid, group in groups.items():
        rate = mean(r["p"] for r in group)
        hard_rate = mean(r["p"] > .5 for r in group)
        sums = defaultdict(float)
        for r in group:
            probabilities = {"activity": 0., "temporal": 1., "credit": r["p"], "training_rate": r["training_rate"],
                             "shuffled_within_dialogue": rate, "greedy": float(r["p"] > .5),
                             "greedy_shuffled_within_dialogue": hard_rate}
            for arm, p in probabilities.items():
                value = r["activity"] + p * (r["temporal"] - r["activity"])
                sums[arm] += value
                total[arm] += value
        per_dialogue[sid] = {k: v/denominator[sid] for k, v in sums.items()}
    return {"aggregate": {k: v/sum(denominator.values()) for k, v in total.items()}, "per_dialogue": per_dialogue}


def freeze(study, corpus, reader):
    spec = json.loads((study / "protocol.json").read_bytes())
    raw = corpus.read_bytes()
    data = {c["sample_id"]: c["conversation"] for c in json.loads(raw) if c["sample_id"] in spec["train"]}
    evidence = json.loads((reader / "evidence.json").read_bytes())
    retrieval = json.loads((reader / "retrieval.json").read_bytes())
    if (hashlib.sha256(raw).hexdigest() != spec["input_sha256"] or evidence["checkpoint"] != spec["checkpoint_sha256"]
            or digest(evidence) != retrieval["evidence"] or data.keys() != evidence["per_dialogue"].keys()
            or len(data) != 6 or set(data) & set(spec["validation"] + spec["test"])):
        raise ValueError("Different training corpus, source choices or held dialogues")
    frozen = reader / "evidence-policy-freeze.json"
    if frozen.exists():
        registered = json.loads(frozen.read_bytes())
        if registered["inputs"] != retrieval or registered["training_corpus_sha256"] != spec["input_sha256"]:
            raise ValueError("A different prospective fit is already registered")
        return
    if (reader / "summary.json").exists():
        raise ValueError("A prospective actor must be registered before the reader panel finishes")
    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA lease to encode the candidate turns and fit the actors")
    processor = LoCoMoProcessor(chunk_mode="full-session")
    set_retriever_flash_attn(False)
    windows = []
    for r in evidence["rows"]:
        sid, n = r["dialogue"], r["session"]
        key = f"session_{n}"
        turns = data[sid][key]
        retained_turns(turns, data[sid][key+"_date_time"], r["selected"], processor, evidence["turn_budget_per_skill"])
        vec = turn_contrast(turns, r["selected"], processor, spec["retriever"], "cuda")
        windows.append({"dialogue": sid, "session": n, "contrast": vec, **r["covered_question_mass"]})
    forecasts, fits = dialogue_fits(windows)
    by_key = {(r["dialogue"], r["session"]): r for r in forecasts}
    if len(by_key) != len(windows) or len(windows) != evidence["sessions"]:
        raise ValueError("Incomplete held-dialogue forecasts")
    rows = [dict(by_key[r["dialogue"], r["session"]], activity=r["activity"], temporal=r["temporal"]) for r in windows]
    counts = {sid: info["questions"] for sid, info in evidence["per_dialogue"].items()}
    report = {"inputs": retrieval, "training_corpus_sha256": spec["input_sha256"], "forecasts": forecasts, "fits": fits,
              "evidence_recall": compare(rows, counts)}
    with TemporaryDirectory(dir=reader) as tmp:
        path = Path(tmp) / frozen.name
        path.write_text(json.dumps(report, indent=2))
        if (reader / "summary.json").exists():
            raise RuntimeError("Reader panel completed before registration")
        os.link(path, frozen)


def score(study, reader):
    frozen = reader / "evidence-policy-freeze.json"
    forecast = json.loads(frozen.read_bytes())
    inputs = json.loads((reader / "retrieval.json").read_bytes())
    protocol = json.loads((study / "protocol.json").read_bytes())
    if (forecast["inputs"] != inputs or forecast["training_corpus_sha256"] != protocol["input_sha256"]
            or digest(json.loads((reader / "jobs.json").read_bytes())) != inputs["plan"]):
        raise ValueError("Completed reader panel differs from the frozen predictions")
    windows = load_outcomes(study, reader)
    by_key = {(r["dialogue"], r["session"]): r for r in forecast["forecasts"]}
    train = set(protocol["train"])
    if len(by_key) != len(forecast["forecasts"]) or set(forecast["fits"]) != train:
        raise ValueError("Invalid or duplicate registered forecasts")
    rows, counts = [], defaultdict(int)
    for sid, pair in windows.items():
        for w in pair:
            r = by_key[sid, w["start"] + 1]
            if (set(forecast["fits"][sid]["training_dialogues"]) != train - {sid}
                    or not math.isclose(r["p"], 1/(1+math.exp(-r["logit"])))):
                raise ValueError("Invalid held-dialogue probability")
            rows.append(dict(r, activity=w["activity"]*w["n"], temporal=w["temporal"]*w["n"], questions=w["n"]))
            counts[sid] += w["n"]
    return {"forecast_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "questions": sum(counts.values()),
            "reader_f1": compare(rows, counts), "windows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze","score"))
    parser.add_argument("study", type=Path)
    parser.add_argument("reader", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        if args.corpus is None:
            parser.error("freeze requires --corpus")
        freeze(args.study, args.corpus, args.reader)
    else:
        print(json.dumps(score(args.study, args.reader), indent=2))
