"""Keep a common single-turn retrieval index and spend six read slots with or without local context."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.factorial_rollout import CELLS
from experiments.controller.paired_grounding import window
from experiments.controller.source_compression import file_hash, register
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn


def retrieve_slots(ranking, lengths, schedule, budget=6):
    if (
        len(lengths) != 2
        or len(schedule) != 2
        or not all(bit in (0, 1) for bit in schedule)
        or min(lengths) <= 0
        or budget < 1
        or sorted(ranking) != list(range(sum(lengths)))
    ):
        raise ValueError("Expected a complete ranked index of two nonempty sessions")
    result = []
    for i in ranking:
        if i in result:
            continue
        result.append(i)
        if len(result) == budget:
            break
        span = int(i >= lengths[0])
        start = lengths[0] if span else 0
        partner = (i - start) ^ 1
        if schedule[span] and partner < lengths[span] and start + partner not in result:
            result.append(start + partner)
            if len(result) == budget:
                break
    if len(result) != min(budget, sum(lengths)) or len(set(result)) != len(result):
        raise ValueError("Read budget was not matched")
    return result


def summarize(rows):
    by_window = defaultdict(list)
    for r in rows:
        by_window[r["dialogue"], r["start"]].append(r)
    schedules = ["".join(map(str, k)) for k in CELLS]
    per_window = []
    for (sid, start), group in by_window.items():
        recall = {k: mean(r["cells"][k]["recall"] for r in group) for k in schedules}
        best = max(schedules, key=recall.get)
        per_window.append(
            {
                "dialogue": sid,
                "start": start,
                "n": len(group),
                "recall": recall,
                "oracle": best,
                "matched": (recall[best] + recall[best[::-1]]) / 2,
            }
        )

    def comparison(g):
        n = sum(w["n"] for w in g)
        a = {k: sum(w["n"] * w["recall"][k] for w in g) / n for k in schedules}
        return {
            **a,
            "oracle": sum(w["n"] * w["recall"][w["oracle"]] for w in g) / n,
            "same_count": sum(w["n"] * w["matched"] for w in g) / n,
        }

    aggregate = comparison(per_window)
    folds = {sid: comparison([w for w in per_window if w["dialogue"] == sid]) for sid, _ in by_window}
    positive = sum(v["oracle"] > v["same_count"] + 1e-12 for v in folds.values())

    def costs(group):
        return {
            k: {
                "all_gold": mean(r["cells"][k]["complete"] for r in group),
                "recall": mean(r["cells"][k]["recall"] for r in group),
                "mean_source_words": mean(r["cells"][k]["words"] for r in group),
            }
            for k in schedules
        }

    by_type = {}
    for cat in sorted({r["type"] for r in rows}):
        part = [r for r in rows if r["type"] == cat]
        by_type[cat] = {"n": len(part), "schedules": costs(part)}
    return {
        "questions": len(rows),
        "windows": per_window,
        "aggregate": aggregate,
        "schedules": costs(rows),
        "per_dialogue": folds,
        "per_type": by_type,
        "positive_placement_dialogues": positive,
        "advance_to_reader": aggregate["oracle"] - max(aggregate["00"], aggregate["11"]) >= 0.02
        and aggregate["oracle"] - aggregate["same_count"] >= 0.01
        and positive >= 4,
    }


def run(study, corpus, dest):
    protocol = json.loads((study / "protocol.json").read_bytes())
    raw = corpus.read_bytes()
    ids = set(protocol["train"])
    if (
        len(ids) != 6
        or ids & set(protocol["validation"] + protocol["test"])
        or hashlib.sha256(raw).hexdigest() != protocol["input_sha256"]
    ):
        raise ValueError("Unexpected training corpus or split")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single CUDA lease")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.76 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise RuntimeError("Insufficient memory for bounded Qwen encoding")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, model = init_context_model(protocol["retriever"])
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("Qwen is not on the leased GPU")

    def embed(text):
        return torch.as_tensor(encode_batch(model, text, 2, normalize=True), device="cuda")

    processor = LoCoMoProcessor(chunk_mode="full-session")
    rows = []
    for c in json.loads(raw):
        sid = c["sample_id"]
        if sid not in ids:
            continue
        for start in (1, 3):
            convo = window(c, start, 2)
            turns, items, lengths = [], [], []
            for n in (start, start + 1):
                key = f"session_{n}"
                group = convo["conversation"][key]
                date = convo["conversation"][key + "_date_time"]
                lengths.append(len(group))
                turns.extend(t["dia_id"] for t in group)
                items.extend(f"{date} | {processor._format_turn(t)}" for t in group)
            if len(set(turns)) != len(turns):
                raise ValueError("Duplicate source turns")
            qa = convo["qa"]
            sims = embed([q["question"] for q in qa]) @ embed(items).T
            for q, indices in zip(
                qa, sims.argsort(dim=1, descending=True, stable=True).tolist(), strict=True
            ):
                gold = set(q["evidence"])
                cells = {}
                for bits in CELLS:
                    slots = retrieve_slots(indices, lengths, bits)
                    found = [turns[i] for i in slots]
                    cells["".join(map(str, bits))] = {
                        "turns": found,
                        "recall": len(gold.intersection(found)) / len(gold),
                        "complete": gold.issubset(found),
                        "words": sum(len(items[i].split()) for i in slots),
                    }
                rows.append(
                    {
                        "dialogue": sid,
                        "start": start,
                        "original_index": q["original_index"],
                        "type": str(q["category"]),
                        "gold": sorted(gold),
                        "single_turn_ranking": [turns[i] for i in indices],
                        "cells": cells,
                    }
                )
    if len(rows) != 123 or len({(r["dialogue"], r["start"]) for r in rows}) != 12:
        raise ValueError("Incomplete early training panel")
    report = {
        "plan": file_hash(dest / "plan.json"),
        "corpus": protocol["input_sha256"],
        "budget_turns": 6,
        "result": summarize(rows),
        "rows": rows,
    }
    register(dest / "screen.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "corpus", "dest"):
        p.add_argument(name, type=Path)
    run(**vars(p.parse_args()))
