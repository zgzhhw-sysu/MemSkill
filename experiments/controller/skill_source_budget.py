"""Bound training evidence recall when either of two frozen skills can retain five source turns."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from types import SimpleNamespace

import torch

from experiments.controller.probe_omitted import atomic_json
from src.data_processing.locomo import LoCoMoProcessor
from src.eval import get_evaluator
from src.rag_utils import get_embeddings, set_retriever_flash_attn


SKILLS = {"activity": "capture_activity_details", "temporal": "capture_temporal_context"}


def retained_turns(turns, date, selected, processor, budget):
    """Keep the selected full turns in source order, with the same date and per-skill count."""
    by_id = {t["dia_id"]: f"{date} | {processor._format_turn(t)}" for t in turns}
    if (len(by_id) != len(turns) or set(selected) != SKILLS.keys()
            or any(len(ids) != min(budget, len(turns)) or len(set(ids)) != len(ids)
                   or not set(ids).issubset(by_id) for ids in selected.values())):
        raise ValueError("Frozen skill choices do not match this source session and turn budget")
    return {arm: [text for tid, text in by_id.items() if tid in ids] for arm, ids in selected.items()}


def dialogue_recall(entries, questions):
    sums = {a: sum(row["covered_question_mass"][a] for row in entries) for a in SKILLS}
    gains = [r["covered_question_mass"]["temporal"] - r["covered_question_mass"]["activity"] for r in entries]
    # Ties choose the stronger fixed skill in this dialogue when determining the oracle's use count.
    tie_choice = sums["temporal"] > sums["activity"]
    choices = [d > 0 if abs(d) > 1e-9 else tie_choice for d in gains]
    recall = {arm: val / questions for arm, val in sums.items()}
    recall["best_per_span"] = (sums["activity"] + sum(d * choice for d, choice in zip(gains, choices))) / questions
    recall["same_count_within_dialogue"] = (sums["activity"] + mean(choices) * sum(gains)) / questions
    return {"questions": questions, "sessions": len(entries), "recall": recall, "best_temporal_count": sum(choices),
            "activity_better": sum(d < -1e-9 for d in gains), "temporal_better": sum(d > 1e-9 for d in gains)}


def screen(study, corpus):
    protocol = json.loads((study / "protocol.json").read_bytes())
    raw = corpus.read_bytes()
    train = set(protocol["train"])
    data = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in train}
    if (hashlib.sha256(raw).hexdigest() != protocol["input_sha256"] or len(train) != 6 or data.keys() != train
            or train.intersection(protocol["validation"] + protocol["test"])):
        raise ValueError("Expected the original six disjoint training dialogues")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA lease is required for the source selection screen")
    skills = {s["name"]: s["description"] for s in protocol["skills"]}
    descriptions = {name: skills[name] for name in SKILLS.values()}
    budget = protocol["retrieval_k"][1]
    if budget != 5:
        raise ValueError("The two skills use a fixed five-turn budget per session")
    processor = LoCoMoProcessor(chunk_mode="full-session")
    evaluator = get_evaluator("locomo", SimpleNamespace())
    set_retriever_flash_attn(False)

    def embed(strings, mode):
        vectors = get_embeddings(protocol["retriever"], strings, mode, batch_size=2)
        return torch.as_tensor(vectors, device="cuda", dtype=torch.float32)

    queries = embed(list(descriptions.values()), "query")
    windows, exclusions, dialogues = [], {}, {}
    for sid in sorted(train):
        conversation = data[sid]["conversation"]
        indices = sorted(int(k.split("_")[1]) for k in conversation if re.fullmatch(r"session_\d+", k))
        ids = [t["dia_id"] for i in indices for t in conversation[f"session_{i}"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate source turn IDs in {sid}")
        eligible = evaluator.filter_qa_list(data[sid]["qa"])
        gold = [set(q["evidence"]) for _, q in eligible if q.get("evidence") and set(q["evidence"]).issubset(ids)]
        exclusions[sid] = len(eligible) - len(gold)
        entries = []
        for i in indices:
            turns = conversation[f"session_{i}"]
            strings = [processor._format_turn(t) for t in turns]
            similarity = queries @ embed(strings, "context").T
            ranks = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :budget].tolist()
            chosen = {arm: [turns[j]["dia_id"] for j in rank] for arm, rank in zip(SKILLS, ranks, strict=True)}
            # Evidence annotations enter only after both selections have been made from the source.
            mass = {arm: sum(len(set(pick) & target) / len(target) for target in gold) for arm, pick in chosen.items()}
            entries.append({"dialogue": sid, "session": i, "source_turns": len(turns), "selected": chosen,
                            "selected_words": {a: sum(len(strings[j].split()) for j in rank)
                                               for a, rank in zip(SKILLS, ranks, strict=True)},
                            "covered_question_mass": mass})
        windows.extend(entries)
        dialogues[sid] = dialogue_recall(entries, len(gold))
    n = sum(v["questions"] for v in dialogues.values())
    comparisons = ("activity", "temporal", "same_count_within_dialogue")
    averages = {a: sum(v["questions"] * v["recall"][a] for v in dialogues.values()) / n
                for a in (*comparisons, "best_per_span")}
    positive = sum(v["recall"]["best_per_span"] > max(v["recall"][a] for a in comparisons) + 1e-9
                   for v in dialogues.values())
    return {"scope": "retained fraction of annotated evidence turns; no reader F1 or deployable policy",
            "checkpoint": protocol["checkpoint_sha256"], "skill_descriptions": descriptions,
            "turn_budget_per_skill": budget, "questions": n, "sessions": len(windows),
            "unmapped_or_missing_evidence": exclusions, "mean_recall": averages,
            "dialogues_with_position_headroom": positive,
            "worth_measuring_reader_f1": averages["best_per_span"] - max(averages[a] for a in comparisons) >= .02 and positive >= 4,
            "per_dialogue": dialogues, "rows": windows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "locomo", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    atomic_json(args.output, screen(args.study, args.locomo))
