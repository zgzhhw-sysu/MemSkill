"""Audit positive and negative training credit caused by equivalent English duration spellings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from types import SimpleNamespace

from experiments.controller.resume_status import load_episode
from src.eval import get_evaluator


SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
NUMBERS = {word: i for i, word in enumerate(SMALL)}
DURATION = re.compile(r"(\w+) (second|minute|hour|day|week|month|year)s?", re.I)


def duration_alias(before, after, reference):
    """Flag a whole-reference duration in two surface forms for conservative attribution."""
    ref = DURATION.fullmatch(str(reference).strip())
    if not ref:
        return False
    number, unit = ref.group(1).lower(), ref.group(2).lower()
    index = NUMBERS.get(number)
    if index is None:
        if number not in {str(i) for i in range(len(SMALL))}:
            return False
        index = int(number)
    def spelling(answer):
        text = answer.lower().strip().strip('".! ')
        qualifiers = r"not|never|no|almost|about|approximately|over|under|nearly|at least|at most|less than|more than|if|might|would|could|will"
        if re.search(rf"\b(?:{qualifiers})\b|n['’]t\b", text):
            return None
        match = re.search(rf"(?:^|\bfor )({index}|{SMALL[index]}) {unit}s?$", text)
        return match.group(1) if match else None
    old, new = spelling(before), spelling(after)
    return old is not None and new is not None and old != new


def audit(study, data_path):
    protocol = json.loads((study / "protocol.json").read_text())
    raw = data_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["input_sha256"]:
        raise ValueError("Training corpus differs from the frozen protocol")
    data = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in protocol["train"]}
    evaluator = get_evaluator("locomo", SimpleNamespace())
    flagged, adjusted, pending = [], {}, []
    counts = Counter()
    for start, end in protocol["train_sessions"]:
        for sid in protocol["train"]:
            path = study / "train" / f"{sid}_s{start}"
            if not path.exists():
                pending.append(path.name)
                continue
            rows = load_episode(path, sid, end-start+1, deployment_writer=protocol.get("deployment_writer", False))
            trace, = (path / "traces").glob("*.jsonl")
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            records = [e["diagnosis"] for e in events if e["kind"] == "span_credit"]
            ids = json.loads((path / "questions.json").read_text())
            counts["windows"] += 1
            for row, rec in zip(rows, records, strict=True):
                deltas = []
                for q in rec["questions"]:
                    delta = q["controller_delta"]
                    if delta:
                        counts["credited_links"] += 1
                        qi, orig = q["qa_index"], q["original_index"]
                        if ids[qi] != orig or q["cause"] != "selection":
                            raise ValueError("Credit is not linked to an observed selection question")
                        qa = data[sid]["qa"][orig]
                        gold = evaluator.get_ground_truth(qa)
                        control, swap = (rec[arm]["qa"][str(qi)] for arm in ("control", "swap"))
                        if duration_alias(control["answer"], swap["answer"], gold):
                            flagged.append({"episode": path.name, "span": row["span"], "original_index": orig,
                                            "reference": gold, "sampled": control, "alternative": swap,
                                            "sampled_grounded": row["use"], "old_delta": delta})
                            delta = 0.
                    deltas.append(delta)
                adjusted[path.name, row["span"]] = mean(deltas) if deltas else 0.
    return {"scope": "training-only explicit duration aliases, with native F1 retained", "counts": dict(counts),
            "flagged_links": flagged, "pending_windows": pending}, adjusted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("locomo", type=Path)
    args = parser.parse_args()
    report, _ = audit(args.study, args.locomo)
    print(json.dumps(report, indent=2))
