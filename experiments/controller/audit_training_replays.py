"""Compare frozen decisions on recorded single-span interventions in held-out training dialogues."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller.paired_policy import fit_anchored_credit, fit_policy, match_training_probability
from experiments.controller.resume_status import load_episode


def read_windows(study, protocol):
    windows = {}
    for start, end in protocol["train_sessions"]:
        for sample in protocol["train"]:
            directory = study / "train" / f"{sample}_s{start}"
            rows = load_episode(directory, sample, end - start + 1,
                                deployment_writer=protocol.get("deployment_writer", False))
            trace, = (directory / "traces").glob("*.jsonl")
            events = map(json.loads, trace.read_text().splitlines())
            diagnoses = [e["diagnosis"] for e in events if e["kind"] == "span_credit"]
            windows[sample, start] = list(zip(rows, diagnoses, strict=True))
    return windows


def compare(study):
    protocol = json.loads((study / "protocol.json").read_text())
    windows = read_windows(study, protocol)
    counts = Counter()
    records = []
    for held in protocol["train"]:
        train = [r for (sample, _), pairs in windows.items() if sample != held for r, _ in pairs]
        terminal = fit_policy(train, 0)
        anchor = fit_anchored_credit(train, terminal)
        weights = {"terminal": terminal, "anchor": anchor, "rate": match_training_probability(train, terminal, anchor)}
        for (sample, start), pairs in windows.items():
            if sample != held:
                continue
            for row, diagnosis in pairs:
                control, swap = diagnosis.get("control", {}), diagnosis.get("swap", {})
                questions = diagnosis["questions"]
                counts["spans"] += 1
                counts["linked"] += len(questions)
                if diagnosis.get("execution_fault") or control.get("contract_fault") or swap.get("contract_fault"):
                    counts["write_fault"] += len(questions)
                    continue
                state = F.normalize(torch.tensor(row["state"]), dim=0)
                actions = {arm: (row["base_logit"] + w[0] + state @ w[1:]).item() >= 0 for arm, w in weights.items()}
                actions.update(always=True, plain=False)
                for question in questions:
                    key = str(question["qa_index"])
                    original = control.get("qa", {}).get(key)
                    other = swap.get("qa", {}).get(key)
                    if (not question["control_matched"] or not original or not original["ok"]
                            or original["f1"] != question["original_f1"]):
                        counts["unmatched_original"] += 1
                        continue
                    if not other or not other["ok"]:
                        counts["no_swap_answer"] += 1
                        continue
                    scores = {arm: (original if row["use"] == decision else other)["f1"]
                              for arm, decision in actions.items()}
                    records.append({"dialogue": sample, "window": start, "span": row["span"],
                                    "question": question["original_index"], "actions": actions, "scores": scores})
    if not records:
        raise ValueError("No linked questions have an observed answer for both actions")
    eligible = {(r["dialogue"], r["window"], r["span"]): r["actions"] for r in records}
    means = {a: mean(r["scores"][a] for r in records) for a in records[0]["scores"]}
    differences = {a: {"paired_delta": means["anchor"] - means[a],
                       "spans_switched": sum(v["anchor"] != v[a] for v in eligible.values()),
                       "improved": sum(r["scores"]["anchor"] > r["scores"][a] for r in records),
                       "worse": sum(r["scores"]["anchor"] < r["scores"][a] for r in records)}
                   for a in means if a != "anchor"}
    return {"scope": "one decision on a held training dialogue, with the suffix exploration noise fixed",
            "coverage": dict(counts), "matched_span_question_links": len(records), "matched_spans": len(eligible),
            "unique_questions": len({(r["dialogue"], r["question"]) for r in records}),
            "mean_link_f1": means, "anchor_minus": differences, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    print(json.dumps(compare(parser.parse_args().study), indent=2))
