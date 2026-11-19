"""Compare frozen decisions on recorded single-span interventions in held-out training dialogues."""
import argparse
from collections import Counter, defaultdict
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


def probability_placement(records):
    """Evaluate a single randomized decision using the two measured outcomes at its state."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r["dialogue"], r["window"], r["span"]].append(r)
    dialogues = defaultdict(list)
    for (dialogue, _, _), links in grouped.items():
        probabilities = links[0]["probabilities"]
        if any(link["probabilities"] != probabilities for link in links):
            raise ValueError("A single span must have the same policy for every linked question")
        plain = mean(link["scores"]["plain"] for link in links)
        gain = mean(link["scores"]["always"] for link in links) - plain
        dialogues[dialogue].append((probabilities, plain, gain))
    by_dialogue = {}
    for dialogue, spans in dialogues.items():
        arms = {}
        for arm in spans[0][0]:
            mean_p = mean(p[arm] for p, _, _ in spans)
            expected = mean(plain + p[arm] * gain for p, plain, gain in spans)
            shuffled = mean(plain + mean_p * gain for _, plain, gain in spans)
            arms[arm] = {"mean_probability": mean_p, "expected_f1": expected,
                         "permuted_expected_f1": shuffled, "placement_delta": expected - shuffled}
        by_dialogue[dialogue] = {"spans": len(spans), "arms": arms}
    n = len(grouped)
    aggregate = {arm: {key: sum(d["spans"] * d["arms"][arm][key] for d in by_dialogue.values()) / n
                       for key in metrics} for arm, metrics in next(iter(by_dialogue.values()))["arms"].items()}
    return {"weighting": "equal spans; average observed linked questions within each span",
            "permutation": "all permutations of the same probabilities among covered spans within each held dialogue",
            "spans": n, "dialogues": by_dialogue, "arms": aggregate}


def compare(study, *, centered_candidate=False):
    protocol = json.loads((study / "protocol.json").read_text())
    windows = read_windows(study, protocol)
    counts = Counter()
    records = []
    for held in protocol["train"]:
        train = [r for (sample, _), pairs in windows.items() if sample != held for r, _ in pairs]
        terminal = fit_policy(train, 0)
        anchor = fit_anchored_credit(train, terminal)
        weights = {"terminal": terminal, "anchor": anchor, "rate": match_training_probability(train, terminal, anchor)}
        if centered_candidate:
            candidate = fit_anchored_credit(train, terminal, center_states=True)
            weights["centered"] = match_training_probability(train, candidate, anchor)
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
                logits = {arm: row["base_logit"] + w[0] + state @ w[1:] for arm, w in weights.items()}
                actions = {arm: logit.item() >= 0 for arm, logit in logits.items()}
                probabilities = {arm: logit.sigmoid().item() for arm, logit in logits.items()}
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
                                    "question": question["original_index"], "actions": actions,
                                    "probabilities": probabilities, "scores": scores})
    if not records:
        raise ValueError("No linked questions have an observed answer for both actions")
    eligible = {(r["dialogue"], r["window"], r["span"]): r["actions"] for r in records}
    means = {a: mean(r["scores"][a] for r in records) for a in records[0]["scores"]}
    differences = {a: {"paired_delta": means["anchor"] - means[a],
                       "spans_switched": sum(v["anchor"] != v[a] for v in eligible.values()),
                       "improved": sum(r["scores"]["anchor"] > r["scores"][a] for r in records),
                       "worse": sum(r["scores"]["anchor"] < r["scores"][a] for r in records)}
                   for a in means if a != "anchor"}
    expectation = probability_placement(records)
    report = {"scope": "one decision on a held training dialogue, with the suffix exploration noise fixed",
            "coverage": dict(counts), "matched_span_question_links": len(records), "matched_spans": len(eligible),
            "unique_questions": len({(r["dialogue"], r["question"]) for r in records}),
            "mean_link_f1": means, "anchor_minus": differences,
            "single_decision_expectation": expectation, "records": records}
    if centered_candidate:
        required = ("expected_f1", "placement_delta")
        contrasts = {a: {k: expectation["arms"]["centered"][k] - expectation["arms"][a][k] for k in required}
                     for a in ("anchor", "rate")}
        report["centered_training_screen"] = {"differences": contrasts,
                                               "advance": all(v > 0 for group in contrasts.values() for v in group.values())}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--centered-candidate", action="store_true",
                        help="Test a centered, dispersion-scaled credit correction at the original training mean probability")
    args = parser.parse_args()
    print(json.dumps(compare(args.study, centered_candidate=args.centered_candidate), indent=2))
