"""Estimate terminal span policies on fixed, measured reader trajectories."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.case_credit import KINDS, corrections, vote
from experiments.controller.factorial_rollout import read
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.retrieval_competition import budgets, summarize
from experiments.controller.source_compression import file_hash, register
from src.evaluation_cache import digest

ARMS = (
    "credit",
    "semantic",
    "verified",
    "permuted",
    "rate",
    "raw",
    "native",
    "terminal",
)


def verified_choice(dest, name, target, registration):
    path = dest / "choices" / (name + ".json")
    choice = read(path)
    if (
        choice["registration"] != registration
        or choice["source"] != target["source"]
        or len(choice["responses"]) != 6
    ):
        raise ValueError("Changed source, model or choice receipts")
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    votes = {}
    for j, kind in enumerate(KINDS):
        votes[kind] = []
        for reverse, text in enumerate(target["prompts"][kind]):
            key = choice["responses"][2 * j + reverse]
            rec = read(dest / "responses" / (key + ".json"))
            req = {
                "model": "llama3-8b",
                "temperature": 0,
                "max_tokens": 3,
                "messages": [{"role": "user", "content": text}],
            }
            if rec["request"] != key or key != digest(req):
                raise ValueError("Wrong intervention response")
            votes[kind].append(vote(rec["reply"], bool(reverse)))
            for k in usage:
                usage[k] += rec["usage"][k]
    if votes != choice["votes"] or corrections(votes) != choice["offsets"]:
        raise ValueError("Frozen decision disagrees with its inference receipts")
    return choice, usage


def endpoints(folder, prefix, target, expected_cases):
    cases, cost, scores = [], [], []
    fixed = None
    for a in (0, 1):
        bits = f"{prefix}{a}"
        cell = folder / bits
        (trace,) = (cell / "traces").glob("*.jsonl")
        spans = [
            s
            for line in trace.read_bytes().splitlines()
            if (s := json.loads(line))["kind"] == "span"
        ]
        first, last = spans
        chosen = last["selected_skills"]
        state = digest(
            [first, last["refs"], last["retrieved"], last["source"], chosen[:-1]]
        )
        if (
            len(chosen) != 3
            or digest(last["source"]) != target["source"]
            or chosen[-1]["name"]
            != ("capture_temporal_context", "capture_activity_details")[a]
            or chosen[-1]["source_rule"]
            != ("dated_single_turns", "dated_adjacent_turns")[a]
            or (fixed is not None and fixed != state)
        ):
            raise ValueError("Mismatched ordered skills, state or focal source")
        fixed = state
        path = cell / "scores.json"
        answers = read(path)["answers"]
        y = [r["f1"] for r in answers]
        if [r["qa_index"] for r in answers] != list(
            range(len(y))
        ) or y != expected_cases[a]:
            raise ValueError("Archived reader outcomes do not reproduce")
        cases.append(y)
        cost.append(budgets(folder, bits))
        scores.append(path)
    return cases, cost, scores


def trajectory(base, name, target, early, old):
    src, late = base / "source-compression", base / "late-pairs/outcomes"
    factorial = target["panel"] == "factorial"
    if factorial:
        w, ref = early[name], old[name]
        indices, types = w["questions"], w["types"]
        folder = src / "factorial" / name
        root = ref["probabilities"]["0"]["native"]
        exposure = [(0, 1 - root), (1, root)]
        expected = w["cases"]
        native_f1 = [r["f1"]["native"] for r in ref["rows"]]
    else:
        folder = late / name
        pair = read(folder / "pair.json")
        ref = read(src / "retrieval-competition/pairs" / (name + ".json"))
        b = target["background"]
        if (
            ref["inputs"]["pair"] != file_hash(folder / "pair.json")
            or pair["paths"] != [[b, 0], [b, 1]]
            or any(
                pair[k] != target[k]
                for k in ("dialogue", "start", "pivot", "background")
            )
        ):
            raise ValueError("Randomized terminal pair changed")
        indices, types = target["questions"], target["types"]
        if [(r["original_index"], r["type"]) for r in pair["questions"]] != list(
            zip(indices, types, strict=True)
        ):
            raise ValueError("Different later questions")
        expected = {
            f"{b}{a}": [r[k] for r in pair["questions"]]
            for a, k in enumerate(("temporal", "activity"))
        }
        native_f1 = [r["focal"]["native"] for r in pair["questions"]]
        exposure = [(b, 1.0)]
    nodes, paths = [], []
    for b, weight in exposure:
        cases, tokens, files = endpoints(
            folder, b, target, [expected[f"{b}{a}"] for a in (0, 1)]
        )
        paths.extend(files)
        prior = ref["probabilities"][str(1 + b)]
        nodes.append(
            {
                "prefix": b,
                "weight": weight,
                "cases": cases,
                "costs": tokens,
                "native": prior["native"],
                "raw": prior["return_only"],
                "terminal": prior["terminal"],
            }
        )
    return {
        "window": name,
        "dialogue": target["dialogue"],
        "indices": indices,
        "types": types,
        "nodes": nodes,
        "native_f1": native_f1,
        "first_score_ns": min(f.stat().st_mtime_ns for f in paths),
        "inputs": {
            "paths": {str(f.relative_to(folder)): file_hash(f) for f in paths},
        },
    }


def integrate(nodes, arm, field, index):
    return sum(
        n["weight"]
        * ((1 - n["p"][arm]) * n[field][0][index] + n["p"][arm] * n[field][1][index])
        for n in nodes
    )


def score(windows, device):
    if not windows:
        return {"questions": 0, "windows": 0}, []
    tasks = [(w, n) for w in windows for n in w["nodes"]]
    native = torch.tensor(
        [n["native"] for _, n in tasks], dtype=torch.float64, device=device
    )
    weights = native.new_tensor([n["weight"] for _, n in tasks])
    logits = native.logit()
    probabilities = {
        arm: (
            logits + native.new_tensor([w["offsets"][arm] for w, _ in tasks])
        ).sigmoid()
        for arm in ("credit", "semantic", "verified", "permuted")
    }
    shift = matched_offset(logits, probabilities["credit"], weights)
    probabilities["rate"] = (logits + shift).sigmoid()
    if abs(float(weights @ (probabilities["credit"] - probabilities["rate"]))) > 1e-8:
        raise ValueError("Rate control failed to match policy visits")
    for i, (_, n) in enumerate(tasks):
        n["p"] = {
            **{arm: float(t[i]) for arm, t in probabilities.items()},
            **{k: n[k] for k in ("raw", "native", "terminal")},
        }
    reports = []
    for w in windows:
        nodes = w["nodes"]

        rows = []
        for i, (idx, kind) in enumerate(zip(w["indices"], w["types"], strict=True)):
            vals = {k: integrate(nodes, k, "cases", i) for k in ARMS}
            if not math.isclose(vals["native"], w["native_f1"][i], abs_tol=1e-10):
                raise ValueError("Published native F1 did not reproduce")
            rows.append({"index": idx, "type": kind, "f1": vals})
        costs = {
            k: {op: integrate(nodes, k, "costs", op) for op in ("reader", "executor")}
            for k in ARMS
        }
        reports.append(
            {
                "dialogue": w["dialogue"],
                "window": w["window"],
                "inputs": w["inputs"],
                "votes": w["votes"],
                "decision_tokens": w["decision_tokens"],
                "strict": w["strict"],
                "expected_tokens": costs,
                "probabilities": {n["prefix"]: n["p"] for n in nodes},
                "focal_probabilities": {
                    k: sum(n["weight"] * n["p"][k] for n in nodes) for k in ARMS
                },
                "rows": rows,
            }
        )
    summary = summarize(
        reports,
        ARMS,
        "credit",
        ("native", "terminal", "semantic", "permuted", "rate", "raw"),
    )
    summary["matched_logit_shift"] = shift
    summary["mean_decision_tokens"] = mean(
        sum(w["decision_tokens"].values()) for w in reports
    )
    summary["spans_sensitive_to_labels"] = sum(
        w["offsets"]["verified"] != w["offsets"]["permuted"] for w in windows
    )
    return summary, reports


def run(base):
    src = base / "source-compression"
    dest = src / "case-credit"
    frozen = dest / "prompts.json"
    registration = file_hash(frozen)
    early = {
        w["window"]: w for w in read(src / "retrieval-competition/fit.json")["training"]
    }
    old = {
        w["window"]: w
        for w in read(src / "retrieval-competition/calibration.json")["windows"]
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one bounded GPU lease for propensity calculations")
    torch.cuda.set_per_process_memory_fraction(0.06)
    from experiments.controller.case_credit_bounds import bounds

    groups = {"factorial": [], "later": []}
    envelopes = {"factorial": [], "later": []}
    for name, t in read(frozen)["targets"].items():
        if (
            t["panel"] == "later"
            and not (src / "retrieval-competition/pairs" / (name + ".json")).exists()
        ):
            continue
        w = trajectory(base, name, t, early, old)
        envelopes[t["panel"]].append(bounds(w, dest, t, "cuda"))
        decision = dest / "choices" / (name + ".json")
        if not decision.exists():
            continue
        choice, usage = verified_choice(dest, name, t, registration)
        w["inputs"]["decision"] = file_hash(decision)
        w.update(
            offsets=choice["offsets"],
            votes=choice["votes"],
            decision_tokens=usage,
            strict=t["panel"] == "later"
            and decision.stat().st_mtime_ns < w["first_score_ns"],
        )
        groups[t["panel"]].append(w)
    for panel, items in envelopes.items():
        if not items:
            continue
        folder = dest / "bounds" / panel
        folder.mkdir(parents=True, exist_ok=True)
        n = sum(v["questions"] for v in items)
        totals = {
            k: sum(v[k] * v["questions"] for v in items) / n
            for k in ("native", "min", "max", "question_oracle")
        }
        result = {
            "registration": registration,
            "windows": items,
            "questions": n,
            "observed_votes": sum(r["observed"] for r in items),
            "affected_questions": sum(r["affected"] for r in items),
            **totals,
        }
        register(folder / (digest(result) + ".json"), result)
    for panel, windows in groups.items():
        if not windows:
            continue
        folder = dest / panel
        folder.mkdir(exist_ok=True)
        stamp = digest([w["inputs"] for w in windows])[:16]
        path = folder / f"through-{len(windows):02d}-{stamp}.json"
        if path.exists():
            continue
        summary, rows = score(windows, "cuda")
        fresh = [w for w in windows if w["strict"]]
        prospect, _ = score(fresh, "cuda")
        result = {
            "registration": registration,
            "registered_questions": 91 if panel == "factorial" else 209,
            "all": summary,
            "strict": prospect,
            "windows": rows,
        }
        if panel == "later":
            f1 = summary["f1"]
            result["eligible_for_development"] = (
                summary["questions"] >= 0.9 * 209
                and sum(
                    g["f1"]["credit"] > g["f1"]["native"]
                    for g in summary["by_dialogue"].values()
                )
                >= 4
                and all(f1["credit"] >= f1[k] + 0.02 for k in ("native", "terminal"))
                and all(
                    f1["credit"] > f1[k]
                    for k in ("semantic", "permuted", "raw", "rate")
                )
                and bool(fresh)
                and prospect["f1"]["credit"] > prospect["f1"]["native"]
            )
        register(path, result)
        if len(windows) == (12 if panel == "factorial" else 25):
            register(folder / "summary.json", result)
        print(panel, len(windows), summary["questions"], summary["f1"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    run(parser.parse_args().base)
