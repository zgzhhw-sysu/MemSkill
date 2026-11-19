"""Recompute paired task F1 and gate probabilities on identical saved states."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from experiments.designer.designer_pilot import prefix
from src.eval import get_evaluator


def read(path):
    return json.loads(path.read_text())


def bootstrap_by_type(deltas, types, seed=20261120):
    """Stratify by question type; LongMemEval has one observation per history."""
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    types = np.asarray(types)
    replicates = np.zeros(10000)
    for group in sorted(set(types)):
        values = deltas[types == group]
        replicates += rng.choice(values, (len(replicates), len(values))).sum(axis=1) / len(deltas)
    return np.quantile(replicates, [.025, .975]).tolist()


def probability(decision, weights):
    state = np.asarray(decision["state"], dtype=np.float64)
    return float(expit(decision["base_logit"] + weights[0] + np.dot(weights[1:], state / max(np.linalg.norm(state), 1e-12))))


def audit(directory, dataset_file, phase, arms):
    cohort = read(Path(__file__).with_name("protocols") / "paired_grounding_cohort.json")
    external = phase == "longmemeval"
    spec = cohort["external" if external else "primary"]
    if hashlib.sha256(dataset_file.read_bytes()).hexdigest() != spec["file_sha256"]:
        raise ValueError("Input fingerprint differs from cohort")
    data = read(dataset_file)
    ev = get_evaluator("longmemeval" if external else "locomo", SimpleNamespace())
    if external:
        by_id = {str(c["question_id"]): c for c in data}
        conversations = [(sid, by_id[sid]) for sid in spec["ids"]]
    else:
        group = [prefix(c, 6) for c in data[6:8]] if phase == "validation" else data[8:10]
        conversations = [(c["sample_id"], c) for c in group]
    if "credit" not in arms or "terminal" not in arms:
        raise ValueError("Both credit and terminal results are required")
    weights = {arm: np.asarray(read(directory / (arm + ".json"))) for arm in ("credit", "terminal")}
    scores, probabilities, extra = {}, {}, {}
    groups, answerable = {}, {}
    choices = defaultdict(dict)
    for arm in arms:
        scored, probabilities[arm] = {}, []
        counts = Counter()
        for sid, convo in conversations:
            parent = directory / phase / arm / sid
            report = read(parent / "scores.json")
            refs = {0: convo} if external else dict(ev.filter_qa_list(convo["qa"]))
            if len(report["answers"]) != len(refs) or {r["qa_index"] for r in report["answers"]} != refs.keys():
                raise ValueError(f"Incomplete data in {parent}")
            ids = read(parent / "question_ids.json")
            for r in report["answers"]:
                index = r["qa_index"]
                q = refs[index]
                key = sid, ids[index]
                f1 = ev.compute_f1(r["prediction"], ev.get_ground_truth(q), q)
                if (key in scored or r["sample_id"] != sid or min(r["tokens"]) < 1
                        or not math.isclose(f1, r["f1"], abs_tol=1e-9)):
                    raise ValueError(f"Invalid score for {key}")
                scored[key], groups[key] = f1, str(q["question_type"] if external else q["category"])
                answerable[key] = not (external and sid.endswith("_abs"))
            decisions = read(parent / "decisions.json")
            expected_spans = len(convo["haystack_sessions"]) if external else sum(
                k.startswith("session_") and not k.endswith("_date_time") for k in convo["conversation"])
            if len(decisions) != expected_spans:
                raise ValueError(f"Partial history at {parent}: {len(decisions)}/{expected_spans}")
            traces = list((parent / "traces").glob("*.jsonl"))
            if len(traces) != 1:
                raise ValueError(f"Missing or duplicate trace at {parent}")
            spans = [e for e in (json.loads(line) for line in traces[0].read_text().splitlines()) if e["kind"] == "span"]
            if len(spans) != expected_spans:
                raise ValueError("Incomplete traced history")
            for i, (row, event) in enumerate(zip(decisions, spans, strict=True)):
                p0, p1 = (probability(row, weights[a]) for a in ("terminal", "credit"))
                observed = p1 if arm == "credit" else p0 if arm == "terminal" else (
                    1.0 if arm == "always" else float(expit(row["base_logit"])))
                if not math.isclose(observed, row["prob"], abs_tol=2e-6) or row["use"] != (observed >= .5):
                    raise ValueError("Logged choice disagrees with deployed probability")
                acts = tuple(s["name"] for s in event["selected_skills"])
                executed = {s["name"] for s in event["skills"]}
                if ("capture_activity_details" in executed) != row["use"]:
                    raise ValueError("Executor did not receive the selected writing skill")
                if ("capture_activity_details" in acts) != row["use"] or len(set(acts)) != 3:
                    raise ValueError("Executor received different skills than the gate selected")
                choices[arm][sid,i] = acts
                probabilities[arm].append([p0, p1])
            if (parent / "reuse.json").is_file():
                reuse = read(parent / "reuse.json")
                ref = directory / phase / reuse["reference"]
                if (reuse["checked_decisions"] != len(decisions)
                        or hashlib.sha256((ref / "scores.json").read_bytes()).hexdigest() != reuse["score_sha256"]):
                    raise ValueError("Reused score provenance is inconsistent")
                counts["identical_rollouts_reused"] += 1
            counts["spans"] += len(decisions)
            counts["grounded_spans"] += sum(d["use"] for d in decisions)
            for event in spans:
                counts["memory_writes"] += len(event["changes"])
            for line in (parent / "usage.jsonl").read_text().splitlines():
                usage = json.loads(line)
                counts[usage["kind"] + "_tokens"] += usage["input"] + usage["output"]
        scores[arm], extra[arm] = scored, dict(counts)
    keys = [k for k in scores["credit"] if answerable[k]]
    if any(s.keys() != scores["credit"].keys() for s in scores.values()):
        raise ValueError("Policies did not answer identical questions")
    by_group = {t: [k for k in keys if groups[k] == t] for t in sorted({groups[k] for k in keys})}
    report = {"phase": phase, "scored_questions": len(keys), "histories": len(conversations),
              "per_type_counts": {t:len(ks) for t,ks in by_group.items()}, "arms": {}, "credit_minus": {}}
    for arm, s in scores.items():
        kind = {t:mean(s[k] for k in ks) for t,ks in by_group.items()}
        report["arms"][arm] = {"f1":mean(s[k] for k in keys), "macro_type_f1":mean(kind.values()),
                                "by_type":kind, **extra[arm]}
        abstentions = [v for k,v in s.items() if not answerable[k]]
        if abstentions:
            report["arms"][arm]["abstention_reference_token_f1"] = mean(abstentions)
    for arm in arms:
        if arm == "credit":
            continue
        d = {k:scores["credit"][k]-scores[arm][k] for k in keys}
        p = np.asarray(probabilities[arm])
        contrast = {"f1_change":mean(d.values()), "improved":sum(v>0 for v in d.values()),
                    "regressed":sum(v<0 for v in d.values()), "tied":sum(v==0 for v in d.values()),
                    "by_type":{t:mean(d[k] for k in ks) for t,ks in by_group.items()},
                    "changed_realized_tuples":sum(a != choices[arm][k] for k,a in choices["credit"].items()),
                    "terminal_to_credit_on_same_states":{"mean_probabilities":p.mean(axis=0).tolist(),
                      "mean_absolute_shift":float(np.abs(p[:,1]-p[:,0]).mean()),
                      "greedy_switches":int(((p[:,0]>=.5)!=(p[:,1]>=.5)).sum()), "states":len(p)}}
        if external:
            contrast["stratified_95pct_ci"] = bootstrap_by_type(list(d.values()), [groups[k] for k in keys])
        else:
            contrast["per_dialogue_delta"] = {sid:mean(v for (s,_),v in d.items() if s==sid)
                                                for sid,_ in conversations}
        report["credit_minus"][arm] = contrast
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("phase", choices=("validation","test","longmemeval"))
    parser.add_argument("--arms", nargs="+", default=["native","terminal","credit"])
    args = parser.parse_args()
    result = audit(args.directory, args.dataset, args.phase, args.arms)
    (args.directory / args.phase / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
