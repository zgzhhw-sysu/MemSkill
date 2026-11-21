"""Recompute paired task F1 and gate probabilities on identical saved states."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from experiments.controller.paired_policy import count_schedule
from experiments.controller.resume_status import check_equal_source_choice, load_episode
from experiments.designer.designer_pilot import prefix
from src.credit_assignment import recovered
from src.eval import get_evaluator
from src.evaluation_cache import read_response
from src.eval_utils import normalize_answer
from src.replay import verified_reader_contexts


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


def answer_change(before, after, truth):
    """Describe a scored change using literal reference evidence, separately from token F1."""
    old, new = (normalize_answer(row["answer"]).split() for row in (before, after))
    refs = truth if isinstance(truth, list) else [truth]
    def contains(words, phrase):
        return bool(phrase) and any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))
    shorter, longer = sorted((old, new), key=len)
    if (len(shorter) < len(longer) and contains(longer, shorter)
            and any(contains(shorter, normalize_answer(str(ref)).split()) for ref in refs)):
        return "credit_shorter_with_shared_reference" if new == shorter else "control_shorter_with_shared_reference"
    if recovered(before, after, truth):
        return "credit_recovered_reference"
    if recovered(after, before, truth):
        return "control_recovered_reference"
    return "other_wording_or_content_change"


def audit_rate_control(study, weights):
    if not np.array_equal(weights["terminal"][1:], weights["rate"][1:]):
        raise ValueError("Rate control changed state-dependent terminal weights")
    protocol = read(study / "protocol.json")
    rows = [r for start, end in protocol["train_sessions"] for sid in protocol["train"]
            for r in load_episode(study / "train" / f"{sid}_s{start}", sid, end - start + 1,
                                  deployment_writer=protocol.get("deployment_writer", False))]
    means = {arm: mean(probability(r, weights[arm]) for r in rows) for arm in ("credit", "rate")}
    if not math.isclose(means["credit"], means["rate"], rel_tol=0, abs_tol=2e-7):
        raise ValueError("Rate control does not match the credit policy on training states")
    return {"training_spans": len(rows), "mean_grounded_probability": means}


def reader_inputs(events, scored_rows, questions, evaluator):
    contexts = verified_reader_contexts(events, scored_rows, questions)
    result = {}
    for index, context in contexts.items():
        question = questions[index]
        prompt = evaluator.build_prompt(question["question"], [text for _, text in context], question)
        result[index] = hashlib.sha256(prompt.encode()).hexdigest(), {span for span, _ in context}
    return result


def audit_writer_ranges(event, inputs):
    """Check that the measured requests cover the source once under one output allowance."""
    end = total = 0
    calls = event["executor_responses"]
    for call in calls:
        start, stop = call["source_range"]
        cap = call["output_cap"]
        if (start != end or stop <= start or not 0 < call["output_tokens"] <= cap
                or not 0 < call["input_tokens"] <= inputs["writer_context"]["context_tokens"] - cap):
            raise ValueError("Invalid writer source range or token counts")
        end = stop
        total += cap
    if end != len(event["source"]) or total != inputs["writer_tokens"]:
        raise ValueError("Writer requests did not cover the session under its output budget")
    return len(calls)


def audit(directory, dataset_file, phase, arms):
    cohort = read(Path(__file__).with_name("protocols") / "paired_grounding_cohort.json")
    protocol = read(directory / "protocol.json")
    external = phase in ("longmemeval", "longmemeval-validation")
    spec = cohort["external" if external else "primary"]
    frozen = None
    if phase == "longmemeval-validation":
        manifest = Path(__file__).with_name("protocols") / "paired_external_validation.json"
        spec = read(manifest)
        frozen = read(directory / phase / "frozen.json")
        fit = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in ("protocol.json", "terminal.json", "credit.json", "rate.json")}
        if (fit != frozen["weights_and_protocol"] or hashlib.sha256(manifest.read_bytes()).hexdigest() != frozen["cohort_sha256"]
                or hashlib.sha256((directory / "validation/selection.json").read_bytes()).hexdigest() != frozen["development_selection_sha256"]):
            raise ValueError("External validation inputs differ from the frozen manifest")
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
    weights = {arm: np.asarray(read(directory / (arm + ".json"))) for arm in arms if arm in ("credit", "terminal", "rate")}
    scores, probabilities, extra = {}, {}, {}
    groups, answerable, references = {}, {}, {}
    predictions = defaultdict(dict)
    choices, readers = defaultdict(dict), defaultdict(dict)
    placements = {}
    common_inputs = {}
    for arm in arms:
        scored, probabilities[arm] = {}, []
        counts = Counter()
        for sid, convo in conversations:
            parent = directory / phase / arm / sid
            inputs = read(parent / "inputs.json")
            if protocol.get("equal_sources") and inputs["skills"] != [s["version"] for s in protocol["skills"]]:
                raise ValueError(f"Evaluation bank differs from the frozen source skills at {parent}")
            shared = {k: v for k, v in inputs.items() if k not in {"weights", "arm", "count_control"}}
            digest = hashlib.sha256(json.dumps(convo, sort_keys=True).encode()).hexdigest()
            weight_digest = hashlib.sha256(weights[arm].astype(np.float32).tobytes()).hexdigest() if arm in weights else None
            if (inputs["arm"] != arm or inputs["weights"] != weight_digest or shared["conversation"] != digest
                    or (frozen and shared["backend"] != frozen["backend"])
                    or shared != common_inputs.setdefault(sid, shared)):
                raise ValueError(f"Policies were evaluated with different inputs or weights at {parent}")
            report = read(parent / "scores.json")
            refs = {0: convo} if external else dict(ev.filter_qa_list(convo["qa"]))
            if len(report["answers"]) != len(refs) or {r["qa_index"] for r in report["answers"]} != refs.keys():
                raise ValueError(f"Incomplete data in {parent}")
            ids = read(parent / "question_ids.json")
            expected_ids = [0] if external else [q.get("original_index", i) for i, q in enumerate(convo["qa"])]
            if ids != expected_ids:
                raise ValueError(f"Question indices differ from the source at {parent}")
            for r in report["answers"]:
                index = r["qa_index"]
                q = refs[index]
                key = sid, ids[index]
                f1 = float(ev.compute_f1(r["prediction"], ev.get_ground_truth(q), q))
                if (key in scored or r["sample_id"] != sid or min(r["tokens"]) < 1
                        or not math.isclose(f1, r["f1"], abs_tol=1e-9)):
                    raise ValueError(f"Invalid score for {key}")
                scored[key], groups[key] = f1, str(q["question_type"] if external else q["category"])
                references[key] = ev.get_ground_truth(q)
                predictions[arm][key] = {"answer": r["prediction"], "f1": f1, "ok": True}
                answerable[key] = not (external and sid.endswith("_abs"))
            decisions = read(parent / "decisions.json")
            expected_spans = len(convo["haystack_sessions"]) if external else sum(
                k.startswith("session_") and not k.endswith("_date_time") for k in convo["conversation"])
            if len(decisions) != expected_spans:
                raise ValueError(f"Partial history at {parent}: {len(decisions)}/{expected_spans}")
            schedule = None
            if arm == "count":
                credit_file = directory / phase / "credit" / sid / "decisions.json"
                source = credit_file.read_bytes()
                credit_choices = [r["use"] for r in json.loads(source)]
                schedule = count_schedule(protocol["seed"], sid, credit_choices)
                expected = {"schedule": schedule, "credit_decisions_sha256": hashlib.sha256(source).hexdigest()}
                if (inputs.get("count_control") != expected or len(schedule) != expected_spans
                        or [r["use"] for r in decisions] != schedule):
                    raise ValueError(f"Count control differs from its preassigned placement at {parent}")
                placements[sid] = {"spans": len(schedule), "grounded_in_each": sum(credit_choices),
                                   "different_positions": sum(a != b for a, b in zip(schedule, credit_choices, strict=True))}
            traces = list((parent / "traces").glob("*.jsonl"))
            if len(traces) != 1:
                raise ValueError(f"Missing or duplicate trace at {parent}")
            events = [json.loads(line) for line in traces[0].read_text().splitlines()]
            if any(event["sample"] != sid for event in events):
                raise ValueError(f"Trace belongs to another history at {parent}")
            qa_prompts = reader_inputs(events, report["answers"], refs, ev)
            for index, evidence in qa_prompts.items():
                readers[arm][sid, ids[index]] = evidence
            spans = [e for e in events if e["kind"] == "span"]
            if [e["span_id"] for e in spans] != list(range(expected_spans)):
                raise ValueError("Incomplete traced history")
            for i, (row, event) in enumerate(zip(decisions, spans, strict=True)):
                p0, p1 = (probability(row, weights[a]) for a in ("terminal", "credit"))
                if arm in weights:
                    observed = probability(row, weights[arm])
                elif arm in ("always", "plain"):
                    observed = float(arm == "always")
                elif arm == "count":
                    observed = float(schedule[i])
                else:
                    observed = float(expit(row["base_logit"]))
                if not math.isclose(observed, row["prob"], abs_tol=2e-6) or row["use"] != (observed >= .5):
                    raise ValueError("Logged choice disagrees with deployed probability")
                acts = tuple(s["name"] for s in event["selected_skills"])
                if protocol.get("equal_sources"):
                    check_equal_source_choice(row, protocol["skills"], event["selected_skills"])
                    if event["selected_skills"][-1] not in event["skills"]:
                        raise ValueError(f"Writer did not receive the frozen source skill at {parent}")
                executed = {s["name"] for s in event["skills"]}
                if inputs.get("writer_context") and executed and event["source"].strip():
                    calls = audit_writer_ranges(event, inputs)
                    counts["writer_calls"] += calls
                    counts["split_writer_spans"] += int(calls > 1)
                if ("capture_activity_details" in executed) != row["use"]:
                    raise ValueError("Executor did not receive the selected writing skill")
                if ("capture_activity_details" in acts) != row["use"] or len(set(acts)) != 3:
                    raise ValueError("Executor received different skills than the gate selected")
                choices[arm][sid,i] = acts
                probabilities[arm].append([p0, p1, observed])
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
            evidence = Counter(("reader", qa_prompts[r["qa_index"]][0], r["prediction"].strip(), *r["tokens"])
                               for r in report["answers"])
            evidence.update(("executor", call["prompt_sha256"], call["response"], call["input_tokens"], call["output_tokens"])
                            for span in spans for call in span.get("executor_responses", ()))
            checked = 0
            for line in (parent / "usage.jsonl").read_text().splitlines():
                usage = json.loads(line)
                tokens = usage["input"] + usage["output"]
                if "cache_key" in usage:
                    measured = read_response(directory / "response-cache" / phase, usage["cache_key"])
                    if (not measured or measured["sha256"] != usage["response_sha256"]
                            or measured["payload"]["tokens"] != [usage["input"], usage["output"]]
                            or usage["cache_hit"] != (usage["attempts"] == 0)):
                        raise ValueError(f"Unverified evaluation cache record in {parent}")
                    payload = measured["payload"]
                    answer = payload["response"].strip() if usage["kind"] == "reader" else payload["response"]
                    proof = (usage["kind"], payload["prompt_sha256"], answer, *payload["tokens"])
                    if evidence[proof] <= 0:
                        raise ValueError(f"Measured request differs from the evaluation trace in {parent}")
                    evidence[proof] -= 1
                    checked += 1
                if usage.get("cache_hit"):
                    counts[usage["kind"] + "_cache_hits"] += 1
                    counts[usage["kind"] + "_cached_tokens"] += tokens
                else:
                    counts[usage["kind"] + "_tokens"] += tokens
            if checked and any(evidence.values()):
                raise ValueError(f"Missing measured request for an evaluation trace in {parent}")
        scores[arm], extra[arm] = scored, dict(counts)
    keys = [k for k in scores["credit"] if answerable[k]]
    if any(s.keys() != scores["credit"].keys() for s in scores.values()):
        raise ValueError("Policies did not answer identical questions")
    by_group = {t: [k for k in keys if groups[k] == t] for t in sorted({groups[k] for k in keys})}
    report = {"phase": phase, "scored_questions": len(keys), "histories": len(conversations),
              "per_type_counts": {t:len(ks) for t,ks in by_group.items()}, "arms": {}, "credit_minus": {}}
    if "rate" in arms:
        report["rate_control"] = audit_rate_control(directory, weights)
    if placements:
        report["count_control"] = placements
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
                    "terminal_to_credit_on_same_states":{"mean_probabilities":p[:, :2].mean(axis=0).tolist(),
                      "mean_absolute_shift":float(np.abs(p[:,1]-p[:,0]).mean()),
                      "greedy_switches":int(((p[:,0]>=.5)!=(p[:,1]>=.5)).sum()), "states":len(p)}}
        contrast["control_to_credit_on_same_states"] = {
            "mean_probabilities": p[:, [2, 1]].mean(axis=0).tolist(),
            "mean_absolute_shift": float(np.abs(p[:, 1] - p[:, 2]).mean()),
            "greedy_switches": int(((p[:, 2] >= .5) != (p[:, 1] >= .5)).sum()),
            "states": len(p),
        }
        changed_spans = defaultdict(set)
        for (sid, span), actions in choices["credit"].items():
            if actions != choices[arm][sid, span]:
                changed_spans[sid].add(span)
        evidence = []
        answer_changes = defaultdict(list)
        unsupported_delta = 0.0
        for (sid, index), delta in d.items():
            if delta == 0:
                continue
            old_prompt, old_origins = readers[arm][sid, index]
            new_prompt, new_origins = readers["credit"][sid, index]
            prompt_changed = old_prompt != new_prompt
            if not changed_spans[sid] or not prompt_changed:
                unsupported_delta += delta
            change = answer_change(predictions[arm][sid, index], predictions["credit"][sid, index], references[sid, index])
            answer_changes[change].append(delta)
            evidence.append({"history": sid, "question_index": index, "f1_delta": delta,
                             "answer_change": change,
                             "actions_changed_in_history": bool(changed_spans[sid]),
                             "reader_prompt_changed": prompt_changed,
                             "switched_spans_read_in_control": sorted(old_origins & changed_spans[sid]),
                             "switched_spans_read_in_credit": sorted(new_origins & changed_spans[sid])})
        contrast["changed_question_evidence"] = evidence
        contrast["answer_changes"] = {kind: {"questions": len(values), "contribution_to_mean_f1_change": sum(values) / len(keys)}
                                      for kind, values in answer_changes.items()}
        contrast["f1_change_without_verified_action_to_prompt_change"] = unsupported_delta / len(keys)
        if external:
            contrast["stratified_95pct_ci"] = bootstrap_by_type(list(d.values()), [groups[k] for k in keys])
        else:
            contrast["per_dialogue_delta"] = {sid:mean(v for (s,_),v in d.items() if s==sid)
                                                for sid,_ in conversations}
        report["credit_minus"][arm] = contrast
    return report


def development_decision(report):
    """Check whether changed span decisions improve development QA beyond global skill usage."""
    if report["phase"] != "validation":
        raise ValueError("Policy selection requires development data")
    reason = []
    controls = ["native", "terminal", "always", "rate"]
    controls.extend(arm for arm in ("plain", "count") if arm in report["credit_minus"])
    for arm in controls:
        diff = report["credit_minus"][arm]
        if diff["f1_change"] <= 0:
            reason.append(f"No positive development F1 difference against {arm}")
        if not diff["control_to_credit_on_same_states"]["greedy_switches"]:
            reason.append(f"No changed greedy choices on the recorded {arm} states")
        supported = [q["f1_delta"] for q in diff["changed_question_evidence"]
                     if q["actions_changed_in_history"] and q["reader_prompt_changed"]]
        if sum(supported) <= 0:
            reason.append(f"No positive net F1 difference with verified choice and reader changes against {arm}")
    return {"advance": not reason, "reasons": reason}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("phase", choices=("validation","test","longmemeval","longmemeval-validation"))
    parser.add_argument("--arms", nargs="+", default=["native","terminal","credit"])
    parser.add_argument("--select-for-test", action="store_true",
                        help="Check the required development controls and publish selection.json before opening held-out data")
    parser.add_argument("--stdout-only", action="store_true", help="Recompute a comparison without replacing an archived report")
    args = parser.parse_args()
    if args.select_for_test and args.phase != "validation":
        parser.error("--select-for-test requires validation")
    if args.select_for_test and args.stdout_only:
        parser.error("--select-for-test publishes a decision and cannot use --stdout-only")
    arms = ["native", "terminal", "credit", "always", "rate"] if args.select_for_test else args.arms
    if args.select_for_test:
        protocol = read(args.directory / "protocol.json")
        if protocol.get("equal_sources"):
            arms.append("plain")
        if protocol.get("all_linked_questions"):
            arms.append("count")
    result = audit(args.directory, args.dataset, args.phase, arms)
    if args.stdout_only:
        print(json.dumps(result, indent=2))
        raise SystemExit(0)
    filename = "comparison-with-count.json" if "count" in arms else "comparison.json"
    if args.select_for_test:
        result["selection"] = development_decision(result)
        filename = "selection.json"
    directory = args.directory / args.phase
    with NamedTemporaryFile(mode="w", dir=directory, delete=False) as temporary:
        json.dump(result, temporary, indent=2)
        temporary.write("\n")
    Path(temporary.name).replace(directory / filename)
    if args.select_for_test:
        print(json.dumps(result["selection"], indent=2))
        raise SystemExit(0 if result["selection"]["advance"] else 3)
    print(json.dumps(result, indent=2))
