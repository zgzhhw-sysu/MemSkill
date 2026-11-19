"""Recheck the losing counterfactual branch of positive sampled-action training credits."""
from collections import Counter
import fcntl
import hashlib
import json
import os

from experiments.controller.probe_omitted import atomic_json, check_responses, omitted_final_indices
from experiments.controller.resume_status import load_episode
from src.credit_assignment import _answer, _digest, _same_branch, _write_branch, hidden_retrieval_candidate, possible_recovery
from src.episode_trace import evidence_turns
from src.rag_utils import get_embeddings


def matching_readers(branch, reference):
    """Compare the shared write trajectory first, then check each measured retrieval and answer."""
    if any(b.get("contract_fault") or b.get("failed_span") is not None or not b.get("qa") for b in (branch, reference)):
        return set()
    if not _same_branch(dict(branch, qa={}), dict(reference, qa={})):
        return set()
    saved = {str(i): a for i, a in reference["qa"].items()}
    return {i for i, a in branch["qa"].items() if a["ok"] and a == saved.get(str(i))}


def matches_recorded_swap(branch, saved):
    """Require the previously measured bank, suffix, and every requested reader response."""
    return bool(branch.get("qa")) and len(matching_readers(branch, saved)) == len(branch["qa"])


def partial_legacy_match(result, diagnosis):
    weak = result["weak"]
    matched = {str(i) for i in matching_readers(weak, diagnosis["swap"])}
    if any((str(i) in matched) != (q["status"] != "unreproduced")
           for i, q in result["questions"].items()):
        return True
    if "repair_repeat" in result:
        repair = result["repair"]
        reproduced = matching_readers(repair, result["repair_repeat"])
        return 0 < len(reproduced) < len(repair.get("qa", {}))
    return False


def publish_reader_result(root, name, data):
    """Retain the older aggregate-reader receipt before updating the standard result path."""
    current = root / name
    if current.is_file():
        old = json.loads(current.read_text())
        if old == data:
            return
        if "reader_match_version" not in old:
            archive = root / "all-readers-v1" / name
            archive.parent.mkdir(exist_ok=True)
            if archive.exists():
                if json.loads(archive.read_text()) != old:
                    raise ValueError(f"Historical reader receipt differs at {archive}")
            else:
                os.link(current, archive)
    atomic_json(current, data)


def check_weaker_span(trainer, conversation, spans, diagnosis):
    idx = diagnosis["span_id"]
    alternative = diagnosis["alternative"]
    questions = diagnosis["questions"]
    targets = {q["qa_index"]: q for q in questions if q["controller_delta"] > 0}
    ids = [q["qa_index"] for q in questions]
    qas = conversation["qa"]
    vectors = get_embeddings(trainer.args.retriever, [qas[i]["question"] for i in ids], "query")
    queries = {i: (qas[i], v) for i, v in zip(ids, vectors, strict=True) if i in targets}
    results = {i: {"original_index": q["original_index"], "original_delta": q["controller_delta"],
                   "status": "unreproduced", "competing": []} for i, q in targets.items()}
    if not _same_branch(diagnosis["swap"], diagnosis["swap_repeat"]):
        raise ValueError("Historical swap and its repeat disagree")
    if any(q["original_index"] != qas[i]["original_index"] or not q["control_matched"]
           or q["original_f1"] <= diagnosis["swap"]["qa"][str(i)]["f1"] for i, q in targets.items()):
        raise ValueError("Counterfactual is not the recorded losing branch for this training question")
    weak = _write_branch(trainer, spans, idx, alternative, queries, include_contents=True)
    check_responses({"questions": [], "control": weak})
    contents = weak.pop("contents", [])
    report = {"questions": results, "weak": weak}
    matched = matching_readers(weak, diagnosis["swap"])
    queries = {i: q for i, q in queries.items() if i in matched}
    if not queries:
        return report

    turns = {i: evidence_turns(conversation["conversation"], qas[i].get("evidence")) for i in ids}
    quotes = list(dict.fromkeys(f'{t["session_date"]} | {t["speaker"]}: {t["text"]}'
                                for i in ids for t in turns[i] if t["dia_id"] in spans[idx]["source_turns"]))
    can_insert = any(op["update_type"].lower() == "insert" for op in alternative)
    truths = {i: trainer.evaluator.get_ground_truth(qas[i]) for i in targets}
    repair = {}
    verified = set(queries)
    if quotes and can_insert:
        report["repair_quotes"] = quotes
        repair = _write_branch(trainer, spans, idx, alternative, queries, quotes)
        report["repair"] = repair
        check_responses({"questions": [], "execution_repair": repair})
        if (repair["actions"] != weak["actions"] or repair["contract_fault"] or repair.get("failed_span") is not None
                or repair.get("writer_prompt_hashes") != weak.get("writer_prompt_hashes")):
            verified.clear()
        verified.intersection_update(repair.get("qa", {}))
        if any(possible_recovery(weak["qa"][i], repair["qa"][i], truths[i]) for i in verified):
            repeat = _write_branch(trainer, spans, idx, alternative, queries, quotes)
            report["repair_repeat"] = repeat
            check_responses({"questions": [], "repair_repeat": repeat})
            verified.intersection_update(matching_readers(repair, repeat))

    live = dict(enumerate(contents))
    for i in queries:
        outcome = results[i]
        before = weak["qa"][i]
        refs = before["indices"]
        if _digest([live[j] for j in refs]) != before["context_hash"]:
            raise ValueError("Counterfactual reader context does not match the replay")
        if (i in verified and repair and repair["bank_hash"] != weak["bank_hash"]
                and repair["qa"][i]["context_hash"] != before["context_hash"]):
            if possible_recovery(before, repair["qa"][i], truths[i]):
                outcome["competing"].append("execution")
        mid = hidden_retrieval_candidate(qas[i], turns[i], refs, live)
        stable = True
        if mid is not None:
            context = [live[j] for j in refs[:-1]] + [live[mid]]
            answer = _answer(trainer, qas[i], context)
            probe = {"inserted_index": mid, "replaced_index": refs[-1], "context_hash": _digest(context), "answer": answer}
            outcome["retrieval"] = probe
            if not answer["ok"]:
                raise RuntimeError("Counterfactual retrieval reader failed")
            if possible_recovery(before, answer, truths[i]):
                probe["repeat"] = _answer(trainer, qas[i], context)
                if not probe["repeat"]["ok"]:
                    raise RuntimeError("Counterfactual retrieval repeat failed")
                stable = probe["repeat"] == answer
                if stable:
                    outcome["competing"].append("retrieval")
        if outcome["competing"]:
            outcome["status"] = "competing_recovery"
        elif i in verified and stable:
            outcome["status"] = "survived_probes"
        else:
            outcome["status"] = "unresolved"
    return report


def supplementary_diagnoses(source, episode, spans, convo, events, trace_hash, backend):
    supplement = json.loads((source / (episode.name + ".json")).read_text())
    original = [e["diagnosis"] for e in events if e["kind"] == "span_credit"][-1]
    observed = {int(e["query_id"].rsplit("_", 1)[1]) for e in events if e["kind"] == "qa" and e["success"]}
    indices = omitted_final_indices(spans, convo["qa"], observed, original)
    signature = {"backend": backend, "episode": episode.name, "trace_sha256": trace_hash,
                 "span": len(spans)-1, "original_question_ids": [convo["qa"][i]["original_index"] for i in indices],
                 "rule": 2}
    if supplement["inputs"] != signature:
        raise ValueError("Omitted-question receipt does not belong to this episode and backend")
    diagnosis = supplement["diagnosis"]
    if not indices and diagnosis is None:
        return []
    if not indices or not diagnosis or [q["qa_index"] for q in diagnosis["questions"]] != indices:
        raise ValueError("Unexpected question coverage in the supplementary receipt")
    if (diagnosis["span_id"] != original["span_id"] or diagnosis["sampled_skills"] != original["sampled_skills"]
            or diagnosis["candidate_versions"] != original["candidate_versions"]):
        raise ValueError("Supplementary probe describes a different training action")
    check_responses(diagnosis)
    return [diagnosis]


def probe_weaker_windows(trainer, study, windows, through, *, omitted=False):
    backend = os.environ.get("MEMSKILL_BACKEND_ID", trainer.args.api_base)
    key = hashlib.sha256(backend.encode()).hexdigest()[:16]
    source = study / "diagnostics" / "omitted-final" / key
    root = study / "diagnostics" / "weaker-branch" / key
    if omitted:
        root = root / "omitted-final"
    canonical_root = root
    root = root / "per-question"
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    counts = Counter()
    for convo, episode in windows:
        trace, = (episode / "traces").glob("*.jsonl")
        raw = trace.read_bytes()
        events = [json.loads(line) for line in raw.splitlines()]
        spans = [e for e in events if e["kind"] == "span"]
        ids = [q["original_index"] for q in convo["qa"]]
        load_episode(episode, convo["sample_id"], len(spans), ids, deployment_writer=trainer.args.span_credit_deployment_writer)
        counts["windows"] += 1
        trace_hash = hashlib.sha256(raw).hexdigest()
        records = (supplementary_diagnoses(source, episode, spans, convo, events, trace_hash, backend) if omitted else
                   [e["diagnosis"] for e in events if e["kind"] == "span_credit"])
        for d in records:
            counts["sampled_losses_checked_by_original_rule"] += sum(q["controller_delta"] < 0 for q in d["questions"])
            positives = [q for q in d["questions"] if q["controller_delta"] > 0]
            if positives:
                if d["evidence_rule_version"] != 2 or any(q["cause"] != "selection" for q in positives):
                    raise ValueError("Expected selection-only rule-2 training credits")
                signature = {"episode": episode.name, "trace_sha256": trace_hash, "backend": backend,
                             "span": d["span_id"], "positive_question_ids": [q["original_index"] for q in positives]}
                if omitted:
                    signature["supplement_sha256"] = hashlib.sha256((source / (episode.name + ".json")).read_bytes()).hexdigest()
                jobs.append((convo, spans, d, signature))
    # End-of-window swaps can be checked before the more expensive suffix replays.
    jobs.sort(key=lambda job: (len(job[1]) - job[2]["span_id"], job[3]["episode"]))
    with (canonical_root / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.environ["MEMSKILL_USAGE_LOG"] = str(root / "usage.jsonl")
        for convo, spans, diagnosis, signature in jobs:
            name = f'{signature["episode"]}-span{signature["span"]}'
            path = root / (name + ".json")
            if path.exists():
                result = json.loads(path.read_text())
                if result["inputs"] != signature:
                    raise ValueError(f"Different training inputs at {path}")
            else:
                legacy = canonical_root / path.name
                prior = json.loads(legacy.read_text()) if legacy.is_file() else None
                if prior and prior["inputs"] != signature:
                    raise ValueError(f"Different training inputs at {legacy}")
                if prior and not partial_legacy_match(prior, diagnosis):
                    result = dict(prior, reader_match_version=2)
                else:
                    result = dict(inputs=signature, reader_match_version=2,
                                  **check_weaker_span(trainer, convo, spans, diagnosis))
                atomic_json(path, result)
            publish_reader_result(canonical_root, path.name, result)
            states = [q["status"] for q in result["questions"].values()]
            counts.update(states)
            print(name, states, flush=True)
        summary = {"backend": backend, "reader_match_version": 2, "counts": dict(counts),
                   "checked": [j[3] for j in jobs]}
        name = f"through-{through}.json"
        atomic_json(root / name, summary)
        publish_reader_result(canonical_root, name, summary)
