"""Evidence-gated feedback for the sampled Controller action on a LoCoMo span."""
import hashlib
import json
import math
import re
from statistics import mean
from types import SimpleNamespace
import numpy as np
import torch

from .episode_trace import evidence_turns, reference_match, reference_recovered, skill_spec
from .replay import rebuild_suffix, restore_text


def single_swap(candidates, selected):
    """Replace the last sampled slot with the first unselected INSERT in bank order."""
    names = {s["name"] for s in selected}
    unused = [op for op in candidates if op["name"] not in names]
    if not selected or not unused:
        return None
    replacement = next((op for op in unused if op["update_type"].lower() == "insert"), unused[0])
    return [*selected[:-1], replacement]


def execution_fault(actions, applied, skills):
    if not actions or not applied or any(not a["success"] for a in actions):
        return "parse_or_apply_failure"
    allowed = {s["update_type"].upper() for s in skills}
    if any(a["action_type"].upper() not in allowed for a in actions):
        return "action_outside_selected_types"
    return None


def recovered(low, high, truth):
    refs = truth if isinstance(truth, list) else [truth]
    return (low["ok"] and high["ok"] and high["f1"] > low["f1"]
            and not any(reference_match(low["answer"], str(ref)) for ref in refs)
            and any(reference_recovered(low["answer"], high["answer"], str(ref)) for ref in refs))


def classify(observed, swap, truth, *, execution, retrieval, fault=None):
    """Positive feedback supports the sampled action. Competing explanations abstain."""
    labels = []
    delta = 0.0
    if not fault and swap is not None and (recovered(observed, swap, truth) or recovered(swap, observed, truth)):
        labels.append("selection")
        delta = observed["f1"] - swap["f1"]
    if execution:
        labels.append("execution")
    if retrieval:
        labels.append("retrieval")
    cause = "mixed" if len(labels) > 1 else next(iter(labels), "unresolved")
    return {"cause": cause, "supported": labels, "selection_delta": delta,
            "controller_delta": delta if cause == "selection" else 0.0}


def possible_recovery(before, after, truth):
    """Whole-reference containment can veto attribution, but never authorize a reward."""
    from src.eval_utils import normalize_answer, ps
    if not after or not before["ok"] or not after["ok"] or after["f1"] <= before["f1"]:
        return False
    def contains(answer, ref):
        if reference_match(answer, ref):
            return True
        words = [ps.stem(w) for w in normalize_answer(answer).split()]
        target = [ps.stem(w) for w in normalize_answer(ref).split()]
        return bool(target) and any(words[i:i+len(target)] == target for i in range(len(words)-len(target)+1))
    refs = [str(ref) for ref in (truth if isinstance(truth, list) else [truth])]
    return not any(contains(before["answer"], ref) for ref in refs) and any(contains(after["answer"], ref) for ref in refs)


def check_competing_evidence(record, question, truth):
    """Apply the same conservative gate to native records and serialized training traces."""
    observed = {"answer":question["original"], "f1":question["original_f1"], "ok":True}
    labels = question["supported"]
    possible = []
    index = question["qa_index"]
    def reply(branch):
        answers = branch.get("qa", {})
        return answers.get(index, answers.get(str(index)))
    control = record.get("control", {})
    original_reply = reply(control)
    if question["control_matched"] and original_reply:
        for stage, key in (("selection", "swap"), ("execution", "execution_repair")):
            branch = record.get(key, {})
            answer = reply(branch)
            if stage in labels or not answer or answer["context_hash"] == original_reply["context_hash"]:
                continue
            if stage == "selection" and (record.get("execution_fault") or control.get("contract_fault") or branch.get("contract_fault")):
                continue
            if stage == "execution" and branch["actions"] != control["actions"]:
                continue
            if possible_recovery(observed, answer, truth) or (stage == "selection" and possible_recovery(answer, observed, truth)):
                possible.append(stage)
    probe = question.get("retrieval_probe")
    if "retrieval" not in labels and question["reader_matched"] and probe and possible_recovery(observed, probe["reply"], truth):
        possible.append("retrieval")
    result = dict(question, possible_causes=possible)
    if len(labels) == 1 and possible:
        result.update(cause="ambiguous", controller_delta=0.0)
    return result


def allocate(buffer, start, steps, diagnostics, weight, clip, final_reward, last_ratio, redistribute):
    """Attach feedback to the original sampled transition after upstream reward allocation."""
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("Invalid span credit scale or clip")
    if len(buffer.rewards) - start != len(steps) or len(steps) != len(diagnostics):
        raise ValueError("Span count does not match the current on-policy episode")
    components = []
    for i, (step, record) in enumerate(zip(steps, diagnostics)):
        j = start + i
        if (step["session_idx"] != i or record["span_id"] != i or step["action_idx"] != buffer.actions[j]
                or step["log_prob"] != buffer.log_probs[j] or step["value"] != buffer.values[j]):
            raise ValueError("Span feedback does not identify its sampled transition")
        upstream = float(buffer.rewards[j])
        shaping = float(step["process_reward"])
        terminal = 0.0
        if i == len(steps)-1:
            fraction = max(0.0, min(1.0, last_ratio)) if redistribute and len(steps)>1 else 1.0
            terminal = final_reward * fraction
        delta = record["mean_controller_delta"]
        if not math.isfinite(delta):
            raise ValueError("Non-finite counterfactual return")
        extra = weight * max(-clip, min(clip, delta))
        components.append({"span_id": i, "sampled_action": buffer.actions[j], "old_log_probability": buffer.log_probs[j],
                           "old_value": buffer.values[j], "shaping": shaping, "terminal": terminal,
                           "redistributed": upstream - shaping - terminal, "upstream_reward": upstream,
                           "credit": extra, "ppo_reward": upstream + extra})
    for i, row in enumerate(components):
        if row["credit"]:
            buffer.rewards[start+i] = row["ppo_reward"]
        steps[i]["span_credit"] = row["credit"]
    return components


def _answer(trainer, qa, memories):
    from src.llm_utils import request_task
    evaluator = trainer.evaluator
    prompt = evaluator.build_prompt(qa["question"], memories, qa)
    _, prediction, tokens, ok = request_task((0, prompt, evaluator.prepare_eval_args()))
    prediction = prediction.strip() if ok and prediction is not None else ""
    return {"answer": prediction, "f1": evaluator.compute_f1(prediction, evaluator.get_ground_truth(qa), qa),
            "ok": bool(ok), "tokens": list(tokens)}


def _write_branch(trainer, events, start, operations, queries, quotes=()):
    from .executor import ExecutionResult, deployment_operations
    event = events[start]
    aligned = getattr(trainer.args, "span_credit_deployment_writer", False)
    selected = [SimpleNamespace(**s) for s in operations]
    writer_ops = deployment_operations(selected) if aligned else selected
    audit_args = {"audit": []} if aligned else {}
    actions = trainer.executor.execute_operation(writer_ops, event["source"], event["retrieved"], **audit_args)
    parsed = [vars(a).copy() for a in actions]
    if quotes:
        written = set(event["retrieved"]) | {a.memory_content for a in actions}
        actions.extend(ExecutionResult("INSERT", True, memory_content=q) for q in quotes if q not in written)
    candidates = trainer.operation_bank.get_candidate_operations()
    versions = [skill_spec(op)["version"] for op in candidates]
    embeddings = torch.as_tensor(np.stack([op.embedding for op in candidates]), device=trainer.device).float()
    def select_suffix(current, embedding, retrieved, memory_embeddings):
        if versions != current["candidate_versions"]:
            raise ValueError("Candidate bank changed within the suffix")
        state = trainer.state_encoder.encode(current["source"], retrieved, session_embedding=embedding,
                                             memory_embeddings=memory_embeddings)
        with torch.no_grad():
            idx, _, _ = trainer.controller(torch.as_tensor(state, device=trainer.device).float(), embeddings,
                                            new_op_mask=current["new_op_mask"], replay_noise=current["sampling_noise"])
        return [candidates[i] for i in (idx if isinstance(idx, list) else [idx])]
    suffix = []
    def observe(current, old, ops, results, success, writer_prompt_hashes=None):
        chosen = [skill_spec(op) for op in ops]
        actual = [vars(a).copy() for a in results]
        executed = [skill_spec(op) for op in deployment_operations(ops)] if aligned else chosen
        suffix.append({"span_id": current["span_id"], "selected": [s["name"] for s in chosen],
                       "retrieval_hash": _digest(old), "actions_hash": _digest(actual),
                       "matches_rollout": (chosen == current.get("selected_skills", current["skills"])
                                           and executed == current["skills"] and old == current["retrieved"]
                                           and actual == current["parsed_actions"] and success == current["applied"]),
                       "fault": execution_fault(actual, success, executed)})
        if aligned:
            suffix[-1]["writer_prompt_hashes"] = writer_prompt_hashes
            observed_hashes = [r["prompt_sha256"] for r in current["executor_responses"]]
            suffix[-1]["matches_rollout"] &= writer_prompt_hashes == observed_hashes
    bank, failed = rebuild_suffix(trainer, events, start, actions, select_suffix=select_suffix, observe_suffix=observe)
    executed = [skill_spec(op) for op in writer_ops] if aligned else operations
    result = {"actions": parsed, "failed_span": failed, "suffix": suffix,
              "contract_fault": execution_fault(parsed, True, executed) or next((s["fault"] for s in suffix if s["fault"]), None)}
    if aligned:
        result["writer_skills"] = executed
        result["writer_prompt_hashes"] = [r["prompt_sha256"] for r in audit_args["audit"]]
    if bank is None:
        return result
    contents = bank.get_all_contents()
    result["bank_hash"] = _digest(contents)
    result["qa"] = {}
    for idx, (qa, embedding) in queries.items():
        memories, indices = bank.retrieve(embedding)
        result["qa"][idx] = dict(_answer(trainer, qa, memories), indices=list(map(int, indices)),
                                 context_hash=_digest(memories))
    return result


def _digest(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def _same_branch(a, b):
    if "qa" not in a or "qa" not in b:
        return False
    return (a["actions"] == b["actions"] and a.get("suffix") == b.get("suffix")
            and a.get("writer_prompt_hashes") == b.get("writer_prompt_hashes")
            and a["bank_hash"] == b["bank_hash"] and a["qa"] == b["qa"]
            and all(q["ok"] for q in a["qa"].values()))


def _retrieval_probe(trainer, qa, row, live, turns, versions):
    """Replace one slot with an omitted live memory. Selection never reads the answer."""
    refs = [mid for mid, _ in row["refs"]]
    visible_texts = {live[mid] for mid in refs}
    hidden = [mid for mid, text in live.items() if text not in visible_texts]
    if not hidden or not refs:
        return False, None
    def words(value):
        return set(re.findall(r"\w+", value.lower()))
    query = words(" ".join([qa["question"], *(t["text"] for t in turns)]))
    def overlap(mid):
        tokens = words(live[mid])
        return len(query & tokens) / max(1, len(query | tokens))
    mid = max(hidden, key=overlap)
    memories = [live[m] for m in refs[:-1]] + [live[mid]]
    reply = _answer(trainer, qa, memories)
    record = {"inserted_ref": [mid, versions[mid][0]], "written_at": versions[mid][1],
              "replaced_ref": row["refs"][-1], "reply": reply}
    if recovered(row, reply, trainer.evaluator.get_ground_truth(qa)):
        repeat = _answer(trainer, qa, memories)
        record["repeat"] = repeat
        return repeat == reply, record
    return False, record


def diagnose_episode(trainer, conversation, trace, steps):
    """Probe evidence-linked spans before PPO. Never use evaluation conversations as training input."""
    from src.rag_utils import get_embeddings
    events = trace.events
    live = restore_text(events)
    expected_hash = _digest(list(live.values()))
    positions = {mid: i for i, mid in enumerate(live)}
    versions = {c["memory_id"]: (c["version"], e["span_id"]) for e in events for c in e["changes"]}
    rows = {}
    for line in trace.path.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "qa":
            if not row["success"]:
                raise RuntimeError("Observed Reader failed; refusing an invalid return for PPO")
            if any(versions.get(mid, (None,))[0] != version or mid not in live for mid, version in row["refs"]):
                raise ValueError("Stale observed Reader reference")
            idx = int(row["query_id"].rsplit("_", 1)[1])
            rows[idx] = dict(row, answer=row["prediction"], ok=True)
    candidates = [skill_spec(op) for op in trainer.operation_bank.get_candidate_operations()]
    candidate_versions = [s["version"] for s in candidates]
    rule_version = getattr(trainer.args, "span_credit_rule_version", 2)
    records = []
    probed = 0
    for start, (event, step) in enumerate(zip(events, steps, strict=True)):
        selected = event.get("selected_skills", event["skills"])
        action = step["action_idx"]
        ids = action if isinstance(action, list) else [action]
        if (candidate_versions != event["candidate_versions"] or [candidates[i] for i in ids] != selected
                or event["old_log_probability"] != step["log_prob"] or start != event["span_id"]):
            raise ValueError("Sampled action or candidate bank changed before span credit")
        fault = execution_fault(event["parsed_actions"], event["applied"], event["skills"])
        if any("API call failed" in a.get("reasoning", "") for a in event["parsed_actions"]):
            raise RuntimeError("Observed Executor failed to reach the model; refusing an invalid rollout")
        record = {"span_id": start, "evidence_rule_version": rule_version, "sampled_skills": selected, "candidate_versions": candidate_versions,
                  "execution_fault": fault, "questions": [], "mean_controller_delta": 0.0}
        records.append(record)
        indices = [i for i in sorted(rows) if set(event.get("source_turns", ())).intersection(
            conversation["qa"][i].get("evidence") or ())]
        if not indices or probed >= trainer.args.span_credit_spans:
            record["status"] = "budget" if indices else "no_linked_training_question"
            continue
        probed += 1
        indices = indices[:trainer.args.span_credit_questions]
        questions = {i: conversation["qa"][i] for i in indices}
        vectors = get_embeddings(trainer.args.retriever, [q["question"] for q in questions.values()], "query")
        queries = {i: (questions[i], v) for i, v in zip(indices, vectors, strict=True)}
        paired = (event.get("sampling_noise") or {}).get("paired_alternative")
        if paired is None:
            alternative = single_swap(candidates, selected)
        else:
            if (len(paired) != len(ids) or len(set(paired)) != len(ids)
                    or any(not isinstance(i, int) or not 0 <= i < len(candidates) for i in paired)
                    or sum(a != b for a, b in zip(ids, paired)) != 1):
                raise ValueError("Invalid paired single-slot intervention")
            alternative = [candidates[i] for i in paired]
        turns = {i: evidence_turns(conversation["conversation"], questions[i].get("evidence")) for i in indices}
        quotes = list(dict.fromkeys(f'{t["session_date"]} | {t["speaker"]}: {t["text"]}'
                                    for group in turns.values() for t in group if t["dia_id"] in event["source_turns"]))
        control = _write_branch(trainer, events, start, selected, queries)
        record.update(control=control, alternative=alternative, status="probed")
        historical = (control.get("bank_hash") == expected_hash and control["actions"] == event["parsed_actions"]
                      and all(s["matches_rollout"] for s in control.get("suffix", [])))
        if getattr(trainer.args, "span_credit_deployment_writer", False):
            historical = historical and control["writer_prompt_hashes"] == [
                call["prompt_sha256"] for call in event["executor_responses"]]
        swap = None
        if historical and alternative is not None and not (fault or control["contract_fault"]):
            a = _write_branch(trainer, events, start, alternative, queries)
            record["swap"] = a
            if ("qa" in a and not a["contract_fault"] and a["bank_hash"] != expected_hash and any(
                    a["qa"][i]["context_hash"] != control["qa"][i]["context_hash"]
                    and (recovered(rows[i], a["qa"][i], trainer.evaluator.get_ground_truth(questions[i]))
                         or recovered(a["qa"][i], rows[i], trainer.evaluator.get_ground_truth(questions[i])))
                    for i in indices)):
                repeat = _write_branch(trainer, events, start, alternative, queries)
                record["swap_repeat"] = repeat
                if _same_branch(a, repeat):
                    swap = a
        repair = None
        repair_available = True
        if historical and quotes and any(s["update_type"].lower() == "insert" for s in selected):
            repair = _write_branch(trainer, events, start, selected, queries, quotes)
            record.update(execution_repair=repair, repair_quotes=quotes)
            if repair["actions"] != control["actions"] or "qa" not in repair:
                repair = None
                repair_available = False
            elif any(recovered(rows[i], repair["qa"][i], trainer.evaluator.get_ground_truth(questions[i])) for i in indices):
                repeat = _write_branch(trainer, events, start, selected, queries, quotes)
                record["repair_repeat"] = repeat
                if not _same_branch(repair, repeat):
                    repair = None
                    repair_available = False
        for i in indices:
            qa, row = questions[i], rows[i]
            truth = trainer.evaluator.get_ground_truth(qa)
            baseline = control.get("qa", {}).get(i)
            expected_indices = [positions[mid] for mid, _ in row["refs"]]
            faithful = bool(historical and baseline and baseline["ok"] and baseline["answer"] == row["answer"]
                            and baseline["indices"] == expected_indices)
            # A retrieval-only diagnosis remains possible even if the writer couldn't be replayed.
            reader_stable = faithful
            if not faithful:
                check = _answer(trainer, qa, [live[mid] for mid, _ in row["refs"]])
                reader_stable = check["ok"] and check["answer"] == row["answer"]
            retrieval, probe = (False, None)
            if reader_stable:
                retrieval, probe = _retrieval_probe(trainer, qa, row, live, turns[i], versions)
            execution = bool(faithful and repair and repair["bank_hash"] != expected_hash
                             and repair["qa"][i]["context_hash"] != baseline["context_hash"]
                             and recovered(row, repair["qa"][i], truth))
            checks_ok = (repair_available and (not repair or repair["qa"][i]["ok"])
                         and (probe is None or (probe["reply"]["ok"] and probe.get("repeat", probe["reply"]) == probe["reply"])))
            selection_tested = (faithful and swap and checks_ok
                                and swap["qa"][i]["context_hash"] != baseline["context_hash"])
            diagnosis = classify(row, swap["qa"][i] if selection_tested else None, truth,
                                 execution=execution, retrieval=retrieval, fault=fault)
            question = dict(diagnosis, qa_index=i, original_index=qa.get("original_index", i),
                            original=row["answer"], original_f1=row["f1"], control_matched=faithful,
                            reader_matched=reader_stable, retrieval_probe=probe, repair_recovered=execution)
            record["questions"].append(check_competing_evidence(record, question, truth) if rule_version == 2 else question)
        record["mean_controller_delta"] = mean(q["controller_delta"] for q in record["questions"])
    return records
