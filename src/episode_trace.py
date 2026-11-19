"""Record applied writes and compare one skill edit at a fixed LoCoMo write."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import re
from datetime import datetime


def skill_spec(op):
    fields = {key: getattr(op, key) for key in
              ("name", "description", "instruction_template", "update_type")}
    if getattr(op, "source_rule", None):
        fields["source_rule"] = op.source_rule
    fields["version"] = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return fields


CALENDAR_INSTRUCTION = (
    " For each event described with yesterday, today, tomorrow or last year, use the utterance's DATE header "
    "as the reference. Write the speaker, event, original relative words and resolved event date together "
    "in the event memory. Yesterday is the previous calendar day, tomorrow is the next day and last year "
    "means the previous year without a known day. Preserve that precision. Keep unknown dates unresolved."
)

DETAIL_INSTRUCTION = (" Preserve the source speaker, explicitly stated event details and time. "
                      "Distinguish plans from completed events and retain still-valid details.")


def reference_match(answer, reference):
    """Conservative evidence for automatic promotion; the benchmark still uses its native F1."""
    from src.eval_utils import normalize_answer
    if not isinstance(answer, str) or not isinstance(reference, str) or not reference.strip():
        return False
    if normalize_answer(answer) == normalize_answer(reference):
        return True
    def full_date(text):
        text = text.strip(' \n\t."“”').replace(",", "")
        text = re.sub(r"^(?:yesterday|today|tomorrow|on)\s+", "", text, flags=re.I)
        text = text.strip("() .")
        for fmt in ("%d %B %Y", "%B %d %Y", "%d %b %Y", "%b %d %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None
    target_date = full_date(reference)
    return target_date is not None and full_date(answer) == target_date


def reference_recovered(before, after, reference):
    """Exclude format-only gains when the old response already contained the reference phrase."""
    from src.eval_utils import normalize_answer, ps
    if not isinstance(before, str) or not reference_match(after, reference) or reference_match(before, reference):
        return False
    target = [ps.stem(w) for w in normalize_answer(reference).split()]
    old = [ps.stem(w) for w in normalize_answer(before).split()]
    return bool(target) and all(old[i:i + len(target)] != target for i in range(len(old) - len(target) + 1))


def calendar_proposal(operation_bank, cases, use_source_rule=False):
    """Only nominate the tested time-normalization rule for a version seen in a rescued case."""
    supported = {}
    for case in cases:
        record = case.replay or {}
        reader = record.get("reader_probes") or {}
        control = reader.get("control") or {}
        probes = {p["type"]: p for p in reader.get("probes", []) if p["ok"]}
        dated, expanded = probes.get("dated_speaker_source"), probes.get("calendar_expansion")
        if not (control.get("ok") and dated and expanded and expanded["f1"] > max(dated["f1"], control["f1"])
                and reference_match(expanded.get("answer"), case.ground_truth)):
            continue
        for name, version in record.get("selected_skills", []):
            if "temporal" in name.lower():
                supported.setdefault((name, version), set()).add(case.query_id)
    candidates = []
    for op in operation_bank.get_all_operations():
        spec = skill_spec(op)
        ids = supported.get((op.name, spec["version"]))
        if use_source_rule and (spec["update_type"].lower() != "insert" or spec.get("source_rule")):
            continue
        if ids and CALENDAR_INSTRUCTION.strip() not in spec["instruction_template"]:
            candidates.append((len(ids), op.name, spec, sorted(ids)))
    if not candidates:
        return {"action": "no_change", "reasoning": "No reproducible calendar-expansion benefit for an active skill"}
    _, name, spec, ids = max(candidates, key=lambda x: (x[0], x[1]))
    changes = ({"source_rule": "dated_relative_turns"} if use_source_rule else
               {"instruction_template": spec["instruction_template"] + CALENDAR_INSTRUCTION})
    return {"action": "refine_existing", "reader_support": ids, "base_version": spec["version"],
            "changes": [{"action": "refine_existing", "refined_operation": {
                "name": name, "changes": changes,
                "reasoning": "Date expansion improved reader F1 relative to the same dated source. "
                             "Test this candidate on independent conversations before promotion."}}]}


def confirmed_proposal(operation_bank, cases):
    """Promote only the exact writer intervention tested in a historically reproduced suffix."""
    groups = {}
    recoveries = {}
    for case in cases:
        r = case.replay or {}
        suffix = r.get("scope_kind") == "suffix" or r.get("scope", "").startswith("final QA,")
        delta = r.get("f1_delta")
        if not (suffix and r.get("unchanged_write_reproduced") and isinstance(delta, (int, float))):
            continue
        target = operation_bank.operations.get(r.get("skill"))
        if not target or skill_spec(target)["version"] != r.get("skill_version") or not r.get("intervention"):
            continue
        key = (target.name, r["skill_version"], r["intervention"])
        groups.setdefault(key, {})[case.query_id] = delta
        answers = r.get("paired_answers", {})
        if delta > 0 and reference_recovered(answers.get("unchanged"), answers.get("edited"), case.ground_truth):
            recoveries.setdefault(key, set()).add(case.query_id)
    eligible = [(sum(values.values()) / len(values), key, values) for key, values in groups.items()
                if sum(values.values()) > 0 and recoveries.get(key)]
    if not eligible:
        return {"action": "no_change", "reasoning": "No current writer revision passed both paired F1 and reference recovery"}
    effect, (name, version, addition), effects = max(eligible, key=lambda x: (x[0], len(x[2]), x[1]))
    op = operation_bank.operations[name]
    return {"action": "refine_existing", "base_version": version, "paired_effects": effects,
            "reference_recoveries": sorted(recoveries[name, version, addition]),
            "changes": [{"action": "refine_existing", "refined_operation": {
                "name": name, "changes": {"instruction_template": op.instruction_template + addition},
                "reasoning": f"Mean final F1 change {effect:.4f} on {len(effects)} reproduced training queries."}}]}


class EpisodeTrace:
    def __init__(self, directory, sample_id, conversation=None):
        self.path = Path(directory) / f"{uuid.uuid4().hex}.jsonl"
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.sample_id = sample_id
        self.events = []
        self.ids = []
        self.next_id = 0
        self.versions = {}
        self.current = None
        conversation = conversation if isinstance(conversation, dict) else {}
        self.turns = [(turn["dia_id"], turn["text"], conversation.get(key + "_date_time", ""))
                      for key, turns in conversation.items() if isinstance(turns, list)
                      for turn in turns if isinstance(turn, dict) and turn.get("dia_id") and turn.get("text")]

    def write(self, kind, payload):
        with self.path.open("a") as out:
            out.write(json.dumps({"kind": kind, "sample": self.sample_id, **payload},
                                 ensure_ascii=False) + "\n")

    def begin(self, span_id, text, skills, retrieved, indices, selected=None):
        self.current = {"span_id": span_id, "source": text,
                        "skills": [skill_spec(op) for op in skills],
                        "retrieved": list(retrieved), "refs": [self.ref(i) for i in indices],
                        "changes": []}
        if selected is not None:
            self.current["selected_skills"] = [skill_spec(op) for op in selected]
        if self.turns:
            # Only claim complete turns visible in this chunk. Overlapping chunks may share a turn.
            self.current["source_turns"] = [tid for tid, words, date in self.turns if words in text
                                            and (not date or text.startswith("DATE: " + date + "\n"))]

    def ref(self, index):
        mid = self.ids[index]
        return [mid, self.versions[mid]]

    def change(self, kind, index, before, after):
        if kind == "INSERT":
            mid = self.next_id
            self.next_id += 1
            self.ids.append(mid)
            self.versions[mid] = 0
        else:
            mid = self.ids[index]
        self.versions[mid] += 1
        self.current["changes"].append({"action": kind, "memory_id": mid,
                                        "version": self.versions[mid], "before": before, "after": after})
        if kind == "DELETE":
            self.ids.pop(index)

    def finish(self, results, applied):
        self.current["parsed_actions"] = [vars(result).copy() for result in results]
        self.current["applied"] = bool(applied)
        self.write("span", self.current)
        self.events.append(self.current)
        self.current = None

    def verify(self, bank):
        from .replay import restore_text
        live = restore_text(self.events)
        if list(live) != self.ids or list(live.values()) != bank.get_all_contents():
            raise ValueError("Journal and live memory bank disagree")
        digest = hashlib.sha256(json.dumps(list(live.items()), ensure_ascii=False).encode()).hexdigest()
        self.write("snapshot", {"memory_refs": [self.ref(i) for i in range(len(live))], "text_sha256": digest})


def evidence_turns(conversation, evidence):
    wanted = set(evidence or ())
    return [dict(turn, session_date=conversation.get(key + "_date_time", "")) for key, turns in conversation.items()
            if key.startswith("session_") and isinstance(turns, list)
            for turn in turns if turn.get("dia_id") in wanted]


def local_diagnosis(trainer, case, qa, conversation, trace):
    """Paired local intervention; both arms see identical old memory, span and question."""
    turns = evidence_turns(conversation, qa.get("evidence"))
    candidates = [event for event in trace.events if any(
        (turn["dia_id"] in event["source_turns"] if "source_turns" in event else
         bool(turn.get("text") and turn["text"] in event["source"])) for turn in turns)]
    if not candidates:
        return {"finding": "unresolved", "reason": "no exact evidence turn in recorded span"}
    event = candidates[0]
    skills = event["skills"]
    editable = next((i for i, s in enumerate(skills) if s["update_type"].lower() in ("insert", "update")), None)
    if editable is None:
        return {"finding": "unresolved", "span_id": event["span_id"], "reason": "no editable skill"}
    ev = trainer.evaluator
    from src.llm_utils import request_task
    from .memory_bank import MemoryBank
    from .replay import replay_suffix, reader_probes
    from src.rag_utils import get_embeddings
    import numpy as np

    # Replay the affected retrieval window. Indices in Executor results are local to this window.
    args = ev.prepare_eval_args()
    old = event["retrieved"]
    scope = getattr(trainer.args, "designer_replay_scope", "local")
    if scope == "local":
        vectors = get_embeddings(trainer.args.retriever, old, "context") if old else []
        query_vector = get_embeddings(trainer.args.retriever, [case.question], "query")[0]

    def read(results):
        bank = MemoryBank(trainer.args.retriever, top_k=trainer.config.mem_top_k)
        for i, memory in enumerate(old):
            bank.add_memory(memory, np.array(vectors[i]))
        if not trainer.executor.apply_to_memory_bank(results, bank, list(range(len(old)))):
            return {"valid": False}
        context, _ = bank.retrieve(query_vector)
        prompt = ev.build_prompt(case.question, context, qa)
        _, answer, _, ok = request_task((0, prompt, args))
        return {"valid": bool(ok), "answer": answer,
                "f1": ev.compute_f1(answer, ev.get_ground_truth(qa), qa), "memories": context}

    arms = {}
    outputs = {}
    intervention = DETAIL_INSTRUCTION
    for label in ("unchanged", "edited"):
        ops = [SimpleNamespace(**skill) for skill in skills]
        if label == "edited":
            ops[editable].instruction_template += intervention
        results = trainer.executor.execute_operation(ops, event["source"], old)
        outputs[label] = [vars(result).copy() for result in results]
        if not results or not all(r.success for r in results):
            arms[label] = {"valid": False}
        elif scope == "suffix":
            arms[label] = replay_suffix(trainer, trace.events, trace.events.index(event), results, qa, case)
        else:
            arms[label] = read(results)
    fields = ("action_type", "success", "memory_index", "memory_content")
    def normalized(actions):
        return [tuple(action[k] for k in fields) for action in actions]
    faithful = normalized(outputs["unchanged"]) == normalized(event["parsed_actions"])
    eligible = faithful and event["applied"] and all(a["valid"] for a in arms.values())
    if eligible and scope == "suffix":
        eligible = (arms["unchanged"]["final_bank_matches_observed"]
                    and arms["unchanged"]["context_matches_observed"]
                    and arms["unchanged"]["answer"].strip() == case.prediction.strip())
    delta = arms["edited"]["f1"] - arms["unchanged"]["f1"] if eligible else None
    finding = "unresolved"
    if delta is not None and delta > 0:
        finding = (f"{scope}_rule_edit_helped" if reference_recovered(arms["unchanged"].get("answer"),
                                                                    arms["edited"].get("answer"), case.ground_truth)
                   else f"{scope}_f1_increased_answer_unconfirmed")
    result = {"finding": finding, "scope_kind": scope, "scope": ("same-span retrieved window, immediate QA" if scope == "local" else
                                           "final QA, later skills fixed; retrieval and writes rerun"),
              "trace": trace.path.name, "span_id": event["span_id"], "source": event["source"],
              "skill": skills[editable]["name"], "skill_version": skills[editable]["version"],
              "unchanged_write_reproduced": faithful, "intervention": intervention,
              "f1_delta": delta, "branches": arms, "actions": outputs}
    if scope == "suffix":
        result["reader_probes"] = reader_probes(trainer, case, qa, trace.events, turns)
    trace.write("replay", {"query_id": case.query_id, **result})
    # Keep prompts small; the full source, memory revisions and interventions are in the journal.
    return {k: result[k] for k in ("finding", "scope", "scope_kind", "span_id", "skill", "skill_version",
                                   "unchanged_write_reproduced", "intervention", "f1_delta", "trace")} | {
        "source_evidence": [turn["text"] for turn in turns if turn["text"] in event["source"]],
        "selected_skills": [(s["name"], s["version"]) for s in skills],
        "candidate_span_ids": [item["span_id"] for item in candidates],
        "reader_probes": result.get("reader_probes"),
        "paired_answers": {k: v.get("answer") for k, v in arms.items()}}
