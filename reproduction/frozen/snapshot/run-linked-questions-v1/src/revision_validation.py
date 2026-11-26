"""Evaluate a proposed skill bank on reserved LoCoMo conversations before promotion."""
import copy
import json
from pathlib import Path
from statistics import mean

from .episode_trace import EpisodeTrace, skill_spec
from .operation_bank import OperationBank


def evaluate_bank(trainer, bank, conversations, directory):
    # Use the same deterministic controller, memory builder, reader and F1 as --eval-only.
    from src.inference.memory import _build_memory_bank_from_sessions, _resolve_sample_id
    from src.inference.text import evaluate_text_dataset_queries
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    previous_bank = trainer.operation_bank
    training = trainer.controller.training
    args = copy.copy(trainer.args)
    args.eval_f1_only = True
    args.out_file = str(directory / "scores.json")
    trainer.operation_bank = bank
    trainer.controller.eval()
    try:
        memories = {}
        for index, convo in enumerate(conversations):
            sid = _resolve_sample_id(trainer, convo, index)
            sessions, _, _ = trainer._prepare_sessions(convo)
            trace = EpisodeTrace(directory / "traces", sid, convo.get("conversation"))
            memories[sid] = _build_memory_bank_from_sessions(trainer, sessions, trace=trace, show_progress=False)
            for event in trace.events:
                if any("API call failed" in a["reasoning"] for a in event["parsed_actions"]):
                    raise RuntimeError(f"Executor request failed in {sid} at span {event['span_id']}")
        return evaluate_text_dataset_queries(trainer, conversations, memories, args)
    finally:
        trainer.operation_bank = previous_bank
        trainer.controller.train(training)


def compare_revisions(control, candidate):
    def index(report):
        rows = report["answers"]
        scores = {(r["sample_id"], r["qa_index"]): float(r["f1"]) for r in rows}
        if len(scores) != len(rows) or not scores:
            raise ValueError("Expected exactly one score per question")
        return scores

    before, after = index(control), index(candidate)
    if before.keys() != after.keys():
        raise ValueError("Control and revised bank cover different questions")
    delta = {key: after[key] - score for key, score in before.items()}
    by_conversation = {sid: mean(v for (s, _), v in delta.items() if s == sid)
                       for sid in sorted({s for s, _ in delta})}
    # Require an aggregate gain and no conversation-level regression on the calibration set.
    return {"accept": mean(delta.values()) > 0 and min(by_conversation.values()) >= 0,
            "delta": mean(delta.values()), "by_conversation": by_conversation,
            "improved": sum(v > 0 for v in delta.values()), "worse": sum(v < 0 for v in delta.values()),
            "tied": sum(v == 0 for v in delta.values())}


def check_exposure(before, after, trace_directory, decision):
    """A score difference supports a revision only if this exact version actually ran."""
    original = {op.name: skill_spec(op)["version"] for op in before.get_all_operations()}
    revised = {op.name: skill_spec(op) for op in after.get_all_operations()
               if original.get(op.name) != skill_spec(op)["version"]}
    exposure = {name: [] for name in revised}
    for path in sorted(Path(trace_directory).glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["kind"] != "span":
                    continue
                for op in event["skills"]:
                    spec = revised.get(op["name"])
                    if spec is None or op["version"] != spec["version"]:
                        continue
                    if spec.get("source_rule"):
                        generated = {a["memory_content"] for a in event["parsed_actions"]
                                     if a["success"] and a["reasoning"] == spec["source_rule"]}
                        if not any(c["action"] == "INSERT" and c["after"] in generated for c in event["changes"]):
                            continue
                    exposure[op["name"]].append([event["sample"], event["span_id"]])
    result = dict(decision, exposure=exposure)
    if not revised or not all(exposure.values()):
        result.update(accept=False, reason="No calibration exposure for one or more changed skill versions")
    return result


def validate_proposal(trainer, proposal, conversations, directory, control=None):
    directory = Path(directory)
    candidate = OperationBank.from_dict(copy.deepcopy(trainer.operation_bank.to_dict()), encoder=trainer.op_encoder)
    if not trainer.designer.apply_evolution(candidate, copy.deepcopy(proposal)):
        return {"accept": False, "reason": "No applicable revision"}
    if control is None:
        control = evaluate_bank(trainer, trainer.operation_bank, conversations, directory / "control")
    scores = evaluate_bank(trainer, candidate, conversations, directory / "candidate")
    result = check_exposure(trainer.operation_bank, candidate, directory / "candidate/traces",
                            compare_revisions(control, scores))
    (directory / "decision.json").write_text(json.dumps(result, indent=2))
    return result
