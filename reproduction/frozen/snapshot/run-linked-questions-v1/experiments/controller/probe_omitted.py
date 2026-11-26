"""Re-probe every unexamined training question linked to the final span of a saved window."""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from experiments.controller.resume_status import load_episode
from src.credit_assignment import diagnose_episode


def omitted_final_indices(spans, questions, observed, original):
    linked = set(spans[-1]["source_turns"])
    seen = {q["qa_index"] for q in original["questions"]}
    return [i for i in sorted(observed) if i not in seen and linked.intersection(questions[i].get("evidence") or ())]


def atomic_json(path, data):
    with TemporaryDirectory(dir=path.parent) as tmp:
        output = Path(tmp) / path.name
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        output.replace(path)


def check_responses(diagnosis):
    for name in ("control", "swap", "swap_repeat", "execution_repair", "repair_repeat"):
        branch = diagnosis.get(name, {})
        if (any("API call failed" in a["reasoning"] for a in branch.get("actions", []))
                or any(not r["ok"] for r in branch.get("qa", {}).values())):
            raise RuntimeError(f"Model request failed in omitted-question {name} replay")
    for question in diagnosis["questions"]:
        probe = question.get("retrieval_probe")
        if probe and not (probe["reply"]["ok"] and probe.get("repeat", probe["reply"])["ok"]):
            raise RuntimeError("Model request failed in omitted-question retrieval probe")


def probe_training_windows(trainer, study, windows, *, completed_only=False):
    backend = os.environ.get("MEMSKILL_BACKEND_ID", trainer.args.api_base)
    work = study / "diagnostics" / "omitted-final" / hashlib.sha256(backend.encode()).hexdigest()[:16]
    work.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    pending = []
    with (work / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.environ["MEMSKILL_USAGE_LOG"] = str(work / "usage.jsonl")
        for conversation, episode in windows:
            if completed_only and not episode.exists():
                pending.append(episode.name)
                continue
            if completed_only and json.loads((episode / "backend.json").read_bytes())["inference"] != backend:
                raise ValueError(f"Live training window and probe use different inference backends at {episode}")
            sid = conversation["sample_id"]
            ids = [q["original_index"] for q in conversation["qa"]]
            trace_file, = (episode / "traces").glob("*.jsonl")
            raw = trace_file.read_bytes()
            events = [json.loads(line) for line in raw.splitlines()]
            spans = [e for e in events if e["kind"] == "span"]
            rows = load_episode(episode, sid, len(spans), ids, deployment_writer=trainer.args.span_credit_deployment_writer)
            original = [e["diagnosis"] for e in events if e["kind"] == "span_credit"][-1]
            observed = {int(e["query_id"].rsplit("_", 1)[1]) for e in events if e["kind"] == "qa" and e["success"]}
            if observed != set(range(len(ids))):
                raise ValueError(f"Missing factual answers in the saved training window at {episode}")
            indices = omitted_final_indices(spans, conversation["qa"], observed, original)
            signature = {"backend": backend, "episode": episode.name, "trace_sha256": hashlib.sha256(raw).hexdigest(),
                         "span": len(spans) - 1, "original_question_ids": [ids[i] for i in indices], "rule": 2}
            output = work / (episode.name + ".json")
            if output.exists():
                result = json.loads(output.read_text())
                if result["inputs"] != signature:
                    raise ValueError(f"Omitted-question probe inputs differ at {output}")
            else:
                result = {"inputs": signature, "diagnosis": None}
                if indices:
                    steps = [{"action_idx": r["grounded"] if r["use"] else r["plain"],
                              "log_prob": r["old_log_probability"]} for r in rows]
                    trace = SimpleNamespace(path=trace_file, events=spans)
                    before = trainer.args.span_credit_questions
                    trainer.args.span_credit_questions = len(indices)
                    try:
                        diagnosis = diagnose_episode(trainer, conversation, trace, steps,
                                                     requested_questions={len(spans) - 1: indices})[-1]
                    finally:
                        trainer.args.span_credit_questions = before
                    if [q["qa_index"] for q in diagnosis["questions"]] != indices:
                        raise ValueError("The probe did not cover every assigned training question")
                    check_responses(diagnosis)
                    result["diagnosis"] = diagnosis
                atomic_json(output, result)
            totals["windows"] += 1
            totals["omitted_questions"] += len(indices)
            if indices:
                totals["windows_with_omissions"] += 1
                for q in result["diagnosis"]["questions"]:
                    totals["factual_reader_matched"] += int(q["control_matched"])
                    totals[q["cause"]] += 1
                    if q["controller_delta"]:
                        signed = q["controller_delta"] * (1 if rows[-1]["use"] else -1)
                        totals["selection_favors_grounded" if signed > 0 else "selection_favors_plain"] += 1
            print(episode.name, dict(totals), flush=True)
        summary = {"backend": backend, "counts": dict(totals)}
        if completed_only:
            atomic_json(work / "progress.json", dict(summary, pending_windows=pending))
        if not pending:
            atomic_json(work / "summary.json", summary)
