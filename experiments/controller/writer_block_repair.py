"""Reparse a measured writer response while holding its memory skill and prefix fixed."""

import json
import os
import re
from collections import Counter
from pathlib import Path

from experiments.controller.factorial_repairs import answer, checked_cell, replay_writes
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.semantic_writer import measured_readers, sha
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import _digest, execution_fault, possible_recovery, recovered
from src.evaluation_cache import history_embeddings
from src.executor import ExecutionResult
from src.rag_utils import get_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "writer_block_credit.json"
HEADER = re.compile(
    r"^[ \t]*(INSERT|UPDATE|DELETE|NOOP)[ \t]+block:[ \t]*\r?\n"
    r"(?=(?:[ \t]*\r?\n)*[ \t]*ACTION[ \t]*:[ \t]*\1[ \t]*\r?$)",
    re.IGNORECASE | re.MULTILINE,
)


def parsed_repair(executor, event):
    old, new = [], []
    for request in event["executor_responses"]:
        raw = request["response"]
        for text, results in ((raw, old), (HEADER.sub("", raw), new)):
            results.extend(
                vars(a) for a in executor._parse_response(text, len(event["refs"]))
            )
    recorded = event["parsed_actions"]
    if not old or old != recorded[: len(old)] or len(old) != len(new):
        raise ValueError(
            "Original action sequence or number of parsed operations changed"
        )
    appended = recorded[len(old) :]
    if any(
        a["reasoning"] not in ("dated_single_turns", "dated_adjacent_turns")
        for a in appended
    ):
        raise ValueError("Unexpected appended source operation")
    if any(
        dict(a, memory_content=b["memory_content"]) != b
        for a, b in zip(old, new, strict=True)
    ):
        raise ValueError("Header repair changed a non-content field")
    return new + appended, sum(
        a["memory_content"] != b["memory_content"]
        for a, b in zip(old, new, strict=True)
    )


def census(executor, archive, known, panel):
    seen = set()
    counts, exposures = {}, {}
    for sid, start in panel["windows"]:
        name = f"{sid}_s{start}"
        source = read(known / (name + ".json"))
        for bits, fingerprint in source["inputs"]["traces"].items():
            (trace,) = (archive / name / bits / "traces").glob("*.jsonl")
            if file_hash(trace) != fingerprint:
                raise ValueError(
                    "Original path trace no longer matches its audited hash"
                )
            events = [json.loads(line) for line in trace.read_bytes().splitlines()]
            affected = []
            for event in (e for e in events if e["kind"] == "span"):
                skill = event["selected_skills"][-1]["name"]
                key = (
                    sid,
                    event["source"],
                    skill,
                    tuple(r["prompt_sha256"] for r in event["executor_responses"]),
                )
                actions, edits = parsed_repair(executor, event)
                dirty = {
                    old["memory_content"]
                    for old, new in zip(event["parsed_actions"], actions, strict=True)
                    if old["memory_content"] != new["memory_content"]
                }
                affected.append(
                    {
                        (c["memory_id"], c["version"])
                        for c in event["changes"]
                        if c["after"] in dirty
                    }
                )
                if key not in seen:
                    seen.add(key)
                    group = counts.setdefault(skill, Counter())
                    group["unique_writes"] += 1
                    group["affected_writes"] += bool(edits)
                    group["changed_generated_items"] += edits
                    group["generated_items"] += sum(
                        bool(a["memory_content"])
                        and a["reasoning"]
                        not in ("dated_single_turns", "dated_adjacent_turns")
                        for a in actions
                    )
            first, last = affected
            group = exposures.setdefault(bits, Counter())
            for qa in (e for e in events if e["kind"] == "qa"):
                refs = set(map(tuple, qa["refs"]))
                group["questions"] += 1
                group["reads_affected_first"] += bool(refs & first)
                group["reads_affected_last"] += bool(refs & last)
                group["reads_either"] += bool(refs & (first | last))
    return {"unique_writes_by_skill": counts, "retrieval_by_path": exposures}


def bounded_credit(old, factual, fixed, truth):
    weak = 0 if old > 0 else 1
    if possible_recovery(factual[weak], fixed[weak], truth):
        return 0.0, "same_skill_recovery"
    if not recovered(fixed[weak], fixed[1 - weak], truth):
        return 0.0, "recovery_lost"
    delta = fixed[1]["f1"] - fixed[0]["f1"]
    return (1 if old > 0 else -1) * min(abs(old), abs(delta)), "retained"


def branch(trainer, convo, folder, bits, archive, specs, queries, indices):
    cell = checked_cell(trainer, convo, folder, bits, specs, archive / "responses")
    prefix, event = cell["spans"]
    raw = replay_writes(trainer, cell["spans"])
    actions, edits = parsed_repair(trainer.executor, event)
    if execution_fault(actions, event["applied"], event["skills"]):
        raise ValueError("Invalid repaired execution")
    fixed = raw
    if edits:
        fixed = replay_writes(trainer, [prefix])
        ids = list(restore_text([prefix]))
        positions = [ids.index(mid) for mid, _ in event["refs"]]
        ops = [s["name"] for s in event["selected_skills"]]
        if not trainer.executor.apply_to_memory_bank(
            [ExecutionResult(**a) for a in actions],
            fixed,
            positions,
            operation_name=ops,
        ):
            raise RuntimeError("Cannot apply repaired actions")
        fixed.step()
    readers = measured_readers(folder, archive / "responses")
    rows = {}
    for i in indices:
        qa, query = convo["qa"][i], queries[i]
        original, ix = raw.retrieve(query, top_k=trainer.config.mem_top_k_eval)
        factual = cell["scores"][i]
        prompt = trainer.evaluator.build_prompt(qa["question"], original, qa)
        receipt = readers.get(sha(prompt))
        if (
            receipt is None
            or receipt["response"].strip() != factual["answer"]
            or receipt["tokens"] != factual["tokens"]
            or list(map(int, ix)) != factual["indices"]
        ):
            rows[str(i)] = {"matched": False, "reason": "control_reader_mismatch"}
            continue
        current, new_ix = (
            fixed.retrieve(query, top_k=trainer.config.mem_top_k_eval)
            if edits
            else (original, ix)
        )
        old = dict(factual, context_hash=_digest(original))
        measured = answer(trainer, qa, current) if current != original else old
        rows[str(i)] = {
            "matched": True,
            "raw": old,
            "fixed": dict(measured, indices=list(map(int, new_ix))),
            "changed_context": current != original,
        }
    return {
        "trace": cell["trace"],
        "bank": _digest(raw.get_all_contents()),
        "fixed_bank": _digest(fixed.get_all_contents()),
        "edited_generated_actions": edits,
        "rows": rows,
    }


def measure(trainer, convo, episode, archive, specs, rows, dest, repair_hash):
    receipts = []
    queries = get_embeddings(
        trainer.args.retriever, [q["question"] for q in convo["qa"]], "query"
    )
    for prefix in (0, 1):
        group = [r for r in rows if r["background"] == prefix]
        if not group:
            continue
        indices = [r["qa_index"] for r in group]
        ends = [
            branch(
                trainer,
                convo,
                episode / f"{prefix}{b}",
                (prefix, b),
                archive,
                specs,
                queries,
                indices,
            )
            for b in (0, 1)
        ]
        paired = []
        for r in group:
            i = r["qa_index"]
            qa = convo["qa"][i]
            if r["original_index"] != qa["original_index"]:
                raise ValueError("Changed original question ID")
            previous = r.get("probe", {}).get("signed_effect", 0.0)
            pair = [e["rows"][str(i)] for e in ends]
            matched = all(v["matched"] for v in pair)
            old, new = None, None
            value, reason = 0.0, "unmatched" if not matched else "no_prior_credit"
            if matched:
                old, new = ([v[k] for v in pair] for k in ("raw", "fixed"))
                if abs((old[1]["f1"] - old[0]["f1"]) - r["delta"]) > 1e-12:
                    raise ValueError(
                        "Original conditional F1 differs from the repair study"
                    )
                if previous:
                    if (
                        r["probe"]["status"] != "survived_probes"
                        or not r["linked"]
                        or previous != r["delta"]
                        or any(
                            r["probe"]["controls"][str(b)] != v
                            for b, v in enumerate(old)
                        )
                    ):
                        raise ValueError("Different original rule-2 controls or credit")
                    value, reason = bounded_credit(
                        previous, old, new, trainer.evaluator.get_ground_truth(qa)
                    )
            paired.append(
                {
                    "qa_index": i,
                    "original_index": r["original_index"],
                    "category": qa["category"],
                    "matched": matched,
                    "original_credit": previous,
                    "credit": value,
                    "status": reason,
                }
            )
        receipts.append({"prefix": prefix, "ends": ends, "questions": paired})
    register(dest, {"repairs": repair_hash, "groups": receipts})


def run(trainer, study, forecast, train, specs, limit):
    base = forecast.parent
    archive, known, dest = (
        base / "factorial",
        base / "factorial-repairs",
        base / "writer-block-repair",
    )
    dest.mkdir(exist_ok=True)
    panel = read(archive / "manifest.json")
    prior = read(known / "summary.json")
    if (
        panel["protocol"] != file_hash(study / "protocol.json")
        or panel["forecast"] != file_hash(forecast)
        or prior["windows"] != 12
        or prior["questions"] != 91
    ):
        raise ValueError("Wrong factorial training study")
    register(
        dest / "manifest.json",
        {
            "plan": file_hash(PLAN),
            "panel": file_hash(archive / "manifest.json"),
            "repairs": file_hash(known / "summary.json"),
            "backend": os.environ["MEMSKILL_BACKEND_ID"],
        },
    )
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(dest / "responses")
    os.environ["MEMSKILL_USAGE_LOG"] = str(dest / "usage.jsonl")
    register(dest / "census.json", census(trainer.executor, archive, known, panel))
    data = {c["sample_id"]: c for c in train}
    # Always finalize the four credit probes before evaluating the broader diagnostic.
    agenda = []
    for sid, start in panel["windows"]:
        name = f"{sid}_s{start}"
        path = known / (name + ".json")
        stored = read(path)
        if stored["inputs"]["window"] != file_hash(archive / name / "measurement.json"):
            raise ValueError("Changed four-path source")
        full = [r for r in stored["rows"] if r["position"] == 1]
        selected = [r for r in full if r.get("probe", {}).get("signed_effect")]
        agenda.append((name, sid, start, path, full, selected))
    for mode in ("credit", "population"):
        folder = dest / mode
        folder.mkdir(exist_ok=True)
        collected = []
        for name, sid, start, path, full, selected in agenda[
            : limit if mode == "population" else None
        ]:
            rows = selected if mode == "credit" else full
            record = folder / (name + ".json")
            if not record.exists():
                if rows:
                    with history_embeddings():
                        measure(
                            trainer,
                            window(data[sid], start, 2),
                            archive / name,
                            archive,
                            specs,
                            rows,
                            record,
                            file_hash(path),
                        )
                else:
                    register(record, {"repairs": file_hash(path), "groups": []})
            value = read(record)
            if value["repairs"] != file_hash(path):
                raise ValueError("Frozen repair labels changed")
            collected.extend(q for g in value["groups"] for q in g["questions"])
            print(mode, name, len(collected), flush=True)
        if len(collected) == (4 if mode == "credit" else 182):
            register(
                folder / "summary.json",
                {
                    "rows": len(collected),
                    "matched": sum(q["matched"] for q in collected),
                    "statuses": dict(Counter(q["status"] for q in collected)),
                    "files": {n: file_hash(folder / (n + ".json")) for n, *_ in agenda},
                },
            )
        if mode == "credit" and not (folder / "summary.json").exists():
            raise ValueError("Missing archived credit probes")
