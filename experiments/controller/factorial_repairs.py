"""Recheck strict terminal action effects against measured same-skill repair controls."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from experiments.controller.factorial_rollout import CELLS, load_cell, read
from experiments.controller.paired_grounding import window
from experiments.controller.resume_status import check_equal_source_choice
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import (
    _answer,
    _digest,
    execution_fault,
    hidden_retrieval_candidate,
    possible_recovery,
    recovered,
)
from src.episode_trace import evidence_turns
from src.evaluation_cache import history_embeddings, read_response
from src.executor import ExecutionResult
from src.rag_utils import get_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "factorial_repairs.json"


def checked_cell(trainer, convo, folder, bits, specs, cache):
    decisions, _, _ = load_cell(
        trainer, convo, folder, bits, [s["version"] for s in specs]
    )
    (trace,) = (folder / "traces").glob("*.jsonl")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    spans = [e for e in events if e["kind"] == "span"]
    if any(e["sample"] != convo["sample_id"] for e in events) or [
        s["span_id"] for s in spans
    ] != [0, 1]:
        raise ValueError("Unexpected dialogue or span ordering")
    usage = [
        json.loads(line) for line in (folder / "usage.jsonl").read_bytes().splitlines()
    ]
    calls = [r for r in usage if r["kind"] == "executor"]
    observed = [r for s in spans for r in s["executor_responses"]]
    for logged, submitted in zip(calls, observed, strict=True):
        measured = read_response(cache, logged["cache_key"])
        if (
            not measured
            or measured["sha256"] != logged["response_sha256"]
            or measured["payload"]["prompt_sha256"] != submitted["prompt_sha256"]
            or measured["payload"]["response"] != submitted["response"]
            or measured["payload"]["tokens"] != [logged["input"], logged["output"]]
        ):
            raise ValueError(
                "Archived writer response does not match its measured request"
            )
    for row, span in zip(decisions, spans, strict=True):
        check_equal_source_choice(row, specs, span["selected_skills"])
    scores = {
        r["qa_index"]: {
            "answer": r["prediction"],
            "ok": True,
            "f1": r["f1"],
            "tokens": r["tokens"],
            "indices": r["retrieved_indices"],
        }
        for r in read(folder / "scores.json")["answers"]
    }
    return {"spans": spans, "scores": scores, "trace": file_hash(trace)}


def contrasts(cells, questions, evaluator):
    rows = []
    for position in (0, 1):
        for background in (0, 1):
            a, b = (
                ((0, background), (1, background))
                if position == 0
                else ((background, 0), (background, 1))
            )
            before, after = [cells[k]["spans"] for k in (a, b)]
            left, right = before[position], after[position]
            state_keys = ("source", "source_turns", "refs", "retrieved")
            if (
                before[:position] != after[:position]
                or any(left[k] != right[k] for k in state_keys)
                or left["selected_skills"][:-1] != right["selected_skills"][:-1]
            ):
                raise ValueError("Action contexts or the unchanged slots are different")
            for i, qa in enumerate(questions):
                temporal, activity = [cells[k]["scores"][i] for k in (a, b)]
                truth = evaluator.get_ground_truth(qa)
                sign = int(recovered(temporal, activity, truth)) - int(
                    recovered(activity, temporal, truth)
                )
                rows.append(
                    {
                        "position": position,
                        "background": background,
                        "qa_index": i,
                        "original_index": qa["original_index"],
                        "category": qa["category"],
                        "linked": bool(
                            set(left["source_turns"]).intersection(qa["evidence"])
                        ),
                        "delta": activity["f1"] - temporal["f1"],
                        "recovery_sign": sign,
                    }
                )
    return rows


def replay_writes(trainer, spans):
    """Reapply the original writer outputs using the same per-session embedding batches."""
    bank = trainer._initialize_memory_bank()
    prefix = {}
    for n, span in enumerate(spans):
        positions = {mid: i for i, mid in enumerate(prefix)}
        indices = [positions[mid] for mid, _ in span["refs"]]
        results = [ExecutionResult(**r) for r in span["parsed_actions"]]
        names = [op["name"] for op in span["selected_skills"]]
        if not trainer.executor.apply_to_memory_bank(
            results, bank, indices, operation_name=names
        ):
            raise RuntimeError("Failed to replay measured write actions")
        bank.step()
        prefix = restore_text(spans[: n + 1])
        if bank.get_all_contents() != list(prefix.values()):
            raise ValueError("Decoded writer actions changed the archived memory bank")
    return bank


def answer(trainer, qa, ctx):
    result = _answer(trainer, qa, ctx)
    if not result["ok"] or min(result["tokens"]) <= 0:
        raise RuntimeError("Missing measured reader answer")
    return dict(result, context_hash=_digest(ctx))


def retrieval_probe(trainer, qa, turns, indices, live):
    mid = hidden_retrieval_candidate(qa, turns, indices, live)
    if mid is None:
        return None
    changed = [*indices[:-1], mid]
    return dict(answer(trainer, qa, [live[j] for j in changed]), indices=changed)


def final_probe(trainer, convo, row, cells, banks, embedding, *, probe_if_write_unavailable=False):
    i, prefix = row["qa_index"], row["background"]
    qa = convo["qa"][i]
    outcome = {
        "status": "unreproduced",
        "signed_effect": 0.0,
        "controls": {},
        "competing": [],
    }
    weak = int(row["recovery_sign"] < 0)
    contexts = {}
    for final in (0, 1):
        key = prefix, final
        spans = cells[key]["spans"]
        if any(
            execution_fault(s["parsed_actions"], s["applied"], s["skills"])
            for s in spans
        ):
            return dict(outcome, status="execution_fault")
        if key not in banks:
            banks[key] = replay_writes(trainer, spans)
        ctx, indices = banks[key].retrieve(
            embedding, top_k=trainer.config.mem_top_k_eval
        )
        indices = list(map(int, indices))
        actual = dict(answer(trainer, qa, ctx), indices=indices)
        outcome["controls"][str(final)] = actual
        expected = cells[key]["scores"][i]
        if any(actual[k] != v for k, v in expected.items()):
            return outcome
        contexts[final] = ctx
    if contexts[0] == contexts[1]:
        return dict(outcome, status="same_context")
    before = outcome["controls"][str(weak)]
    weak_bank = banks[prefix, weak]
    event = cells[prefix, weak]["spans"][-1]
    turns = evidence_turns(convo["conversation"], qa["evidence"])
    quoted = [
        f"{t['session_date']} | {t['speaker']}: {t['text']}"
        for t in turns
        if t["dia_id"] in event["source_turns"]
    ]
    live = dict(enumerate(weak_bank.get_all_contents()))
    truth = trainer.evaluator.get_ground_truth(qa)
    if not quoted or not any(
        op["update_type"].lower() == "insert" for op in event["skills"]
    ):
        if probe_if_write_unavailable:
            probe = retrieval_probe(trainer, qa, turns, before["indices"], live)
            if probe is not None:
                outcome["retrieval"] = probe
                if possible_recovery(before, probe, truth):
                    outcome["competing"].append("retrieval")
                    return dict(outcome, status="competing_recovery")
        return dict(outcome, status="unavailable_repair")
    seen = set(live.values())
    missing = [t for t in dict.fromkeys(quoted) if t not in seen]
    # Rebuild a private copy only when a question adds quotes. All original vectors stay in their native batches.
    repaired = weak_bank
    if missing:
        repaired = replay_writes(trainer, cells[prefix, weak]["spans"])
        actions = [ExecutionResult("INSERT", True, memory_content=t) for t in missing]
        if not trainer.executor.apply_to_memory_bank(actions, repaired, []):
            raise RuntimeError("Failed to append the evidence quotes")
        if repaired.get_all_contents() != [*live.values(), *missing]:
            raise ValueError("The repair changed an existing memory")
    ctx, idx = repaired.retrieve(embedding, top_k=trainer.config.mem_top_k_eval)
    repair = dict(
        answer(trainer, qa, ctx), indices=list(map(int, idx)), added_quotes=missing
    )
    outcome["execution"] = repair
    if repair["context_hash"] != before["context_hash"] and possible_recovery(
        before, repair, truth
    ):
        outcome["competing"].append("execution")
    probe = retrieval_probe(trainer, qa, turns, before["indices"], live)
    if probe is not None:
        outcome["retrieval"] = probe
        if possible_recovery(before, probe, truth):
            outcome["competing"].append("retrieval")
    if outcome["competing"]:
        return dict(outcome, status="competing_recovery")
    return dict(outcome, status="survived_probes", signed_effect=row["delta"])


def run(trainer, study, forecast, train, specs):
    archive = forecast.parent / "factorial"
    manifest = read(archive / "manifest.json")
    targets = [(c["sample_id"], n) for c in train for n in (5, 7)]
    if (
        manifest["protocol"] != file_hash(study / "protocol.json")
        or manifest["forecast"] != file_hash(forecast)
        or targets != list(map(tuple, manifest["windows"]))
        or len(targets) != 12
    ):
        raise ValueError("Expected the frozen four-path training cohort")
    out = archive.parent / "factorial-repairs"
    out.mkdir(exist_ok=True)
    identity = {
        "plan": file_hash(PLAN),
        "source": file_hash(archive / "manifest.json"),
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
    }
    register(out / "manifest.json", identity)
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "usage.jsonl")
    data = {c["sample_id"]: c for c in train}
    reports = []
    for sid, start in targets:
        name = f"{sid}_s{start}"
        folder = archive / name
        if not (folder / "measurement.json").is_file():
            continue
        convo = window(data[sid], start, 2)
        cells = {
            k: checked_cell(
                trainer,
                convo,
                folder / f"{k[0]}{k[1]}",
                k,
                specs,
                archive / "responses",
            )
            for k in CELLS
        }
        inputs = dict(
            identity,
            window=file_hash(folder / "measurement.json"),
            traces={f"{a}{b}": cells[a, b]["trace"] for a, b in CELLS},
        )
        dest = out / (name + ".json")
        if dest.exists():
            report = read(dest)
            if report["inputs"] != inputs:
                raise ValueError("Completed window inputs changed")
        else:
            rows = contrasts(cells, convo["qa"], trainer.evaluator)
            candidates = [
                r
                for r in rows
                if r["position"] == 1 and r["linked"] and r["recovery_sign"]
            ]
            if candidates:
                with history_embeddings():
                    queries = get_embeddings(
                        trainer.args.retriever,
                        [q["question"] for q in convo["qa"]],
                        "query",
                    )
                    banks = {}
                    for r in candidates:
                        r["probe"] = final_probe(
                            trainer, convo, r, cells, banks, queries[r["qa_index"]]
                        )
            report = {
                "inputs": inputs,
                "dialogue": sid,
                "start": start,
                "questions": len(convo["qa"]),
                "rows": rows,
            }
            register(dest, report)
        print(
            name,
            Counter(r["probe"]["status"] for r in report["rows"] if "probe" in r),
            flush=True,
        )
        reports.append(report)
    if len(reports) == len(targets):
        register(out / "summary.json", summarize(reports))


def summarize(reports):
    summary = {
        "windows": len(reports),
        "questions": sum(v["questions"] for v in reports),
        "positions": {},
    }
    for pos in (0, 1):
        pairs = [(w, r) for w in reports for r in w["rows"] if r["position"] == pos]
        linked = [(w, r) for w, r in pairs if r["linked"] and r["recovery_sign"]]
        by_question = defaultdict(list)
        for w, r in linked:
            by_question[w["dialogue"], w["start"], r["original_index"]].append(
                r["recovery_sign"]
            )
        summary["positions"][str(pos)] = {
            "comparisons": len(pairs),
            "nonzero_f1": sum(r["delta"] != 0 for _, r in pairs),
            "recoveries": sum(bool(r["recovery_sign"]) for _, r in pairs),
            "linked": len(linked),
            "unique_linked_questions": len({(k[0], k[2]) for k in by_question}),
            "both_backgrounds_agree": sum(
                len(v) == 2 and v[0] == v[1] for v in by_question.values()
            ),
            "by_dialogue": {
                w["dialogue"]: sum(x["dialogue"] == w["dialogue"] for x, _ in linked)
                for w in reports
            },
            "probes": dict(
                Counter(r["probe"]["status"] for _, r in linked if "probe" in r)
            ),
        }
    return summary
