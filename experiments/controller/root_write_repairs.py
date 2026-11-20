"""Counterfactual selection at the first span with the observed suffix decisions held fixed."""

import json
import os
from collections import Counter
from pathlib import Path

from experiments.controller.factorial_repairs import answer, checked_cell
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import (
    _digest,
    execution_fault,
    hidden_retrieval_candidate,
    possible_recovery,
    recovered,
)
from src.episode_trace import evidence_turns
from src.evaluation_cache import history_embeddings, publish_response, read_response
from src.executor import ExecutionResult
from src.rag_utils import get_embeddings
from src.replay import rebuild_suffix, restore_text

PLAN = Path(__file__).with_name("protocols") / "root_write_repairs.json"


def seed_controls(cell, archive, cache):
    for line in (cell / "usage.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        key = r["cache_key"]
        receipt = read_response(archive, key)
        if not receipt or receipt["sha256"] != r["response_sha256"]:
            raise ValueError("Control cache mismatch")
        v = receipt["payload"]
        publish_response(cache, key, v["prompt_sha256"], v["response"], *v["tokens"])


def replay(trainer, spans, extra=()):
    actions = [ExecutionResult(**r) for r in spans[0]["parsed_actions"]]
    actions.extend(ExecutionResult("INSERT", True, memory_content=t) for t in extra)
    audit = []

    def observe(event, retrieved, ops, writes, applied, writer_prompt_hashes):
        decoded = [vars(action).copy() for action in writes]
        audit.append(
            {
                "retrieved": list(retrieved),
                "writer_requests": writer_prompt_hashes,
                "parsed_actions": decoded,
                "applied": bool(applied),
                "fault": execution_fault(decoded, applied, event["skills"]),
                "matches_original": retrieved == event["retrieved"]
                and decoded == event["parsed_actions"]
                and writer_prompt_hashes
                == [r["prompt_sha256"] for r in event["executor_responses"]],
            }
        )

    bank, failed = rebuild_suffix(trainer, spans, 0, actions, observe_suffix=observe)
    return bank, {
        "suffix": audit,
        "failed": failed,
        "contents_sha256": _digest(bank.get_all_contents())
        if bank is not None
        else None,
    }


def check_root(trainer, convo, row, cells, embedding):
    i = row["qa_index"]
    b = row["background"]
    qa = convo["qa"][i]
    weak = int(row["recovery_sign"] < 0)
    branches = [cells[a, b] for a in (0, 1)]
    events = [c["spans"] for c in branches]
    out = {
        "controls": {},
        "competing": [],
        "signed_effect": 0.0,
        "status": "unreproduced",
        "choices": [[s["selected_skills"] for s in spans] for spans in events],
    }
    if (
        events[0][1]["selected_skills"] != events[1][1]["selected_skills"]
        or events[0][0]["selected_skills"][:-1] != events[1][0]["selected_skills"][:-1]
        or any(
            events[0][0][k] != events[1][0][k]
            for k in ("source", "source_turns", "retrieved", "refs")
        )
    ):
        return dict(out, status="different_fixed_context")
    banks = []
    for a, (cell, spans) in enumerate(zip(branches, events, strict=True)):
        if any(
            execution_fault(s["parsed_actions"], s["applied"], s["skills"])
            for s in spans
        ):
            return dict(out, status="execution_fault")
        bank, audit = replay(trainer, spans)
        out["controls"][str(a)] = audit
        if (
            bank is None
            or bank.get_all_contents() != list(restore_text(spans).values())
            or len(audit["suffix"]) != 1
            or not audit["suffix"][0]["matches_original"]
            or audit["suffix"][0]["fault"]
        ):
            return out
        ctx, idx = bank.retrieve(embedding, top_k=trainer.config.mem_top_k_eval)
        check = dict(answer(trainer, qa, ctx), indices=list(map(int, idx)), context=ctx)
        audit["reader"] = check
        if any(check[k] != v for k, v in cell["scores"][i].items()):
            return out
        banks.append(bank)
    before = out["controls"][str(weak)]["reader"]
    better = out["controls"][str(1 - weak)]["reader"]
    truth = trainer.evaluator.get_ground_truth(qa)
    if not recovered(before, better, truth) or before["context"] == better["context"]:
        return out
    bank = banks[weak]
    spans = events[weak]
    root = spans[0]
    turns = evidence_turns(convo["conversation"], qa["evidence"])
    quotes = list(
        dict.fromkeys(
            f"{t['session_date']} | {t['speaker']}: {t['text']}"
            for t in turns
            if t["dia_id"] in root["source_turns"]
        )
    )
    if not quotes or not any(
        op["update_type"].lower() == "insert" for op in root["skills"]
    ):
        return dict(out, status="unavailable_repair")
    original = set(restore_text(spans[:1]).values())
    missing = [t for t in quotes if t not in original]
    if missing:
        fixed, audit = replay(trainer, spans, missing)
        out["repaired_suffix"] = audit
        if fixed is None or len(audit["suffix"]) != 1 or audit["suffix"][0]["fault"]:
            return dict(out, status="invalid_repaired_suffix")
        ctx, idx = fixed.retrieve(embedding, top_k=trainer.config.mem_top_k_eval)
        probe = dict(answer(trainer, qa, ctx), indices=list(map(int, idx)), context=ctx)
    else:
        probe = before
    out["execution"] = {"added": missing, "reader": probe}
    if possible_recovery(before, probe, truth):
        out["competing"].append("execution")
    live = dict(enumerate(bank.get_all_contents()))
    idx = before["indices"]
    mid = hidden_retrieval_candidate(qa, turns, idx, live)
    if mid is not None:
        chosen = [*idx[:-1], mid]
        ctx = [live[n] for n in chosen]
        probe = dict(answer(trainer, qa, ctx), indices=chosen, context=ctx)
        out["retrieval"] = probe
        if possible_recovery(before, probe, truth):
            out["competing"].append("retrieval")
    if out["competing"]:
        return dict(out, status="competing_recovery")
    return dict(out, status="survived_probes", signed_effect=row["delta"])


def run(trainer, study, forecast, train, specs):
    src = forecast.parent
    archive = src / "factorial"
    known = src / "factorial-repairs"
    out = src / "root-write-repairs"
    out.mkdir(exist_ok=True)
    panel = read(archive / "manifest.json")
    summary = read(known / "summary.json")
    if (
        panel["backend"] != os.environ["MEMSKILL_BACKEND_ID"]
        or panel["protocol"] != file_hash(study / "protocol.json")
        or panel["forecast"] != file_hash(forecast)
        or not trainer.args.span_credit_deployment_writer
        or summary["windows"] != 12
        or summary["questions"] != 91
    ):
        raise ValueError("Different original archive or deployment writer")
    identity = {
        "plan": file_hash(PLAN),
        "repairs": file_hash(known / "summary.json"),
        "panel": file_hash(archive / "manifest.json"),
    }
    register(out / "manifest.json", identity)
    cache = out / "responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    data = {c["sample_id"]: c for c in train}
    finished = []
    for sid, n in panel["windows"]:
        key = f"{sid}_s{n}"
        stored = read(known / (key + ".json"))
        d = out / key
        d.mkdir(exist_ok=True)
        os.environ["MEMSKILL_USAGE_LOG"] = str(d / "usage.jsonl")
        inputs = dict(identity, repairs_window=file_hash(known / (key + ".json")))
        receipt = d / "window.json"
        if not receipt.exists():
            convo = window(data[sid], n, 2)
            candidates = [
                r
                for r in stored["rows"]
                if r["position"] == 0 and r["linked"] and r["recovery_sign"]
            ]
            rows = []
            cells = {}
            if candidates:
                with history_embeddings():
                    vecs = get_embeddings(
                        trainer.args.retriever,
                        [q["question"] for q in convo["qa"]],
                        "query",
                    )
                    for r in candidates:
                        for a in (0, 1):
                            bits = (a, r["background"])
                            label = "".join(map(str, bits))
                            if bits not in cells:
                                cell = archive / key / label
                                cells[bits] = checked_cell(
                                    trainer,
                                    convo,
                                    cell,
                                    bits,
                                    specs,
                                    archive / "responses",
                                )
                                if (
                                    cells[bits]["trace"]
                                    != stored["inputs"]["traces"][label]
                                ):
                                    raise ValueError("Changed original writer trace")
                                seed_controls(cell, archive / "responses", cache)
                        path = (
                            d / f"q{r['original_index']}-suffix{r['background']}.json"
                        )
                        if not path.exists():
                            register(
                                path,
                                {
                                    "inputs": inputs,
                                    "original": r,
                                    "probe": check_root(
                                        trainer, convo, r, cells, vecs[r["qa_index"]]
                                    ),
                                },
                            )
                        v = read(path)
                        if v["inputs"] != inputs or v["original"] != r:
                            raise ValueError("Changed underlying candidate")
                        rows.append(
                            {
                                "index": r["original_index"],
                                "suffix": r["background"],
                                "status": v["probe"]["status"],
                                "signed_effect": v["probe"]["signed_effect"],
                                "receipt": file_hash(path),
                            }
                        )
            register(
                receipt,
                {
                    "inputs": inputs,
                    "dialogue": sid,
                    "questions": stored["questions"],
                    "comparisons": 2 * stored["questions"],
                    "rows": rows,
                },
            )
        v = read(receipt)
        if v["inputs"] != inputs:
            raise ValueError("Changed earlier-span inputs")
        finished.append(v)
        print(key, Counter(r["status"] for r in v["rows"]), flush=True)
    rows = [r for w in finished for r in w["rows"]]
    if len(rows) != summary["positions"]["0"]["linked"]:
        raise ValueError("Missing linked earlier-span comparison")
    register(
        out / "summary.json",
        {
            "windows": len(finished),
            "questions": sum(w["questions"] for w in finished),
            "comparisons": sum(w["comparisons"] for w in finished),
            "candidates": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "unique_surviving_questions": len(
                {
                    (w["dialogue"], r["index"])
                    for w in finished
                    for r in w["rows"]
                    if r["signed_effect"]
                }
            ),
        },
    )
