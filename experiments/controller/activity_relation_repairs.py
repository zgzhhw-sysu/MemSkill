"""Recover authenticated activity-arm outcomes on the relation trial's fixed prefix."""

import json
from types import SimpleNamespace

from experiments.controller.capped_source_repair import diagnose
from experiments.controller.factorial_rollout import read
from experiments.controller.legacy_terminal_symmetry import make_writer
from experiments.controller.semantic_writer import measured_readers, sha
from src.credit_assignment import _digest, execution_fault
from src.eval.locomo import LoCoMoEvaluator
from src.replay import restore_text


def activity_rows(cell, cache, semantic_spans, questions, rows, proto):
    (trace,) = (cell / "traces").glob("*.jsonl")
    events = list(map(json.loads, trace.read_bytes().splitlines()))
    spans = [e for e in events if e["kind"] == "span"]
    prefix, end = spans
    first, temporal = semantic_spans
    if (
        any(
            prefix[k] != first[k]
            for k in ("selected_skills", "source", "parsed_actions", "changes")
        )
        or restore_text([prefix]) != restore_text([first])
        or any(
            end[k] != temporal[k]
            for k in ("source", "source_turns", "refs", "retrieved")
        )
        or end["selected_skills"][:-1] != temporal["selected_skills"][:-1]
    ):
        raise ValueError("Different factual state or fixed slots on the third edge")
    activity = end["selected_skills"][-1]
    if (
        activity["name"] != "capture_activity_details"
        or activity["source_rule"] != "dated_adjacent_turns"
        or temporal["selected_skills"][-1]["source_rule"] != "dated_single_turns"
    ):
        raise ValueError("Unexpected two deployed source rules")
    ex = make_writer(proto)
    if any(
        execution_fault(e["parsed_actions"], e["applied"], e["skills"])
        or diagnose(ex, e)
        for e in spans
    ):
        return spans, None
    saved = measured_readers(cell, cache)
    live = list(restore_text(spans).items())
    versions = {c["memory_id"]: c["version"] for e in spans for c in e["changes"]}
    ev = LoCoMoEvaluator(SimpleNamespace())
    out = []
    answers = [e for e in events if e["kind"] == "qa"]
    for i, (q, r, a, t) in enumerate(
        zip(
            questions, rows, read(cell / "scores.json")["answers"], answers, strict=True
        )
    ):
        ix = a["retrieved_indices"]
        ctx = [live[j][1] for j in ix]
        prompt = ev.build_prompt(q["question"], ctx, q)
        z = saved.get(sha(prompt))
        if (
            a["qa_index"] != i
            or r["original_index"] != q["original_index"]
            or t["question"] != q["question"]
            or t["refs"] != [[live[j][0], versions[live[j][0]]] for j in ix]
            or not t["success"]
            or not z
            or z["response"].strip() != a["prediction"]
            or z["tokens"] != a["tokens"]
            or t["prediction"] != a["prediction"]
            or t["f1"] != a["f1"]
        ):
            raise ValueError("Unmatched archived activity reader")
        out.append(
            dict(
                r,
                activity={
                    "answer": a["prediction"],
                    "f1": a["f1"],
                    "ok": True,
                    "indices": ix,
                    "tokens": a["tokens"],
                    "context_hash": _digest(ctx),
                },
            )
        )
    return spans, out
