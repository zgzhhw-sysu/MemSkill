"""Test competing same-action explanations for localized relation-writer effects."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.activity_relation_repairs import activity_rows
from experiments.controller.capped_source_repair import diagnose
from experiments.controller.exchange_paths import Ranking
from experiments.controller.factorial_rollout import read
from experiments.controller.legacy_terminal_symmetry import (
    make_writer,
    measure,
    ops,
    prepare,
)
from experiments.controller.paired_grounding import window
from experiments.controller.relation_prefix_interaction import PREFIX_NAMES
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import _digest, execution_fault, recovered
from src.episode_trace import skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings, read_response
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "relation_skill_repairs.json"


def candidates(questions, rows, source_turns, baseline="temporal"):
    ev = LoCoMoEvaluator(SimpleNamespace())
    jobs = []
    linked = 0
    for i, (q, r) in enumerate(zip(questions, rows, strict=True)):
        if q["original_index"] != r["original_index"] or q["category"] != r["category"]:
            raise ValueError("Different QA order or category")
        if not set(q["evidence"]).intersection(source_turns):
            continue
        linked += 1
        t, rel = r[baseline], r["relation"]
        truth = ev.get_ground_truth(q)
        if recovered(t, rel, truth) or recovered(rel, t, truth):
            jobs.append(
                {
                    "qa": q,
                    "question_index": i,
                    "weak": "control" if t["f1"] < rel["f1"] else "swap",
                    "signed_credit": rel["f1"] - t["f1"],
                }
            )
    return linked, jobs


def check_response(receipts, folder, *, cache=None):
    pending = {
        (r["prompt_sha256"], r["response"], r["input_tokens"], r["output_tokens"])
        for r in receipts
    }
    for row in map(json.loads, (folder / "usage.jsonl").read_bytes().splitlines()):
        if row["kind"] != "executor":
            continue
        result = read_response(
            cache if cache is not None else folder / "responses", row["cache_key"]
        )
        if not result or result["sha256"] != row["response_sha256"]:
            raise ValueError("Changed stored relation write")
        p = result["payload"]
        pending.discard((p["prompt_sha256"], p["response"], *p["tokens"]))
        if not pending:
            return
    raise ValueError("Missing archived relation writer response")


def forecast_window(
    parent, archive, data, sid, start, bit, protocol, ranker, *, baseline="temporal"
):
    folder = parent / f"{sid}_s{start}"
    pair = read(folder / "pair.json")
    qa = window(data, start, 2)["qa"]
    if pair["dialogue"] != sid or pair["questions"] != len(qa):
        raise ValueError("Changed training population")
    (trace,) = (archive / folder.name / f"{bit}0" / "traces").glob("*.jsonl")
    if pair["source_trace"] != file_hash(trace):
        raise ValueError("Changed control trace")
    spans = [
        v
        for v in map(json.loads, trace.read_bytes().splitlines())
        if v["kind"] == "span"
    ]
    prefix, end = spans
    if prefix["selected_skills"][-1]["name"] != PREFIX_NAMES[bit]:
        raise ValueError("Changed first action")
    base = {
        "matched": pair["matched"],
        "questions": len(qa),
        "linked": sum(
            bool(set(q["evidence"]).intersection(end["source_turns"])) for q in qa
        ),
        "jobs": [],
    }
    if not pair["matched"]:
        return dict(base, reason=pair["reason"])
    semantic_end = end
    rows = pair["rows"]
    if baseline == "activity":
        cell = archive / folder.name / f"{bit}1"
        spans, rows = activity_rows(
            cell, archive / "responses", spans, qa, rows, protocol
        )
        prefix, end = spans
        if rows is None:
            return dict(base, matched=False, reason="activity_write_fault")
        base["raw_pairs"] = [
            {
                "original_index": q["original_index"],
                "relation": q["relation"]["f1"],
                "activity": q["activity"]["f1"],
            }
            for q in rows
        ]
    linked, jobs = candidates(qa, rows, end["source_turns"], baseline)
    if linked != base["linked"]:
        raise ValueError("Changed focal support")
    if not jobs and baseline == "temporal":
        return base
    choice = read(folder / "choice.json")
    write = read(folder / "write.json")
    if (
        choice != pair["choice"]
        or choice["control_trace"] != file_hash(trace)
        or choice["fixed_prefix"] != _digest(list(restore_text([prefix]).items()))
        or file_hash(folder / "choice.json") != write["choice"]
    ):
        raise ValueError("Unbound relation response")
    check_response(write["writer"], parent)
    relation = choice["selected"]
    if (
        relation[:-1] != end["selected_skills"][:-1]
        or relation[-1]["name"] != "handle_entity_relationships"
        or semantic_end["selected_skills"][-1]["name"] != "capture_temporal_context"
        or semantic_end["selected_skills"][-1]["source_rule"]
        != relation[-1]["source_rule"]
    ):
        raise ValueError("Different terminal intervention")
    skill = [skill_spec(op) for op in ops(relation)]
    alt = dict(
        end,
        selected_skills=relation,
        skills=skill,
        parsed_actions=write["actions"],
        applied=write["applied"],
        executor_responses=write["writer"],
    )
    ex = make_writer(protocol)
    fault = any(
        diagnose(ex, e)
        or execution_fault(e["parsed_actions"], e["applied"], e["skills"])
        for e in (*spans, alt)
    )
    if baseline == "activity" and fault:
        base.pop("raw_pairs")
        return dict(base, matched=False, reason="direct_edge_write_fault")
    if not jobs:
        return base
    d = {
        "questions": [
            {"qa_index": i, "original_index": q["original_index"]}
            for i, q in enumerate(qa)
        ]
    }
    for arm, e, bank, key in (
        ("control", end, list(restore_text(spans).values()), baseline),
        ("swap", alt, write["contents"], "relation"),
    ):
        d[arm] = {
            "bank_hash": _digest(bank),
            "actions": e["parsed_actions"],
            "writer_skills": e["skills"],
            "qa": {str(i): r[key] for i, r in enumerate(rows)},
        }
    for job in jobs:
        result = (
            {"status": "write_fault"}
            if fault
            else prepare(job, ranker(), spans, d, protocol, data, strict_indices=True)
        )
        base["jobs"].append({"job": job, "result": result})
    return base


def run(base, study, corpus, bit, phase, *, baseline="temporal"):
    semantic = base / "source-compression/semantic-writer"
    parent = semantic / "activity-prefix" if bit else semantic
    archive = semantic.parent / "factorial"
    protocol = read(study / "protocol.json")
    manifest = read(archive / "manifest.json")
    if (
        file_hash(corpus) != protocol["input_sha256"]
        or manifest["protocol"] != file_hash(study / "protocol.json")
        or set(protocol["train"]) & set(protocol["validation"] + protocol["test"])
        or protocol["rule"] != 2
        or {sid for sid, _ in manifest["windows"]} != set(protocol["train"])
    ):
        raise ValueError("Different corpus, split or evidence rule")
    data = {
        r["sample_id"]: r for r in read(corpus) if r["sample_id"] in protocol["train"]
    }
    direct = baseline == "activity"
    out = (
        semantic.parent
        / ("activity-relation-repairs" if direct else "relation-skill-repairs")
        / str(bit)
    )
    dest = out / phase
    dest.mkdir(exist_ok=True, parents=True)
    frozen = parent / ("registration.json" if bit else "summary.json")
    registration = {
        "plan": file_hash(
            PLAN.with_name("activity_relation_repairs.json") if direct else PLAN
        ),
        "manifest": file_hash(archive / "manifest.json"),
        "parent": file_hash(frozen),
        "protocol": file_hash(study / "protocol.json"),
        "prefix": bit,
    }
    reg = out / "registration.json"
    register(reg, registration)
    rank = None

    def ranker():
        nonlocal rank
        if rank is None:
            rank = Ranking(protocol["retriever"])
        return rank

    panel = {}
    for sid, n in manifest["windows"]:
        key = f"{sid}_s{n}"
        f = dest / (key + ".json")
        before = (
            parent / key / "pair.json"
            if phase == "prepared"
            else out / "prepared" / f.name
        )
        if not before.exists():
            continue
        inputs = {"registration": file_hash(reg), "parent": file_hash(before)}
        if direct and phase == "prepared":
            cell = archive / key / f"{bit}1"
            (trace,) = (cell / "traces").glob("*.jsonl")
            inputs.update(
                activity_trace=file_hash(trace),
                activity_scores=file_hash(cell / "scores.json"),
                activity_usage=file_hash(cell / "usage.jsonl"),
            )
        if f.exists():
            if read(f)["inputs"] != inputs:
                raise ValueError("Changed upstream window")
        elif phase == "prepared":
            with history_embeddings():
                result = forecast_window(
                    parent,
                    archive,
                    data[sid],
                    sid,
                    n,
                    bit,
                    protocol,
                    ranker,
                    baseline=baseline,
                )
            register(f, dict(result, inputs=inputs))
        else:
            p = read(before)
            jobs = []
            for item in p["jobs"]:
                j, a = item["job"], item["result"]
                observed = (
                    measure(out, j, a)
                    if a["status"] == "prepared"
                    else {"status": a["status"], "credit": 0.0}
                )
                jobs.append(
                    {
                        "original_index": j["qa"]["original_index"],
                        "weak": j["weak"],
                        "raw": j["signed_credit"],
                        **observed,
                    }
                )
            register(
                f,
                {
                    **{k: v for k, v in p.items() if k not in ("jobs", "inputs")},
                    "jobs": jobs,
                    "inputs": inputs,
                },
            )
        panel[key] = file_hash(f)
    if len(panel) == len(manifest["windows"]):
        register(out / (phase + "-complete.json"), panel)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "study", "corpus"):
        p.add_argument(key, type=Path)
    p.add_argument("bit", type=int, choices=(0, 1))
    p.add_argument("phase", choices=("prepared", "outcomes"))
    p.add_argument("--baseline", choices=("temporal", "activity"), default="temporal")
    a = p.parse_args()
    run(a.base, a.study, a.corpus, a.bit, a.phase, baseline=a.baseline)
