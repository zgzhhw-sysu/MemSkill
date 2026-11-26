"""Measure whole-window terminal action effects in the earliest six training histories."""

import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.capped_source_repair import diagnose, writer
from experiments.controller.exchange_paths import Ranking
from experiments.controller.exchange_questions import call
from experiments.controller.legacy_terminal_symmetry import make_bank, make_writer, ops
from experiments.controller.resume_status import check_equal_source_choice
from experiments.controller.semantic_writer import sha
from src.credit_assignment import _digest, execution_fault
from src.episode_trace import skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "earlier_terminal_returns.json"
PHASES = ("writers", "prepared", "outcomes")


def replay_writer(out, spans, diag, row, proto):
    _prefix, event = spans
    ex = make_writer(proto)
    for e in spans:
        if diagnose(ex, e) or execution_fault(
            e["parsed_actions"], e["applied"], e["skills"]
        ):
            return {"status": "factual_write_fault", "span": e["span_id"]}
    alt = [proto["skills"][i] for i in row["plain" if row["use"] else "grounded"]]
    check_equal_source_choice(row, proto["skills"], event["selected_skills"], alt)
    old = diag.get("swap")
    if diag.get("alternative") not in (None, alt):
        raise ValueError("Different historical intervention")
    if old and old["contract_fault"]:
        return {"status": "archived_alternative_fault"}
    c = diag.get("control")
    if c and (c["actions"] != event["parsed_actions"] or c["contract_fault"]):
        return {"status": "archived_control_fault"}
    if not old and diag["status"] != "no_linked_training_question":
        return {
            "status": "unavailable_historical_alternative",
            "reason": diag["status"],
        }
    selected = ops(alt)
    prompt = ex._build_executor_prompt(selected, event["source"], event["retrieved"])
    if old and old["writer_prompt_hashes"] != [sha(prompt)]:
        return {"status": "changed_alternative_prompt"}
    resp = call(out, prompt, "executor", proto["writer_tokens"])
    receipt = {
        "prompt_sha256": sha(prompt),
        "response": resp["text"],
        "input_tokens": resp["tokens"][0],
        "output_tokens": resp["tokens"][1],
    }
    e = dict(event, executor_responses=[receipt])
    acts = writer(ex, selected, e, repair=False)
    if old and acts != old["actions"]:
        return {"status": "changed_alternative_actions", "response": resp}
    deployed = [skill_spec(s) for s in selected]
    if execution_fault(acts, True, deployed) or acts != writer(
        ex, selected, e, repair=True
    ):
        return {"status": "alternative_write_fault", "response": resp}
    return {
        "status": "matched_writer",
        "response": resp,
        "actions": acts,
        "writer_skills": deployed,
        "previously_probed": bool(old),
    }


def prepare_window(rank, spans, diag, trace_qa, questions, proto, measured):
    prefix, event = spans
    if len(trace_qa) != len(questions):
        raise ValueError("Incomplete original question list")
    vectors = rank.encode(tuple(q["question"] for q in questions))
    native = list(restore_text(spans).items())
    versions = {c["memory_id"]: c["version"] for span in spans for c in span["changes"]}
    refs = [[mid, versions[mid]] for mid, _ in native]
    branches = [
        {"actions": event["parsed_actions"], "writer_skills": event["skills"]},
        measured,
    ]
    ev = LoCoMoEvaluator(SimpleNamespace())
    packets = [{} for _ in questions]
    banks = []
    for key, branch, history in zip(
        ("native", "alternative"),
        branches,
        (diag.get("control"), diag.get("swap")),
        strict=True,
    ):
        bank = make_bank(rank, make_writer(proto), prefix, event, branch, [])
        contents = bank.get_all_contents()
        bank_hash = _digest(contents)
        banks.append(bank_hash)
        if (key == "native" and contents != [t for _, t in native]) or (
            history and bank_hash != history["bank_hash"]
        ):
            return {"status": "changed_final_bank", "arm": key}
        embs = [m.embedding for m in bank.memories]
        for i, (q, v, logged) in enumerate(
            zip(questions, vectors, trace_qa, strict=True)
        ):
            ix, ctx = rank.rank(v, contents, embs)
            orig = (history or {}).get("qa", {}).get(str(i))
            if orig and (orig["indices"] != ix or orig["context_hash"] != _digest(ctx)):
                return {
                    "status": "changed_archived_retrieval",
                    "arm": key,
                    "original_index": q["original_index"],
                }
            if key == "native":
                if (
                    logged["question"] != q["question"]
                    or not logged["success"]
                    or int(logged["query_id"].rsplit("_", 1)[1]) != i
                ):
                    raise ValueError("Changed original QA identity")
                if [refs[j] for j in ix] != logged["refs"]:
                    return {
                        "status": "changed_native_order",
                        "original_index": q["original_index"],
                    }
                orig = {
                    "answer": logged["prediction"],
                    "f1": logged["f1"],
                    "tokens": logged["tokens"],
                }
                packets[i].update(
                    qa=q,
                    linked=bool(set(q["evidence"]).intersection(event["source_turns"])),
                )
            prompt = ev.build_prompt(q["question"], ctx, q)
            packets[i][key] = {
                "prompt": prompt,
                "hash": sha(prompt),
                "indices": ix,
                "old": orig,
            }
    return {"status": "prepared", "banks": banks, "questions": packets}


def read_window(out, prepared, use):
    ev = LoCoMoEvaluator(SimpleNamespace())
    signed = 1 if use else -1
    rows = []
    for q in prepared["questions"]:
        qa = q["qa"]
        truth = ev.get_ground_truth(qa)
        arms = {}
        for k in ("native", "alternative"):
            spec = q[k]
            if sha(spec["prompt"]) != spec["hash"]:
                raise ValueError("Prepared reader prompt changed")
            response = call(out, spec["prompt"], "reader", 32)
            ans = response["text"].strip()
            score = ev.compute_f1(ans, truth, qa)
            old = spec["old"]
            arms[k] = {"answer": ans, "f1": score, "receipt": response}
            if old and (ans != old["answer"] or abs(score - old["f1"]) > 1e-12):
                return {
                    "status": "changed_reader_answer",
                    "arm": k,
                    "original_index": qa["original_index"],
                    "archived": old,
                    "observed": arms[k],
                    "completed": rows,
                }
        rows.append(
            {
                "original_index": qa["original_index"],
                "category": qa["category"],
                "linked": q["linked"],
                "same_prompt": q["native"]["hash"] == q["alternative"]["hash"],
                "arms": arms,
                "delta": signed * (arms["native"]["f1"] - arms["alternative"]["f1"]),
            }
        )
    return {
        "status": "matched",
        "rows": rows,
        "mean_activity_minus_temporal": mean(r["delta"] for r in rows),
    }


def run(base, study, corpus, phase):
    src = base / "source-compression"
    out = src / "earlier-terminal-returns"
    dest = out / phase
    dest.mkdir(parents=True, exist_ok=True)
    old = src / "legacy-terminal-symmetry/registration.json"
    original = read(old)
    proto = read(study / "protocol.json")
    if (
        original["corpus"] != file_hash(corpus)
        or original["protocol"] != file_hash(study / "protocol.json")
        or set(proto["train"]) & set(proto["validation"] + proto["test"])
    ):
        raise ValueError("Changed earlier cohort")
    train = {
        c["sample_id"]: c for c in read(corpus) if c["sample_id"] in proto["train"]
    }
    windows = {}
    for name, h in original["windows"].items():
        ep = study / "train" / name
        f = ep / "questions.json"
        if file_hash(f) != h["questions"] or name.rsplit("_s", 1)[0] not in train:
            raise ValueError("Changed earlier questions")
        windows[name] = read(f)
    if len(windows) != 12 or sum(map(len, windows.values())) != 123:
        raise ValueError("Incomplete earlier cohort")
    sequence = sorted(windows, key=lambda w: (len(windows[w]), w))
    reg = out / "registration.json"
    register(
        reg,
        {
            "plan": file_hash(PLAN),
            "archive": file_hash(old),
            "windows": windows,
            "order": sequence,
        },
    )
    rank = None
    hashes = {}
    for name in sequence:
        f = dest / (name + ".json")
        parent = (
            reg
            if phase == "writers"
            else out / PHASES[PHASES.index(phase) - 1] / f.name
        )
        if not parent.exists():
            continue
        inputs = {"registration": file_hash(reg), "parent": file_hash(parent)}
        if not f.exists():
            ep = study / "train" / name
            h = original["windows"][name]
            (trace,) = (ep / "traces").glob("*.jsonl")
            if (
                file_hash(trace) != h["trace"]
                or file_hash(ep / "rows.json") != h["rows"]
            ):
                raise ValueError("Historical decisions changed")
            es = list(map(json.loads, trace.read_bytes().splitlines()))
            spans = [e for e in es if e["kind"] == "span"]
            diag = [e["diagnosis"] for e in es if e["kind"] == "span_credit"][1]
            row = read(ep / "rows.json")[1]
            prior = read(parent)
            if phase == "writers":
                result = replay_writer(out, spans, diag, row, proto)
            elif prior["status"] != (
                "matched_writer" if phase == "prepared" else "prepared"
            ):
                result = {"status": prior["status"]}
            elif phase == "prepared":
                if rank is None:
                    rank = Ranking(proto["retriever"])
                sid = name.rsplit("_s", 1)[0]
                qa = [
                    dict(train[sid]["qa"][i], original_index=i) for i in windows[name]
                ]
                with history_embeddings():
                    result = prepare_window(
                        rank,
                        spans,
                        diag,
                        [e for e in es if e["kind"] == "qa"],
                        qa,
                        proto,
                        prior,
                    )
            else:
                result = read_window(out, prior, row["use"])
            register(f, dict(result, inputs=inputs))
            print(name, phase, result["status"], flush=True)
        elif read(f)["inputs"] != inputs:
            raise ValueError("Changed stage input")
        hashes[name] = file_hash(f)
    if len(hashes) == len(sequence):
        register(out / (phase + "-complete.json"), hashes)
    if phase == "outcomes" and hashes:
        completed = [(name, read(dest / (name + ".json"))) for name in hashes]
        matched = [(n, r) for n, r in completed if r["status"] == "matched"]
        rows = [r for _, w in matched for r in w["rows"]]
        register(
            out / f"through-{len(hashes):02d}.json",
            {
                "sources": hashes,
                "registered_windows": 12,
                "registered_questions": 123,
                "attempted": len(hashes),
                "matched_windows": len(matched),
                "matched_questions": len(rows),
                "excluded": {
                    n: r["status"] for n, r in completed if r["status"] != "matched"
                },
                "by_evidence": {
                    group: {
                        "questions": len(v),
                        "mean_effect": mean(r["delta"] for r in v) if v else None,
                    }
                    for group, v in (
                        ("linked", [r for r in rows if r["linked"]]),
                        ("other", [r for r in rows if not r["linked"]]),
                    )
                },
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("phase", choices=PHASES)
    a = p.parse_args()
    run(a.base, a.study, a.corpus, a.phase)
