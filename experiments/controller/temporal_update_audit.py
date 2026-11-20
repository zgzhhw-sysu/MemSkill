"""Separate changes to old memories from new writes and audit strict skill recoveries."""

import argparse
import fcntl
import json
import os
from collections import Counter
from copy import copy
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller.factorial_repairs import final_probe, replay_writes
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.root_write_repairs import seed_controls
from experiments.controller.semantic_writer import measured_readers, sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.temporal_update_writer import verify_pair
from experiments.controller.writer_granularity import measured_answer
from src.credit_assignment import _digest
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings
from src.memory_bank import MemoryBank
from src.rag_utils import get_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "temporal_update_audit.json"
ARMS = ("temporal", "update")
CROSSES = {"old_update": (1, 0), "new_update": (0, 1)}


def setup(forecast):
    source = forecast.parent
    out = source.parent / "temporal-update-audit"
    out.mkdir(exist_ok=True)
    reg = out / "registration.json"
    identity = {"plan": file_hash(PLAN), "forecast": file_hash(forecast)}
    if not reg.is_file():
        register(
            reg,
            dict(
                identity,
                baseline={
                    p.parent.name: file_hash(p)
                    for p in sorted(source.glob("conv-*/pair.json"))
                },
            ),
        )
    saved = read(reg)
    if any(saved[k] != v for k, v in identity.items()) or any(
        file_hash(source / n / "pair.json") != h for n, h in saved["baseline"].items()
    ):
        raise ValueError("Changed audit registration or pre-existing physical endpoint")
    return source, out


def parent_cell(source, w):
    archive = (
        source.parent / "factorial"
        if w["panel"] == "early"
        else source.parents[1] / "late-pairs/outcomes"
    )
    return archive / w["name"] / f"{w['prefix']}0"


def endpoints(source, w):
    folder = parent_cell(source, w)
    (trace,) = (folder / "traces").glob("*.jsonl")
    prefix, last = [
        e
        for line in trace.read_bytes().splitlines()
        if (e := json.loads(line))["kind"] == "span"
    ]
    r = read(source / w["name"] / "pair.json")
    p = source / w["name"] / "writes.json"
    if file_hash(trace) != r["trace"] or file_hash(p) != r["writes"]:
        raise ValueError("Parent endpoint receipts changed")
    writes = read(p)
    events = []
    for key in ("control", "update"):
        event = writes[key]
        if any(event[k] != last[k] for k in ("source", "refs", "retrieved")):
            raise ValueError("Terminal source or pre-write input mismatch")
        events.append(dict(event, source_turns=last["source_turns"]))
    return prefix, events


def bindings(prefix, event):
    spans = [prefix, event]
    live = restore_text(spans)
    versions = {c["memory_id"]: c["version"] for s in spans for c in s["changes"]}
    return [(mid, versions[mid], content) for mid, content in live.items()]


def mixed_banks(banks, parts, prefix_ids, old, new):
    bank = MemoryBank(retriever_name=banks[old].retriever_name, top_k=banks[old].top_k)
    refs = []
    for branch, take_old in ((old, True), (new, False)):
        for i, (mid, version, text) in enumerate(parts[branch]):
            if (mid in prefix_ids) != take_old:
                continue
            item = banks[branch].get_memory_at(i)
            if item.content != text or item.embedding is None:
                raise ValueError(
                    "Replayed item differs from the physical memory or lacks its retrieval vector"
                )
            bank.memories.append(copy(item))
            refs.append([ARMS[branch], mid, version])
    return bank, refs


def contribution(factual, old_update, new_update, treated):
    old = ((old_update - factual) + (treated - new_update)) / 2
    new = ((new_update - factual) + (treated - old_update)) / 2
    return {
        "old": old,
        "new": new,
        "interaction": treated - old_update - new_update + factual,
        "total": treated - factual,
    }


def check_endpoint(evaluator, q, observed, branch, parts, context, indices, readers):
    arm = ARMS[branch]
    prompt = evaluator.build_prompt(q["question"], context, q)
    same = observed["temporal"]["context_hash"] == observed["update"]["context_hash"]
    payload = readers[0 if same else branch][sha(prompt)]
    prediction = payload["response"].strip()
    actual = {
        "answer": prediction,
        "ok": True,
        "tokens": payload["tokens"],
        "context_hash": _digest(context),
        "f1": evaluator.compute_f1(prediction, evaluator.get_ground_truth(q), q),
        "indices": indices,
        "refs": [list(parts[j][:2]) for j in indices],
    }
    if actual != observed[arm]:
        raise ValueError(
            f"Uncrossed {arm} endpoint failed to reproduce for question {observed['index']}"
        )


def prepare_window(trainer, source, out, w, full, dest):
    name = w["name"]
    parent = source / name
    pair = read(parent / "pair.json")
    packet = {
        "name": name,
        "panel": w["panel"],
        "dialogue": w["dialogue"],
        "questions": w["questions"],
        "pair": file_hash(parent / "pair.json"),
        "registration": file_hash(out / "registration.json"),
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
        "parent_status": pair["status"],
    }
    if (
        pair["questions"] != w["questions"]
        or pair["dialogue"] != w["dialogue"]
        or pair["panel"] != w["panel"]
    ):
        raise ValueError("Parent pair is outside the registered window")
    if pair["status"] != "matched":
        return packet
    cell = parent_cell(source, w)
    verify_pair(source, pair, {w["dialogue"]: full}, trainer.evaluator, cell)
    prefix, events = endpoints(source, w)
    candidates = read(parent / "recovery-candidates.json")
    if candidates["pair"] != packet["pair"] or candidates["writes"] != pair["writes"]:
        raise ValueError("Different candidate ledger")
    convo = window(full, w["start"], 2)
    if [q["original_index"] for q in convo["qa"]] != w["questions"]:
        raise ValueError("Window questions changed")
    cache = out / "responses"
    archived_readers = []
    for directory, archive in (
        (cell, cell.parents[1] / "responses"),
        (parent, source / "responses"),
    ):
        archived_readers.append(measured_readers(directory, archive))
        if candidates["candidates"]:
            seed_controls(directory, archive, cache)
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    os.environ["MEMSKILL_USAGE_LOG"] = str(dest / "usage.jsonl")
    prefix_ids = set(restore_text([prefix]))
    parts = [bindings(prefix, e) for e in events]
    banks = [replay_writes(trainer, [prefix, e]) for e in events]
    mixes = {k: mixed_banks(banks, parts, prefix_ids, *v) for k, v in CROSSES.items()}
    vectors = get_embeddings(
        trainer.args.retriever, [q["question"] for q in convo["qa"]], "query"
    )
    strict = {r["index"]: r for r in candidates["candidates"]}
    cells = {
        (w["prefix"], a): {
            "spans": [prefix, events[a]],
            "scores": {
                i: {k: v for k, v in r[arm].items() if k != "refs"}
                for i, r in enumerate(pair["rows"])
            },
        }
        for a, arm in enumerate(ARMS)
    }
    probe_banks = {(w["prefix"], a): b for a, b in enumerate(banks)}
    rows = []
    for i, (q, observed, vec) in enumerate(
        zip(convo["qa"], pair["rows"], vectors, strict=True)
    ):
        idx = q["original_index"]
        if observed["index"] != idx:
            raise ValueError("Unexpected parent question order")
        for a, bank in enumerate(banks):
            ctx, ix = bank.retrieve(vec, top_k=trainer.config.mem_top_k_eval)
            check_endpoint(
                trainer.evaluator, q, observed, a, parts[a], ctx, ix, archived_readers
            )
        mixed = {}
        for k, (bank, refs) in mixes.items():
            ctx, ix = bank.retrieve(vec, top_k=trainer.config.mem_top_k_eval)
            mixed[k] = {"refs": [refs[j] for j in ix], "context_hash": _digest(ctx)}
        probe = None
        if idx in strict:
            c = strict[idx]
            if c["status"] == "execution_fault":
                probe = {"status": c["status"], "signed_effect": 0.0}
            else:
                args = {
                    "qa_index": i,
                    "background": w["prefix"],
                    "delta": c["delta"],
                    "recovery_sign": 1 if c["recovered_by"] == "update" else -1,
                }
                probe = final_probe(
                    trainer,
                    convo,
                    args,
                    cells,
                    probe_banks,
                    vec,
                    probe_if_write_unavailable=True,
                )
                if probe["signed_effect"] != (
                    c["delta"] if probe["status"] == "survived_probes" else 0.0
                ):
                    raise ValueError("Invalid signed selection credit")
        rows.append(
            {"index": idx, "type": observed["type"], "mixed": mixed, "probe": probe}
        )
    touched = {
        c["memory_id"]
        for e in events
        for c in e["changes"]
        if c["memory_id"] in prefix_ids
    }
    if not touched:
        for row, observed in zip(rows, pair["rows"], strict=True):
            for k, (_, new) in CROSSES.items():
                if (
                    row["mixed"][k]["context_hash"]
                    != observed[ARMS[new]]["context_hash"]
                ):
                    raise ValueError(
                        "An untouched old partition produced a different context"
                    )
    return dict(
        packet, rows=rows, touched_old=sorted(touched), model=trainer.args.model
    )


def prepare(trainer, study, forecast, train, specs):
    source, out = setup(forecast)
    locked = read(forecast)
    backend = read(source / "backend.json")
    if (
        locked["study"] != file_hash(study / "protocol.json")
        or backend["forecast"] != file_hash(forecast)
        or backend["skills"] != [s["version"] for s in specs]
        or backend["backend"] != os.environ["MEMSKILL_BACKEND_ID"]
    ):
        raise ValueError("Different original encoder, checkpoint or skill bank")
    data = {c["sample_id"]: c for c in train}
    with (out / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for w in locked["windows"]:
            dest = out / w["name"]
            if (dest / "inputs.json").exists() or not (
                source / w["name"] / "pair.json"
            ).exists():
                continue
            dest.mkdir(exist_ok=True)
            with history_embeddings():
                packet = prepare_window(
                    trainer, source, out, w, data[w["dialogue"]], dest
                )
            register(dest / "inputs.json", packet)
            print(w["name"], packet["parent_status"], flush=True)
            break


def score_window(source, out, w, data, ev):
    name = w["name"]
    path = out / name / "inputs.json"
    packet = read(path)
    parent = source / name / "pair.json"
    if (
        packet["pair"] != file_hash(parent)
        or packet["backend"] != os.environ["MEMSKILL_BACKEND_ID"]
        or packet["registration"] != file_hash(out / "registration.json")
        or packet["questions"] != w["questions"]
    ):
        raise ValueError("Prepared audit inputs differ from the registered trial")
    result = {
        "inputs": file_hash(path),
        "status": packet["parent_status"],
        "panel": w["panel"],
        "name": name,
        "dialogue": w["dialogue"],
        "questions": w["questions"],
    }
    if result["status"] != "matched":
        return result
    pair = read(parent)
    prefix, events = endpoints(source, w)
    items = {
        (a, mid, v): t
        for a, e in zip(ARMS, events, strict=True)
        for mid, v, t in bindings(prefix, e)
    }
    os.environ["MEMSKILL_USAGE_LOG"] = str(path.parent / "reader-usage.jsonl")
    rows = []
    for r, obs in zip(packet["rows"], pair["rows"], strict=True):
        idx = r["index"]
        if idx != obs["index"]:
            raise ValueError("Prepared question order changed")
        q = data[w["dialogue"]]["qa"][idx]
        answers = {}
        for k, info in r["mixed"].items():
            ctx = [items[tuple(ref)] for ref in info["refs"]]
            if _digest(ctx) != info["context_hash"]:
                raise ValueError(
                    "Hybrid memory versions do not reproduce frozen reader input"
                )
            known = next(
                (a for a in ARMS if obs[a]["context_hash"] == info["context_hash"]),
                None,
            )
            if known:
                answers[k] = {
                    field: obs[known][field] for field in ("answer", "f1", "tokens")
                }
            else:
                prompt = ev.build_prompt(q["question"], ctx, q)
                answers[k] = measured_answer(
                    prompt, packet["model"], q, ev, out / "responses"
                )
                if min(answers[k]["tokens"]) <= 0:
                    raise RuntimeError("No measured hybrid reader response")
            answers[k]["same_endpoint"] = known
        effects = contribution(
            obs["temporal"]["f1"],
            answers["old_update"]["f1"],
            answers["new_update"]["f1"],
            obs["update"]["f1"],
        )
        rows.append(
            {
                "index": idx,
                "type": r["type"],
                "answers": answers,
                "effects": effects,
                "probe": r["probe"],
            }
        )
    return dict(
        result, touched_old=packet["touched_old"], rows=rows, choice=pair["choice"]
    )


def summarize(records):
    good = [r for r in records if r["status"] == "matched"]
    rows = [(r, q) for r in good for q in r["rows"]]
    result = {
        "windows": len(records),
        "registered_questions_arrived": sum(len(r["questions"]) for r in records),
        "matched_questions": len(rows),
        "statuses": dict(Counter(r["status"] for r in records)),
    }
    if rows:
        result.update(
            old_touched_windows=sum(bool(r["touched_old"]) for r in good),
            mean_effect={
                k: mean(q["effects"][k] for _, q in rows)
                for k in ("old", "new", "interaction", "total")
            },
            source_minus_native={
                k: mean(
                    (r["choice"]["source"] - r["choice"]["native"]) * q["effects"][k]
                    for r, q in rows
                )
                for k in ("old", "new", "total")
            },
            credit_statuses=dict(
                Counter(q["probe"]["status"] for _, q in rows if q["probe"])
            ),
            positive=sum(
                bool(q["probe"]) and q["probe"]["signed_effect"] > 0 for _, q in rows
            ),
            negative=sum(
                bool(q["probe"]) and q["probe"]["signed_effect"] < 0 for _, q in rows
            ),
        )
    return result


def finish(forecast, study, corpus):
    source, out = setup(forecast)
    locked = read(forecast)
    protocol = read(study / "protocol.json")
    if (
        locked["study"] != file_hash(study / "protocol.json")
        or file_hash(corpus) != protocol["input_sha256"]
    ):
        raise ValueError("Wrong split or benchmark")
    data = {
        d["sample_id"]: d for d in read(corpus) if d["sample_id"] in protocol["train"]
    }
    ev = LoCoMoEvaluator(SimpleNamespace())
    with (out / "reader.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = []
        for w in locked["windows"]:
            p = out / w["name"] / "result.json"
            if not (p.parent / "inputs.json").exists():
                continue
            if not p.exists():
                register(p, score_window(source, out, w, data, ev))
            r = read(p)
            if r["inputs"] != file_hash(p.parent / "inputs.json"):
                raise ValueError("Completed result refers to different inputs")
            results.append(r)
        if results:
            panel = {
                k: summarize([r for r in results if r["panel"] == k])
                for k in ("early", "later")
            }
            report = {
                "panels": panel,
                "files": {
                    r["name"]: file_hash(out / r["name"] / "result.json")
                    for r in results
                },
            }
            register(
                out / f"through-{len(results):02d}-{_digest(report)[:12]}.json", report
            )
            if len(results) == len(locked["windows"]):
                register(out / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("forecast", "study", "corpus"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    finish(args.forecast, args.study, args.corpus)
