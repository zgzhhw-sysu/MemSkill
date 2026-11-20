"""Measure a real terminal UPDATE against the original temporal INSERT on the same prefix."""

import fcntl
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import numpy as np

from experiments.controller.factorial_repairs import checked_cell, replay_writes
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.root_write_repairs import seed_controls
from experiments.controller.semantic_writer import (
    TEMP,
    measured_readers,
    probabilities,
    sha,
)
from experiments.controller.source_compression import file_hash, register
from experiments.controller.temporal_update_forecast import PLAN, UPDATE
from experiments.controller.writer_granularity import measured_answer
from src.credit_assignment import _digest, execution_fault, recovered
from src.episode_trace import EpisodeTrace, skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest, history_embeddings
from src.executor import deployment_operations
from src.rag_utils import get_embeddings
from src.replay import restore_text


def choices(original, upgrade):
    if (
        len(original) != 3
        or original[-1]["name"] != TEMP
        or original[-1].get("source_rule") != "dated_single_turns"
    ):
        raise ValueError("Unexpected archived temporal control")
    fixed = original[:2]
    if sum(
        s["name"] == "insert" and s["update_type"].lower() == "insert" for s in fixed
    ) != 1 or any(s["name"] in (TEMP, UPDATE) or s.get("source_rule") for s in fixed):
        return None
    common = [
        SimpleNamespace(
            **(
                dict(s, source_rule="dated_single_turns")
                if s["name"] == "insert"
                else s
            )
        )
        for s in fixed
    ]
    control = {k: v for k, v in original[-1].items() if k != "source_rule"}
    if (
        upgrade["name"] != UPDATE
        or upgrade["update_type"].lower() != "update"
        or upgrade.get("source_rule")
    ):
        raise ValueError("Unexpected native time-refinement skill")
    return [*common, SimpleNamespace(**control)], [*common, SimpleNamespace(**upgrade)]


def replay_provider(responses):
    lookup = {r["prompt_sha256"]: r for r in responses}

    def provide(*, prompt, MAX_TOKENS, **kwargs):
        r = lookup[sha(prompt)]
        if "output_cap" in r and r["output_cap"] != MAX_TOKENS:
            raise ValueError("Archived output budget changed")
        return r["response"], r["input_tokens"], r["output_tokens"]

    return provide


def execute(trainer, bank, prefix, event, idx, selected, out, provider=None):
    source, ctx = event["source"], event["retrieved"]
    trace = EpisodeTrace(out, event["sample"])
    trace.ids = list(restore_text([prefix]))
    trace.versions = {r["memory_id"]: r["version"] for r in prefix["changes"]}
    trace.next_id = max(trace.versions, default=-1) + 1
    ops = deployment_operations(selected)
    trace.begin(1, source, ops, ctx, idx, selected=selected)
    if trace.current["refs"] != event["refs"]:
        raise ValueError("Different version of a retrieved memory")
    bank.trace = trace
    audit = []
    actions = trainer.executor.execute_operation(
        ops, source, ctx, audit=audit, response_provider=provider
    )
    if any("API call failed" in a.reasoning for a in actions):
        if provider is not None:
            return None
        raise RuntimeError("Update writer request failed")
    applied = trainer.executor.apply_to_memory_bank(
        actions, bank, idx, operation_name=[o.name for o in selected]
    )
    bank.step()
    record = dict(
        trace.current,
        executor_responses=audit,
        parsed_actions=[vars(a).copy() for a in actions],
        applied=bool(applied),
    )
    if bank.get_all_contents() != list(restore_text([prefix, record]).values()):
        raise ValueError(
            "Changed version trace disagrees with measured memory contents"
        )
    return record


def measure(trainer, convo, root, folder, specs, w, out):
    original = checked_cell(
        trainer, convo, folder, (w["prefix"], 0), specs, root / "responses"
    )
    prefix, event = original["spans"]
    if [digest(s["source"]) for s in (prefix, event)] != w["sources"]:
        raise ValueError("Actor source differs from actual writes")
    receipt = {
        "name": w["name"],
        "trace": original["trace"],
        "panel": w["panel"],
        "dialogue": w["dialogue"],
        "questions": w["questions"],
    }
    by_name = {s["name"]: s for s in specs}
    pair = choices(event["selected_skills"], by_name[UPDATE])
    if pair is None:
        return dict(receipt, status="ineligible_fixed_slots")
    low, high = pair
    bank = replay_writes(trainer, [prefix])
    session = trainer.state_encoder._encode_texts(event["source"]).reshape(-1)
    ctx, idx, mem = bank.retrieve(
        session, use_state_encoder=True, return_embeddings=True
    )
    ids = list(restore_text([prefix]))
    if ctx != event["retrieved"] or [ids[j] for j in idx] != [
        mid for mid, _ in event["refs"]
    ]:
        return dict(receipt, status="unmatched_retrieval")
    state = trainer.state_encoder.encode(
        event["source"], ctx, session_embedding=session, memory_embeddings=mem
    )
    ps = probabilities(
        trainer,
        state,
        {s["name"]: i for i, s in enumerate(specs)},
        w["offset"],
        alternative=UPDATE,
    )
    choice = dict(
        ps, source=w["source_p"], sham=w["sham_p"], fixed_update=1.0, fixed_temporal=0.0
    )
    register(
        out / "choice.json",
        {
            "probabilities": choice,
            "selected": [[skill_spec(s) for s in a] for a in pair],
            "prefix": digest(list(restore_text([prefix]).items())),
            "state": sha(np.asarray(state).tobytes().hex()),
            "sources": w["sources"],
            "control": original["trace"],
        },
    )
    control = execute(
        trainer,
        bank,
        prefix,
        event,
        idx,
        low,
        out,
        replay_provider(event["executor_responses"]),
    )
    if control is None or any(
        control[k] != event[k]
        for k in ("parsed_actions", "changes", "applied", "executor_responses")
    ):
        return dict(receipt, status="unmatched_temporal_write")
    queries = get_embeddings(
        trainer.args.retriever, [q["question"] for q in convo["qa"]], "query"
    )
    readers = measured_readers(folder, root / "responses")
    baseline = []
    for i, (q, vec) in enumerate(zip(convo["qa"], queries, strict=True)):
        context, ix = bank.retrieve(vec, top_k=trainer.config.mem_top_k_eval)
        saved = readers.get(
            sha(trainer.evaluator.build_prompt(q["question"], context, q))
        )
        old = original["scores"][i]
        if (
            not saved
            or old["indices"] != list(map(int, ix))
            or old["tokens"] != saved["tokens"]
            or old["answer"] != saved["response"].strip()
            or abs(
                trainer.evaluator.compute_f1(
                    old["answer"], trainer.evaluator.get_ground_truth(q), q
                )
                - old["f1"]
            )
            > 1e-10
        ):
            return dict(receipt, status="unmatched_temporal_reader")
        baseline.append(
            dict(
                old, context_hash=_digest(context), refs=[bank.trace.ref(j) for j in ix]
            )
        )
    bank = replay_writes(trainer, [prefix])
    treated = execute(trainer, bank, prefix, event, idx, high, out)
    old_updates = [
        r
        for r in treated["changes"]
        if r["memory_id"] in ids
        and r["action"] == "UPDATE"
        and r["before"] != r["after"]
    ]
    register(
        out / "writes.json",
        {
            "control": control,
            "update": treated,
            "older_updates": old_updates,
            "choice": file_hash(out / "choice.json"),
        },
    )
    rows = []
    for q, vec, lo in zip(convo["qa"], queries, baseline, strict=True):
        context, ix = bank.retrieve(vec, top_k=trainer.config.mem_top_k_eval)
        context_hash = _digest(context)
        prompt = (
            None
            if context_hash == lo["context_hash"]
            else trainer.evaluator.build_prompt(q["question"], context, q)
        )
        hi = {
            "indices": list(map(int, ix)),
            "refs": [bank.trace.ref(j) for j in ix],
            "context_hash": context_hash,
        }
        first_only = set(q["evidence"]) <= set(prefix["source_turns"])
        rows.append(
            {
                "index": q["original_index"],
                "type": int(q["category"]),
                "first_only": first_only,
                "temporal": lo,
                "update": hi,
                "prompt": prompt,
                "prompt_hash": sha(prompt) if prompt is not None else None,
            }
        )
    return dict(
        receipt,
        status="prepared",
        model=trainer.args.model,
        backend=os.environ["MEMSKILL_BACKEND_ID"],
        choice=choice,
        rows=rows,
        edited=len(old_updates),
        writes=file_hash(out / "writes.json"),
        update_fault=execution_fault(
            treated["parsed_actions"], treated["applied"], treated["skills"]
        ),
    )


def summarize(reports):
    good = [r for r in reports if r["status"] == "matched"]
    population = sum(len(r["questions"]) for r in reports)
    summary = {
        "windows": len(reports),
        "arrived_questions": population,
        "matched_windows": len(good),
        "eligible_questions": sum(
            len(r["questions"])
            for r in reports
            if r["status"] != "ineligible_fixed_slots"
        ),
        "statuses": dict(Counter(r["status"] for r in reports)),
    }
    if not good:
        return summary
    rate = mean(r["choice"]["source"] for r in good)
    rows = [(r, q) for r in good for q in r["rows"]]
    metrics = (
        "source",
        "sham",
        "native",
        "proposed",
        "fixed_update",
        "fixed_temporal",
        "rate",
    )

    def chance(record, arm):
        return rate if arm == "rate" else record["choice"][arm]

    def scores(group):
        values = {}
        for arm in metrics:
            returns = []
            for r, q in group:
                p = chance(r, arm)
                lo, hi = q["temporal"]["f1"], q["update"]["f1"]
                returns.append(lo + p * (hi - lo))
            values[arm] = mean(returns)
        return {"questions": len(group), "f1": values}

    return dict(
        summary,
        **scores(rows),
        matched_source_rate=rate,
        reader_tokens_per_question={
            arm: mean(
                (1 - chance(r, arm)) * sum(q["temporal"]["tokens"])
                + chance(r, arm) * sum(q["update"]["tokens"])
                for r, q in rows
            )
            for arm in metrics
        },
        writer_tokens_per_window={
            arm: mean(
                (1 - chance(r, arm)) * r["writer_tokens"]["control"]
                + chance(r, arm) * r["writer_tokens"]["update"]
                for r in good
            )
            for arm in metrics
        },
        strictly_prospective_questions=sum(
            len(r["questions"]) for r in good if r["strict"]
        ),
        strictly_prospective=scores([(r, q) for r, q in rows if r["strict"]])
        if any(r["strict"] for r in good)
        else {"questions": 0},
        mean_abs_probability_shift=mean(
            abs(r["choice"]["source"] - r["choice"]["native"]) for r in good
        ),
        edited_windows=sum(bool(r["edited"]) for r in good),
        execution_faults=sum(bool(r["update_fault"]) for r in good),
        by_dialogue={
            s: scores([(r, q) for r, q in rows if r["dialogue"] == s])
            for s in sorted({r["dialogue"] for r in good})
        },
        by_type={
            str(t): scores([(r, q) for r, q in rows if q["type"] == t])
            for t in sorted({q["type"] for r, q in rows})
        },
        by_first_only={
            str(k): scores([(r, q) for r, q in rows if q["first_only"] == k])
            for k in {q["first_only"] for r, q in rows}
        },
    )


def verify_pair(out, r, data, evaluator, cell):
    if r["status"] != "matched":
        return None
    directory = out / r["name"]
    if file_hash(directory / "writes.json") != r["writes"]:
        raise ValueError("Physical write receipt changed")
    writes = read(directory / "writes.json")
    choice = directory / "choice.json"
    if (
        file_hash(choice) != writes["choice"]
        or read(choice)["probabilities"] != r["choice"]
    ):
        raise ValueError("Measured actions no longer match the frozen probabilities")
    if r["questions"] != [q["index"] for q in r["rows"]]:
        raise ValueError("Question population or order changed")
    (trace,) = (cell / "traces").glob("*.jsonl")
    if file_hash(trace) != r["trace"]:
        raise ValueError("Changed factual trace")
    prefix = next(
        event
        for event in map(json.loads, trace.read_bytes().splitlines())
        if event["kind"] == "span"
    )
    banks = {}
    for arm, event in (("temporal", writes["control"]), ("update", writes["update"])):
        trajectory = [prefix, event]
        live = restore_text(trajectory)
        versions = {
            c["memory_id"]: c["version"] for step in trajectory for c in step["changes"]
        }
        banks[arm] = (list(live.values()), [[mid, versions[mid]] for mid in live])
    jobs = {}
    if "reader_inputs" in r:
        path = directory / "reader-inputs.json"
        if file_hash(path) != r["reader_inputs"]:
            raise ValueError("Frozen reader inputs changed")
        jobs = {j["index"]: j for j in read(path)["rows"]}
    queue = []
    edited = {(c["memory_id"], c["version"]) for c in writes["older_updates"]}
    for row in r["rows"]:
        question = data[r["dialogue"]]["qa"][row["index"]]
        ref = evaluator.get_ground_truth(question)
        lo, hi = row["temporal"], row["update"]
        sign = int(recovered(lo, hi, ref)) - int(recovered(hi, lo, ref))
        if sign:
            queue.append(
                {
                    "index": row["index"],
                    "type": row["type"],
                    "recovered_by": "update" if sign > 0 else "temporal",
                    "delta": hi["f1"] - lo["f1"],
                    "read_updated_ref": bool(edited & {tuple(v) for v in hi["refs"]}),
                    "status": "execution_fault"
                    if r["update_fault"]
                    else "requires_same_skill_controls",
                    "additional_credit": 0.0,
                }
            )
        for arm in ("temporal", "update"):
            observed = row[arm]
            texts, refs = banks[arm]
            positions = observed["indices"]
            if any(j < 0 or j >= len(refs) for j in positions):
                raise ValueError("Invalid read position")
            context = [texts[j] for j in positions]
            if [refs[j] for j in positions] != observed["refs"] or _digest(
                context
            ) != observed["context_hash"]:
                raise ValueError("Versioned read differs from actual bank")
            if jobs and arm == "update":
                prompt = jobs[row["index"]]["prompt"]
                if prompt is None and lo["context_hash"] != hi["context_hash"]:
                    raise ValueError("Changed reader input marked reusable")
                if prompt is not None and prompt != evaluator.build_prompt(
                    question["question"], context, question
                ):
                    raise ValueError("Scored prompt differs from the versioned read")
            f1 = evaluator.compute_f1(
                observed["answer"], evaluator.get_ground_truth(question), question
            )
            if abs(observed["f1"] - f1) > 1e-10 or min(observed["tokens"]) <= 0:
                raise ValueError("Stored answer or F1 did not reproduce")

    register(
        directory / "recovery-candidates.json",
        {
            "pair": file_hash(directory / "pair.json"),
            "writes": r["writes"],
            "candidates": queue,
        },
    )
    return writes


def passed_screen(report):
    if report["windows"] != report["total_windows"] or not report.get("questions"):
        return False
    f = report["f1"]
    wins = sum(
        g["f1"]["source"] > g["f1"]["rate"] for g in report["by_dialogue"].values()
    )
    return (
        report["questions"] >= 0.9 * report["eligible_questions"]
        and f["source"] >= f["native"] + 0.02
        and f["source"] > max(f["sham"], f["rate"])
        and wins >= 4
    )


def run(trainer, study, forecast, train, specs):
    base = forecast.parent.parent.parent
    out = forecast.parent
    locked = read(forecast)
    if locked["plan"] != file_hash(PLAN) or locked["study"] != file_hash(
        study / "protocol.json"
    ):
        raise ValueError("Changed forecast protocol")
    data = {r["sample_id"]: r for r in train}
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    register(
        out / "backend.json",
        {
            "backend": os.environ["MEMSKILL_BACKEND_ID"],
            "forecast": file_hash(forecast),
            "study": locked["study"],
            "skills": [s["version"] for s in specs],
        },
    )
    with (out / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        processed = False
        # Alternate the two registered cohorts so later controls can be measured as they arrive.
        early = [w for w in locked["windows"] if w["panel"] == "early"]
        later = [w for w in locked["windows"] if w["panel"] == "later"]
        entries = sorted([*enumerate(early), *enumerate(later)], key=lambda v: v[0])
        for _, w in entries:
            root = base / (
                "source-compression/factorial"
                if w["panel"] == "early"
                else "late-pairs/outcomes"
            )
            folder = root / w["name"] / f"{w['prefix']}0"
            dest = out / w["name"]
            result = dest / "pair.json"
            if (
                result.exists()
                or (dest / "reader-inputs.json").exists()
                or not (folder / "scores.json").exists()
            ):
                continue
            dest.mkdir(exist_ok=True)
            convo = window(data[w["dialogue"]], w["start"], 2)
            if [q["original_index"] for q in convo["qa"]] != w["questions"]:
                raise ValueError("Question cohort changed")
            for archived in sorted(folder.parent.glob("[01][01]/scores.json")):
                seed_controls(archived.parent, root / "responses", out / "responses")
            os.environ["MEMSKILL_USAGE_LOG"] = str(dest / "usage.jsonl")
            with history_embeddings():
                record = measure(trainer, convo, root, folder, specs, w, dest)
            register(
                dest / "reader-inputs.json"
                if record["status"] == "prepared"
                else result,
                record,
            )
            processed = True
            break
        print(
            "completed a new window" if processed else "awaiting original controls",
            flush=True,
        )


def score_prepared(path, out, corpus, evaluator):
    packet = read(path)
    if packet["backend"] != os.environ["MEMSKILL_BACKEND_ID"] or packet[
        "writes"
    ] != file_hash(path.parent / "writes.json"):
        raise ValueError("Different physical writer or reader backend")
    os.environ["MEMSKILL_USAGE_LOG"] = str(path.parent / "usage.jsonl")
    rows = []
    for job in packet["rows"]:
        q = corpus[packet["dialogue"]]["qa"][job["index"]]
        prompt = job["prompt"]
        if prompt is None:
            if job["temporal"]["context_hash"] != job["update"]["context_hash"]:
                raise ValueError("Reader reuse requires identical contexts")
            hi = dict(job["temporal"])
        else:
            if sha(prompt) != job["prompt_hash"]:
                raise ValueError("Saved update prompt changed")
            hi = measured_answer(
                prompt, packet["model"], q, evaluator, out / "responses"
            )
            hi["answer"] = hi["answer"].strip()
            hi["ok"] = True
        rows.append(
            {
                **{k: v for k, v in job.items() if k not in ("prompt", "prompt_hash")},
                "update": dict(hi, **job["update"]),
            }
        )
    result = path.parent / "pair.json"
    register(
        result, dict(packet, rows=rows, status="matched", reader_inputs=file_hash(path))
    )


def finish(base, study, corpus):
    out = base / "source-compression/temporal-update-writer"
    forecast = out / "forecast.json"
    locked = read(forecast)
    if (
        locked["study"] != file_hash(study / "protocol.json")
        or file_hash(corpus) != read(study / "protocol.json")["input_sha256"]
    ):
        raise ValueError("Reader split or corpus changed")
    train = set(read(study / "protocol.json")["train"])
    data = {r["sample_id"]: r for r in read(corpus) if r["sample_id"] in train}
    evaluator = LoCoMoEvaluator(SimpleNamespace())
    with (out / "reader.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in sorted(out.glob("conv*/reader-inputs.json")):
            if not (path.parent / "pair.json").exists():
                score_prepared(path, out, data, evaluator)
        publish(out, base, locked, forecast, data, evaluator)


def publish(out, base, locked, forecast, data, evaluator):
    for panel in ("early", "later"):
        names = [w["name"] for w in locked["windows"] if w["panel"] == panel]
        files = [
            out / n / "pair.json" for n in names if (out / n / "pair.json").exists()
        ]
        if not files:
            continue
        records = [read(p) for p in files]
        for record in records:
            spec = next(w for w in locked["windows"] if w["name"] == record["name"])
            archive = base / (
                "source-compression/factorial"
                if panel == "early"
                else "late-pairs/outcomes"
            )
            control = archive / record["name"] / f"{spec['prefix']}0" / "scores.json"
            writes = verify_pair(out, record, data, evaluator, control.parent)
            if writes is None:
                continue
            record["writer_tokens"] = {
                arm: sum(
                    call["input_tokens"] + call["output_tokens"]
                    for call in writes[arm]["executor_responses"]
                )
                for arm in ("control", "update")
            }
            record["strict"] = forecast.stat().st_mtime_ns < min(
                control.stat().st_mtime_ns,
                (out / record["name"] / "writes.json").stat().st_mtime_ns,
            )
        report = dict(
            summarize(records),
            archive={p.parent.name: file_hash(p) for p in files},
            total_windows=len(names),
        )
        report["registered_questions"] = sum(
            len(w["questions"]) for w in locked["windows"] if w["panel"] == panel
        )
        report["screen_passed"] = panel == "later" and passed_screen(report)
        register(out / f"{panel}-{len(files):02d}-{digest(report)[:12]}.json", report)
        if len(files) == len(names):
            register(out / (panel + ".json"), report)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Score frozen prompts on the existing local Llama GPU server."
    )
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    finish(a.base, a.study, a.corpus)
