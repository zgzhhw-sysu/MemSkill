"""Replay fixed skill choices when an empty action at the output cap suppressed source writes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller.factorial_repairs import checked_cell, replay_writes
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.semantic_writer import measured_readers, sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.temporal_update_writer import replay_provider
from experiments.controller.writer_granularity import measured_answer
from src.credit_assignment import _digest, recovered
from src.episode_trace import EpisodeTrace, skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings, read_response
from src.executor import Executor, deployment_operations
from src.llm_utils import get_llm_response_via_api
from src.rag_utils import get_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "capped_source_credit.json"
BACKEND = "251-hf-llama3-bf16-pre16-duty22-gpu7-qwen1-shared-b2-duty30"


def archive(base, panel):
    return base / (
        "source-compression/factorial" if panel == "early" else "late-pairs/outcomes"
    )


def writer(executor, ops, event, *, repair):
    calls = []
    actions = executor.execute_operation(
        ops,
        event["source"],
        event["retrieved"],
        audit=calls,
        response_provider=replay_provider(event["executor_responses"]),
        repair_capped_source=repair,
    )
    if calls != event["executor_responses"]:
        raise ValueError("Archived writer input, output or token count changed")
    return [vars(a) for a in actions]


def diagnose(executor, event):
    ops = deployment_operations(
        [SimpleNamespace(**s) for s in event["selected_skills"]]
    )
    original = writer(executor, ops, event, repair=False)
    if original != event["parsed_actions"]:
        raise ValueError("Historical writer actions did not reproduce")
    fixed = writer(executor, ops, event, repair=True)
    if fixed == original:
        return None
    retained = len(original) - 1
    if (
        original[-1]["success"]
        or fixed[:retained] != original[:retained]
        or any(
            not a["success"]
            or a["reasoning"] not in ("dated_single_turns", "dated_adjacent_turns")
            for a in fixed[retained:]
        )
    ):
        raise ValueError("Intervention changed a completed action")
    return {
        "kept": retained,
        "restored_items": len(fixed) - retained,
        "requests": [r["prompt_sha256"] for r in event["executor_responses"]],
    }


def census(base, study):
    out = base / "source-compression/capped-source-credit"
    out.mkdir(exist_ok=True)
    proto = read(study / "protocol.json")
    late = base / "late-pairs"
    split = read(late / "selection-labels/split.json")["windows"]["donor"]
    early = {
        w["window"]: w["questions"]
        for w in read(base / "source-compression/retrieval-competition/fit.json")[
            "training"
        ]
    }
    if [
        len(early),
        sum(map(len, early.values())),
        len(split),
        sum(map(len, split.values())),
    ] != [12, 91, 12, 88]:
        raise ValueError("Wrong fixed training population")
    identity = {
        "plan": file_hash(PLAN),
        "protocol": file_hash(study / "protocol.json"),
        "ledger": file_hash(late / "selection-labels/through-12.json"),
        "cap": proto["writer_tokens"],
    }
    path = out / "registration.json"
    if path.exists():
        saved = read(path)
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError("Registration or donor population changed")
        return path
    ex = Executor(
        SimpleNamespace(
            retriever=proto["retriever"],
            model=proto["model"],
            max_new_tokens=proto["writer_tokens"],
            temperature=0,
            api_base="local",
            api_key="local",
        )
    )
    windows = {}
    for panel, cohort in (("early", early), ("donor", split)):
        for name, ids in cohort.items():
            sid, start = name.rsplit("_s", 1)
            if sid not in proto["train"]:
                raise ValueError("A nontraining history entered the census")
            folder = archive(base, panel) / name
            paths = (
                ["00", "01", "10", "11"]
                if panel == "early"
                else ["".join(map(str, b)) for b in read(folder / "pair.json")["paths"]]
            )
            cells = {}
            for bits in paths:
                cell = folder / bits
                (trace,) = (cell / "traces").glob("*.jsonl")
                spans = [
                    v
                    for line in trace.read_bytes().splitlines()
                    if (v := json.loads(line))["kind"] == "span"
                ]
                if len(spans) != 2:
                    raise ValueError("Expected both physical writes")
                # The shared archive must contain the raw content actually returned for each writer request.
                calls = [
                    r
                    for line in (cell / "usage.jsonl").read_bytes().splitlines()
                    if (r := json.loads(line))["kind"] == "executor"
                ]
                for logged, raw in zip(
                    calls,
                    [r for step in spans for r in step["executor_responses"]],
                    strict=True,
                ):
                    hit = read_response(
                        folder.parent / "responses", logged["cache_key"]
                    )
                    if (
                        not hit
                        or hit["sha256"] != logged["response_sha256"]
                        or hit["payload"]["response"] != raw["response"]
                        or hit["payload"]["prompt_sha256"] != raw["prompt_sha256"]
                        or hit["payload"]["tokens"]
                        != [raw["input_tokens"], raw["output_tokens"]]
                    ):
                        raise ValueError("Writer receipt failed its archive check")
                cells[bits] = {
                    "trace": file_hash(trace),
                    "scores": file_hash(cell / "scores.json"),
                    "triggers": [diagnose(ex, s) for s in spans],
                }
            windows[name] = {
                "name": name,
                "dialogue": sid,
                "start": int(start),
                "panel": panel,
                "questions": ids,
                "cells": cells,
            }
    register(path, dict(identity, windows=windows))
    print(
        "registered",
        sum(len(w["cells"]) for w in windows.values()),
        "paths",
        flush=True,
    )
    return path


def affected(manifest):
    return sorted(
        [
            (w, b)
            for w in manifest["windows"].values()
            for b, c in w["cells"].items()
            if any(c["triggers"])
        ],
        key=lambda pair: (pair[0]["name"] != "conv-43_s11", pair[0]["name"], pair[1]),
    )


def suffix_provider(dest, request):
    """Use only a measured reply to the exact corrected prefix and suffix prompt."""
    path = dest / "suffix-request.json"
    register(path, request)
    response = dest / "suffix-response.json"
    if not response.exists():
        return None
    reply = read(response)
    raw = reply["call"]
    if reply["request"] != file_hash(path) or raw["prompt_sha256"] != sha(
        request["prompt"]
    ):
        raise ValueError("Different corrected suffix response")
    for line in (dest / "usage.jsonl").read_bytes().splitlines():
        logged = json.loads(line)
        if logged["kind"] != "executor":
            continue
        hit = read_response(dest.parents[1] / "responses", logged["cache_key"])
        if hit and hit["sha256"] == logged["response_sha256"]:
            payload = hit["payload"]
            if (
                payload["prompt_sha256"] == raw["prompt_sha256"]
                and payload["response"] == raw["response"]
                and payload["tokens"] == [raw["input_tokens"], raw["output_tokens"]]
                and payload["tokens"] == [logged["input"], logged["output"]]
            ):
                return replay_provider([raw])
    raise ValueError("Corrected suffix lacks its physical executor receipt")


def repaired_writes(trainer, spans, dest, trigger):
    trace = EpisodeTrace(dest, spans[0]["sample"])
    bank = trainer._initialize_memory_bank()
    bank.trace = trace
    repaired = []
    for i, event in enumerate(spans):
        selected = [SimpleNamespace(**s) for s in event["selected_skills"]]
        ops = deployment_operations(selected)
        if [skill_spec(op) for op in ops] != event["skills"]:
            raise ValueError("Changed ordered deployment skills")
        if i == 0:
            ctx, ix = [], []
        else:
            vec = trainer.state_encoder._encode_texts(event["source"]).reshape(-1)
            ctx, ix = bank.retrieve(vec, use_state_encoder=True)
        trace.begin(i, event["source"], ops, ctx, ix, selected=selected)
        prompt = trainer.executor._build_executor_prompt(ops, event["source"], ctx)
        unchanged = ctx == event["retrieved"]
        if unchanged:
            if (
                len(event["executor_responses"]) != 1
                or sha(prompt) != event["executor_responses"][0]["prompt_sha256"]
            ):
                raise ValueError(
                    "Identical input did not reproduce the physical prompt"
                )
            provider = replay_provider(event["executor_responses"])
        else:
            if i != 1 or trigger[0] is None:
                raise ValueError(
                    "Changed writer context without a changed earlier action"
                )
            request = {
                "prompt": prompt,
                "model": trainer.args.model,
                "cap": trainer.args.max_new_tokens,
                "prefix": _digest(repaired[0]),
                "backend": os.environ["MEMSKILL_BACKEND_ID"],
            }
            provider = suffix_provider(dest, request)
            if provider is None:
                return None
        calls = []
        acts = trainer.executor.execute_operation(
            ops,
            event["source"],
            ctx,
            audit=calls,
            response_provider=provider,
            repair_capped_source=True,
        )
        if len(calls) != 1 or any("API call failed" in a.reasoning for a in acts):
            raise RuntimeError("No measured corrected write")
        applied = trainer.executor.apply_to_memory_bank(
            acts, bank, ix, operation_name=[s.name for s in selected]
        )
        if not applied:
            raise RuntimeError("Failed to apply a corrected memory operation")
        bank.step()
        rec = dict(
            trace.current,
            parsed_actions=[vars(a) for a in acts],
            executor_responses=calls,
            applied=bool(applied),
            source_turns=event["source_turns"],
        )
        if (
            unchanged
            and trigger[i] is None
            and rec["parsed_actions"] != event["parsed_actions"]
        ):
            raise ValueError("Untriggered write changed with identical input")
        repaired.append(rec)
        if bank.get_all_contents() != list(restore_text(repaired).values()):
            raise ValueError("Repaired writes differ from replayed versions")
    register(dest / "writes.json", {"spans": repaired})
    return bank


def prepare(trainer, study, forecast, train, specs):
    out = forecast.parent
    base = out.parents[1]
    locked = read(forecast)
    if (
        locked["protocol"] != file_hash(study / "protocol.json")
        or os.environ["MEMSKILL_BACKEND_ID"] != BACKEND
    ):
        raise ValueError("Unexpected encoder or study")
    data = {c["sample_id"]: c for c in train}
    with (out / "encoder.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for w, bits in affected(locked):
            dest = out / w["name"] / bits
            if (dest / "inputs.json").exists() or (
                (dest / "suffix-request.json").exists()
                and not (dest / "suffix-response.json").exists()
            ):
                continue
            dest.mkdir(exist_ok=True, parents=True)
            root = archive(base, w["panel"])
            cell = root / w["name"] / bits
            src = w["cells"][bits]
            (trace,) = (cell / "traces").glob("*.jsonl")
            if (
                file_hash(trace) != src["trace"]
                or file_hash(cell / "scores.json") != src["scores"]
            ):
                raise ValueError("Registered physical trajectory changed")
            convo = window(data[w["dialogue"]], w["start"], 2)
            ids = [q["original_index"] for q in convo["qa"]]
            if ids != w["questions"]:
                raise ValueError("Training question population changed")
            control = checked_cell(
                trainer, convo, cell, tuple(map(int, bits)), specs, root / "responses"
            )
            readers = measured_readers(cell, root / "responses")
            with history_embeddings():
                bank = repaired_writes(trainer, control["spans"], dest, src["triggers"])
                if bank is not None:
                    factual = replay_writes(trainer, control["spans"])
                    vectors = get_embeddings(
                        trainer.args.retriever,
                        [q["question"] for q in convo["qa"]],
                        "query",
                    )
                    jobs = []
                    for i, (q, vec) in enumerate(
                        zip(convo["qa"], vectors, strict=True)
                    ):
                        ctx, ix = factual.retrieve(
                            vec, top_k=trainer.config.mem_top_k_eval
                        )
                        obs = control["scores"][i]
                        prompt = trainer.evaluator.build_prompt(q["question"], ctx, q)
                        r = readers[sha(prompt)]
                        if (
                            ix != obs["indices"]
                            or r["response"].strip() != obs["answer"]
                            or r["tokens"] != obs["tokens"]
                            or abs(
                                trainer.evaluator.compute_f1(
                                    obs["answer"],
                                    trainer.evaluator.get_ground_truth(q),
                                    q,
                                )
                                - obs["f1"]
                            )
                            > 1e-10
                        ):
                            raise ValueError("Native question did not reproduce")
                        new, nx = bank.retrieve(
                            vec, top_k=trainer.config.mem_top_k_eval
                        )
                        nprompt = trainer.evaluator.build_prompt(q["question"], new, q)
                        jobs.append(
                            {
                                "index": ids[i],
                                "type": int(q["category"]),
                                "factual": obs,
                                "context": _digest(ctx),
                                "repaired_context": _digest(new),
                                "refs": [bank.trace.ref(j) for j in nx],
                                "indices": nx,
                                "prompt": None if nprompt == prompt else nprompt,
                                "prompt_hash": sha(nprompt),
                            }
                        )
                    register(
                        dest / "inputs.json",
                        {
                            "jobs": jobs,
                            "writes": file_hash(dest / "writes.json"),
                            "trace": src["trace"],
                            "model": trainer.args.model,
                            "backend": BACKEND,
                            "registration": file_hash(forecast),
                        },
                    )
            print("prepared", w["name"], bits, flush=True)
            break


def repair_effects(rows, questions, evaluator):
    shifts = []
    gains, losses = [], []
    for row in rows:
        idx = row["index"]
        original, fixed = row["factual"], dict(row["fixed"], ok=True)
        truth = evaluator.get_ground_truth(questions[idx])
        shifts.append(fixed["f1"] - original["f1"])
        if recovered(original, fixed, truth):
            gains.append(idx)
        if recovered(fixed, original, truth):
            losses.append(idx)
    return {
        "questions": len(rows),
        "factual_f1": mean(r["factual"]["f1"] for r in rows),
        "fixed_f1": mean(r["fixed"]["f1"] for r in rows),
        "mean_delta": mean(shifts),
        "increased": sum(d > 0 for d in shifts),
        "unchanged": shifts.count(0.0),
        "decreased": sum(d < 0 for d in shifts),
        "strict_recovered_indices": gains,
        "strict_lost_indices": losses,
    }


def finish(base, study, corpus):
    reg = census(base, study)
    out = reg.parent
    locked = read(reg)
    proto = read(study / "protocol.json")
    if (
        os.environ.get("MEMSKILL_BACKEND_ID") != BACKEND
        or file_hash(corpus) != proto["input_sha256"]
    ):
        raise ValueError("Unexpected physical generator or benchmark")
    data = {x["sample_id"]: x for x in read(corpus) if x["sample_id"] in proto["train"]}
    ev = LoCoMoEvaluator(SimpleNamespace())
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    with (out / "reader.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for w, bits in affected(locked):
            d = out / w["name"] / bits
            rq = d / "suffix-request.json"
            resp = d / "suffix-response.json"
            if rq.is_file() and not resp.is_file():
                r = read(rq)
                if (
                    r["backend"] != BACKEND
                    or r["cap"] != proto["writer_tokens"]
                    or r["model"] != proto["model"]
                ):
                    raise ValueError("Changed request parameters")
                os.environ["MEMSKILL_USAGE_LOG"] = str(d / "usage.jsonl")
                ans, ni, no = get_llm_response_via_api(
                    r["prompt"],
                    LLM_MODEL=r["model"],
                    MAX_TOKENS=r["cap"],
                    TAU=0.0,
                    base_url=os.environ["MEMSKILL_API_BASE"],
                    api_key=os.environ.get("MEMSKILL_API_KEY", "local-only"),
                    usage_kind="executor",
                )
                if min(ni, no) <= 0:
                    raise RuntimeError("Missing corrected suffix response")
                register(
                    resp,
                    {
                        "request": file_hash(rq),
                        "call": {
                            "prompt_sha256": sha(r["prompt"]),
                            "response": ans,
                            "input_tokens": ni,
                            "output_tokens": no,
                        },
                    },
                )
        results = {}
        for w, bits in affected(locked):
            d = out / w["name"] / bits
            p = d / "inputs.json"
            if not p.is_file():
                continue
            packet = read(p)
            wp = d / "writes.json"
            if (
                packet["registration"] != file_hash(reg)
                or packet["writes"] != file_hash(wp)
                or packet["backend"] != BACKEND
                or packet["trace"] != w["cells"][bits]["trace"]
            ):
                raise ValueError("Different prepared reader inputs")
            spans = read(wp)["spans"]
            live = restore_text(spans)
            versions = {
                c["memory_id"]: c["version"] for step in spans for c in step["changes"]
            }
            jobs = packet["jobs"]
            if [j["index"] for j in jobs] != w["questions"]:
                raise ValueError("Incomplete registered question set")
            os.environ["MEMSKILL_USAGE_LOG"] = str(d / "usage.jsonl")
            folder = d / "questions"
            folder.mkdir(exist_ok=True)
            rows = {}
            for j in sorted(
                jobs,
                key=lambda j: not (w["name"] == "conv-43_s11" and j["index"] == 112),
            ):
                idx = j["index"]
                q = data[w["dialogue"]]["qa"][idx]
                dest = folder / f"{idx}.json"
                mids = [list(live)[k] for k in j["indices"]]
                ctx = [live[m] for m in mids]
                prompt = ev.build_prompt(q["question"], ctx, q)
                if (
                    j["refs"] != [[m, versions[m]] for m in mids]
                    or _digest(ctx) != j["repaired_context"]
                    or sha(prompt) != j["prompt_hash"]
                ):
                    raise ValueError("Versioned memory does not reproduce reader input")
                if not dest.exists():
                    if j["prompt"] is None:
                        if j["context"] != j["repaired_context"]:
                            raise ValueError(
                                "Original reader reuse with different contexts"
                            )
                        ans = {k: j["factual"][k] for k in ("answer", "tokens", "f1")}
                    else:
                        if j["prompt"] != prompt:
                            raise ValueError("Changed repaired prompt")
                        ans = measured_answer(
                            prompt, packet["model"], q, ev, out / "responses"
                        )
                    if min(ans["tokens"]) <= 0:
                        raise RuntimeError("Missing measured reader answer")
                    register(
                        dest,
                        {
                            "job": _digest(j),
                            "fixed": ans,
                            "factual": j["factual"],
                            "index": idx,
                        },
                    )
                row = read(dest)
                if (
                    row["job"] != _digest(j)
                    or abs(
                        ev.compute_f1(row["fixed"]["answer"], ev.get_ground_truth(q), q)
                        - row["fixed"]["f1"]
                    )
                    > 1e-10
                ):
                    raise ValueError("Scored repair does not reproduce")
                rows[idx] = row
            result = d / "scores.json"
            register(
                result,
                {"inputs": file_hash(p), "rows": [rows[i] for i in w["questions"]]},
            )
            results[w["name"] + "/" + bits] = file_hash(result)
            register(
                d / "effects.json",
                {
                    "scores": file_hash(result),
                    **repair_effects(rows.values(), data[w["dialogue"]]["qa"], ev),
                },
            )
        if len(results) == len(affected(locked)):
            register(
                out / "complete.json",
                {"registration": file_hash(reg), "scores": results},
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "finish"))
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    if a.phase == "register":
        census(a.base, a.study)
    else:
        finish(a.base, a.study, a.corpus)
