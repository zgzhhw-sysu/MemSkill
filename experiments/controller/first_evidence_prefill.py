"""Measure the changed first write from an already verified physical prefix, ahead of its continuation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.capped_source_repair import diagnose, writer
from experiments.controller.factorial_rollout import read
from experiments.controller.first_evidence_credit import (
    BACKEND,
    events,
    paths,
    register_trial,
)
from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash, register
from src.evaluation_cache import publish_response, read_response, request_cache
from src.executor import Executor, deployment_operations
from src.llm_utils import get_llm_response_via_api


def pivot_name(g):
    return f"{g['dialogue']}_s{g['first']:02d}"


def request_for(group, src, pair, specs, model, cap):
    first = group["first"]
    factual, _ = paths(group["dialogue"], first, group["end"])
    bits = "".join(map(str, factual))
    cell = src / bits
    (trace,) = (cell / "traces").glob("*.jsonl")
    if pair["trajectories"][bits][str(trace.relative_to(cell))] != file_hash(
        trace
    ) or pair["trajectories"][bits]["decisions.json"] != file_hash(
        cell / "decisions.json"
    ):
        raise ValueError("First write lacks its archived decision and memory trace")
    dec = read(cell / "decisions.json")[first - 1]
    span = events(cell)[0][first - 1]
    used = int(dec["use"])
    menu = [dec["plain"], dec["grounded"]]
    old, alt = (
        [SimpleNamespace(**specs[i]) for i in menu[j]] for j in (used, 1 - used)
    )
    if (
        used != factual[first - 1]
        or len(old) != 3
        or len(alt) != 3
        or menu[0][:2] != menu[1][:2]
        or [s.version for s in old] != [s["version"] for s in span["selected_skills"]]
    ):
        raise ValueError("Archived common skills or first intervention changed")
    ex = Executor(SimpleNamespace(retriever="unused"))
    prompts = [
        ex._build_executor_prompt(
            deployment_operations(ops), span["source"], span["retrieved"]
        )
        for ops in (old, alt)
    ]
    (call,) = span["executor_responses"]
    if sha(prompts[0]) != call["prompt_sha256"] or prompts[0] == prompts[1]:
        raise ValueError("Could not reproduce the observed earliest writer input")
    return {
        "pivot": pivot_name(group),
        "backend": BACKEND,
        "model": model,
        "cap": cap,
        "prompt": prompts[1],
        "prompt_hash": sha(prompts[1]),
        "factual_prompt": call["prompt_sha256"],
        "selected": [s.version for s in alt],
    }


def audit_write(group, src, proto, cache, request, receipt):
    r, saved = read(request), read(receipt)
    hit = read_response(cache, saved["key"])
    if (
        saved["request"] != file_hash(request)
        or not hit
        or saved["response_sha256"] != hit["sha256"]
    ):
        raise ValueError("Different physical first write")
    v = hit["payload"]
    if v["prompt_sha256"] != r["prompt_hash"] or v["tokens"] != saved["tokens"]:
        raise ValueError("Different physical prompt or budget")
    bits, _ = paths(group["dialogue"], group["first"], group["end"])
    factual = events(src / "".join(map(str, bits)))[0][group["first"] - 1]
    specs = {s["version"]: s for s in proto["skills"]}
    changed = dict(
        factual,
        selected_skills=[specs[k] for k in r["selected"]],
        executor_responses=[
            {
                "prompt_sha256": v["prompt_sha256"],
                "response": v["response"],
                "input_tokens": v["tokens"][0],
                "output_tokens": v["tokens"][1],
            }
        ],
    )
    ex = Executor(
        SimpleNamespace(
            retriever=proto["retriever"],
            model=proto["model"],
            api_key="local",
            api_base="local",
            temperature=0.0,
            max_new_tokens=proto["writer_tokens"],
        )
    )
    ops = deployment_operations(
        [SimpleNamespace(**s) for s in changed["selected_skills"]]
    )
    changed["parsed_actions"] = writer(ex, ops, changed, repair=False)

    def count(span):
        actions = span["parsed_actions"]
        return {
            "skill": span["selected_skills"][-1]["name"],
            "successful_actions": sum(a["success"] for a in actions),
            "inserted_actions": sum(
                a["success"] and a["action_type"] == "INSERT" for a in actions
            ),
            "failed_actions": sum(not a["success"] for a in actions),
            "capped_source_repair": diagnose(ex, span),
        }

    register(
        receipt.parent / "writes.json",
        {
            "response": file_hash(receipt),
            "factual": count(factual),
            "switched": count(changed),
        },
    )


def next_write(locked, root, cross, proto):
    """Complete exact archived responses before using the shared writer for a new one."""
    seen = set()
    fallback = None
    for g in locked["groups"]:
        src = cross / g["endpoint"]
        key = pivot_name(g)
        directory = root / key
        receipt = directory / "response.json"
        if (
            key in seen
            or (directory / "writes.json").exists()
            or not (src / "pair.json").exists()
        ):
            continue
        seen.add(key)
        pair = read(src / "pair.json")
        if pair["manifest"] != locked["manifest"]:
            raise ValueError("Different long-prefix source")
        r = request_for(
            g, src, pair, proto["skills"], proto["model"], proto["writer_tokens"]
        )
        request = directory / "request.json"
        params = {
            "model": r["model"],
            "messages": [{"role": "user", "content": r["prompt"]}],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": r["cap"],
        }
        _, request_id = request_cache(
            os.environ["MEMSKILL_API_BASE"], params, "executor"
        )
        upstream = read_response(cross / "responses", request_id)
        item = (g, src, r, request, receipt, request_id, upstream)
        if upstream or receipt.exists():
            return item
        if fallback is None:
            fallback = item
    return fallback


def prefill(base, study):
    if os.environ["MEMSKILL_BACKEND_ID"] != BACKEND:
        raise ValueError("Expected original bounded physical server")
    reg = register_trial(base)
    locked = read(reg)
    out = reg.parent
    cross = out.parent / "cross-window-prefix"
    proto = read(study / "protocol.json")
    manifest = read(cross / "manifest.json")
    if manifest["protocol"] != file_hash(study / "protocol.json"):
        raise ValueError("Unexpected experiment configuration")
    cache = out / "responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    root = out / "first-writes"
    root.mkdir(exist_ok=True)
    with (root / "measure.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = next_write(locked, root, cross, proto)
        if pending is None:
            return
        g, src, r, request, receipt, request_id, upstream = pending
        receipt.parent.mkdir(exist_ok=True)
        register(request, r)
        if receipt.exists():
            audit_write(g, src, proto, cache, request, receipt)
            return
        if upstream:
            v = upstream["payload"]
            if v["prompt_sha256"] != r["prompt_hash"]:
                raise ValueError("Different ancestor writer prompt")
            publish_response(
                cache, request_id, r["prompt_hash"], v["response"], *v["tokens"]
            )
        log = receipt.parent / "usage.jsonl"
        os.environ["MEMSKILL_USAGE_LOG"] = str(log)
        ans, ni, no = get_llm_response_via_api(
            r["prompt"],
            LLM_MODEL=r["model"],
            MAX_TOKENS=r["cap"],
            TAU=0.0,
            base_url=os.environ["MEMSKILL_API_BASE"],
            api_key=os.getenv("MEMSKILL_API_KEY", "local-only"),
            usage_kind="executor",
        )
        hit = read_response(cache, request_id)
        logged = json.loads(log.read_bytes().splitlines()[-1])
        if (
            not hit
            or hit["payload"]["response"] != ans
            or hit["payload"]["tokens"] != [ni, no]
            or logged["response_sha256"] != hit["sha256"]
        ):
            raise ValueError("Missing physical response for the changed first action")
        register(
            receipt,
            {
                "request": file_hash(request),
                "key": request_id,
                "response_sha256": hit["sha256"],
                "tokens": [ni, no],
                "parent_cache_hit": bool(upstream),
                "generation_attempts": logged["attempts"],
            },
        )
        audit_write(g, src, proto, cache, request, receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("study", type=Path)
    a = p.parse_args()
    prefill(a.base, a.study)
