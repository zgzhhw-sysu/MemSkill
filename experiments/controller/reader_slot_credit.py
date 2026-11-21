"""Measure paired reader effects carried by focal-write-only retrieved texts."""

import argparse
import fcntl
import json
import math
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.audit_paired import reader_inputs
from experiments.controller.capped_source_policy import baseline, joint_evidence
from experiments.controller.capped_source_repair import BACKEND as EARLY
from experiments.controller.capped_source_repair import archive
from experiments.controller.first_evidence_credit import BACKEND as DONOR
from experiments.controller.semantic_writer import sha
from experiments.controller.writer_granularity import measured_answer
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import read_response, request_cache
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "reader_slot_credit.json"


def slots(left, right, banks, pivot):
    """Read entries are (last writing span, text). Preserve rank, including duplicates."""
    groups = []
    mixed = [[], []]
    for lo, hi in zip_longest(left, right):
        lt, ht = lo and lo[1], hi and hi[1]
        direct = (lt != ht) and any(
            entry is not None and entry[0] == pivot and entry[1] not in other
            for entry, other in ((lo, banks[1]), (hi, banks[0]))
        )
        groups.append("same" if lt == ht else "direct" if direct else "other")
        for acc, text in zip(mixed, (ht, lt) if direct else (lt, ht), strict=True):
            if text is not None:
                acc.append(text)
    return groups, mixed


def components(low, high, hybrids, credit):
    a, b = hybrids  # direct from high in low background, then vice versa
    direct = ((a - low) + (high - b)) / 2
    other = high - low - direct
    sign = (credit > 0) - (credit < 0)
    pos = max(0.0, sign * direct)
    total = pos + max(0.0, sign * other)
    if credit and not total:
        raise ValueError("Verified direction disagrees with both physical components")
    return direct, credit * pos / total if total else 0.0


def reader_key(model, prompt):
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 32,
    }
    return request_cache(os.environ["MEMSKILL_API_BASE"], params, "reader")[1]


def endpoint(cell, cache, expected, qa, ev, model):
    (trace,) = (cell / "traces").glob("*.jsonl")
    scores = cell / "scores.json"
    if expected["trace"] != file_hash(trace) or expected["scores"] != file_hash(scores):
        raise ValueError("Original trace or scores changed")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    spans = [s for s in events if s["kind"] == "span"]
    queries = {q["query_id"]: q for q in events if q["kind"] == "qa"}
    rows = read(scores)["answers"]
    inputs = reader_inputs(events, rows, dict(enumerate(qa)), ev)
    versions = {
        (c["memory_id"], c["version"]): (i, c["after"])
        for i, s in enumerate(spans)
        for c in s["changes"]
    }
    calls = [
        r
        for l in (cell / "usage.jsonl").read_bytes().splitlines()
        if (r := json.loads(l))["kind"] == "reader"
    ]
    contexts, answers = [], []
    by_key = {}
    for call in calls:
        by_key.setdefault(call["cache_key"], []).append(call)
    for i, (q, row) in enumerate(zip(qa, rows, strict=True)):
        ctx = [versions[tuple(ref)] for ref in queries[i]["refs"]]
        prompt = ev.build_prompt(q["question"], [text for _, text in ctx], q)
        key = reader_key(model, prompt)
        if not by_key.get(key):
            raise ValueError(f"Missing physical reader request at {cell}, reader {i}")
        logged = by_key[key].pop()
        if (
            row["qa_index"] != i
            or key != logged["cache_key"]
            or sha(prompt) != inputs[i][0]
        ):
            raise ValueError(
                f"Physical reader request differs at {cell}, reader {i}: key={key == logged['cache_key']}, prompt={sha(prompt) == inputs[i][0]}"
            )
        hit = read_response(cache, key)
        if not hit or hit["sha256"] != logged["response_sha256"]:
            raise ValueError("Physical response has changed")
        v = hit["payload"]
        score = ev.compute_f1(v["response"], ev.get_ground_truth(q), q)
        if (
            v["response"].strip() != row["prediction"]
            or v["tokens"] != row["tokens"]
            or v["prompt_sha256"] != inputs[i][0]
            or not math.isclose(score, row["f1"], abs_tol=1e-12)
        ):
            raise ValueError("Reader endpoint does not reproduce")
        contexts.append(ctx)
        answers.append(
            {
                "answer": v["response"],
                "tokens": v["tokens"],
                "f1": score,
                "prompt_sha256": inputs[i][0],
                "key": key,
                "response_sha256": hit["sha256"],
            }
        )
    if any(by_key.values()):
        raise ValueError("Unmatched physical reader calls")
    return spans, contexts, answers, read(cell / "decisions.json")


def prepare(base, study, corpus, *, verify=False):
    base / "late-pairs"
    parent = base / "source-compression/capped-source-credit"
    out = base / "source-compression/reader-slot-credit"
    out.mkdir(exist_ok=True)
    reg = out / "registration.json"
    manifest = read(parent / "registration.json")
    fit = parent / "policy/joint-mask/fit.json"
    proto = read(study / "protocol.json")
    inputs = {
        "plan": file_hash(PLAN),
        "manifest": file_hash(parent / "registration.json"),
        "fit": file_hash(fit),
        "corpus": file_hash(corpus),
        "protocol": file_hash(study / "protocol.json"),
    }
    if (
        inputs["corpus"] != proto["input_sha256"]
        or inputs["protocol"] != manifest["protocol"]
    ):
        raise ValueError("Wrong training data or reader")
    register(reg, inputs)
    if not verify and (out / "cohort.json").exists():
        if read(out / "cohort.json")["registration"] != file_hash(reg):
            raise ValueError("Different intervention registration")
        return out
    data = {
        c["sample_id"]: c["qa"]
        for c in read(corpus)
        if c["sample_id"] in proto["train"]
    }
    original, contrasts, _ = baseline(base, manifest)
    checked, contrasts, _ = joint_evidence(base, original, contrasts)
    fitted = read(fit)["observations"]
    ev = LoCoMoEvaluator(SimpleNamespace())
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    cells = {}
    contrasts_out = []
    counts = Counter()
    for idx, (obs, (w, bits, credits), target) in enumerate(
        zip(checked, contrasts, fitted, strict=True)
    ):
        sid, j = obs["dialogue"], obs["stage"]
        masked = any(any(w["cells"][k]["triggers"]) for k in bits)
        if (dict(obs, raw=0.0, credit=0.0) if masked else obs) != target:
            raise ValueError("Wrong joint-masked physical target")
        backend = EARLY if w["panel"] == "early" else DONOR
        os.environ["MEMSKILL_BACKEND_ID"] = backend
        ar = archive(base, w["panel"])
        qs = [data[sid][i] for i in w["questions"]]
        pair = []
        for bit in bits:
            k = (w["name"], bit)
            if k not in cells:
                cells[k] = endpoint(
                    ar / w["name"] / bit,
                    ar / "responses",
                    w["cells"][bit],
                    qs,
                    ev,
                    proto["model"],
                )
            pair.append(cells[k])
        (sa, ca, aa, da), (sb, cb, ab, db) = pair
        chosen = [s[j]["selected_skills"] for s in (sa, sb)]
        if (
            any(
                len(ops) != 3 or len({op["version"] for op in ops}) != 3
                for ops in chosen
            )
            or chosen[0][:2] != chosen[1][:2]
            or [ops[-1]["name"] for ops in chosen]
            != ["capture_temporal_context", "capture_activity_details"]
            or any(
                [int(d["use"]) for d in ds] != list(map(int, bit))
                for ds, bit in zip((da, db), bits, strict=True)
            )
        ):
            raise ValueError(
                "Archived intervention differs from its fixed ordered K=3 decision"
            )
        if (
            sa[:j] != sb[:j]
            or any(
                sa[j][k] != sb[j][k]
                for k in ("source", "refs", "retrieved", "source_turns")
            )
            or any(
                da[j][k] != db[j][k]
                for k in ("state", "base_logit", "plain", "grounded")
            )
            or da[j]["use"]
            or not db[j]["use"]
            or da[j]["state"] != obs["state"]
        ):
            raise ValueError("Invalid physical one-slot intervention")
        banks = [set(restore_text(s[: j + 1]).values()) for s in (sa, sb)]
        rows = []
        for i, (q, l, h, low, high, c) in enumerate(
            zip(qs, ca, cb, aa, ab, credits, strict=True)
        ):
            tags, mix = slots(l, h, banks, j)
            prompts = [ev.build_prompt(q["question"], ls, q) for ls in mix]
            hashes = [sha(p) for p in prompts]
            known = {a["prompt_sha256"]: a for a in (low, high)}
            pure = [k for k in ("direct", "other") if k in tags]
            group = "+".join(pure) or "same"
            counts[group] += 1
            unresolved = {
                d: p for d, p in zip(hashes, prompts, strict=True) if d not in known
            }
            if len(pure) < 2 and unresolved:
                raise ValueError("Pure-component hybrid does not reproduce an endpoint")
            rows.append(
                {
                    "index": w["questions"][i],
                    "type": q.get("category"),
                    "slots": tags,
                    "group": group,
                    "credit": 0.0 if masked else c,
                    "low": low,
                    "high": high,
                    "hybrids": hashes,
                    "novel": unresolved,
                }
            )
        if not math.isclose(
            mean(r["high"]["f1"] - r["low"]["f1"] for r in rows),
            obs["raw"],
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Reader returns disagree with the observed training target"
            )
        contrasts_out.append(
            {
                "observation": idx,
                "window": w["name"],
                "stage": j,
                "dialogue": sid,
                "bits": bits,
                "masked": masked,
                "backend": backend,
                "rows": rows,
            }
        )
    if len(contrasts_out) != 60 or sum(counts.values()) != 4 * 91 + 88:
        raise ValueError("Wrong paired question population")
    register(
        out / "cohort.json",
        {
            "registration": file_hash(reg),
            "model": proto["model"],
            "groups": dict(counts),
            "contrasts": contrasts_out,
        },
    )
    return out


def measure_response(base, out, backend, model, q, h, p, ev):
    os.environ["MEMSKILL_BACKEND_ID"] = backend
    cache = out / "responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    k = reader_key(model, p)
    dest = out / "hybrids" / (k + ".json")
    dest.parent.mkdir(exist_ok=True)
    original_cache = (
        archive(base, "early" if backend == EARLY else "donor") / "responses"
    )
    found = read_response(cache, k) or read_response(original_cache, k)
    if not found:
        if dest.is_file():
            raise ValueError("Existing reader receipt lost its physical response")
        measured = measured_answer(p, model, q, ev, cache)
        found = read_response(cache, k)
        if (
            not found
            or found["payload"]["response"] != measured["answer"]
            or found["payload"]["tokens"] != measured["tokens"]
        ):
            raise ValueError("Physical reader did not reproduce generated answer")
    v = found["payload"]
    if sha(p) != h or v["prompt_sha256"] != h or v["key"] != k:
        raise ValueError("Physical response differs from the rank intervention")
    receipt = {
        "prompt_sha256": h,
        "key": k,
        "response_sha256": found["sha256"],
        "answer": v["response"],
        "tokens": v["tokens"],
    }
    register(dest, receipt)
    return receipt


def measured_hybrids(base, out, corpus):
    plan = read(out / "cohort.json")
    data = {
        c["sample_id"]: c["qa"]
        for c in read(corpus)
        if c["sample_id"] in {v["dialogue"] for v in plan["contrasts"]}
    }
    ev = LoCoMoEvaluator(SimpleNamespace())
    jobs = [(c, r) for c in plan["contrasts"] for r in c["rows"] if r["novel"]]
    jobs.sort(key=lambda pair: (not bool(pair[1]["credit"]), pair[0]["masked"]))
    for c, r in jobs:
        q = data[c["dialogue"]][r["index"]]
        for h, p in r["novel"].items():
            measure_response(base, out, c["backend"], plan["model"], q, h, p, ev)
    register(
        out / "readers-complete.json",
        {
            "cohort": file_hash(out / "cohort.json"),
            "files": {
                p.stem: file_hash(p) for p in sorted((out / "hybrids").glob("*.json"))
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("study", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("--read", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    o = prepare(a.base, a.study, a.corpus, verify=a.verify)
    if a.read:
        with (o / "reader.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            measured_hybrids(a.base, o, a.corpus)
