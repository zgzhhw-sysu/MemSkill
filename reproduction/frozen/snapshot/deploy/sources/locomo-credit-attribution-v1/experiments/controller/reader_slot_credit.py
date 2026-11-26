"""Measure paired reader effects carried by focal-write-only retrieved texts."""

import argparse
import fcntl
import json
import math
import os
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller import file_hash, measured_answer, read, register, text_hash as sha
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import read_response, request_cache
from src.replay import restore_text, verified_reader_contexts

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


def reader_key(model, prompt, max_tokens=32):
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": max_tokens,
    }
    return request_cache(os.environ["MEMSKILL_API_BASE"], params, "reader")[1]


def corrected_marker_contents(spans):
    """Reparse each logged response; only remove an action heading repeated before its matching ACTION line."""
    from src.executor import Executor

    parser = Executor(SimpleNamespace(retriever="trace"))
    double = re.compile(r"(?im)^[ \t]*(INSERT|UPDATE|DELETE|NOOP)[ \t]*\n(?=[ \t]*ACTION[ \t]*:[ \t]*\1[ \t]*$)")
    tail = re.compile(r"(?i)\n[ \t]*(INSERT|UPDATE|DELETE|NOOP)[ \t]*\Z")
    lookup = {}
    counts = Counter()
    for span in spans:
        original, revised = [], []
        for receipt in span["executor_responses"]:
            response = receipt["response"]
            clean, n = double.subn("", response)
            counts["repeated_headings"] += n
            for dest, text in ((original,response),(revised,clean)):
                dest.extend(vars(a) for a in parser._parse_response(text,len(span["retrieved"])))
        if original != span["parsed_actions"] or len(original) != len(revised):
            raise ValueError("Parser replay did not reproduce the original operation list")
        counts["parsed_records"] += len(original)
        writes = {}
        for op in ("INSERT","UPDATE"):
            physical = [c for c in span["changes"] if c["action"] == op]
            parsed = [a["memory_content"] for a in original if a["success"] and a["action_type"] == op]
            if [c["after"] for c in physical] != parsed:
                raise ValueError("Parser records do not match physical versions within an operation")
            writes[op] = iter(physical)
        for a,b in zip(original,revised,strict=True):
            applied = a["success"] and a["action_type"] in writes
            change = next(writes[a["action_type"]]) if applied else None
            if a==b:
                continue
            before,after = a["memory_content"],b["memory_content"]
            if ({**a,"memory_content":after} != b or a["action_type"] not in ("INSERT","UPDATE")
                    or not a["success"] or not tail.search(before) or tail.sub("",before).rstrip()!=after):
                raise ValueError("Reparsing changed more than the redundant trailing action word")
            key = (change["memory_id"],change["version"])
            value = (after,tail.sub("\nNOTE",before))
            if lookup.setdefault(key,value) != value:
                raise ValueError("Ambiguous provenance for retrieved memory")
            counts["cleaned_records"] += 1
    return lookup,dict(counts)


def archive_digests(cell):
    trace, = (cell / "traces").glob("*.jsonl")
    return {"trace": file_hash(trace), "scores": file_hash(cell / "scores.json")}


def measured_contexts(cell, cache, expected, questions, ev, model, max_tokens=32):
    (trace,) = (cell / "traces").glob("*.jsonl")
    scores = cell / "scores.json"
    if expected["trace"] != file_hash(trace) or expected["scores"] != file_hash(scores):
        raise ValueError("Original trace or scores changed")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    spans = [s for s in events if s["kind"] == "span"]
    rows = read(scores)["answers"]
    verified = verified_reader_contexts(events, rows, questions)
    calls = [
        r
        for l in (cell / "usage.jsonl").read_bytes().splitlines()
        if (r := json.loads(l))["kind"] == "reader"
    ]
    contexts, answers = [], []
    by_key = {}
    for call in calls:
        by_key.setdefault(call["cache_key"], []).append(call)
    by_question = {row["qa_index"]: row for row in rows}
    if len(rows) != len(by_question) or by_question.keys() != questions.keys():
        raise ValueError("Missing or duplicate native question")
    for i, q in questions.items():
        row = by_question[i]
        ctx = verified[i]
        prompt = ev.build_prompt(q["question"], [text for _, text in ctx], q)
        prompt_hash = sha(prompt)
        key = reader_key(model, prompt, max_tokens)
        if not by_key.get(key):
            raise ValueError(f"Missing physical reader request at {cell}, reader {i}")
        logged = by_key[key].pop()
        hit = read_response(cache, key)
        if not hit or hit["sha256"] != logged["response_sha256"]:
            raise ValueError("Physical response has changed")
        v = hit["payload"]
        score = ev.compute_f1(v["response"], ev.get_ground_truth(q), q)
        if (
            v["response"].strip() != row["prediction"]
            or v["tokens"] != row["tokens"]
            or v["prompt_sha256"] != prompt_hash
            or not math.isclose(score, row["f1"], abs_tol=1e-12)
        ):
            raise ValueError("Reader endpoint does not reproduce")
        contexts.append(ctx)
        answers.append(
            {
                "answer": v["response"],
                "tokens": v["tokens"],
                "f1": score,
                "prompt_sha256": prompt_hash,
                "key": key,
                "response_sha256": hit["sha256"],
            }
        )
    if any(by_key.values()):
        raise ValueError("Unmatched physical reader calls")
    return spans, contexts, answers


def endpoint(cell, cache, expected, qa, ev, model):
    values = measured_contexts(cell, cache, expected, dict(enumerate(qa)), ev, model)
    return *values, read(cell / "decisions.json")


def prepare(base, study, corpus, *, verify=False):
    from experiments.controller.capped_source_policy import baseline, joint_evidence
    from experiments.controller.capped_source_repair import BACKEND as EARLY, archive
    from experiments.controller.first_evidence_credit import BACKEND as DONOR

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
    from experiments.controller.capped_source_repair import BACKEND as EARLY, archive

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


def matched_reader_slots(target, lengths, *, removed=()):
    """Match disjoint slots by total absolute token-length difference, then rank distance."""
    from scipy.optimize import linear_sum_assignment

    ranks = {slot: rank for rank, slot in enumerate(i for i in range(len(lengths)) if i not in removed)}
    others = [i for i in ranks if i not in target]
    if len(others) < len(target):
        return None
    if not target:
        return []
    stride = len(lengths) ** 2 + 1
    cost = [[abs(lengths[i] - lengths[j]) * stride + abs(ranks[i] - ranks[j]) for j in others] for i in target]
    _, selected = linear_sum_assignment(cost)
    return [others[j] for j in selected]


def focal_reader_slots(context, pivot, exact_copies=False):
    """Find actual focal versions or all identical displayed texts of those versions."""
    direct = [j for j, (origin, _) in enumerate(context) if origin == pivot]
    if not exact_copies:
        return direct
    texts = {context[j][1] for j in direct}
    return [j for j, (_, text) in enumerate(context) if text in texts]


def ablation_answer(ctx, remove, q, ev, model, cache, generate):
    prompt = ev.build_prompt(q["question"], [text for j, (_, text) in enumerate(ctx) if j not in remove], q)
    return measured_prompt(prompt, q, ev, model, cache, generate)


def measured_prompt(prompt, q, ev, model, cache, generate, max_tokens=32):
    key = reader_key(model, prompt, max_tokens)
    if generate:
        measured_answer(prompt, model, q, ev, cache, max_tokens)
    receipt = read_response(cache, key)
    prompt_hash = sha(prompt)
    if not receipt or receipt["payload"]["prompt_sha256"] != prompt_hash:
        raise ValueError("Missing measured Reader response")
    payload = receipt["payload"]
    return {"answer": payload["response"], "tokens": payload["tokens"],
            "f1": float(ev.compute_f1(payload["response"], ev.get_ground_truth(q), q)),
            "key": key, "response_sha256": receipt["sha256"], "prompt_sha256": prompt_hash}


def paired_reader_repeats(prompts, ordinal, trial, question, evaluator, model, generate):
    arms = list(prompts)
    cut = ordinal % len(arms)
    rotated = arms[cut:] + arms[:cut]
    for repeat in range(2):
        cache = trial / f"cache-{repeat}"
        os.environ["MEMSKILL_USAGE_LOG"] = str(trial / f"usage-{repeat}.jsonl")
        order = rotated if repeat == 0 else reversed(rotated)
        yield {arm: measured_prompt(prompts[arm], question, evaluator, model, cache, generate) for arm in order}


def copy_contrasts(answers):
    f = {arm: row["f1"] for arm, row in answers.items()}
    present = f["full"] - f["no_focal"]
    absent = f["no_copies"] - f["neither"]
    focal = (present + absent) / 2
    joint = f["full"] - f["neither"]
    effects = {"focal_with_copies": present, "focal_without_copies": absent, "focal_order_average": focal,
               "copies_order_average": joint - focal, "joint": joint}
    if "no_control" in f:
        effects["copies_vs_control_given_no_focal"] = f["no_focal_control"] - f["neither"]
        effects["interaction_vs_control"] = absent - (f["no_control"] - f["no_focal_control"])
    return effects


def measure_copy_interaction(version_dir, expected, protocol, sid, pivot, questions, contexts, ev, model, verify):
    exact = version_dir / "exact-text"
    out = version_dir / "copy-interaction"
    if verify and not (out / "summary.json").is_file():
        raise ValueError("No completed copy interaction to verify")
    out.mkdir(exist_ok=True)
    cache = version_dir / "greedy-cache"
    anchors = []
    for directory in (version_dir, exact):
        summary = read(directory / "summary.json")
        plan = read(directory / "plan.json")
        if (summary["plan_sha256"] != file_hash(directory / "plan.json") or plan["native"] != expected
                or summary["scores_sha256"] != file_hash(directory / "scores.json")):
            raise ValueError("Reader interaction inputs differ from the measured native and copy deletions")
        anchors.append((plan, {r["qa_index"]: r["answers"] for r in read(directory / "scores.json")}))
    if anchors[1][0]["version_summary_sha256"] != file_hash(version_dir / "summary.json"):
        raise ValueError("Copy deletion used a different focal version measurement")
    design = {}
    for (i, q), ctx in zip(questions.items(), contexts, strict=True):
        direct = focal_reader_slots(ctx, pivot)
        target = focal_reader_slots(ctx, pivot, True)
        original, copied = (plan["rows"][str(i)] for plan, _ in anchors)
        if (original["target"] != direct or copied["target"] != target or original["lengths"] != copied["lengths"]
                or anchors[0][1][i]["repeat"] != anchors[1][1][i]["repeat"]):
            raise ValueError("Comparison does not share the same native Reader context")
        extra = [j for j in target if j not in direct]
        if extra:
            control = matched_reader_slots(extra, original["lengths"], removed=direct)
            design[i] = {"focal": direct, "copies": extra, "control": control}
    register(out / "plan.json", {"protocol": protocol, "history": sid, "span": pivot, "questions": len(questions),
             "version_summary_sha256": file_hash(version_dir / "summary.json"), "copy_summary_sha256": file_hash(exact / "summary.json"),
             "rows": design})
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "usage.jsonl")
    generate = not verify and not (out / "scores.json").is_file()
    rows = []
    for (i, q), ctx in zip(questions.items(), contexts, strict=True):
        if i not in design:
            continue
        focal, copies, control = (design[i][k] for k in ("focal", "copies", "control"))
        answers = {}
        for arm, remove, anchor in (("full", [], anchors[0][1][i]["repeat"]),
                                    ("no_focal", focal, anchors[0][1][i]["target"]),
                                    ("neither", focal + copies, anchors[1][1][i]["target"])):
            answers[arm] = ablation_answer(ctx, remove, q, ev, model, cache, False)
            if answers[arm] != anchor:
                raise ValueError("Measured factorial endpoint differs from the previous deletion arm")
        arms = [("no_copies", copies)]
        if control is not None:
            arms.extend([("no_control", control), ("no_focal_control", focal + control)])
        start = len(rows) % len(arms)
        for arm, remove in arms[start:] + arms[:start]:
            answers[arm] = ablation_answer(ctx, remove, q, ev, model, cache, generate)
        rows.append({"qa_index": i, "answers": answers, "effects": copy_contrasts(answers)})
    register(out / "scores.json", rows)
    effects = {}
    for name in sorted({k for row in rows for k in row["effects"]}):
        values = [r["effects"][name] for r in rows if name in r["effects"]]
        # Subtracting paired F1 differences can leave roundoff on a mathematical tie.
        positive = sum(v > 1e-12 for v in values)
        negative = sum(v < -1e-12 for v in values)
        effects[name] = {"questions": len(values), "points": 100 * mean(values),
                         "positive": positive, "negative": negative, "zero": len(values) - positive - negative}
    register(out / "summary.json", {"history": sid, "span": pivot, "questions": len(questions),
             "expanded_questions": len(rows), "matched_questions": sum(r["control"] is not None for r in design.values()),
             "effects": effects, "plan_sha256": file_hash(out / "plan.json"), "scores_sha256": file_hash(out / "scores.json")})


def writer_read_paths(spans, contexts, origins, device):
    """Shortest path via exact versions displayed to each writer and Reader."""
    import torch

    n = len(spans)
    if [e["span_id"] for e in spans] != list(range(n)) or any(not 0 <= i < n for i in origins):
        raise ValueError("Expected a complete ordered sequence of writer spans")
    produced = {}
    distance = torch.full((len(origins), n + 1), n + 1, dtype=torch.int32, device=device)
    distance[range(len(origins)), origins] = 0
    for j, event in enumerate(spans):
        parents = sorted({produced[tuple(ref)] for ref in event["refs"]})
        if parents:
            through_writer = 1 + distance[:, parents].amin(dim=1)
            distance[:, j] = torch.minimum(distance[:, j], through_writer)
        for change in event["changes"]:
            produced[change["memory_id"], change["version"]] = j
    # The last column denotes an empty context or padding. Its distance stays infinite.
    width = max(1, max(map(len, contexts.values())))
    indices = [[k for k, text in ctx] + [n] * (width - len(ctx)) for ctx in contexts.values()]
    closest = distance[:, torch.tensor(indices, device=device)].amin(dim=-1).tolist()
    return {str(origin): {str(i): (h if h <= n else None) for i, h in zip(contexts, row, strict=True)}
            for origin, row in zip(origins, closest, strict=True)}


def matched_path_placement(hops, eligible, switched, questions, total):
    """Expected distinct questions reached by uniform switches within chronological quarters."""
    quarters = [[j for j in eligible if 4 * j // total == b] for b in range(4)]
    draws = [len(set(q) & switched) for q in quarters]
    coverage = {}
    for name, maximum in (("direct", 0), ("one_or_less", 1), ("any", total)):
        observed, expected = [], []
        for i in questions:
            reachable = {j for j in eligible if (h := hops[str(j)][str(i)]) is not None and h <= maximum}
            if reachable & switched:
                observed.append(i)
            miss = math.prod(math.comb(len(q) - len(reachable.intersection(q)), k) / math.comb(len(q), k)
                             for q, k in zip(quarters, draws, strict=True))
            expected.append(1 - miss)
        coverage[name] = {"observed_questions": observed, "expected_count": math.fsum(expected)}
    return {"eligible_per_quarter": list(map(len, quarters)), "switches_per_quarter": draws, "coverage": coverage}


def measure_writer_paths(native, output, sid, questions, cell, spec, *, swaps=None, projection_study=None, verify=False):
    from experiments.controller.paper_f1 import question_scores

    folder = native / "locomo/native" / sid
    target = output / sid / "writer-paths.json"
    if verify and not target.is_file():
        raise ValueError("No completed writer path study to verify")
    trace, = (folder / "traces").glob("*.jsonl")
    events = [json.loads(line) for line in trace.open()]
    spans = [e for e in events if e["kind"] == "span"]
    scores = folder / "scores.json"
    contexts = verified_reader_contexts(events, read(scores)["answers"], questions)
    if len(spans) != cell["total_spans"] or contexts.keys() != questions.keys():
        raise ValueError("Incomplete writer or Reader cohort")
    pivot = cell["span"]
    origins = [pivot - 1, pivot, pivot + 1]
    all_hops = writer_read_paths(spans, contexts, list(range(len(spans))) if projection_study is not None else origins, "cuda")
    hops = {str(o): all_hops[str(o)] for o in origins}
    linked = set(cell["linked_question_indices"])
    if linked != {i for i, q in questions.items() if set(q.get("evidence") or []) & set(spans[pivot]["source_turns"])}:
        raise ValueError("Native pivot and registered evidence links differ")
    def group(h):
        return "unreachable" if h is None else "direct" if h == 0 else "one" if h == 1 else "longer"
    groups = {name: [i for i in questions if group(hops[str(pivot)][str(i)]) == name]
              for name in ("direct", "one", "longer", "unreachable")}
    target.parent.mkdir(parents=True, exist_ok=True)
    register(target, {"history": sid, "split": "train", "pivot": pivot, "protocol": spec["native_writer_read_paths"],
                      "trace_sha256": file_hash(trace), "scores_sha256": file_hash(scores), "questions": len(questions),
                      "spans": len(spans), "hops": hops, "groups": groups,
                      "linked_in_groups": {k: sorted(linked.intersection(v)) for k, v in groups.items()}})
    if projection_study is not None:
        import torch
        from experiments.controller.paper_f1 import read_locked_json
        proof = read(projection_study / f"{sid}-upstream-verification.json")
        run = read_locked_json(projection_study / f"{sid}.json", proof["report_sha256"])
        rule = spec["forward_rank_boundary"] | spec["forward_rank_skill_projection"]
        native_protocol = read(native / "locomo/protocol.json")
        if (file_hash(projection_study / "frozen-weight.json") != proof["weight_sha256"]
                or run["checkpoint_sha256"] != native_protocol["checkpoint"] or run["method"] != rule or proof["upstream"] != spec["upstream_revision"] or run["history"] != sid
                or proof["verified_frozen_forwards"] != len(spans) or run["input_sha256"] != file_hash(folder / "controller-inputs.npz")
                or run["selected_rank"] is None or proof["weight_sha256"] != run["weight_sha256"]):
            raise ValueError("Expected the author-verified frozen skill projection on these native states")
        outgoing, incoming = (spec["state_credit"][k] for k in ("outgoing", "incoming"))
        eligible, switched, other = [], set(), []
        names = run["candidate_order"]
        k = native_protocol["settings"]["action_top_k"]
        logits = torch.tensor([e["policy"]["logits"] for e in spans], dtype=torch.float32, device="cuda")
        next_ids = logits.topk(k + 1).indices[:, -1].tolist()
        for j, (event, row, eighth) in enumerate(zip(spans, run["rows"], next_ids, strict=True)):
            original = [op["name"] for op in event["selected_skills"]]
            changed = row["ordered_skills"]
            if row["span"] != j or event["policy"]["candidates"] != names:
                raise ValueError("The projected and native states use different candidate rows")
            if original[-1] == outgoing and names[eighth] == incoming:
                eligible.append(j)
                if changed == original[:-1] + [incoming]:
                    switched.add(j)
            if changed != original and j not in switched:
                other.append(j)
        placement = matched_path_placement(all_hops, eligible, switched, questions, len(spans))
        register(target.with_name("writer-path-placement.json"), {**placement, "history": sid, "rank": run["selected_rank"],
                 "protocol": spec["native_rank_path_placement"], "weight_sha256": proof["weight_sha256"],
                 "paths_sha256": file_hash(target), "rank_report_sha256": proof["report_sha256"],
                 "eligible": eligible, "switched": sorted(switched), "other_changes": other})
    if swaps is not None:
        deltas = {}
        subset = {(sid, i): q for i, q in questions.items()}
        for repeat in range(1, spec["training_boundary_swap"]["independent_repeats"] + 1):
            root = swaps / sid / f"span-{pivot}" / "repeats" / str(repeat) / "locomo"
            mpath = root / "swap" / sid / "measurement.json"
            m = read(mpath)
            arm_files = [root / arm / sid / "scores.json" for arm in ("native", "swap")]
            if ((m["sample_id"], m["span_id"], m["repeat"]) != (sid, pivot, repeat) or not m["registered_span"]
                    or m["state_sha256"] != spans[pivot]["policy"]["state_sha256"]
                    or [file_hash(f) for f in arm_files] != [m["native_scores_sha256"], m["artifacts"]["scores.json"]]):
                raise ValueError("The measured swap differs from the registered native pivot")
            left, right = [question_scores("locomo", subset, [f]) for f in arm_files]
            effect = {str(i): right[sid, i] - left[sid, i] for i in questions}
            if effect != m["f1_deltas"]:
                raise ValueError("Question contrasts differ from the measured paired answers")
            deltas[str(repeat)] = {"measurement_sha256": file_hash(mpath), "points": {
                k: 100 * mean(effect[str(i)] for i in ids) if ids else None for k, ids in groups.items()}}
        register(target.with_name("writer-path-outcomes.json"), {"paths_sha256": file_hash(target), "repeats": deltas})


def competing_probes(trial, sample, cell, questions, linked, spans, contexts, archived, base, ev,
                     model, tokenizer, spec, verify):
    """Compare repairs at a fixed Reader context for the evidence-linked training panel."""
    from src.credit_assignment import hidden_retrieval_candidate
    from src.episode_trace import evidence_turns

    trial.mkdir(parents=True, exist_ok=True)
    conversation = sample["conversation"]
    turns = [dict(t, session_date=conversation[key + "_date_time"])
             for key, session in conversation.items() if key.startswith("session_") and isinstance(session, list)
             for t in session]
    def format_turn(t):
        return f"{t['session_date']} | {t['speaker']}: {t['text']}"

    by_id = {t["dia_id"]: t for t in turns}
    live = restore_text(spans)
    bank = list(live)
    texts = list(dict.fromkeys([*live.values(), *(format_turn(t) for t in turns)]))
    tokens = tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
    lengths = {t: len(ids) for t, ids in zip(texts, tokens, strict=True)}
    by_index = {r["qa_index"]: r for r in read(base / "scores.json")["answers"]}
    native = dict(zip(questions, zip(contexts, archived, strict=True), strict=True))
    probes, records, prompts = {}, {}, {}
    for i in linked:
        q = questions[i]
        evidence = evidence_turns(conversation, q["evidence"])
        if {t["dia_id"] for t in evidence} != set(q["evidence"]):
            raise ValueError(f"Missing annotated source for {sample['sample_id']}/{i}")
        ctx, observed = native[i]
        refs = [bank[j] for j in by_index[i]["retrieved_indices"]]
        visible = [t for _, t in ctx]
        if visible != [live[m] for m in refs] or len(visible) != 20:
            raise ValueError("Repair requires the original ordered top-20 context")
        chosen = hidden_retrieval_candidate(q, evidence, refs, live)
        if chosen is None:
            raise ValueError("No omitted live memory available")
        hidden = set(visible) | {live[chosen]}
        foil = min((m for m in bank if live[m] not in hidden), key=lambda m: abs(lengths[live[m]] - lengths[live[chosen]]))
        excluded = set(q["evidence"])
        foils = []
        for target in evidence:
            matched = min((t for t in turns if t["dia_id"] not in excluded), key=lambda t: (
                t["session_date"] != target["session_date"], t["speaker"] != target["speaker"],
                abs(lengths[format_turn(t)] - lengths[format_turn(target)]), t["dia_id"]))
            foils.append(matched["dia_id"])
            excluded.add(matched["dia_id"])
        replacements = {"retrieval": live[chosen], "retrieval_sham": live[foil],
                        "source": "\n".join(map(format_turn, evidence)),
                        "source_sham": "\n".join(format_turn(by_id[k]) for k in foils)}
        if spec.get("query_only"):
            blind = hidden_retrieval_candidate({"question": q["question"]}, [], refs, live)
            excluded_texts = set(visible) | {live[blind]}
            control = min((m for m in bank if live[m] not in excluded_texts),
                          key=lambda m: abs(lengths[live[m]] - lengths[live[blind]]))
            replacements.update(query_only=live[blind], query_only_sham=live[control])
        original_prompt = ev.build_prompt(q["question"], visible, q)
        if sha(original_prompt) != observed["prompt_sha256"]:
            raise ValueError("Native prompt does not match original service receipt")
        variants = {"native": original_prompt,
                    **{k: ev.build_prompt(q["question"], visible[:-1] + [v], q) for k,v in replacements.items()},
                    **{k: v + "\n" + original_prompt for k,v in spec["reader_instructions"].items()}}
        prompts[i] = variants
        probes[i] = {"archived": observed, "refs": refs, "inserted_ref": chosen, "foil_ref": foil,
                     "evidence": [t["dia_id"] for t in evidence], "foil_turns": foils,
                     "prompt_sha256": {k: sha(v) for k,v in variants.items()}}
        if spec.get("query_only"):
            probes[i].update(query_only_ref=blind, query_only_foil=control)
    reg = {"protocol": spec, "model": model, "span": cell["span"], "native_trace_sha256": file_hash(next((base/"traces").glob("*.jsonl"))),
           "native_scores_sha256": file_hash(base/"scores.json"), "rows": {str(i): p for i,p in probes.items()}}
    register(trial/"plan.json", reg)
    generate = not verify and not (trial/"scores.json").exists()
    for ordinal, i in enumerate(linked):
        repeats = list(paired_reader_repeats(prompts[i], ordinal, trial, questions[i], ev, model, generate))
        controls = {"retrieval": "retrieval_sham", "source":"source_sham", "reader":"reader_sham"}
        if spec.get("query_only"):
            controls["query_only"] = "query_only_sham"
        supported = [arm for arm, sham in controls.items()
                     if all(a[arm]["f1"] == 1 and max(a["native"]["f1"], a[sham]["f1"]) < 1 for a in repeats)]
        records[str(i)] = {"repeats": repeats, "supported_repairs": supported}
        print(f"{sample['sample_id']} question {i}: complete", flush=True)
    register(trial/"scores.json", records)
    register(trial/"summary.json", {"history":sample["sample_id"], "questions":len(linked),
             "plan_sha256":file_hash(trial/"plan.json"), "scores_sha256":file_hash(trial/"scores.json"),
             "patterns":dict(Counter("+".join(r["supported_repairs"]) or "unresolved" for r in records.values()))})


def registered_swap_contexts(swapped, native_spans, cell, questions, cache, ev, model):
    from src.replay import verify_native_swap

    meta = read(swapped / "measurement.json")
    trace, = (swapped / "traces").glob("*.jsonl")
    expected = archive_digests(swapped)
    pivot = cell["span"]
    if (not meta["registered_span"] or meta["sample_id"] != swapped.name or meta["span_id"] != pivot
            or meta["questions"] != len(questions) or meta["matched_prefix_spans"] != pivot
            or meta["artifacts"]["scores.json"] != expected["scores"]
            or meta["artifacts"][str(trace.relative_to(swapped))] != expected["trace"]):
        raise ValueError("Swap measurement disagrees with the registered cohort")
    data = measured_contexts(swapped, cache, expected, questions, ev, model)
    verify_native_swap(native_spans, data[0], pivot)
    return meta, expected, data


def training_swap_slots(trial, native_data, swapped, questions, cell, cache, model, ev, protocol, verify, repair=None):
    """Cross focal-version ranks with remaining ranks in complete real training branches."""
    native_spans, native_ctx, native_answers = native_data
    marker = swapped / "measurement.json"
    pivot = cell["span"]
    meta, _, changed_data = registered_swap_contexts(swapped, native_spans, cell, questions, cache, ev, model)
    changed, swapped_ctx, swapped_answers = changed_data
    banks = [set(restore_text(events).values()) for events in (native_spans, changed)]
    plan, panel = {}, {}
    arms = ("native", "swap", "direct_in_native", "native_in_swap")
    for i, left, right, original, alternate in zip(questions, native_ctx, swapped_ctx, native_answers, swapped_answers, strict=True):
        if len(left) != 20 or len(right) != 20:
            raise ValueError("Expected complete top-20 contexts on both branches")
        tags, hybrids = slots(left, right, banks, pivot)
        layouts = [[text for _, text in left], [text for _, text in right], *hybrids]
        prompts = {a: ev.build_prompt(questions[i]["question"], ctx, questions[i]) for a, ctx in zip(arms, layouts, strict=True)}
        prompt_hashes = {a: sha(p) for a, p in prompts.items()}
        if ([prompt_hashes[a] for a in arms[:2]] != [original["prompt_sha256"], alternate["prompt_sha256"]]
                or not math.isclose(alternate["f1"] - original["f1"], meta["f1_deltas"][str(i)], abs_tol=1e-12)):
            raise ValueError("Archived endpoints differ from real swap")
        plan[str(i)] = {"tags": tags, "prompt_sha256": prompt_hashes}
        panel[i] = (prompts, original, alternate)
    trial.mkdir(parents=True, exist_ok=True)
    if repair is not None:
        sample, original_dir, tokenizer, design = repair
        selected = {"native": [], "swap": []}
        missing = []
        for i, (_, original, alternate) in panel.items():
            if {original["f1"], alternate["f1"]} != {0.0, 1.0} or "direct" in plan[str(i)]["tags"]:
                continue
            if not questions[i].get("evidence"):
                missing.append(i)
                continue
            selected["native" if original["f1"] == 0 else "swap"].append(i)
        register(trial/"plan.json", {"protocol": design, "swap_sha256": file_hash(marker), "selected": selected,
                                     "missing_evidence": missing, "cohort": len(panel)})
        receipts = {}
        for arm, directory, data in (("native", original_dir, native_data), ("swap", swapped, (changed, swapped_ctx, swapped_answers))):
            if selected[arm]:
                competing_probes(trial/arm, sample, cell, questions, selected[arm], *data, directory, ev, model, tokenizer, design, verify)
                receipts[arm] = file_hash(trial/arm/"scores.json")
        register(trial/"summary.json", {"plan_sha256": file_hash(trial/"plan.json"), "receipts": receipts})
        return
    register(trial/"plan.json", {"protocol": protocol, "swap_sha256": file_hash(marker), "model": model, "questions": plan})
    generate = not verify and not (trial/"scores.json").exists()
    rows = {}
    for ordinal, (i, (prompts, original, alternate)) in enumerate(panel.items()):
        tags = plan[str(i)]["tags"]
        mixed = "direct" in tags and "other" in tags
        observed = []
        if mixed:
            responses = paired_reader_repeats(prompts, ordinal, trial, questions[i], ev, model, generate)
        else:
            endpoints = {a["prompt_sha256"]: a for a in (original, alternate)}
            responses = [{a: endpoints[sha(prompts[a])] for a in arms}]
        for answers in responses:
            low, high, left, right = [answers[a]["f1"] for a in arms]
            direct, _ = components(low, high, (left, right), 0)
            observed.append({"answers": answers, "direct": direct, "other": high - low - direct,
                             "interaction": (high - right) - (left - low)})
        rows[str(i)] = {"mixed": mixed, "archived_delta": alternate["f1"] - original["f1"], "observed": observed}
    register(trial/"scores.json", rows)
    groups = {"all": list(rows), "mixed": [i for i,r in rows.items() if r["mixed"]],
              "direct_only": [i for i in rows if set(plan[i]["tags"]) - {"same"} == {"direct"}],
              "other_only": [i for i in rows if set(plan[i]["tags"]) - {"same"} == {"other"}],
              "same": [i for i in rows if set(plan[i]["tags"]) == {"same"}]}
    def describe(ids):
        if not ids:
            return {"questions": 0}
        return {"questions": len(ids), "archived_delta": 100*mean(rows[i]["archived_delta"] for i in ids),
                **{key: 100*mean(mean(v[key] for v in rows[i]["observed"]) for i in ids)
                   for key in ("direct", "other", "interaction")}}
    register(trial/"summary.json", {"history": meta["sample_id"], "span": pivot, "groups": {k: describe(v) for k,v in groups.items()},
                                    "plan_sha256": file_hash(trial/"plan.json"), "scores_sha256": file_hash(trial/"scores.json")})


def paper_reader_ablation(native, output, corpus, sid, *, verify=False, exact_copies=False, copy_interaction=False, writer_paths=False, swaps=None, projection_study=None, repairs=False, swap_slots=False, downstream=False):
    from transformers import AutoTokenizer
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort

    spec = read(PROTOCOL)
    cell = spec["training_boundary_swap"]["histories"][sid]
    meta = read(native / "locomo/protocol.json")
    setting = spec["datasets"]["locomo"]
    samples, training = load_split_cohort("locomo", corpus, setting, "train")
    questions = {i: q for (sample, i), q in training.items() if sample == sid}
    if (meta["split"] != "train" or meta["question_keys"] != setting["development_splits"]["train"]["scored_key_sha256"]
            or meta["backend"] != os.environ["MEMSKILL_BACKEND_ID"] or meta["encode_batch_size"] != 64
            or meta["model"]["model"] != "meta-llama/Llama-3.1-8B-Instruct" or len(questions) != cell["questions"]):
        raise ValueError("Reader diagnostic requires the locked native training run")
    if writer_paths:
        return measure_writer_paths(native, output, sid, questions, cell, spec, swaps=swaps, projection_study=projection_study, verify=verify)
    model = meta["model"]["model"]
    tokenizer_dir = Path(os.environ["MEMSKILL_MODEL_MANIFEST"]).parent
    if (read(tokenizer_dir / "verified-model.json") != meta["model"]
            or file_hash(tokenizer_dir / "tokenizer.json") != meta["model"]["sha256"]["tokenizer.json"]):
        raise ValueError("Reader tokenizer or model revision differs from the native run")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    cache = native / "greedy-cache"
    version_trial = output / sid
    trial = version_trial / "exact-text" if exact_copies else version_trial
    if verify and not (repairs or swap_slots or downstream) and not all((trial / name).is_file() for name in ("plan.json", "scores.json", "summary.json")):
        raise ValueError("The archived Reader comparison is incomplete")
    trial.mkdir(parents=True, exist_ok=True)
    fresh = version_trial / "greedy-cache"
    if fresh.resolve() == cache.resolve():
        raise ValueError("The native repeat requires an independent request cache")
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(fresh)
    base = native / "locomo/native" / sid
    trace, = (base / "traces").glob("*.jsonl")
    expected = {"trace": file_hash(trace), "scores": file_hash(base / "scores.json")}
    ev = LoCoMoEvaluator(SimpleNamespace())
    spans, contexts, archived = measured_contexts(base, cache, expected, questions, ev, model)
    pivot = cell["span"]
    turns = set(spans[pivot]["source_turns"])
    linked = [i for i, q in questions.items() if turns.intersection(q.get("evidence") or [])]
    if (len(spans) != cell["total_spans"] or spans[pivot]["span_id"] != pivot
            or sha(spans[pivot]["source"]) != cell["source_sha256"] or linked != cell["linked_question_indices"]):
        raise ValueError("Focal span or evidence question registration differs")
    if swap_slots or downstream:
        trial = version_trial / ("downstream-repairs" if downstream else "swap-slots")
        repair = None
        if downstream:
            sample, = [s for s in samples if s["sample_id"] == sid]
            design = dict(spec["downstream_swap_repairs"], reader_instructions=spec["competing_repair_panel"]["reader_instructions"])
            repair = (sample, base, tokenizer, design)
        swapped = swaps / sid / f"span-{pivot}" / "locomo" / "swap" / sid
        return training_swap_slots(trial, (spans, contexts, archived), swapped, questions, cell,
                                   cache, model, ev, spec["training_swap_reader_slots"], verify, repair)
    if copy_interaction:
        return measure_copy_interaction(version_trial, expected, spec["native_reader_copy_interaction"], sid,
                                        pivot, questions, contexts, ev, model, verify)
    rows = {}
    protocol = spec["native_reader_ablation"]
    if repairs:
        sample, = [s for s in samples if s["sample_id"] == sid]
        return competing_probes(output / sid / "competing-repairs", sample, cell, questions, linked, spans, contexts,
                                archived, base, ev, model, tokenizer, spec["competing_repair_panel"], verify)
    links = {}
    if exact_copies:
        original_plan = read(version_trial / "plan.json")
        original_result = read(version_trial / "summary.json")
        if (original_plan["protocol"] != protocol or original_plan["native"] != expected
                or original_result["plan_sha256"] != file_hash(version_trial / "plan.json")
                or original_result["scores_sha256"] != file_hash(version_trial / "scores.json")):
            raise ValueError("Exact-copy measurement requires the completed native version ablation")
        links["version_summary_sha256"] = file_hash(version_trial / "summary.json")
        previous = {r["qa_index"]: r for r in read(version_trial / "scores.json")}
        protocol = spec["native_reader_copy_ablation"]
    expanded = []
    for i, ctx, orig in zip(questions, contexts, archived, strict=True):
        target = focal_reader_slots(ctx, pivot, exact_copies)
        if exact_copies:
            direct = focal_reader_slots(ctx, pivot)
            if original_plan["rows"][str(i)]["target"] != direct:
                raise ValueError("Original Reader exposure differs from the native trace")
            if len(target) > len(direct):
                expanded.append(i)
        lengths = [len(ids) for ids in tokenizer([text for _, text in ctx], add_special_tokens=False)["input_ids"]]
        rows[i] = {"target": target, "sham": matched_reader_slots(target, lengths), "lengths": lengths,
                   "linked": i in linked, "archived": orig}
    register(trial / "plan.json", {"protocol": protocol, "native": expected, **links,
             "backend": meta["backend"], "history": sid, "span": pivot, "rows": rows})
    os.environ["MEMSKILL_USAGE_LOG"] = str(trial / "usage.jsonl")
    rows_out = []
    generate = not verify and not (trial / "scores.json").is_file()
    arms = ["repeat", "target", "sham"]
    for ordinal, ((i, q), ctx) in enumerate(zip(questions.items(), contexts, strict=True)):
        record = rows[i]
        answers = {"archived": record["archived"]}
        for arm in arms[ordinal % 3:] + arms[:ordinal % 3]:
            remove = [] if arm == "repeat" else record[arm]
            if remove is None:
                continue
            if not record["target"]:
                answers[arm] = record["archived"]
                continue
            answers[arm] = ablation_answer(ctx, remove, q, ev, model, fresh, generate)
            if arm == "repeat" and answers[arm]["key"] != record["archived"]["key"]:
                raise ValueError("Native repeat request changed")
        rows_out.append({"qa_index": i, "linked": record["linked"], "direct": bool(record["target"]), "answers": answers})
    register(trial / "scores.json", rows_out)
    groups = {"all": rows_out, "direct": [r for r in rows_out if r["direct"]],
              "linked": [r for r in rows_out if r["linked"]],
              "direct_unlinked": [r for r in rows_out if r["direct"] and not r["linked"]]}
    if exact_copies:
        groups["extra_copies"] = [r for r in rows_out if r["qa_index"] in expanded]
    summary = {}
    for name, group in groups.items():
        matched = [r for r in group if "sham" in r["answers"]]
        summary[name] = {"questions": len(group), "matched_questions": len(matched),
                         "f1": {a: 100 * mean(r["answers"][a]["f1"] for r in group) for a in ("archived", "repeat", "target")} if group else {},
                         "matched_f1": {a: 100 * mean(r["answers"][a]["f1"] for r in matched) for a in arms} if matched else {}}
        if exact_copies:
            diffs = []
            for row in group:
                old = previous[row["qa_index"]]["answers"]
                if old["repeat"] != row["answers"]["repeat"]:
                    raise ValueError("Copy and version arms must share the same native Reader response")
                diffs.append(float(row["answers"]["target"]["f1"]) - old["target"]["f1"])
            summary[name]["copy_minus_version"] = {
                "points": 100 * mean(diffs) if diffs else None,
                "higher": sum(v > 0 for v in diffs), "equal": sum(v == 0 for v in diffs), "lower": sum(v < 0 for v in diffs)
            }
    register(trial / "summary.json", {"history": sid, "span": pivot, "groups": summary,
             "plan_sha256": file_hash(trial / "plan.json"), "scores_sha256": file_hash(trial / "scores.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("study", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("--read", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--paper-history", help="Measure the preassigned training span using a native run, output directory and corpus")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--downstream-repairs", action="store_true", help="Diagnose weaker branches of full F1 reversals outside directly displayed focal writes")
    mode.add_argument("--swap-slots", action="store_true", help="Cross the directly written and other retrieved ranks in a measured full-history training swap")
    mode.add_argument("--competing-repairs", action="store_true", help="Test retrieval, raw evidence and Reader controls on linked training questions")
    mode.add_argument("--copy-interaction", action="store_true", help="Cross focal-version, identical-copy and matched background deletions")
    mode.add_argument("--exact-copies", action="store_true", help="Also remove exact text copies from other source spans in a completed paper history")
    mode.add_argument("--writer-paths", action="store_true", help="Measure paths through versioned writer and Reader inputs")
    p.add_argument("--paired-swaps", type=Path, help="Group both completed swap repeats by frozen native writer paths")
    p.add_argument("--projection-study", type=Path, help="Compare frozen switch placement with a time- and count-matched control")
    a = p.parse_args()
    if (a.swap_slots or a.downstream_repairs) and not (a.paper_history and a.paired_swaps):
        p.error("--swap-slots requires --paper-history and --paired-swaps")
    if (a.writer_paths and not a.paper_history) or (a.paired_swaps and not (a.writer_paths or a.swap_slots or a.downstream_repairs)) or (a.projection_study and not a.writer_paths):
        p.error("--writer-paths requires --paper-history; --paired-swaps and --projection-study require --writer-paths")
    if (a.exact_copies or a.copy_interaction or a.competing_repairs) and not a.paper_history:
        p.error("Copy measurements require --paper-history")
    if a.paper_history:
        paper_reader_ablation(a.base, a.study, a.corpus, a.paper_history, verify=a.verify, exact_copies=a.exact_copies, copy_interaction=a.copy_interaction,
                             writer_paths=a.writer_paths, swaps=a.paired_swaps, projection_study=a.projection_study, repairs=a.competing_repairs, swap_slots=a.swap_slots, downstream=a.downstream_repairs)
        raise SystemExit(0)
    o = prepare(a.base, a.study, a.corpus, verify=a.verify)
    if a.read:
        with (o / "reader.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            measured_hybrids(a.base, o, a.corpus)
