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


def measured_contexts(cell, cache, expected, questions, ev, model):
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
        key = reader_key(model, prompt)
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
    key = reader_key(model, prompt)
    if generate:
        measured_answer(prompt, model, q, ev, cache)
    receipt = read_response(cache, key)
    prompt_hash = sha(prompt)
    if not receipt or receipt["payload"]["prompt_sha256"] != prompt_hash:
        raise ValueError("Missing measured Reader response")
    payload = receipt["payload"]
    return {"answer": payload["response"], "tokens": payload["tokens"],
            "f1": float(ev.compute_f1(payload["response"], ev.get_ground_truth(q), q)),
            "key": key, "response_sha256": receipt["sha256"], "prompt_sha256": prompt_hash}


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


def paper_reader_ablation(native, output, corpus, sid, *, verify=False, exact_copies=False, copy_interaction=False):
    from transformers import AutoTokenizer
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort

    spec = read(PROTOCOL)
    cell = spec["training_boundary_swap"]["histories"][sid]
    meta = read(native / "locomo/protocol.json")
    setting = spec["datasets"]["locomo"]
    _, training = load_split_cohort("locomo", corpus, setting, "train")
    questions = {i: q for (sample, i), q in training.items() if sample == sid}
    if (meta["split"] != "train" or meta["question_keys"] != setting["development_splits"]["train"]["scored_key_sha256"]
            or meta["backend"] != os.environ["MEMSKILL_BACKEND_ID"] or meta["encode_batch_size"] != 64
            or meta["model"]["model"] != "meta-llama/Llama-3.1-8B-Instruct" or len(questions) != cell["questions"]):
        raise ValueError("Reader diagnostic requires the locked native training run")
    model = meta["model"]["model"]
    tokenizer_dir = Path(os.environ["MEMSKILL_MODEL_MANIFEST"]).parent
    if (read(tokenizer_dir / "verified-model.json") != meta["model"]
            or file_hash(tokenizer_dir / "tokenizer.json") != meta["model"]["sha256"]["tokenizer.json"]):
        raise ValueError("Reader tokenizer or model revision differs from the native run")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    cache = native / "greedy-cache"
    version_trial = output / sid
    trial = version_trial / "exact-text" if exact_copies else version_trial
    if verify and not all((trial / name).is_file() for name in ("plan.json", "scores.json", "summary.json")):
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
    if copy_interaction:
        return measure_copy_interaction(version_trial, expected, spec["native_reader_copy_interaction"], sid,
                                        pivot, questions, contexts, ev, model, verify)
    rows = {}
    protocol = spec["native_reader_ablation"]
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
    mode.add_argument("--copy-interaction", action="store_true", help="Cross focal-version, identical-copy and matched background deletions")
    mode.add_argument("--exact-copies", action="store_true", help="Also remove exact text copies from other source spans in a completed paper history")
    a = p.parse_args()
    if (a.exact_copies or a.copy_interaction) and not a.paper_history:
        p.error("Copy measurements require --paper-history")
    if a.paper_history:
        paper_reader_ablation(a.base, a.study, a.corpus, a.paper_history, verify=a.verify, exact_copies=a.exact_copies, copy_interaction=a.copy_interaction)
        raise SystemExit(0)
    o = prepare(a.base, a.study, a.corpus, verify=a.verify)
    if a.read:
        with (o / "reader.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            measured_hybrids(a.base, o, a.corpus)
