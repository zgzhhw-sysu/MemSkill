"""Separate generated writes from source-format writes in two-skill LoCoMo training trajectories."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.controller.audit_source_compression import panel
from experiments.controller.conditional_credit import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.writer_granularity import measured_answer
from src.credit_assignment import _digest
from src.eval import get_evaluator
from src.evaluation_cache import encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn
from src.replay import restore_text


CELLS = ("00", "01", "10", "11")


def trace_exposure(events, ids, questions, evaluator):
    spans = [e for e in events if e["kind"] == "span"]
    bank = restore_text(spans)
    (snapshot,) = (e for e in events if e["kind"] == "snapshot")
    versions, written = {}, defaultdict(set)
    for span in spans:
        for c in span["changes"]:
            mid, v = c["memory_id"], c["version"]
            versions[mid] = v
            if c["after"] is not None:
                written[span["span_id"]].add((mid, v))
    final = [(mid, versions[mid]) for mid in bank]
    if snapshot["text_sha256"] != _digest(list(bank.items())) or snapshot["memory_refs"] != list(
        map(list, final)
    ):
        raise ValueError("Final snapshot disagrees with write versions")
    factual = {}
    for q in (e for e in events if e["kind"] == "qa"):
        i = int(q["query_id"].rsplit("_", 1)[1])
        if not 0 <= i < len(ids):
            raise ValueError("Unknown factual question index")
        qa = questions[ids[i]]
        refs = list(map(tuple, q["refs"]))
        if (
            i in factual
            or not q["success"]
            or q["question"] != qa["question"]
            or len(refs) != len(set(refs))
            or not set(refs).issubset(final)
            or not math.isclose(
                q["f1"], evaluator.compute_f1(q["prediction"], evaluator.get_ground_truth(qa), qa)
            )
        ):
            raise ValueError("Question, read versions or factual F1 changed")
        factual[i] = {
            "refs": refs,
            "context_hash": _digest([bank[mid] for mid, _ in refs]),
            "answer": q["prediction"],
        }
    if set(factual) != set(range(len(ids))):
        raise ValueError("Incomplete factual question set")
    used = {ref for q in factual.values() for ref in q["refs"]}
    return factual, [
        {
            "span": s["span_id"],
            "writes": len(written[s["span_id"]]),
            "overwritten_or_deleted": len(written[s["span_id"]].difference(final)),
            "survived_unread": len(written[s["span_id"]].intersection(final).difference(used)),
            "read_by_any_question": len(written[s["span_id"]].intersection(used)),
        }
        for s in spans
    ]


def split_writes(branch, active):
    source_tag = "dated_adjacent_turns" if active else "dated_single_turns"
    generated, source = [], []
    for action in branch["actions"]:
        if action["action_type"] == "NOOP" and action["success"]:
            continue
        if not action["success"] or action["action_type"] != "INSERT":
            raise ValueError("Hybrid intervention requires append-only branch writes")
        reason = action["reasoning"]
        if reason == source_tag:
            source.append(action["memory_content"])
        elif reason == "No reasoning provided" and not source:
            generated.append(action["memory_content"])
        else:
            raise ValueError("Unexpected write origin or order")
    return generated, source


def crossed_banks(prefix, endpoints):
    parts = [split_writes(b, bit) for bit, b in enumerate(endpoints)]
    banks = {k: prefix + parts[int(k[0])][0] + parts[int(k[1])][1] for k in CELLS}
    if any(_digest(banks[k]) != b["bank_hash"] for k, b in zip(("00", "11"), endpoints, strict=True)):
        raise ValueError("Reconstructed pure bank differs from the archived intervention")
    return banks


def build(study, corpus, source, dest):
    protocol = read(study / "protocol.json")
    features = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    training, receipts, backend = panel(study, corpus, features, (1, 3), supplement=True)
    parent = read(source / "frozen.json")
    if receipts != parent["prior_hashes"] or parent["backend"] != backend:
        raise ValueError("Different historical rule-2 trajectories")
    data = {c["sample_id"]: c["qa"] for c in read(corpus) if c["sample_id"] in protocol["train"]}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one leased CUDA device for Qwen encoding and retrieval")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.76 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise RuntimeError("Insufficient room for the bounded Qwen encoder")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, model = init_context_model(protocol["retriever"])
    if not all(p.is_cuda for p in model.parameters()) or protocol["retriever"] != "qwen3-embedding-0.6b":
        raise ValueError("Expected the frozen CUDA Qwen query/context encoder")

    def embed(texts):
        vectors = encode_batch(model, texts, 2, normalize=True)
        return F.normalize(torch.as_tensor(vectors, device="cuda").float(), dim=1)

    evaluator = get_evaluator("locomo", SimpleNamespace())
    windows, jobs = {}, []
    for row in training:
        if row["span"] != 1:
            continue
        sid, name = row["dialogue"], row["episode"]
        ep = study / "train" / name
        (path,) = (ep / "traces").glob("*.jsonl")
        events = list(map(json.loads, path.read_bytes().splitlines()))
        ids = read(ep / "questions.json")
        factual, exposure = trace_exposure(events, ids, data[sid], evaluator)
        info = windows[name] = {"exposure": exposure, "questions": len(ids), "credits": []}
        for earlier in [r for r in training if r["episode"] == name]:
            for orig, effect in earlier["credits"].items():
                refs = set(factual[ids.index(orig)]["refs"])
                (span,) = [e for e in events if e["kind"] == "span" and e["span_id"] == earlier["span"]]
                writes = {(c["memory_id"], c["version"]) for c in span["changes"] if c["after"] is not None}
                info["credits"].append(
                    {
                        "span": earlier["span"],
                        "original_index": orig,
                        "effect": effect,
                        "retrieved_write_count": len(refs & writes),
                    }
                )
        original = [e["diagnosis"] for e in events if e["kind"] == "span_credit"][-1]
        folder = hashlib.sha256(backend.encode()).hexdigest()[:16]
        supplement = read(study / "diagnostics/omitted-final" / folder / (name + ".json"))["diagnosis"]
        branches = [
            d
            for d in (original, supplement)
            if d
            and not d["execution_fault"]
            and all(d.get(a, {}).get("qa") and not d[a]["contract_fault"] for a in ("control", "swap"))
        ]
        if not branches:
            info["exclusion"] = "no eligible final-span control and swap banks"
            continue
        sampled = int(row["use"])
        order = ("swap", "control") if sampled else ("control", "swap")
        spans = [e for e in events if e["kind"] == "span"]
        prefix = list(restore_text(spans[:-1]).values())
        banks = crossed_banks(prefix, [branches[0][a] for a in order])
        if banks[str(sampled) * 2] != list(restore_text(spans).values()):
            raise ValueError("Sampled pure bank disagrees with factual rollout")
        by_question = {}
        for d in branches:
            if crossed_banks(prefix, [d[a] for a in order]) != banks:
                raise ValueError("Original and supplementary interventions differ")
            for q in d["questions"]:
                i = q["qa_index"]
                if i in by_question or q["original_index"] != ids[i]:
                    raise ValueError("Repeated or mismatched branch question")
                endpoints = {str(b) * 2: d[a]["qa"][str(i)] for b, a in enumerate(order)}
                if q["control_matched"] and (
                    factual[i]["context_hash"] != endpoints[str(sampled) * 2]["context_hash"]
                    or factual[i]["answer"] != endpoints[str(sampled) * 2]["answer"]
                ):
                    raise ValueError("Archived matched control has different context or answer")
                by_question[i] = {"matched": q["control_matched"], "endpoints": endpoints}
        texts = list(dict.fromkeys(t for bank in banks.values() for t in bank))
        index = {t: i for i, t in enumerate(texts)}
        vectors = embed(texts)
        queries = embed([data[sid][orig]["question"] for orig in ids])
        ranked = {}
        for key, bank in banks.items():
            sims = queries @ vectors[[index[t] for t in bank]].T
            ranked[key] = sims.argsort(dim=1, descending=True, stable=True)[
                :, : protocol["retrieval_k"][1]
            ].tolist()
        for i, orig in enumerate(ids):
            qa = data[sid][orig]
            ctx = {k: [banks[k][j] for j in ranked[k][i]] for k in CELLS}
            stored = by_question.get(i)
            pure = (
                {
                    k: {"context_reproduced": _digest(ctx[k]) == v["context_hash"], "answer": v["answer"]}
                    for k, v in stored["endpoints"].items()
                }
                if stored
                else {}
            )
            jobs.append(
                {
                    "dialogue": sid,
                    "episode": name,
                    "original_index": orig,
                    "question": qa,
                    "sampled_context_reproduced": _digest(ctx[str(sampled) * 2])
                    == factual[i]["context_hash"],
                    "archived": pure,
                    "control_matched": bool(stored and stored["matched"]),
                    "strict_credit": row["credits"].get(orig, 0.0),
                    "prompts": {k: evaluator.build_prompt(qa["question"], ctx[k], qa) for k in CELLS},
                }
            )
    register(
        dest / "jobs.json",
        {
            "plan": file_hash(dest / "plan.json"),
            "parent": file_hash(source / "frozen.json"),
            "corpus": file_hash(corpus),
            "model": protocol["model"],
            "windows": windows,
            "jobs": jobs,
        },
    )


def contributions(values):
    source = 0.5 * (values["01"] - values["00"] + values["11"] - values["10"])
    generated = 0.5 * (values["10"] - values["00"] + values["11"] - values["01"])
    if not math.isclose(source + generated, values["11"] - values["00"], abs_tol=1e-12):
        raise ValueError("Attribution does not sum to the total skill effect")
    return {"source": source, "generated": generated, "total": values["11"] - values["00"]}


def retain_credit(original, source, faithful):
    return (
        math.copysign(min(abs(original), abs(source)), original)
        if faithful and original * source > 0
        else 0.0
    )


def score(dest, *, credit_only=False):
    jobs = read(dest / "jobs.json")
    if jobs["plan"] != file_hash(dest / "plan.json"):
        raise ValueError("Different registered mechanism or training questions")
    manifest = {"jobs": file_hash(dest / "jobs.json"), "backend": os.environ["MEMSKILL_BACKEND_ID"]}
    register(dest / "reader.json", manifest)
    ev = get_evaluator("locomo", SimpleNamespace())
    folder = dest / "answers"
    folder.mkdir(exist_ok=True)
    rows = []
    for job in jobs["jobs"]:
        if credit_only and not job["strict_credit"]:
            continue
        qa = job["question"]
        path = folder / f"{job['episode']}-q{job['original_index']}.json"
        if path.exists():
            saved = read(path)
            if saved["manifest"] != manifest:
                raise ValueError("Different reader manifest")
            answers = saved["answers"]
        else:
            answers = {
                k: measured_answer(prompt, jobs["model"], qa, ev, dest / "responses")
                for k, prompt in job["prompts"].items()
            }
        for k in CELLS:
            r = answers[k]
            if min(r["tokens"]) <= 0 or not math.isclose(
                r["f1"], ev.compute_f1(r["answer"], ev.get_ground_truth(qa), qa)
            ):
                raise ValueError("Missing reader output or invalid F1")
        register(path, {"manifest": manifest, "answers": answers})
        values = {k: r["f1"] for k, r in answers.items()}
        parts = contributions(values)
        faithful = (
            job["control_matched"]
            and len(job["archived"]) == 2
            and all(
                v["context_reproduced"] and answers[k]["answer"].strip() == v["answer"]
                for k, v in job["archived"].items()
            )
        )
        rows.append(
            {
                "dialogue": job["dialogue"],
                "episode": job["episode"],
                "original_index": job["original_index"],
                "type": str(qa["category"]),
                "endpoints_reproduced": faithful,
                "strict_credit": job["strict_credit"],
                "retained": retain_credit(job["strict_credit"], parts["source"], faithful),
                "f1": values,
                **parts,
            }
        )

    eligible = [r for r in rows if r["strict_credit"]]
    register(dest / "eligible-labels.json", {"manifest": manifest, "rows": eligible})
    if credit_only:
        return

    def summary(group):
        if not group:
            return {"questions": 0}
        groups = defaultdict(list)
        for r in group:
            groups[r["episode"]].append(r)
        return {
            "questions": len(group),
            "f1": {k: mean(r["f1"][k] for r in group) for k in CELLS},
            **{k: mean(r[k] for r in group) for k in ("source", "generated", "total")},
            "pure_oracle": sum(
                len(g) * max(mean(r["f1"][k] for r in g) for k in ("00", "11")) for g in groups.values()
            )
            / len(group),
        }

    register(
        dest / "summary.json",
        {
            "manifest": manifest,
            "all": summary(rows),
            "faithful": summary([r for r in rows if r["endpoints_reproduced"]]),
            "per_type": {
                t: summary([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})
            },
            "per_dialogue": {
                s: summary([r for r in rows if r["dialogue"] == s])
                for s in sorted({r["dialogue"] for r in rows})
            },
            "retained_by_dialogue": dict(Counter(r["dialogue"] for r in rows if r["retained"])),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("build", "score-credit", "score"))
    for name in ("study", "corpus", "source", "dest"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if a.phase == "build":
        build(a.study, a.corpus, a.source, a.dest)
    else:
        score(a.dest, credit_only=a.phase == "score-credit")
