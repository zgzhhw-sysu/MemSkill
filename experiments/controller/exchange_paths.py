"""Measure auxiliary questions against the original matched writer paths on 251."""

import argparse
import hashlib
import json
import os
import subprocess
from functools import cache, lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.capped_source_repair import archive
from experiments.controller.complete_donor_paths import decision_context
from experiments.controller.exchange_questions import (
    call,
    population,
    quote,
    verify_question,
)
from experiments.controller.resume_status import check_equal_source_choice
from src.credit_assignment import (
    execution_fault,
    hidden_retrieval_candidate,
    possible_recovery,
    recovered,
)
from src.data_processing.locomo import LoCoMoProcessor
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn
from src.replay import restore_text


class Ranking:
    def __init__(self, retriever):
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            != subprocess.check_output(
                ["nvidia-smi", "-i", "3", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True,
            ).strip()
            or float(os.environ["MEMSKILL_ENCODER_DUTY"]) != 0.30
            or int(os.environ["MEMSKILL_ENCODER_MAX_BATCH"]) != 2
        ):
            raise ValueError("Expected guarded physical GPU 3 with bounded encoder")
        free, total = torch.cuda.mem_get_info()
        budget = min(total * 0.65, free - 2048 * 2**20)
        if budget < 2048 * 2**20:
            raise SystemExit(75)
        torch.cuda.set_per_process_memory_fraction(budget / total)
        set_retriever_flash_attn(False)
        _, self.model = init_context_model(retriever)
        if self.model.device.type != "cuda":
            raise ValueError("Expected GPU Qwen")

        model = self.model

        @lru_cache(maxsize=64)
        def encode(texts):
            return encode_batch(model, list(texts), 2, normalize=True)

        self.encode = encode

    def snapshots(self, spans):
        live = {}
        snapshots = []
        for span in spans:
            # Executor encodes successful updates then inserts, including inserts later skipped.
            texts = tuple(
                a["memory_content"]
                for kind in ("UPDATE", "INSERT")
                for a in span["parsed_actions"]
                if a["success"] and a["action_type"] == kind
            )
            batch = dict(zip(texts, self.encode(texts), strict=True)) if texts else {}
            for change in span["changes"]:
                mid = change["memory_id"]
                if change["action"] == "DELETE":
                    del live[mid]
                else:
                    live[mid] = batch[change["after"]]
            snapshots.append(list(live.values()))
        return snapshots

    @torch.inference_mode()
    def rank(self, query_vector, texts, vectors):
        if len(vectors) != len(texts):
            raise ValueError("Missing memory embedding")
        if not texts:
            return [], []
        matrix = torch.as_tensor(np.stack(vectors), device="cuda", dtype=torch.float32)
        q = torch.as_tensor(query_vector, device="cuda", dtype=torch.float32)
        sim = matrix @ F.normalize(q, dim=0)
        idx = np.argsort(sim.cpu().numpy())[::-1][:5].tolist()
        return idx, [texts[i] for i in idx]


def clean_marker(text):
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].strip() == "INSERT block:":
        return "".join(lines[:-1]).rstrip("\r\n")
    return text


def physical(base, job, spec, data, rank):
    pivot = job["pivot"]
    name = f"{job['dialogue']}_s{job['start']}"
    folder = archive(base, job["panel"]) / name
    control = read(base / "source-compression/capped-source-credit/registration.json")[
        "windows"
    ][name]
    h = data[job["dialogue"]]
    proc = LoCoMoProcessor(chunk_mode="full-session")
    source_hashes = []
    for n in (job["start"], job["start"] + 1):
        sk = f"session_{n}"
        (source,) = proc._extract_session_chunks(h[sk], h[sk + "_date_time"])
        source_hashes.append(digest(source))
    result = []
    contexts = []
    for bits in job["bits"]:
        cell = folder / bits
        (trace,) = (cell / "traces").glob("*.jsonl")
        scores = cell / "scores.json"
        known = control["cells"][bits]
        if file_hash(trace) != known["trace"] or file_hash(scores) != known["scores"]:
            raise ValueError("Archived physical path changed")
        events = [json.loads(line) for line in trace.read_bytes().splitlines()]
        spans = [e for e in events if e["kind"] == "span"]
        decisions = read(cell / "decisions.json")
        ctx = []
        for d, span, src, b in zip(decisions, spans, source_hashes, bits, strict=True):
            check_equal_source_choice(d, spec["skills"], span["selected_skills"])
            if int(d["use"]) != int(b):
                raise ValueError("Changed action")
            ctx.append(decision_context(span, d, src, int(b)))
        current = spans[pivot]
        if not {job["ask_id"], job["reply_id"]} <= set(current["source_turns"]):
            raise ValueError("Auxiliary evidence outside pivot")
        contexts.append((digest(spans[:pivot]), ctx[pivot]))
        bank = restore_text(spans)
        (snap,) = [e for e in events if e["kind"] == "snapshot"]
        if snap["text_sha256"] != hashlib.sha256(
            json.dumps(list(bank.items()), ensure_ascii=False).encode()
        ).hexdigest() or [ref[0] for ref in snap["memory_refs"]] != list(bank):
            raise ValueError("Final writer snapshot did not reproduce")
        qa = next(e for e in events if e["kind"] == "qa")
        sc = read(scores)["answers"][0]
        final = list(bank.values())
        expected = [bank[i] for i, _ in qa["refs"]]
        if [final[i] for i in sc["retrieved_indices"]] != expected or sc[
            "qa_index"
        ] != qa["query_id"]:
            raise ValueError("Archived reader context mismatch")
        queries = tuple(e["question"] for e in events if e["kind"] == "qa")
        matrices = rank.snapshots(spans)
        _, observed = rank.rank(rank.encode(queries)[0], final, matrices[-1])
        result.append(
            {
                "matrix": matrices[pivot],
                "final_matrix": matrices[-1],
                "immediate": list(restore_text(spans[: pivot + 1]).values()),
                "final": final,
                "parity": {
                    "expected": digest(expected),
                    "actual": digest(observed),
                    "question": digest(qa["question"]),
                },
                "faults": [
                    execution_fault(e["parsed_actions"], e["applied"], e["skills"])
                    for e in spans
                ],
                "capped": known["triggers"],
                "trace": known["trace"],
                "decisions": file_hash(cell / "decisions.json"),
            }
        )
    if contexts[0] != contexts[1] or len(result) != 2:
        raise ValueError("Unmatched physical pivot states")
    return result, contexts[0]


def prepare(base, job, qa, spec, data, rank):
    banks, common = physical(base, job, spec, data, rank)
    receipt = [
        {
            k: v
            for k, v in bank.items()
            if k not in ("immediate", "final", "matrix", "final_matrix")
        }
        for bank in banks
    ]
    info = {"actions": job["bits"], "common": common, "paths": receipt}
    if any(v["parity"]["expected"] != v["parity"]["actual"] for v in receipt):
        return dict(info, status="control_parity_failed")
    arms = []
    quotes = [
        quote(job, job["asker"], job["ask_text"]),
        quote(job, job["responder"], job["reply"]),
    ]
    query = rank.encode((qa["question"],))[0]
    for b in banks:
        texts = b["immediate"]
        matrix = b["matrix"]
        indices, native = rank.rank(query, texts, matrix)
        extra = [t for t in quotes if t not in texts]
        augmented = matrix + list(rank.encode(tuple(extra))) if extra else matrix
        cleaned = [clean_marker(t) for t in texts]
        changed = [
            i for i, (a, c) in enumerate(zip(texts, cleaned, strict=True)) if a != c
        ]
        clean_vectors = matrix.copy()
        if changed:
            for i, v in zip(
                changed, rank.encode(tuple(cleaned[i] for i in changed)), strict=True
            ):
                clean_vectors[i] = v
        mid = hidden_retrieval_candidate(
            qa,
            [{"text": t} for t in (job["ask_text"], job["reply"])],
            indices,
            dict(enumerate(texts)),
        )
        arms.append(
            {
                "native": native,
                "delayed": rank.rank(query, b["final"], b["final_matrix"])[1]
                if job["pivot"] == 0
                else native,
                "execution": rank.rank(query, texts + extra, augmented)[1],
                "retrieval": native[:-1] + [texts[mid]] if mid is not None else None,
                "format": rank.rank(query, cleaned, clean_vectors)[1]
                if changed
                else native,
            }
        )
    return dict(info, status="prepared", arms=arms)


def sign(pair, truth):
    left, right = pair
    return int(recovered(left, right, truth)) - int(recovered(right, left, truth))


def measure(out, obj, qa, ev, pivot):
    if obj["status"] != "prepared":
        return {"status": obj["status"], "raw": 0.0, "credit": 0.0}

    @cache
    def ask(ctx):
        v = call(out, ev.build_prompt(qa["question"], list(ctx), qa), "reader", 32)
        ans = v["text"].strip()
        return {
            "answer": ans,
            "ok": True,
            "f1": ev.compute_f1(ans, qa["answer"], qa),
            "receipt": v,
        }

    reads = {
        label: [ask(tuple(arm[label])) for arm in obj["arms"]]
        for label in ("native", "delayed")
    }
    raw = reads["native"][1]["f1"] - reads["native"][0]["f1"]
    direction = sign(reads["native"], qa["answer"])
    problems = []
    if direction == 0:
        problems.append("no_strict_recovery")
    if any(any(p["faults"]) or any(p["capped"]) for p in obj["paths"]):
        problems.append("writer_defect")
    if pivot == 0 and sign(reads["delayed"], qa["answer"]) != direction:
        problems.append("delayed_disagrees")
    if not problems:
        index = int(direction < 0)
        low = reads["native"][index]
        for label in ("execution", "retrieval", "format"):
            ctx = obj["arms"][index][label]
            reads[label] = ask(tuple(ctx)) if ctx is not None else None
            if possible_recovery(low, reads[label], qa["answer"]):
                problems.append(label)
    return {
        "status": "measured",
        "raw": raw,
        "credit": 0.0 if problems else raw,
        "veto": problems,
        "reads": reads,
    }


def pending(reg, role):
    out = reg.parent
    for job in read(reg)["jobs"]:
        name = job["id"] + ".json"
        stage = out / ("questions" if role == "prepare" else "prepared") / name
        dest = out / ("prepared" if role == "prepare" else "outcomes") / name
        if (
            dest.is_file()
            or not stage.is_file()
            or (role == "prepare" and read(stage)["status"] != "accepted")
        ):
            continue
        yield job, stage, dest


def run(base, study, corpus, role, limit):
    reg = population(base, study, corpus)
    out = reg.parent
    if role != "audit" and not next(pending(reg, role), None):
        return
    spec = read(study / "protocol.json")
    if spec["retrieval_k"] != [5, 5]:
        raise ValueError("Expected five memory slots")
    if role in ("prepare", "audit"):
        rank = Ranking(spec["retriever"])
        data = {
            d["sample_id"]: d["conversation"]
            for d in read(corpus)
            if d["sample_id"] in spec["train"]
        }
    if role == "audit":
        entries = {}
        for job in read(reg)["jobs"]:
            key = job["panel"] + str(job["pivot"])
            if key in entries:
                continue
            banks, common = physical(base, job, spec, data, rank)
            entries[key] = {
                "id": job["id"],
                "common": common,
                "control": [b["parity"] for b in banks],
            }
            print(
                key,
                job["id"],
                [b["parity"]["expected"] == b["parity"]["actual"] for b in banks],
                flush=True,
            )
        register(out / "control-audit-native-batches.json", entries)
        return
    ev = LoCoMoEvaluator(SimpleNamespace())
    for i, (job, inp, dest) in enumerate(pending(reg, role)):
        if i == limit:
            break
        dest.parent.mkdir(exist_ok=True)
        question = out / "questions" / dest.name
        packet = read(question)
        verify_question(reg, job, packet)
        if packet["status"] != "accepted":
            raise ValueError("Unscreened auxiliary question")
        if role == "prepare":
            result = prepare(base, job, packet["qa"], spec, data, rank)
        else:
            obj = read(inp)
            if obj["question"] != file_hash(question):
                raise ValueError("Physical paths for a different QA")
            result = measure(out, obj, packet["qa"], ev, job["pivot"])
        register(
            dest, dict(result, inputs=file_hash(inp), question=file_hash(question))
        )
        print(role, job["id"], result["status"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("role", choices=("prepare", "read", "audit"))
    p.add_argument("--limit", type=int, default=8)
    a = p.parse_args()
    run(a.base, a.study, a.corpus, a.role, a.limit)
