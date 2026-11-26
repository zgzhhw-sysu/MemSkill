"""Measure weaker-skill repairs after recomputing the fixed suffix of a root intervention."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.capped_source_repair import writer
from experiments.controller.exchange_paths import Ranking, clean_marker
from experiments.controller.exchange_questions import call
from experiments.controller.legacy_root_pivots import PLAN
from experiments.controller.legacy_terminal_symmetry import (
    make_writer,
    measure,
    ops,
    source,
)
from experiments.controller.semantic_writer import sha
from src.credit_assignment import _digest, execution_fault, hidden_retrieval_candidate
from src.episode_trace import evidence_turns
from src.evaluation_cache import encode_batch, history_embeddings
from src.executor import ExecutionResult
from src.memory_bank import MemoryBank

PHASES = ("suffix-inputs", "suffix-writes", "reader-inputs", "outcomes")
READY = ("matched_writer", "prepared_suffix", "matched_suffix", "prepared")
ARMS = ("control", "swap", "repair")


def bank_after_root(rank, ex, branch, quotes):
    def state(texts):
        return encode_batch(rank.model, list(texts), 2)

    bank = MemoryBank(
        ex.retriever_name, top_k=5, state_encoder=SimpleNamespace(_encode_texts=state)
    )
    actions = [ExecutionResult(**r) for r in branch["actions"]]
    written = {a.memory_content for a in actions}
    added = [q for q in dict.fromkeys(quotes) if q not in written]
    actions.extend(ExecutionResult("INSERT", True, memory_content=q) for q in added)
    if not ex.apply_to_memory_bank(actions, bank, []):
        raise ValueError("Verified root actions could not be applied")
    bank.step()
    return bank, added


def focal_evidence(job, data, root):
    turns = [
        t
        for t in evidence_turns(data["conversation"], job["qa"]["evidence"])
        if t["dia_id"] in root["source_turns"]
    ]
    if not turns:
        raise ValueError("Missing root evidence")
    return turns, [f"{t['session_date']} | {t['speaker']}: {t['text']}" for t in turns]


def suffix_inputs(job, data, spans, d, p, rank):
    root, suffix = spans
    _, quotes = focal_evidence(job, data, root)
    ex = make_writer(p)
    selected = ops(suffix["selected_skills"])
    query = encode_batch(rank.model, [suffix["source"]], 2)[0]
    prepared = {}
    for arm in ARMS:
        branch = job["weak"] if arm == "repair" else arm
        bank, added = bank_after_root(
            rank, ex, d[branch], quotes if arm == "repair" else []
        )
        contents = bank.get_all_contents()
        vectors = torch.as_tensor(
            np.stack([m.state_encoder_embedding for m in bank.memories]),
            device="cuda",
            dtype=torch.float32,
        )
        q = torch.as_tensor(query, device="cuda", dtype=torch.float32)
        scores = F.normalize(vectors) @ F.normalize(q, dim=0)
        idx = np.argsort(scores.cpu().numpy())[::-1][:5].tolist()
        ctx = [contents[i] for i in idx]
        prompt = ex._build_executor_prompt(selected, suffix["source"], ctx)
        prepared[arm] = {
            "bank": _digest(contents),
            "added": added,
            "indices": idx,
            "retrieved": ctx,
            "prompt": sha(prompt),
        }
        if arm != "repair":
            (logged,) = d[branch]["suffix"]
            if (
                logged["retrieval_hash"] != _digest(ctx)
                or logged["writer_prompt_hashes"] != [sha(prompt)]
                or (arm == "control" and ctx != suffix["retrieved"])
            ):
                return {
                    "status": "suffix_input_mismatch",
                    "arm": arm,
                    "observed": prepared[arm],
                    "expected": logged,
                }
    return {"status": "prepared_suffix", "branches": prepared}


def reissue(out, spans, d, p, prepared):
    _, suffix = spans
    ex = make_writer(p)
    selected = ops(suffix["selected_skills"])
    measured = {}
    for arm in ARMS:
        inp = prepared["branches"][arm]
        prompt = ex._build_executor_prompt(selected, suffix["source"], inp["retrieved"])
        if sha(prompt) != inp["prompt"]:
            raise ValueError("Prepared suffix prompt changed")
        response = call(out, prompt, "executor", p["writer_tokens"])
        receipt = {
            "prompt_sha256": sha(prompt),
            "response": response["text"],
            "input_tokens": response["tokens"][0],
            "output_tokens": response["tokens"][1],
        }
        event = dict(suffix, retrieved=inp["retrieved"], executor_responses=[receipt])
        actions = writer(ex, selected, event, repair=False)
        measured[arm] = {"actions": actions, "response": response, "input": inp}
        if arm != "repair" and _digest(actions) != d[arm]["suffix"][0]["actions_hash"]:
            return {
                "status": "suffix_actions_changed",
                "arm": arm,
                "branches": measured,
            }
        if execution_fault(actions, True, suffix["skills"]) or actions != writer(
            ex, selected, event, repair=True
        ):
            return {"status": "suffix_write_fault", "arm": arm, "branches": measured}
    return {"status": "matched_suffix", "branches": measured}


def reader_inputs(job, data, spans, d, p, rank, suffixes):
    root, _ = spans
    ex = make_writer(p)
    turns, quotes = focal_evidence(job, data, root)
    ids = [q["qa_index"] for q in d["questions"]]
    texts = tuple(data["qa"][q["original_index"]]["question"] for q in d["questions"])
    qvec = rank.encode(texts)[ids.index(job["question_index"])]
    packets = {}
    for arm in ARMS:
        branch = job["weak"] if arm == "repair" else arm
        suf = suffixes["branches"][arm]
        inp = suf["input"]
        bank, added = bank_after_root(
            rank, ex, d[branch], quotes if arm == "repair" else []
        )
        before = bank.get_all_contents()
        if (
            added != inp["added"]
            or _digest(before) != inp["bank"]
            or [before[i] for i in inp["indices"]] != inp["retrieved"]
        ):
            raise ValueError("Changed root bank or suffix indices")
        actions = [ExecutionResult(**r) for r in suf["actions"]]
        if not ex.apply_to_memory_bank(actions, bank, inp["indices"]):
            return {"status": "suffix_application_failed", "arm": arm}
        bank.step()
        contents = bank.get_all_contents()
        vectors = [m.embedding for m in bank.memories]
        ids, ctx = rank.rank(qvec, contents, vectors)
        if arm == "repair":
            packets[branch]["execution"] = ctx
            repair = {
                "final_bank": _digest(contents),
                "added": added,
                "indices": ids,
                "context": _digest(ctx),
            }
            continue
        orig = d[branch]["qa"][str(job["question_index"])]
        if (
            _digest(contents) != d[branch]["bank_hash"]
            or ids != orig["indices"]
            or _digest(ctx) != orig["context_hash"]
        ):
            return {
                "status": "reader_input_mismatch",
                "arm": arm,
                "bank": _digest(contents),
                "indices": ids,
                "context": _digest(ctx),
            }
        packets[arm] = {"native": ctx, "original": orig}
        if arm == job["weak"]:
            mid = hidden_retrieval_candidate(
                job["qa"], turns, ids, dict(enumerate(contents))
            )
            packets[arm]["retrieval"] = (
                ctx[:-1] + [contents[mid]] if mid is not None else None
            )
            cleaned = [clean_marker(t) for t in contents]
            changed = [
                i
                for i, (t, c) in enumerate(zip(contents, cleaned, strict=True))
                if t != c
            ]
            if changed:
                vectors = vectors.copy()
                for i, v in zip(
                    changed,
                    rank.encode(tuple(cleaned[i] for i in changed)),
                    strict=True,
                ):
                    vectors[i] = v
            packets[arm]["format"] = (
                rank.rank(qvec, cleaned, vectors)[1] if changed else ctx
            )
    return {"status": "prepared", "branches": packets, "repaired_suffix": repair}


def run(base, study, corpus, phase):
    out = base / "source-compression/legacy-root-symmetry"
    reg = out / "registration.json"
    earlier = out.parent / "legacy-terminal-symmetry/registration.json"
    packet = read(reg)
    old = read(earlier)
    p = read(study / "protocol.json")
    if (
        packet["plan"] != file_hash(PLAN)
        or packet["terminal_registration"] != file_hash(earlier)
        or packet["jobs"] != old["root_separate"]
        or old["protocol"] != file_hash(study / "protocol.json")
        or old["corpus"] != file_hash(corpus)
    ):
        raise ValueError("Root study provenance changed")
    data = {d["sample_id"]: d for d in read(corpus) if d["sample_id"] in p["train"]}
    parent = ("writers", *PHASES)[PHASES.index(phase)]
    dest = out / phase
    dest.mkdir(exist_ok=True)
    rank = None
    hashes = {}
    for j in packet["jobs"]:
        name = j["id"]
        prev = out / parent / (name + ".json")
        f = dest / (name + ".json")
        if not prev.is_file():
            continue
        a = read(prev)
        inputs = {"registration": file_hash(reg), "parent": file_hash(prev)}
        if f.exists():
            if read(f)["inputs"] != inputs:
                raise ValueError("Changed upstream replay")
        else:
            spans, d = source(j, old, study)
            status = a["writer"]["status"] if parent == "writers" else a["status"]
            if status != READY[PHASES.index(phase)]:
                v = {"status": status, "credit": 0.0}
            elif phase in ("suffix-inputs", "reader-inputs"):
                if rank is None:
                    rank = Ranking(p["retriever"])
                with history_embeddings():
                    if phase == "suffix-inputs":
                        v = suffix_inputs(j, data[j["history"]], spans, d, p, rank)
                    else:
                        v = reader_inputs(j, data[j["history"]], spans, d, p, rank, a)
            elif phase == "suffix-writes":
                v = reissue(out, spans, d, p, a)
            else:
                v = measure(out, j, a)
            register(f, dict(v, inputs=inputs))
            print(phase, name, v["status"], flush=True)
        hashes[name] = file_hash(f)
    if len(hashes) == len(packet["jobs"]):
        register(out / (phase + "-complete.json"), hashes)
        if phase == "outcomes":
            cases = {j["id"]: read(dest / (j["id"] + ".json")) for j in packet["jobs"]}
            register(
                out / "summary.json",
                {
                    "registration": file_hash(reg),
                    "outcomes": hashes,
                    "cases": {
                        k: {
                            "status": v["status"],
                            "credit": v.get("credit", 0.0),
                            "veto": v.get("veto", []),
                        }
                        for k, v in cases.items()
                    },
                },
            )


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        arg.add_argument(k, type=Path)
    arg.add_argument("phase", choices=PHASES)
    a = arg.parse_args()
    run(a.base, a.study, a.corpus, a.phase)
