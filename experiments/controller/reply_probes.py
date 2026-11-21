"""Measure within-span question-to-reply placement without benchmark QA annotations."""

import argparse
import os
from pathlib import Path
import re
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn
from src.replay import dated_group_parts

PLAN = Path(__file__).with_name("protocols") / "reply_placement.json"


def exchanges(turns):
    speakers = [turn["speaker"] for turn in turns]
    probes = []
    for ask in range(len(turns) - 1):
        target = ask + 1
        if speakers[ask] == speakers[target]:
            continue
        questions = [s.strip() for s in re.findall(r"[^.!?]*\?", turns[ask]["text"])
                     if s.strip(" \t\n?")]
        peers = [i for i, name in enumerate(speakers) if name == speakers[target]]
        if questions and len(peers) >= 2:
            sham = peers[(peers.index(target) + 1) % len(peers)]
            probes.append((questions[-1], ask, target, sham, peers))
    return probes


def candidate_text(parent, parts, ask):
    if all(i != ask for i, _ in parts):
        return parent
    remaining = [text for i, text in parts if i != ask]
    if len(remaining) > 1:
        raise ValueError("Expected at most two source turns per item")
    return remaining[0] if remaining else None


def ranked_turns(records, vectors, query, ask):
    items = [(text, [i for i, _ in parts if i != ask])
             for parent, parts in records
             if (text := candidate_text(parent, parts, ask)) is not None]
    scores = torch.stack([vectors[text] for text, _ in items]) @ vectors[query]
    positions = scores.argsort(descending=True, stable=True)[:5].tolist()
    ranks = {}
    for rank, position in enumerate(positions, 1):
        for turn in items[position][1]:
            ranks.setdefault(turn, 1 / rank)
    return ranks


def excess(ranks, target, peers):
    return ranks.get(target, 0.) - mean(ranks.get(i, 0.) for i in peers if i != target)


def measure(source, turns, encoder):
    probes = exchanges(turns)
    groups = [dated_group_parts(source, n) for n in (1, 2)]
    if any({i for _, parts in group for i, _ in parts} != set(range(len(turns))) for group in groups):
        raise ValueError("Turn ordinals in dated items do not match the dialogue")
    per_target = {"actual": [], "sham": []}
    if probes:
        texts = list(dict.fromkeys([q[0] for q in probes] +
                    [v for records in groups for parent, parts in records
                     for v in [parent, *(text for _, text in parts)]]))
        enc = encode_batch(encoder, texts, 2, normalize=True)
        vectors = dict(zip(texts, F.normalize(torch.as_tensor(enc, device="cuda").float(), dim=1), strict=True))
        for query, ask, reply, sham, peers in probes:
            banks = [[ranked_turns(records, vectors, query, remove) for remove in (None, ask)]
                     for records in groups]
            for arm, target in (("actual", reply), ("sham", sham)):
                delta = [excess(pair, target, peers) - excess(single, target, peers)
                         for single, pair in zip(*banks, strict=True)]
                per_target[arm].append([*delta, float(ask // 2 == target // 2) - .5])
    density = len(probes) / max(1, len(turns) - 1)
    return {"source": digest(source), "turns": len(turns), "probes": len(probes),
            "within_pair": sum(ask // 2 == reply // 2 for _, ask, reply, _, _ in probes),
            **{arm: [*[mean(v) for v in zip(*rows, strict=True)], density] if rows else [0.] * 4
               for arm, rows in per_target.items()}}


def collect(study, corpus, dest, early):
    protocol = read(study / "protocol.json")
    train = set(protocol["train"])
    if (len(train) != 6 or train.intersection(protocol["validation"] + protocol["test"])
            or file_hash(corpus) != protocol["input_sha256"] or not protocol["equal_sources"]):
        raise ValueError("Expected the original disjoint six-history training cohort")
    sources = dest / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    register(dest / "manifest.json", {
        "plan": file_hash(PLAN), "code": file_hash(Path(__file__)),
        "protocol": file_hash(study / "protocol.json"), "retriever": protocol["retriever"],
        "backend": os.environ["MEMSKILL_BACKEND_ID"]})
    pending = []
    for convo in read(corpus):
        if convo["sample_id"] not in train:
            continue
        history = convo["conversation"]
        for k in history:
            if k.startswith("session_") and k[8:].isdigit():
                n = int(k[8:])
                if early and not 5 <= n <= 8:
                    continue
                path = sources / f"{convo['sample_id']}_s{n}.json"
                if not path.exists():
                    pending.append((n, path, history[k], history[k + "_date_time"]))
    if not pending:
        return
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= .30):
        raise RuntimeError("Use one leased CUDA device at <= 30% encoder duty")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(.76 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, encoder = init_context_model(protocol["retriever"])
    if not all(p.is_cuda for p in encoder.parameters()):
        raise RuntimeError("Encoder must reside on the leased GPU")
    processor = LoCoMoProcessor(chunk_mode="full-session")
    for _, path, turns, date in sorted(pending, key=lambda item: (item[0] < 5, item[0], item[1])):
        (source,) = processor._extract_session_chunks(turns, date)
        result = measure(source, turns, encoder)
        register(path, result)
        print(path.stem, result["probes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "corpus", "dest"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--early", action="store_true")
    args = parser.parse_args()
    collect(args.study, args.corpus, args.dest, args.early)
