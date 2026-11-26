"""Cache normalized, pre-write utterance embeddings of the training histories."""

import argparse
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn

PLAN = Path(__file__).with_name("protocols") / "evidence_transport.json"


def texts(history, turns):
    roles = {
        history["speaker_a"].casefold(): "Speaker 1",
        history["speaker_b"].casefold(): "Speaker 2",
    }
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(s) for s in sorted(roles, key=len, reverse=True))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    result = []
    for t in turns:
        body = t["text"]
        if t.get("blip_caption"):
            body += " [Image: " + t["blip_caption"] + "]"
        result.append(
            roles[t["speaker"].casefold()]
            + ": "
            + pattern.sub(lambda m: roles[m[0].casefold()], body)
        )
    return result


def collect(base, study, corpus, phase):
    out = base / "source-compression/evidence-transport"
    storage = out / "turns"
    storage.mkdir(parents=True, exist_ok=True)
    protocol = read(study / "protocol.json")
    train = set(protocol["train"])
    if (
        len(train) != 6
        or train & set(protocol["validation"] + protocol["test"])
        or file_hash(corpus) != protocol["input_sha256"]
        or protocol["retriever"] != "qwen3-embedding-0.6b"
    ):
        raise ValueError("Expected fixed training histories and retriever")
    register(
        out / "sources.json",
        {
            "plan": file_hash(PLAN),
            "encoder": file_hash(Path(__file__)),
            "study": file_hash(study / "protocol.json"),
            "corpus": protocol["input_sha256"],
            "model": protocol["retriever"],
            "backend": os.environ["MEMSKILL_BACKEND_ID"],
        },
    )
    data = {
        c["sample_id"]: c["conversation"]
        for c in read(corpus)
        if c["sample_id"] in train
    }
    windows = [
        (sid, n)
        for sid, n in read(base / "source-compression/factorial/manifest.json")[
            "windows"
        ]
    ]
    if phase == "all":
        windows.extend(
            (w["dialogue"], w["start"])
            for w in read(base / "late-pairs/cohort.json")["assignments"]
        )
    jobs = [
        (sid, j)
        for sid, n in windows
        for j in (n, n + 1)
        if not (storage / f"{sid}_s{j}.npz").exists()
    ]
    if not jobs:
        return
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= 0.30
    ):
        raise RuntimeError("Select a single bounded CUDA encoder")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.76 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, encoder = init_context_model(protocol["retriever"])
    if not all(p.is_cuda for p in encoder.parameters()):
        raise ValueError("Encoder is not fully on CUDA")
    processor = LoCoMoProcessor(chunk_mode="full-session")
    for sid, n in jobs:
        history = data[sid]
        key = f"session_{n}"
        turns = history[key]
        ids = [t["dia_id"] for t in turns]
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError("Missing or duplicate turns")
        (source,) = processor._extract_session_chunks(
            turns, history[key + "_date_time"]
        )
        arr = F.normalize(
            torch.as_tensor(
                encode_batch(encoder, texts(history, turns), 2, normalize=True),
                device="cuda",
            ).float(),
            dim=1,
        )
        path = storage / f"{sid}_s{n}.npz"
        with TemporaryDirectory(dir=storage) as tmp:
            part = Path(tmp) / path.name
            np.savez(part, ids=ids, source=digest(source), vectors=arr.cpu().numpy())
            os.link(part, path)
        print(path.stem, len(turns), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "study", "corpus"):
        p.add_argument(key, type=Path)
    p.add_argument("phase", choices=("early", "all"))
    args = p.parse_args()
    collect(args.base, args.study, args.corpus, args.phase)
