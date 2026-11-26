"""Complete the visible turn archive for the registered full-prefix endpoints."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.evidence_turns import texts
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn


def encode(base, study, corpus):
    group = base / "source-compression/witness-passages"
    original = group.parent / "evidence-transport/turns"
    ext = group / "evidence-advantage/prefix-sources"
    folder = ext / "turns"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = base / "late-pairs/cross-window-prefix/manifest.json"
    proto = read(study / "protocol.json")
    if (
        file_hash(corpus) != proto["input_sha256"]
        or read(manifest)["corpus"] != proto["input_sha256"]
    ):
        raise ValueError("Different full-prefix data")
    training = set(proto["train"])
    histories = {
        r["sample_id"]: r["conversation"]
        for r in read(corpus)
        if r["sample_id"] in training
    }
    targets = {(r["dialogue"], r["end"]) for r in read(manifest)["endpoints"]}
    if len(targets) != 99 or {sid for sid, n in targets} != training:
        raise ValueError("Different prefix endpoints")
    missing = sorted(
        (sid, n) for sid, n in targets if not (original / f"{sid}_s{n}.npz").exists()
    )
    lock = ext / "complete.json"
    if lock.exists():
        rec = read(lock)
        if rec["manifest"] != file_hash(manifest) or rec["files"] != {
            f"{sid}_s{n}": file_hash(folder / f"{sid}_s{n}.npz") for sid, n in missing
        }:
            raise ValueError("Changed prefix source archive")
        return
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= 0.30
        or int(os.environ["MEMSKILL_ENCODER_MAX_BATCH"]) != 2
    ):
        raise ValueError("Use the registered bounded CUDA encoder")
    free, total = torch.cuda.mem_get_info()
    cap = min(int(0.65 * total), free - 2048 * 2**20)
    if cap < 2048 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    set_retriever_flash_attn(False)
    _, encoder = init_context_model(proto["retriever"])
    if not all(v.is_cuda for v in encoder.parameters()):
        raise ValueError("Model not on leased GPU")
    proc = LoCoMoProcessor(chunk_mode="full-session")

    def values(sid, n):
        hist = histories[sid]
        k = f"session_{n}"
        turns = hist[k]
        ids = [t["dia_id"] for t in turns]
        (src,) = proc._extract_session_chunks(turns, hist[k + "_date_time"])
        vec = F.normalize(
            torch.as_tensor(
                encode_batch(encoder, texts(hist, turns), 2, normalize=True),
                device="cuda",
            ).float(),
            dim=1,
        )
        return ids, digest(src), vec

    sid = min(training)
    ids, source, v = values(sid, 5)
    with np.load(original / f"{sid}_s5.npz", allow_pickle=False) as a:
        reference = torch.as_tensor(a["vectors"], device="cuda")
        diff = float((reference - v).abs().max())
        cosine = float(F.cosine_similarity(reference, v).min())
        if (
            ids != a["ids"].tolist()
            or source != str(a["source"])
            or diff > 1e-4
            or cosine < 0.999999
        ):
            raise ValueError("Encoder parity failed")
    files = {}
    for sid, n in missing:
        name = f"{sid}_s{n}"
        path = folder / (name + ".npz")
        ids, source, v = values(sid, n)
        if not path.exists():
            with TemporaryDirectory(dir=folder) as tmp:
                dst = Path(tmp) / path.name
                np.savez(dst, ids=ids, source=source, vectors=v.cpu().numpy())
                try:
                    os.link(dst, path)
                except FileExistsError:
                    pass
        with np.load(path, allow_pickle=False) as a:
            if (
                ids != a["ids"].tolist()
                or str(a["source"]) != source
                or not torch.equal(torch.as_tensor(a["vectors"], device="cuda"), v)
            ):
                raise ValueError("Changed prefix turn vectors")
        files[name] = file_hash(path)
    register(
        lock,
        {
            "manifest": file_hash(manifest),
            "corpus": file_hash(corpus),
            "files": files,
            "parity_max_error": diff,
            "parity_min_cosine": cosine,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    o = p.parse_args()
    encode(o.base, o.study, o.corpus)
