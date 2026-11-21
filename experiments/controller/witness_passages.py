"""Encode the paired, physically observed reader passages for a training-only actor."""

import argparse
import os
import re
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.capped_source_policy import (
    baseline,
    cell_scores,
    joint_evidence,
    surviving_credit,
)
from experiments.controller.evidence_turns import texts
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn

PLAN = Path(__file__).with_name("protocols") / "witness_passages.json"


def clean(text, history, date):
    if not text:
        return None
    raw = text.removeprefix(date + " | ").removesuffix("INSERT block:").strip()
    roles = {
        history[k].casefold(): f"Speaker {i}"
        for i, k in enumerate(("speaker_a", "speaker_b"), 1)
    }
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(n) for n in sorted(roles, key=len, reverse=True))
        + r")(?!\w)"
    )
    return (
        re.sub(pattern, lambda m: roles[m[0].casefold()], raw, flags=re.IGNORECASE)
        or None
    )


def registration(base, study, corpus):
    src = base / "source-compression"
    out = src / "witness-passages"
    out.mkdir(exist_ok=True)
    cap = src / "capped-source-credit"
    ranks = src / "reader-slot-credit/single-rank/requests.json"
    parent = cap / "policy/joint-repaired/fit.json"
    proto = read(study / "protocol.json")
    ledger = src / "evidence-transport/sources.json"
    key = read(ledger)
    if (
        key["corpus"] != file_hash(corpus)
        or key["model"] != proto["retriever"]
        or key["study"] != file_hash(study / "protocol.json")
        or len(proto["train"]) != 6
    ):
        raise ValueError("Turn encoder or donor split changed")
    inputs = {
        "plan": file_hash(PLAN),
        "ranks": file_hash(ranks),
        "cases": file_hash(ranks.parent.parent / "verified-cases.json"),
        "parent": file_hash(parent),
        "complete": file_hash(cap / "complete.json"),
        "turns": file_hash(ledger),
        "study": key["study"],
        "corpus": key["corpus"],
    }
    jobs = read(ranks)
    fitted = read(parent)
    if (
        jobs["inputs"]["cases"] != inputs["cases"]
        or fitted["evidence"]["measured"] != inputs["complete"]
    ):
        raise ValueError("Missing same-skill or group measurements")
    reg = out / "registration.json"
    if reg.exists():
        if read(reg)["inputs"] != inputs:
            raise ValueError("Different witness trial")
        return reg
    data = {v["sample_id"]: v for v in read(corpus) if v["sample_id"] in proto["train"]}
    manifest = read(cap / "registration.json")
    initial, comparisons, _ = baseline(base, manifest)
    _, paired, _ = joint_evidence(base, initial, comparisons)
    corrected = read(cap / "complete.json")["scores"]
    ev = LoCoMoEvaluator(SimpleNamespace())
    confirmed = {}
    for row, (w, bits, old) in zip(fitted["observations"], paired, strict=True):
        low, high = [cell_scores(base, cap, w, k, corrected)[1] for k in bits]
        ds = [h["f1"] - l["f1"] for l, h in zip(low, high, strict=True)]
        after = [
            surviving_credit(
                v, l, h, ev.get_ground_truth(data[w["dialogue"]]["qa"][idx])
            )
            for v, l, h, idx in zip(old, low, high, w["questions"], strict=True)
        ]
        if (
            row["window"] != w["name"]
            or abs(mean(ds) - row["raw"]) > 1e-12
            or abs(mean(after) - row["credit"]) > 1e-12
        ):
            raise ValueError("Per-question repaired targets differ from the frozen fit")
        confirmed.update(
            {
                (w["name"], tuple(bits), i): v
                for i, v in zip(w["questions"], after, strict=True)
                if v
            }
        )
    expected = {
        (j["window"], tuple(j["bits"]), j["index"]): j["credit"] for j in jobs["jobs"]
    }
    if confirmed != expected:
        raise ValueError(
            "Physical rank inventory differs from repaired per-question labels"
        )
    records = []
    vocab = {}
    for i, j in enumerate(jobs["jobs"]):
        win = j["winner"]
        sid, start = j["window"].rsplit("_s", 1)
        history = data[sid]["conversation"]
        pivots = [k for k, (a, b) in enumerate(zip(*j["bits"], strict=True)) if a != b]
        (pivot,) = pivots
        pair = []
        for arm in (win, 1 - win):
            item = j["texts"][arm]
            t = (
                clean(
                    item[1],
                    history,
                    history[f"session_{int(start) + item[0]}_date_time"],
                )
                if item
                else None
            )
            pair.append(vocab.setdefault(t, len(vocab)) if t else None)
        if j["exclusive_focal"][win] and (
            j["texts"][win][0] != pivot or pair[0] is None
        ):
            raise ValueError("Missing actual focal text")
        records.append(
            {
                "job": i,
                "dialogue": sid,
                "window": j["window"],
                "index": j["index"],
                "bits": j["bits"],
                "rank": j["rank"],
                "credit": j["credit"],
                "eligible": j["exclusive_focal"][win],
                "keys": pair,
            }
        )
    register(reg, {"inputs": inputs, "records": records, "texts": list(vocab)})
    return reg


def encode(base, study, corpus):
    reg = registration(base, study, corpus)
    out = reg.parent
    v = read(reg)
    blob = out / "passages.npz"
    receipt = out / "encoder.json"
    if receipt.exists():
        if (
            read(receipt)["registration"] != file_hash(reg)
            or file_hash(blob) != read(receipt)["vectors"]
        ):
            raise ValueError("Passage archive changed")
        return
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= 0.30
        or int(os.environ["MEMSKILL_ENCODER_MAX_BATCH"]) != 2
    ):
        raise ValueError("Expected one duty-limited CUDA encoder")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.65 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    proto = read(study / "protocol.json")
    set_retriever_flash_attn(False)
    _, model = init_context_model(proto["retriever"])
    if not all(p.is_cuda for p in model.parameters()):
        raise ValueError("Encoder not on leased GPU")

    def normalized(ls):
        return F.normalize(
            torch.as_tensor(
                encode_batch(model, ls, 2, normalize=True), device="cuda"
            ).float(),
            dim=1,
        )

    sid = min(proto["train"])
    history = next(v["conversation"] for v in read(corpus) if v["sample_id"] == sid)
    sample = out.parent / "evidence-transport/turns" / f"{sid}_s5.npz"
    with np.load(sample, allow_pickle=False) as arr:
        if arr["ids"].tolist() != [t["dia_id"] for t in history["session_5"]]:
            raise ValueError("Calibration turn order changed")
        old = torch.as_tensor(arr["vectors"], device="cuda")
    check = normalized(texts(history, history["session_5"]))
    error = float((check - old).abs().max())
    cosine = float(F.cosine_similarity(check, old).min())
    if error > 1e-4 or cosine < 0.999999:
        raise ValueError(f"Turn encoding parity failed: {error}, {cosine}")
    matrix = normalized(v["texts"])
    if (
        matrix.shape != (len(v["texts"]), old.shape[1])
        or not torch.isfinite(matrix).all()
    ):
        raise ValueError("Incomplete passage vectors")
    with TemporaryDirectory(dir=out) as tmp:
        f = Path(tmp) / blob.name
        np.savez(f, vectors=matrix.cpu().numpy(), texts=digest(v["texts"]))
        if not blob.exists():
            os.link(f, blob)
        else:
            with np.load(blob, allow_pickle=False) as a:
                if str(a["texts"]) != digest(v["texts"]) or not np.array_equal(
                    a["vectors"], matrix.cpu().numpy()
                ):
                    raise ValueError("Conflicting cached passage vectors")
    register(
        receipt,
        {
            "registration": file_hash(reg),
            "vectors": file_hash(blob),
            "parity_session": file_hash(sample),
            "max_error": error,
            "min_cosine": cosine,
            "device": torch.cuda.get_device_name(),
            "backend": os.environ["MEMSKILL_BACKEND_ID"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("--encode", action="store_true")
    a = p.parse_args()
    (encode if a.encode else registration)(a.base, a.study, a.corpus)
