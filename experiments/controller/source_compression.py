"""Measure what adjacent-turn grouping preserves before the writer sees a training span."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from experiments.controller.granularity_placement import fit_preference
from src.data_processing.locomo import LoCoMoProcessor
from src.rag_utils import get_embeddings, init_context_model, set_retriever_flash_attn
from src.replay import dated_turn_groups


FEATURES = ("mean_cosine_loss", "worst_quarter_cosine_loss", "change_in_item_crowding")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(path, report):
    """Atomically create a receipt. Concurrent repeat runs must agree."""
    with TemporaryDirectory(dir=path.parent) as temp:
        fresh = Path(temp) / path.name
        fresh.write_text(json.dumps(report, indent=2))
        try:
            os.link(fresh, path)
        except FileExistsError:
            if json.loads(path.read_bytes()) != json.loads(fresh.read_bytes()):
                raise ValueError(f"Different registration already exists at {path}")


def compression_features(singles, pairs):
    if singles.ndim != 2 or pairs.ndim != 2 or singles.shape[1] != pairs.shape[1] or min(len(singles), len(pairs)) == 0:
        raise ValueError("Expected nonempty embedding matrices of the same width")
    singles, pairs = (F.normalize(v.double(), dim=1) for v in (singles, pairs))
    loss = 1. - (singles @ pairs.T).amax(dim=1).clamp(-1., 1.)

    def crowding(v):
        if len(v) == 1:
            return v.new_zeros(())
        similarities = v @ v.T
        similarities.fill_diagonal_(-torch.inf)
        return similarities.amax(dim=1).mean()

    return torch.stack((loss.mean(), loss.topk(math.ceil(len(loss)/4)).values.mean(), crowding(pairs) - crowding(singles)))


def collect_features(study, corpus, dest):
    protocol = json.loads((study / "protocol.json").read_bytes())
    raw = corpus.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["input_sha256"] or not protocol.get("equal_sources"):
        raise ValueError("Expected the original LoCoMo corpus and equal-source skill pair")
    identity = {"features": list(FEATURES), "corpus": protocol["input_sha256"], "retriever": protocol["retriever"]}
    train = set(protocol["train"])
    if len(train) != 6 or train & set(protocol["validation"] + protocol["test"]):
        raise ValueError("Expected six disjoint training dialogues")
    if dest.exists():
        saved = json.loads(dest.read_bytes())
        keys = [(r["dialogue"], r["session"]) for r in saved["rows"]]
        if (any(saved.get(k) != v for k,v in identity.items()) or len(keys) != 48
                or set(keys) != {(sid,n) for sid in train for n in range(1,9)}):
            raise ValueError("Different source feature registration already exists")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select one CUDA GPU for source embeddings")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(.76*total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise RuntimeError("Insufficient GPU memory for bounded Qwen encoding")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    retriever = protocol["retriever"]
    _, model = init_context_model(retriever)
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("Expected Qwen on the leased GPU")
    proc = LoCoMoProcessor(chunk_mode="full-session")
    rows = []
    for convo in json.loads(raw):
        sid = convo["sample_id"]
        if sid not in train:
            continue
        for n in range(1, 9):
            k = f"session_{n}"
            source, = proc._extract_session_chunks(convo["conversation"][k], convo["conversation"][k + "_date_time"])
            a, b = (dated_turn_groups(source, size) for size in (1, 2))
            texts = list(dict.fromkeys(a + b))
            vectors = torch.as_tensor(get_embeddings(retriever, texts, "context", batch_size=2), device="cuda")
            index = {text: i for i, text in enumerate(texts)}
            features = compression_features(vectors[[index[t] for t in a]], vectors[[index[t] for t in b]])
            rows.append({"dialogue": sid, "session": n, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "singles": len(a), "pairs": len(b), "features": features.tolist()})
    if len({(r["dialogue"], r["session"]) for r in rows}) != 48:
        raise ValueError("Incomplete training features")
    register(dest, {**identity,
                    "allocator_limit_mib": budget // 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20,
                    "rows": rows})


def fit_actor(x, effects):
    """Signed verified paired F1, standardization on training spans, symmetric logit prior."""
    center = x.mean(0)
    scale = x.std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    design = torch.cat((torch.ones_like(x[:, :1]), (x - center) / scale), dim=1)
    active = effects != 0
    weights = (fit_preference(design[active], effects[active], torch.ones_like(effects[active]))
               if active.any() else design.new_zeros(design.shape[1]))
    return center, scale, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "corpus", "dest"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    collect_features(args.study, args.corpus, args.dest)
