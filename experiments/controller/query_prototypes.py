"""Transfer verified question-level skill credit to similar source spans without target questions."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

from experiments.controller import file_hash, read as read_json, register
from experiments.controller.audit_source_compression import check_forecasts, panel
from experiments.controller.paired_policy import _shift_for_mean_probability, _training_states
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn
from src.replay import dated_turn_groups


def read(p):
    return read_json(p)


def mask_speakers(text, conversation):
    names = [conversation[k] for k in ("speaker_a", "speaker_b")]
    pattern = (
        r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?!\w)"
    )
    return re.sub(pattern, "the speaker", text, flags=re.IGNORECASE)


def prototype_logits(scores, center, scale, effects):
    affinity = ((scores - center) / scale).clamp(0.0, 3.0)
    return 2 * (affinity @ effects) / (1.0 + affinity @ effects.abs())


def freeze(study, corpus, source):
    dest = source / "query-prototypes"
    parent = read(source / "frozen.json")
    raw = corpus.read_bytes()
    proto = read(study / "protocol.json")
    inputs = {
        "parent": file_hash(source / "frozen.json"),
        "plan": file_hash(dest / "plan.json"),
        "corpus": hashlib.sha256(raw).hexdigest(),
    }
    if inputs["corpus"] != proto["input_sha256"] or parent["inputs"]["features.json"] != file_hash(
        source / "features.json"
    ):
        raise ValueError("Unexpected source or corpus for question prototypes")
    if (dest / "frozen.json").exists():
        if read(dest / "frozen.json")["inputs"] != inputs:
            raise ValueError("Different question-prototype registration")
        return
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("Use one CUDA lease for prototype embeddings and fitting")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.76 * total), free - 2048 * 2**20)
    if budget < 2048 * 2**20:
        raise RuntimeError("Insufficient memory for bounded Qwen encoding")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    features = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    data = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in proto["train"]}
    training, receipts, backend = panel(study, corpus, features, (1, 3), supplement=True)
    if receipts != parent["prior_hashes"] or backend != parent["backend"]:
        raise ValueError("Verified prototype labels differ from the registered study")
    witnesses = []
    for r in training:
        sid = r["dialogue"]
        for orig, effect in sorted(r["credits"].items()):
            if effect:
                q = data[sid]["qa"][orig]["question"]
                witnesses.append(
                    {
                        "dialogue": sid,
                        "session": r["session"],
                        "original_index": orig,
                        "effect": effect,
                        "query": mask_speakers(q, data[sid]["conversation"]),
                    }
                )
    if not witnesses:
        raise ValueError("No verified training prototypes")
    set_retriever_flash_attn(False)
    _, model = init_context_model(proto["retriever"])
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("Expected a GPU-backed frozen encoder")

    def embed(texts):
        return torch.as_tensor(
            encode_batch(model, texts, 2, normalize=True), device="cuda", dtype=torch.float64
        )

    queries = embed([r["query"] for r in witnesses])
    proc = LoCoMoProcessor(chunk_mode="full-session")
    keys = sorted(features)
    similarity = []
    for sid, n in keys:
        convo = data[sid]["conversation"]
        k = f"session_{n}"
        (text,) = proc._extract_session_chunks(convo[k], convo[k + "_date_time"])
        if hashlib.sha256(text.encode()).hexdigest() != features[sid, n]["source_sha256"]:
            raise ValueError("Question probe source does not match the pre-action text")
        items = [mask_speakers(t, convo) for t in dated_turn_groups(text, 1)]
        similarity.append((queries @ embed(items).T).amax(dim=1))
    scores = torch.stack(similarity)
    fits, probs = {}, []
    with torch.device("cuda"):
        check_forecasts(features, parent)
        for held in sorted(data):
            fold = [r for r in training if r["dialogue"] != held]
            columns = [j for j, r in enumerate(witnesses) if r["dialogue"] != held]
            v = scores[:, columns]
            idx = [keys.index((r["dialogue"], r["session"])) for r in fold]
            center, scale = v[idx].mean(0), v[idx].std(0, unbiased=False)
            scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
            effects = v.new_tensor([witnesses[j]["effect"] for j in columns])
            logits = prototype_logits(v, center, scale, effects)
            weights = v.new_tensor(parent["fits"][held]["terminal"])
            x, offset = _training_states(fold)
            base = offset + x @ weights
            shift = _shift_for_mean_probability(base, logits[idx] - base)
            fits[held] = {
                "training_dialogues": sorted(set(data) - {held}),
                "training_witnesses": columns,
                "center": center.tolist(),
                "scale": scale.tolist(),
                "terminal": weights.tolist(),
                "training_mean": float(logits[idx].sigmoid().mean()),
                "matched_logit_shift": shift,
            }
            for i, (sid, n) in enumerate(keys):
                if sid == held:
                    probs.append({"dialogue": sid, "session": n, "p_activity": float(logits[i].sigmoid())})
    published = sorted(p.parent.name for p in (source / "factorial").glob("*/measurement.json"))
    report = {
        "inputs": inputs,
        "witnesses": witnesses,
        "fits": fits,
        "forecasts": probs,
        "already_published_windows": published,
        "similarities": [
            {"dialogue": s, "session": n, "values": v.tolist()}
            for (s, n), v in zip(keys, scores, strict=True)
        ],
        "verified_links": sum(len(r["credits"]) for r in training),
        "peak_mib": torch.cuda.max_memory_reserved() // 2**20,
    }
    register(dest / "frozen.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("study", "corpus", "source"):
        p.add_argument(arg, type=Path)
    freeze(**vars(p.parse_args()))
