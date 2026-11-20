"""Build source-only retrieval simulations and checked question-level donor outcomes."""

import argparse
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from experiments.controller.capped_source_policy import (
    baseline,
    cell_scores,
    joint_evidence,
    surviving_credit,
)
from experiments.controller.exchange_paths import Ranking
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passages import clean
from src.data_processing.locomo import LoCoMoProcessor
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest
from src.replay import dated_group_parts

PLAN = Path(__file__).with_name("protocols") / "query_displacement.json"


def registration(base, study, corpus):
    src = base / "source-compression"
    cap = src / "capped-source-credit"
    out = src / "query-displacement"
    out.mkdir(exist_ok=True)
    parent = src / "witness-passages/evidence-advantage/group"
    fitted = cap / "policy/joint-repaired/fit.json"
    known = src / "witness-passages/registration.json"
    inputs = {
        k: file_hash(p)
        for k, p in {
            "plan": PLAN,
            "protocol": study / "protocol.json",
            "corpus": corpus,
            "local": parent / "forecast.json",
            "prefix": parent / "long-prefix/forecast.json",
            "cases": known,
            "fit": fitted,
            "cap": cap / "registration.json",
            "complete": cap / "complete.json",
        }.items()
    }
    reg = out / "registration.json"
    if reg.is_file():
        if read(reg)["inputs"] != inputs:
            raise ValueError("Changed query experiment")
        return reg
    proto = read(study / "protocol.json")
    ids = set(proto["train"])
    if (
        len(ids) != 6
        or ids & set(proto["validation"] + proto["test"])
        or proto["input_sha256"] != inputs["corpus"]
    ):
        raise ValueError("Different training split")
    data = {v["sample_id"]: v for v in read(corpus) if v["sample_id"] in ids}
    saved = read(known)
    if (
        saved["inputs"]["parent"] != inputs["fit"]
        or saved["inputs"]["corpus"] != inputs["corpus"]
    ):
        raise ValueError("Different verified donor bank")
    expected = {
        (r["window"], tuple(r["bits"]), r["index"]): r["credit"]
        for r in saved["records"]
    }
    initial, contrasts, _ = baseline(base, read(cap / "registration.json"))
    _, paired, _ = joint_evidence(base, initial, contrasts)
    corrected = read(cap / "complete.json")["scores"]
    ev = LoCoMoEvaluator(SimpleNamespace())
    rows = []
    queries = {}
    for obs, (w, bits, old) in zip(read(fitted)["observations"], paired, strict=True):
        sid = w["dialogue"]
        h = data[sid]["conversation"]
        pivot = next(i for i, (l, r) in enumerate(zip(*bits, strict=True)) if l != r)
        if bits[0][pivot] != "0" or bits[1][pivot] != "1" or obs["stage"] != pivot:
            raise ValueError("Wrong ordered physical action")
        n = w["start"] + pivot
        turns = [t["dia_id"] for t in h[f"session_{n}"]]
        low, high = [cell_scores(base, cap, w, k, corrected)[1] for k in bits]
        subset = []
        for idx, l, r, orig in zip(w["questions"], low, high, old, strict=True):
            q = data[sid]["qa"][idx]
            key = f"{sid}/{idx}"
            phrase = clean(q["question"], h, "")
            if queries.setdefault(key, phrase) != phrase:
                raise ValueError("Ambiguous donor question")
            c = surviving_credit(orig, l, r, ev.get_ground_truth(q))
            if expected.get((w["name"], tuple(bits), idx), 0.0) != c:
                raise ValueError("Wrong rule-2 label")
            subset.append(
                {
                    "history": sid,
                    "window": w["name"],
                    "bits": bits,
                    "index": idx,
                    "key": key,
                    "node": f"{sid}_s{n}",
                    "stage": pivot,
                    "linked": [i for i, t in enumerate(turns) if t in q["evidence"]],
                    "raw": r["f1"] - l["f1"],
                    "credit": c,
                }
            )
        if any(
            abs(mean(j[k] for j in subset) - obs[k]) > 1e-12 for k in ("raw", "credit")
        ):
            raise ValueError("Did not reproduce measured effects")
        rows.extend(subset)
    local = read(parent / "forecast.json")["sources"]
    prefix = read(parent / "long-prefix/forecast.json")["sources"]
    nodes = {n: v["text"] for n, v in local.items()}
    if any(n in nodes and nodes[n] != v for n, v in prefix.items()):
        raise ValueError("Mismatched shared source")
    nodes.update(prefix)
    if (
        len(rows),
        len(queries),
        sum(r["credit"] != 0 for r in rows),
        len(local),
        len(prefix),
    ) != (452, 179, 7, 128, 99):
        raise ValueError("Incomplete registered source or donor coverage")
    register(reg, {"inputs": inputs, "rows": rows, "queries": queries, "nodes": nodes})
    return reg


def store_vectors(path, **values):
    with TemporaryDirectory(dir=path.parent) as d:
        fresh = Path(d) / path.name
        np.savez(fresh, **values)
        try:
            os.link(fresh, path)
        except FileExistsError:
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(values) or any(
                    not np.array_equal(old[k], v) for k, v in values.items()
                ):
                    raise ValueError("Inconsistent repeated embedding")


def encode_questions(queries, encoder):
    """Keep the BF16 padding and batch order independent across held-history folds."""
    keys = sorted(queries)
    groups = {}
    for key in keys:
        groups.setdefault(key.split("/", 1)[0], []).append(key)
    blocks = [encoder(tuple(queries[k] for k in group)) for group in groups.values()]
    return keys, np.concatenate(blocks)


def encode_groups(source, expected, h, sk, model):
    record = {"source": expected, "ids": [t["dia_id"] for t in h[sk]]}
    for k in (1, 2):
        chunks = dated_group_parts(source, k)
        texts = tuple(clean(t, h, h[sk + "_date_time"]) for t, _ in chunks)
        memberships = np.zeros((len(chunks), len(h[sk])), dtype=bool)
        for i, (_, parts) in enumerate(chunks):
            for j, _ in parts:
                memberships[i, j] = True
        if not memberships.any(axis=0).all():
            raise ValueError("Missing source turns")
        record[f"texts{k}"] = digest(texts)
        record[f"map{k}"] = memberships
        record[f"vec{k}"] = model.encode(texts)
    return record


def encode(base, study, corpus, limit):
    reg = registration(base, study, corpus)
    out = reg.parent
    bank = out / "sources"
    bank.mkdir(exist_ok=True)
    packet = read(reg)
    proto = read(study / "protocol.json")
    data = {
        d["sample_id"]: d["conversation"]
        for d in read(corpus)
        if d["sample_id"] in proto["train"]
    }
    query = out / "queries_by_history.npz"
    complete = out / "encoded.json"
    if complete.is_file():
        return
    model = Ranking(proto["retriever"])
    if not query.exists():
        keys, vectors = encode_questions(packet["queries"], model.encode)
        texts = tuple(packet["queries"][k] for k in keys)
        store_vectors(query, keys=keys, vectors=vectors, fingerprint=digest(texts))
    proc = LoCoMoProcessor(chunk_mode="full-session")
    count = 0
    donors = {r["node"] for r in packet["rows"]}
    for name, expected in sorted(
        packet["nodes"].items(), key=lambda item: (item[0] not in donors, item[0])
    ):
        f = bank / (name + ".npz")
        if f.exists():
            continue
        sid, n = name.rsplit("_s", 1)
        h = data[sid]
        sk = "session_" + n
        (source,) = proc._extract_session_chunks(h[sk], h[sk + "_date_time"])
        if digest(source) != expected:
            raise ValueError("Different pre-write source")
        store_vectors(f, **encode_groups(source, expected, h, sk, model))
        count += 1
        if count == limit:
            break
    files = sorted(bank.glob("*.npz"))
    print("encoded", len(files), "of", len(packet["nodes"]), flush=True)
    if {f.stem for f in files} == set(packet["nodes"]):
        register(
            complete,
            {
                "registration": file_hash(reg),
                "queries": file_hash(query),
                "sources": {f.stem: file_hash(f) for f in files},
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.limit:
        encode(a.base, a.study, a.corpus, a.limit)
    else:
        print(registration(a.base, a.study, a.corpus))
