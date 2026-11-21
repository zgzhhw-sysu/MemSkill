"""Learn when to index paired turns from leave-one-dialogue-out, full-history retrieval effects."""

import argparse
import hashlib
import math
import os
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn
from src.replay import dated_group_parts

PLAN = Path(__file__).with_name("protocols") / "turn_key_credit.json"


def keyed_scores(parent_scores, children, parents):
    """Only two-turn parents get extra keys. Rotate the same key sets within the current span."""
    keyed, rotated = parent_scores.clone(), parent_scores.clone()
    if parents:
        best = children.amax(dim=2)
        keyed[:, parents] = 0.5 * (parent_scores[:, parents] + best)
        rotated[:, parents] = 0.5 * (parent_scores[:, parents] + best.roll(1, dims=1))
    return keyed, rotated


def source_features(parent, parts, paired_fraction):
    if not len(parent):
        return [0.0, 0.0, paired_fraction]
    losses = (1 - (parent[:, None, :] * parts).sum(-1).clamp(-1, 1)).flatten()
    return [
        float(losses.mean()),
        float(losses.topk(math.ceil(len(losses) / 4)).values.mean()),
        paired_fraction,
    ]


def fit(rows):
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float64)
    y = x.new_tensor([r["target"] for r in rows])
    mu, scale = x.mean(0), x.std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    z = (x - mu) / scale
    slope = torch.linalg.solve(
        z.T @ z + len(x) * torch.eye(x.shape[1], dtype=x.dtype, device=x.device),
        z.T @ (y - y.mean()),
    )
    return {
        "mean": mu.tolist(),
        "scale": scale.tolist(),
        "slope": slope.tolist(),
        "intercept": float(y.mean()),
    }


def offset(model, features):
    pred = model["intercept"] + sum(
        w * (v - m) / s
        for v, m, s, w in zip(
            features, model["mean"], model["scale"], model["slope"], strict=True
        )
    )
    return max(-2.0, min(2.0, pred / 0.05))


def read_metrics(rank, turns, words, gold):
    # Each row of turns has one or two source IDs. The extra gold column represents padding.
    ids = turns[rank].flatten(1).sort(dim=1).values
    distinct = torch.ones_like(ids, dtype=torch.bool)
    distinct[:, 1:] = ids[:, 1:] != ids[:, :-1]
    hits = (gold.gather(1, ids) & distinct).sum(1)
    denom = gold.sum(1)
    return hits / denom, hits == denom, words[rank].sum(1)


def history(convo, model):
    c = convo["conversation"]
    proc = LoCoMoProcessor(chunk_mode="full-session")
    numbers = sorted(
        int(k[8:]) for k in c if k.startswith("session_") and k[8:].isdigit()
    )
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("Incomplete chronological history")
    texts, groups, turn_ids = [], [], []
    for n in numbers:
        k = f"session_{n}"
        (src,) = proc._extract_session_chunks(c[k], c[k + "_date_time"])
        single, pair = [dated_group_parts(src, k) for k in (1, 2)]
        if {i for _, parts in single for i, _ in parts} != set(range(len(c[k]))):
            raise ValueError("Some source turns were not indexed")
        pos = len(turn_ids)
        turn_ids.extend(t["dia_id"] for t in c[k])
        groups.append((n, src, single, pair, pos))
        for parent, parts in single + pair:
            texts.append(parent)
            texts.extend(text for _, text in parts)
    unique = list(dict.fromkeys(texts))
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError("Repeated source turn ID")
    turn_pos = {name: i for i, name in enumerate(turn_ids)}
    valid = []
    omitted = Counter()
    for i, q in enumerate(convo["qa"]):
        if int(q["category"]) == 5:
            continue
        ev = set(q.get("evidence") or [])
        if not ev or not ev <= turn_pos.keys():
            omitted[str(q["category"])] += 1
            continue
        valid.append((i, q, [turn_pos[t] for t in sorted(ev)]))

    def vectors(strings):
        arr = encode_batch(model, strings, 2, normalize=True)
        return F.normalize(torch.as_tensor(arr, device="cuda").float(), dim=1)

    enc = vectors(unique)
    lookup = {t: i for i, t in enumerate(unique)}
    queries = vectors([q["question"] for _, q, _ in valid])
    sim = queries @ enc.T
    gold = torch.zeros((len(valid), len(turn_ids) + 1), dtype=torch.bool, device="cuda")
    for j, (_, _, ids) in enumerate(valid):
        gold[j, ids] = True
    metric_types = [str(q["category"]) for _, q, _ in valid]
    mats = []
    doc_ids = []
    lengths = []
    singles = []
    pairs = []
    permuted = []
    meta = []
    head = 0
    for n, src, single, pair, pos in groups:
        for coll in (single, pair):
            for parent, parts in coll:
                ids = [pos + i for i, _ in parts]
                doc_ids.append([*ids, *([len(turn_ids)] * (2 - len(ids)))])
                lengths.append(len(parent.split()))
        parent = [lookup[t] for t, _ in pair]
        multi = [i for i, (_, parts) in enumerate(pair) if len(parts) == 2]
        children = torch.tensor(
            [[lookup[t] for _, t in pair[i][1]] for i in multi],
            dtype=torch.long,
            device="cuda",
        ).reshape(-1, 2)
        base = sim[:, parent]
        keyed, rot = keyed_scores(base, sim[:, children], multi)
        a = head + len(single)
        end = a + len(pair)
        singles.append((head, a))
        pairs.append((a, end))
        mats.extend([sim[:, [lookup[t] for t, _ in single]], base])
        permuted.append((keyed, rot))
        meta.append(
            {
                "dialogue": convo["sample_id"],
                "session": n,
                "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                "features": source_features(
                    enc[[parent[j] for j in multi]],
                    enc[children],
                    len(multi) / len(pair),
                ),
                "linked": int(gold[:, pos : pos + len(c[f"session_{n}"])].any(1).sum()),
                "keys": 2 * len(multi),
            }
        )
        head = end
    raw = torch.cat(mats, dim=1)
    turns = torch.tensor(doc_ids, device="cuda")
    words = torch.tensor(lengths, device="cuda")

    # A stable ranking within each session suffices: no item below the local fifth can enter the global top five.
    def best(scores, begin):
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :5]
        return scores.gather(1, order), order + begin

    choices = []
    for (a, b), (c0, d), (keyed, rot) in zip(singles, pairs, permuted, strict=True):
        choices.append(
            [
                best(raw[:, a:b], a),
                best(raw[:, c0:d], c0),
                best(keyed, c0),
                best(rot, c0),
            ]
        )

    def retrieve(schedule):
        parts = [row[action] for row, action in zip(choices, schedule, strict=True)]
        values, indices = [torch.cat(v, dim=1) for v in zip(*parts, strict=True)]
        rank = indices.gather(
            1, values.argsort(dim=1, descending=True, stable=True)[:, :5]
        )
        return read_metrics(rank, turns, words, gold)

    off = [0] * len(groups)
    base = retrieve(off)[0]
    for j, row in enumerate(meta):
        change = off.copy()
        change[j] = 2
        diff = retrieve(change)[0] - base
        row["target"] = float(diff.sum()) / max(row["linked"], 1)
        row["changed_questions"] = int((diff != 0).sum())
    return meta, retrieve, metric_types, dict(omitted)


def paired_interval(counts, differences):
    """Question-weighted paired interval from resampling complete dialogues."""
    n = len(counts)
    if not 2 <= n <= 6:
        raise ValueError("Expected two to six training dialogues")
    draws = np.array(list(product(range(n), repeat=n)))
    counts = np.asarray(counts)
    weighted = counts * np.asarray(differences)
    return np.quantile(
        weighted[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
    ).tolist()


def screen_summary(dest):
    histories = list(read(dest / "screen.json")["histories"].values())

    def combine(groups):
        n = sum(g["n"] for g in groups)
        values = {
            arm: {
                k: sum(g["n"] * g["arms"][arm][k] for g in groups) / n
                for k in ("recall", "complete", "read_words")
            }
            for arm in groups[0]["arms"]
        }
        return {"questions": n, "arms": values}

    full = [h["all"] for h in histories]
    groups = sorted({k for h in histories for k in h["types"]})
    diffs = {}
    for first, other in (
        ("gate", "same_count"),
        ("gate", "single"),
        ("keyed", "paired"),
        ("keyed", "rotated_keys"),
    ):
        d = [g["arms"][first]["recall"] - g["arms"][other]["recall"] for g in full]
        diffs[first + "-" + other] = {
            "dialogues_positive": sum(v > 1e-12 for v in d),
            "interval": paired_interval([g["n"] for g in full], d),
        }
    register(
        dest / "screen-summary.json",
        {
            "screen": file_hash(dest / "screen.json"),
            "all": combine(full),
            "by_type": {
                k: combine([h["types"][k] for h in histories if k in h["types"]])
                for k in groups
            },
            "unresolved_evidence": sum(sum(h["excluded"].values()) for h in histories),
            "spans": sum(h["spans"] for h in histories),
            "keyed_spans": sum(h["keyed_spans"] for h in histories),
            "paired_dialogue_differences": diffs,
        },
    )


def screen(study, corpus, dest):
    dest.mkdir(exist_ok=True, parents=True)
    protocol = read(study / "protocol.json")
    if (
        file_hash(corpus) != protocol["input_sha256"]
        or len(protocol["train"]) != 6
        or set(protocol["train"]) & set(protocol["validation"] + protocol["test"])
        or not protocol["equal_sources"]
    ):
        raise ValueError(
            "Expected six training-only histories and the frozen skill versions"
        )
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or float(os.environ["MEMSKILL_ENCODER_DUTY"]) > 0.30
    ):
        raise RuntimeError("Use a bounded single GPU")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.65 * total), free - 2048 * 2**20)
    if budget < 1800 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, model = init_context_model(protocol["retriever"])
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("Encoder must reside on CUDA")
    identity = {
        "plan": file_hash(PLAN),
        "protocol": file_hash(study / "protocol.json"),
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
    }
    register(dest / "manifest.json", identity)
    data = {
        c["sample_id"]: c for c in read(corpus) if c["sample_id"] in protocol["train"]
    }
    panels = {}
    for sid in protocol["train"]:
        panels[sid] = history(data[sid], model)
        print(sid, len(panels[sid][0]), len(panels[sid][2]), flush=True)
    all_rows = [r for rows, *_ in panels.values() for r in rows]
    fits = {}
    forecasts = []
    results = {}
    with torch.device("cuda"):
        for sid, (rows, retrieve, categories, omitted) in panels.items():
            fit_rows = [r for r in all_rows if r["dialogue"] != sid]
            model_fit = fit(fit_rows)
            fits[sid] = dict(
                model_fit, donors=sorted({r["dialogue"] for r in fit_rows})
            )
            changes = [offset(model_fit, r["features"]) for r in rows]
            use = [2 if v > 0 else 0 for v in changes]
            for row, shift in zip(rows, changes, strict=True):
                forecasts.append(
                    {
                        k: row[k]
                        for k in ("dialogue", "session", "source_sha256", "features")
                    }
                    | {"offset": shift}
                )

            def metrics(schedule, retrieve=retrieve):
                vals = retrieve(schedule)
                return torch.stack([v.double() for v in vals], dim=1)

            sums = {
                "single": metrics([0] * len(rows)),
                "paired": metrics([1] * len(rows)),
                "keyed": metrics([2] * len(rows)),
                "rotated_keys": metrics([3] * len(rows)),
                "gate": metrics(use),
            }
            sums["same_count"] = torch.stack(
                [metrics(use[i:] + use[:i]) for i in range(len(rows))]
            ).mean(0)

            def summary(indices, sums=sums):
                return {
                    "n": len(indices),
                    "arms": {
                        k: dict(
                            zip(
                                ("recall", "complete", "read_words"),
                                v[indices].mean(0).tolist(),
                                strict=True,
                            )
                        )
                        for k, v in sums.items()
                    },
                }

            results[sid] = {
                "all": summary(list(range(len(categories)))),
                "excluded": omitted,
                "spans": len(rows),
                "keyed_spans": use.count(2),
                "types": {
                    k: summary([i for i, v in enumerate(categories) if v == k])
                    for k in set(categories)
                },
            }
    register(
        dest / "forecast.json",
        {
            "manifest": file_hash(dest / "manifest.json"),
            "fits": fits,
            "forecasts": forecasts,
            "training_rows": all_rows,
        },
    )
    register(
        dest / "screen.json",
        {"forecast": file_hash(dest / "forecast.json"), "histories": results},
    )
    screen_summary(dest)
    print("Frozen full-history forecasts", len(forecasts), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "corpus", "dest"):
        p.add_argument(key, type=Path)
    screen(**vars(p.parse_args()))
