"""Forecast paired writer utility from counterfactual query rankings in the current span."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from experiments.controller import file_hash, read, register
from experiments.controller.evidence_transport_data import sources
from experiments.controller.passage_advantage import evaluate
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.passage_evidence_prefix import evaluate as evaluate_prefix
from experiments.controller.query_displacement_data import registration
from experiments.controller.witness_passage_policy import limited
from src.evaluation_cache import digest

ARMS = (
    "credit",
    "raw",
    "raw_mechanism",
    "untranslated",
    "source_only",
    "reassigned",
    "parent",
    "native",
    "terminal",
)


def reciprocal_ranks(queries, group):
    vectors, member = group
    indices = (queries @ vectors.T).argsort(dim=1, descending=True, stable=True)[:, :5]
    weights = 1 / torch.arange(
        1, indices.shape[1] + 1, device=queries.device, dtype=queries.dtype
    )
    return (member[indices] * weights[None, :, None]).amax(dim=1)


def turn_mass(queries, group):
    vectors, members = group
    logits = (queries @ vectors.T) / 0.08
    values = (logits - logits.amax(1, keepdim=True)).exp()
    weights = (values @ members.to(values.dtype)) / members.sum(0)
    return weights / weights.sum(1, keepdim=True)


def contrast(queries, groups, indices, mask, metric="rank"):
    utility = {"rank": reciprocal_ranks, "mass": turn_mass}[metric]
    ranks = [utility(queries, g).gather(1, indices) for g in groups]
    return ((ranks[1] - ranks[0]) * mask).sum(dim=1) / mask.sum(dim=1)


def load_banks(base, reg, device):
    src = base / "source-compression"
    out = reg.parent
    manifest = read(out / "encoded.json")
    packet = read(reg)
    if manifest["registration"] != file_hash(reg) or manifest["queries"] != file_hash(
        out / "queries_by_history.npz"
    ):
        raise ValueError("Changed question bank")
    original = sources(src / "evidence-transport", device)
    extra = sources(src / "witness-passages/evidence-advantage/prefix-sources", device)
    parent = read(src / "witness-passages/evidence-advantage/group/forecast.json")
    locked = read(
        src / "witness-passages/evidence-advantage/prefix-sources/complete.json"
    )
    if (
        {k: {"text": v[2], "vectors": v[3]} for k, v in original.items()}
        != parent["sources"]
        or {k: v[3] for k, v in extra.items()} != locked["files"]
        or original.keys() & extra.keys()
    ):
        raise ValueError("Frozen turn embeddings changed")
    turns = original | extra
    banks = {}
    with np.load(out / "queries_by_history.npz", allow_pickle=False) as f:
        keys = f["keys"].tolist()
        if keys != sorted(packet["queries"]) or str(f["fingerprint"]) != digest(
            [packet["queries"][k] for k in keys]
        ):
            raise ValueError("Question order mismatch")
        qvec = F.normalize(
            torch.as_tensor(f["vectors"], device=device, dtype=torch.float64)
        )
        queries = dict(zip(keys, qvec, strict=True))
    if set(turns) != set(manifest["sources"]) or set(turns) != set(packet["nodes"]):
        raise ValueError("Source coverage differs")
    for name, (ids, vec, src_hash, _) in turns.items():
        path = out / "sources" / (name + ".npz")
        if file_hash(path) != manifest["sources"][name]:
            raise ValueError("Different source group vectors")
        with np.load(path, allow_pickle=False) as f:
            if (
                src_hash != packet["nodes"][name]
                or str(f["source"]) != src_hash
                or ids != f["ids"].tolist()
            ):
                raise ValueError(
                    "Cannot align simulated groups with actual source turns"
                )
            groups = []
            for k in (1, 2):
                matrix = F.normalize(
                    torch.as_tensor(f[f"vec{k}"], device=device, dtype=torch.float64)
                )
                membership = torch.as_tensor(f[f"map{k}"], device=device)
                if (
                    membership.shape != (len(matrix), len(ids))
                    or not membership.any(dim=0).all()
                ):
                    raise ValueError("Broken chunk-to-turn mapping")
                groups.append((matrix, membership))
        banks[name] = (F.normalize(vec.double()), groups)
    return queries, banks


def index_mask(links, device):
    values = pad_sequence(
        [torch.tensor(v, dtype=torch.long, device=device) for v in links],
        batch_first=True,
        padding_value=-1,
    )
    return values.clamp_min(0), values >= 0


def donor_advantages(rows, queries, banks, metric="rank"):
    donor = {}
    by_node = defaultdict(list)
    for i, r in enumerate(rows):
        if r["linked"]:
            by_node[r["node"]].append(i)
    for node, indices in by_node.items():
        with limited():
            q = torch.stack([queries[rows[i]["key"]] for i in indices])
            links, mask = index_mask([rows[i]["linked"] for i in indices], q.device)
            vals = contrast(q, banks[node][1], links, mask, metric).tolist()
            donor.update(zip(indices, vals, strict=True))
    return donor


def compile_fold(rows, held, stage, histories, queries, banks, donor):
    ordered = sorted(
        (i for i, r in enumerate(rows) if r["history"] != held and r["stage"] == stage),
        key=lambda i: (rows[i]["window"], tuple(rows[i]["bits"]), rows[i]["index"]),
    )
    nz = [i for i in ordered if rows[i]["credit"] != 0]
    reassigned = {
        i: rows[j]["credit"] for i, j in zip(nz, nz[1:] + nz[:1], strict=True)
    }
    idx = [i for i in ordered if rows[i]["linked"]]
    lab = [rows[i] for i in idx]
    tensors = [banks[r["node"]][0][r["linked"]] for r in lab]
    ev = pad_sequence(tensors, batch_first=True)
    _, mask = index_mask([r["linked"] for r in lab], ev.device)
    fold = {
        "histories": sorted(set(histories) - {held}),
        "owners": [r["history"] for r in lab],
        "evidence": ev,
        "mask": mask,
        "average": (ev * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True),
        "queries": torch.stack([queries[r["key"]] for r in lab]),
        "D": ev.new_tensor([donor[i] for i in idx]),
        "raw": ev.new_tensor([r["raw"] for r in lab]),
        "credit": ev.new_tensor([r["credit"] for r in lab]),
        "reassigned": ev.new_tensor([reassigned.get(i, 0.0) for i in idx]),
    }
    if len(set(histories)) != 6 or set(fold["owners"]) - set(fold["histories"]):
        raise ValueError("Held history entered the fold")
    return fold


def vote(labels, donor, target, similarity, owners, histories, bandwidth=0.2):
    keep = labels * donor > 0
    vote_sum = labels.new_zeros(())
    for history in histories:
        take = keep & labels.new_tensor(
            [r == history for r in owners], dtype=torch.bool
        )
        if take.any():
            weights = (similarity[take] / 0.08).softmax(0)
            values = (
                labels[take].abs()
                * torch.tanh(donor[take].abs() / bandwidth)
                * torch.tanh(target[take] / bandwidth)
            )
            vote_sum += weights @ values
    offset = (vote_sum / (len(histories) * 0.05)).clamp(-2.0, 2.0)
    return float(offset), int(keep.sum()), int((keep & (target * donor < 0)).sum())


def projected_queries(fold, target, metric="rank"):
    turns, groups = target
    mask = fold["mask"]
    sims = fold["evidence"] @ turns.T
    scores, mapped = sims.max(dim=2)
    sim = (scores * mask).sum(1) / mask.sum(1)
    centroid = (turns[mapped] * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
    questions = fold["queries"]
    translates = {
        "credit": F.normalize(questions + centroid - fold["average"]),
        "untranslated": questions,
        "source_only": F.normalize(centroid),
    }
    target_delta = {
        k: contrast(q, groups, mapped, mask, metric) for k, q in translates.items()
    }
    return mapped, sim, target_delta


def predict(fold, target, metric="rank"):
    _, sim, target_delta = projected_queries(fold, target, metric)
    offsets = {}
    stats = {}
    for arm, (lab, geom) in {
        "credit": ("credit", "credit"),
        "raw_mechanism": ("raw", "credit"),
        "reassigned": ("reassigned", "credit"),
        "untranslated": ("credit", "untranslated"),
        "source_only": ("credit", "source_only"),
    }.items():
        offsets[arm], n, rev = vote(
            fold[lab],
            fold["D"],
            target_delta[geom],
            sim,
            fold["owners"],
            fold["histories"],
            0.05 if metric == "mass" else 0.2,
        )
        stats[arm] = {"admissible": n, "reversed": rev}
    if metric == "mass":
        offsets["mild"] = vote(
            fold["credit"],
            fold["D"],
            target_delta["credit"],
            sim,
            fold["owners"],
            fold["histories"],
        )[0]
    return offsets, stats


def freeze(base, study, corpus, metric="rank"):
    reg = registration(base, study, corpus)
    out = reg.parent
    full = (out if metric == "rank" else out / "support-mass") / "full"
    prefix = full / "long-prefix/forecast.json"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    local = full / "forecast.json"
    packet = read(reg)
    src = base / "source-compression/witness-passages/evidence-advantage/group"
    a = read(src / "forecast.json")
    b = read(src / "long-prefix/forecast.json")
    identity = {
        "registration": file_hash(reg),
        "encoded": file_hash(out / "encoded.json"),
        "manifest": b["inputs"]["manifest"],
    }
    arms = ARMS
    if metric == "mass":
        earlier = out / "full/forecast.json"
        old_prefix = out / "full/long-prefix/forecast.json"
        identity.update(
            geometry=file_hash(
                Path(__file__).with_name("protocols") / "query_support_mass.json"
            ),
            rank=file_hash(earlier),
            rank_prefix=file_hash(old_prefix),
        )
        rank = read(earlier)["by_stage"]
        rank_ext = read(old_prefix)["offsets"]
        arms = (*ARMS, "rank_parent", "mild")
    if local.exists() and prefix.exists():
        if read(local)["inputs"] != identity or read(prefix)["inputs"] != {
            **identity,
            "local": file_hash(local),
        }:
            raise ValueError("Different frozen query policy")
        return local, prefix
    queries, banks = load_banks(base, reg, "cuda")
    rows = packet["rows"]
    donors = donor_advantages(rows, queries, banks, metric)
    cases = [
        {
            **{
                k: r[k]
                for k in (
                    "history",
                    "window",
                    "bits",
                    "index",
                    "stage",
                    "raw",
                    "credit",
                )
            },
            "retrieval_delta": donors.get(i),
        }
        for i, r in enumerate(rows)
    ]
    register(full / "donors.json", {"registration": file_hash(reg), "rows": cases})
    histories = sorted({r["history"] for r in rows})
    folds = {
        (h, j): compile_fold(rows, h, j, histories, queries, banks, donors)
        for h in histories
        for j in (0, 1)
    }
    stages = {"0": {}, "1": {}}
    ext = {}
    audit = {}
    for node, target in sorted(banks.items()):
        held = node.rsplit("_s", 1)[0]
        audit[node] = {
            "cost": {
                "turn_vectors": len(target[0]),
                "single_chunks": len(target[1][0][0]),
                "paired_chunks": len(target[1][1][0]),
                "target_questions_encoded": 0,
            }
        }
        for j in (0, 1):
            with limited():
                vals, stats = predict(folds[held, j], target, metric)
            audit[node][str(j)] = stats
            if node in a["sources"]:
                stages[str(j)][node] = {**vals, "parent": a["offsets"][node]["parent"]}
                if metric == "mass":
                    stages[str(j)][node]["rank_parent"] = rank[str(j)][node]["credit"]
            if node in b["sources"] and j == 1:
                ext[node] = {**vals, "parent": b["offsets"][node]["parent"]}
                if metric == "mass":
                    ext[node]["rank_parent"] = rank_ext[node]["credit"]
    register(full / "mechanism.json", {"inputs": identity, "sources": audit})
    register(
        local,
        {"inputs": identity, "sources": a["sources"], "by_stage": stages, "arms": arms},
    )
    register(
        prefix,
        {
            "inputs": {**identity, "local": file_hash(local)},
            "sources": b["sources"],
            "offsets": ext,
            "arms": arms,
        },
    )
    return local, prefix


def run(base, study, corpus, metric="rank"):
    from experiments.controller.policy_capacity import summarize_capacity

    local, prefix = freeze(base, study, corpus, metric)
    evaluate_prefix(base, prefix)
    evaluate(
        base,
        study,
        corpus,
        local,
        final_stage="full",
        required_prefix=prefix.parent / "summary.json",
        gate_controls=(
            "raw",
            "raw_mechanism",
            "reassigned",
            "source_only",
            "untranslated",
        )
        + (("rank_parent",) if metric == "mass" else ()),
    )
    summarize_capacity(local.parent)
    summarize_capacity(
        local.parent, local.parent / "direction-capacity", directional=True
    )
    from experiments.controller.query_displacement_audit import perform

    perform(
        base / "source-compression/query-displacement/registration.json",
        local,
        prefix,
        metric,
    )
    print("scored", local, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("--metric", choices=("rank", "mass"), default="rank")
    a = p.parse_args()
    gpu()
    run(a.base, a.study, a.corpus, a.metric)
