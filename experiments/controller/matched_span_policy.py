"""Transfer a skill preference learned from within-history pairs of measured span effects."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.matched_span_data import (
    PLAN,
    extra_units,
    features,
    original_units,
)
from experiments.controller.passage_advantage import evaluate
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.passage_evidence_prefix import evaluate as evaluate_prefix
from experiments.controller.policy_capacity import summarize_capacity
from experiments.controller.query_displacement_data import registration
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited

ARMS = (
    "credit",
    "raw",
    "raw_rank",
    "rotated",
    "fixed_scale",
    "joint",
    "parent",
    "native",
    "terminal",
)


def paired_edges(units):
    strata = defaultdict(list)
    for u in units:
        strata[u["history"], u["background"], u["stage"]].append(u)
    edges = []
    for group in strata.values():
        group.sort(key=lambda u: u["node"])
        if len({r["node"] for r in group}) != len(group):
            raise ValueError("Repeated decision in a matched stratum")
        for i, j in combinations(range(len(group)), 2):
            a, b = group[i], group[j]
            edges.append(
                {
                    "history": a["history"],
                    "background": a["background"],
                    "stage": a["stage"],
                    "a": a["node"],
                    "b": b["node"],
                    "credit": b["credit"] - a["credit"],
                    "rotated": group[(j + 1) % len(group)]["credit"]
                    - group[(i + 1) % len(group)]["credit"],
                    "raw": b["raw"] - a["raw"] if "raw" in a and "raw" in b else None,
                }
            )
    return edges


def fit(units, x, held, stage, labels):
    """Center and scale using exclusively the donor histories that make up the fit."""
    training = [r for r in units if r["history"] != held and r["stage"] == stage]
    by_history = defaultdict(set)
    for r in training:
        by_history[r["history"]].add(r["node"])
    nodes = [n for h, ns in sorted(by_history.items()) for n in sorted(ns)]
    design = torch.stack([x[n] for n in nodes])
    weights = design.new_tensor(
        [
            1 / (len(by_history) * len(ns))
            for h, ns in sorted(by_history.items())
            for n in sorted(ns)
        ]
    )
    center = weights @ design
    scale = (weights @ (design - center).square()).sqrt().clamp_min(0.02)
    edges = paired_edges(training)
    freq = defaultdict(int)
    for edge in edges:
        freq[edge["history"]] += 1
    dw = design.new_tensor([1 / (len(freq) * freq[e["history"]]) for e in edges])
    delta = torch.stack([(x[e["b"]] - x[e["a"]]) / scale for e in edges])
    truth = design.new_tensor([[e[k] for k in labels] for e in edges])
    gram = delta.T @ (dw[:, None] * delta)
    ridge = 0.5 * torch.eye(len(center), device=design.device, dtype=design.dtype)
    coef = torch.linalg.solve(gram + ridge, delta.T @ (dw[:, None] * truth))
    train_scores = ((design - center) / scale) @ coef
    q = torch.quantile(train_scores, design.new_tensor([0.25, 0.75]), dim=0)
    width = (q[1] - q[0]).clamp_min(0.01)
    return {
        "center": center,
        "scale": scale,
        "coef": coef,
        "width": width,
        "histories": sorted(by_history),
        "nodes": nodes,
        "edges": len(edges),
    }


def predict(model, v):
    score = ((v - model["center"]) / model["scale"]) @ model["coef"]
    return score, score / model["width"]


def deploy(v, raw_model, credit_model):
    raw, raw_shift = predict(raw_model, v)
    vcredit, credit_shift = predict(credit_model, v)
    credit, rotated = credit_shift
    vals = torch.stack(
        [credit, raw_shift[0], rotated, vcredit[0] / 0.05, credit + raw_shift[0]]
    )
    return dict(
        zip(
            ("credit", "raw_rank", "rotated", "fixed_scale", "joint"),
            vals.clamp(-2, 2).tolist(),
            strict=True,
        )
    ), {
        "credit": float(vcredit[0]),
        "rotated": float(vcredit[1]),
        "raw_rank": float(raw[0]),
    }


def fit_receipt(model):
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in model.items()
    }


def concordance(edges, forecasts):
    rows = []
    for e in edges:
        f = forecasts[str(e["stage"])]
        a, b = f[e["a"]], f[e["b"]]
        rows.append({**e, "predicted": {k: b[k] - a[k] for k in a}})

    def stats(items):
        result = {}
        for target in ("raw", "credit"):
            known = [r for r in items if r[target] is not None]
            moving = [r for r in known if abs(r[target]) > 1e-12]
            if not known:
                continue
            result[target] = {
                "edges": len(known),
                "non_tied": len(moving),
                "mae_zero": mean(abs(r[target]) for r in known),
                "predictors": {},
            }
            for k in ("credit", "raw_rank", "rotated"):
                result[target]["predictors"][k] = {
                    "correct": sum(
                        r["predicted"][k] * r[target] > 0
                        for r in moving
                        if abs(r["predicted"][k]) > 1e-12
                    ),
                    "predicted_ties": sum(
                        abs(r["predicted"][k]) <= 1e-12 for r in moving
                    ),
                    "mae": mean(abs(r["predicted"][k] - r[target]) for r in known),
                }
        return result

    return {
        "all": stats(rows),
        "by_history": {
            h: stats([r for r in rows if r["history"] == h])
            for h in sorted({r["history"] for r in rows})
        },
        "rows": rows,
    }


def freeze(base, study, corpus, stage):
    qreg = registration(base, study, corpus)
    q = qreg.parent
    root = q.parent / "matched-span-rank"
    trial = root / stage
    trial.mkdir(parents=True, exist_ok=True)
    local = trial / "forecast.json"
    prefix = trial / "long-prefix/forecast.json"
    prefix.parent.mkdir(exist_ok=True)
    parent = q / "full/forecast.json"
    long_parent = q / "full/long-prefix/forecast.json"
    a = read(parent)
    b = read(long_parent)
    enc = q / "encoded.json"
    vals = features(root, q / "sources", enc)
    original = original_units(read(qreg))
    units = original
    inputs = {
        "plan": file_hash(PLAN),
        "registration": file_hash(qreg),
        "features": file_hash(root / "features.json"),
        "local_parent": file_hash(parent),
        "prefix_parent": file_hash(long_parent),
        "manifest": b["inputs"]["manifest"],
    }
    arms = ARMS
    if stage != "original":
        extra = root / "earlier-sources"
        more = features(extra, extra, extra / "complete.json")
        if any(vals[n] != more[n] for n in vals.keys() & more):
            raise ValueError("Group geometry changed for a shared source")
        vals = vals | more
        units = original + extra_units(base, study, corpus)
        orig_local = root / "original/forecast.json"
        orig_prefix = root / "original/long-prefix/forecast.json"
        old = read(orig_local)["by_stage"]
        ext_old = read(orig_prefix)["offsets"]
        inputs.update(
            earlier=file_hash(q.parent / "legacy-terminal-symmetry/summary.json"),
            earlier_features=file_hash(extra / "features.json"),
            original=file_hash(orig_local),
            original_prefix=file_hash(orig_prefix),
        )
        arms = (*arms, "original")
    if stage == "root-augmented":
        from experiments.controller.matched_root_credit import root_units

        root_dir = root / "earlier-root-sources"
        more = features(root_dir, root_dir, root_dir / "complete.json")
        if any(vals[n] != more[n] for n in vals.keys() & more):
            raise ValueError("Changed earlier root source geometry")
        vals = vals | more
        extra_roots = root_units(base, study, corpus)
        zero_units = units + [dict(u, credit=0.0) for u in extra_roots]
        units = units + extra_roots
        prior_local = root / "expanded/forecast.json"
        prior_prefix = root / "expanded/long-prefix/forecast.json"
        prior = read(prior_local)["by_stage"]
        ext_prior = read(prior_prefix)["offsets"]
        inputs.update(
            root_registration=file_hash(trial / "registration.json"),
            root_summary=file_hash(q.parent / "legacy-root-symmetry/summary.json"),
            root_features=file_hash(root_dir / "features.json"),
            expanded=file_hash(prior_local),
            expanded_prefix=file_hash(prior_prefix),
        )
        arms = (*arms, "expanded", "zero_roots")
    if local.exists() and prefix.exists() and (trial / "edge-audit.json").exists():
        if read(local)["inputs"] != inputs or read(prefix)["inputs"] != {
            **inputs,
            "local": file_hash(local),
        }:
            raise ValueError("Different frozen matched actor")
        return local, prefix, arms
    x = {
        n: torch.tensor(v, device="cuda", dtype=torch.float64) for n, v in vals.items()
    }
    offsets = {"0": {}, "1": {}}
    ext = {}
    fits = {}
    predicted = {"0": {}, "1": {}}
    histories = sorted({r["history"] for r in original})
    if len(histories) != 6:
        raise ValueError("Expected six training histories")
    for h in histories:
        fits[h] = {}
        for j in (0, 1):
            with limited():
                raw = fit(original, x, h, j, ("raw",))
                model = fit(units, x, h, j, ("credit", "rotated"))
                fits[h][str(j)] = {
                    "raw": fit_receipt(raw),
                    "credit": fit_receipt(model),
                }
                if stage == "root-augmented" and j == 0:
                    zero_model = fit(zero_units, x, h, j, ("credit", "rotated"))
                    fits[h][str(j)]["zero_roots"] = fit_receipt(zero_model)
                for n, v in x.items():
                    if n.rsplit("_s", 1)[0] != h:
                        continue
                    p, physical = deploy(v, raw, model)
                    predicted[str(j)][n] = physical
                    if n in a["sources"]:
                        item = {**p, "parent": a["by_stage"][str(j)][n]["parent"]}
                        if stage != "original":
                            item["original"] = old[str(j)][n]["credit"]
                        if stage == "root-augmented":
                            old_credit = prior[str(j)][n]["credit"]
                            if j == 1 and item["credit"] != old_credit:
                                raise ValueError(
                                    "Terminal forecast changed during root repair"
                                )
                            item.update(
                                expanded=old_credit,
                                zero_roots=deploy(v, raw, zero_model)[0]["credit"]
                                if j == 0
                                else old_credit,
                            )
                        offsets[str(j)][n] = item
                    if j == 1 and n in b["sources"]:
                        item = {**p, "parent": b["offsets"][n]["parent"]}
                        if stage != "original":
                            item["original"] = ext_old[n]["credit"]
                        if stage == "root-augmented":
                            old_credit = ext_prior[n]["credit"]
                            if item["credit"] != old_credit:
                                raise ValueError(
                                    "Long-prefix terminal forecast changed"
                                )
                            item.update(expanded=old_credit, zero_roots=old_credit)
                        ext[n] = item
    if any(set(d) != set(a["sources"]) for d in offsets.values()) or set(ext) != set(
        b["sources"]
    ):
        raise ValueError("Failed to freeze every registered target")
    register(trial / "models.json", {"inputs": inputs, "folds": fits, "units": units})
    register(
        local,
        {"inputs": inputs, "sources": a["sources"], "by_stage": offsets, "arms": arms},
    )
    register(
        prefix,
        {
            "inputs": {**inputs, "local": file_hash(local)},
            "sources": b["sources"],
            "offsets": ext,
            "arms": arms,
        },
    )
    register(
        trial / "edge-audit.json",
        {
            "forecast": file_hash(local),
            "prefix": file_hash(prefix),
            "original": concordance(paired_edges(original), predicted),
            **(
                {"expanded": concordance(paired_edges(units), predicted)}
                if stage != "original"
                else {}
            ),
        },
    )
    return local, prefix, arms


def score_trial(base, study, corpus, stage, local, prefix, controls):
    evaluate_prefix(base, prefix)
    evaluate(
        base,
        study,
        corpus,
        local,
        final_stage=stage,
        gate_controls=controls,
        required_prefix=prefix.parent / "summary.json",
    )
    summarize_capacity(
        local.parent, local.parent / "direction-capacity", directional=True
    )
    print("scored", local, flush=True)


def run(base, study, corpus, stage):
    local, prefix, arms = freeze(base, study, corpus, stage)
    controls = ("raw", "raw_rank", "rotated") + (
        ("original",) if "original" in arms else ()
    )
    if stage == "root-augmented":
        controls += ("expanded", "zero_roots")
    score_trial(base, study, corpus, stage, local, prefix, controls)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("stage", choices=("original", "expanded", "root-augmented"))
    a = p.parse_args()
    gpu()
    run(a.base, a.study, a.corpus, a.stage)
