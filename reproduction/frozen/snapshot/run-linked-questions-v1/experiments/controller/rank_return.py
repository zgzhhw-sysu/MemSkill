"""Optimize finite, source-dependent changes of odds on observed two-action outcomes."""

from itertools import product

import torch

from experiments.controller.late_pair_policy import correction
from experiments.controller.paired_policy import _shift_for_mean_probability
from experiments.controller.prefix_return import masses, predicted_offsets


def fit_rules(windows, direction):
    if not windows or any(w["n"] <= 0 for w in windows):
        raise ValueError("Need complete calibration windows")
    base = torch.tensor([w["base"] for w in windows], dtype=torch.float64)
    scores = base.new_tensor([w["f1"] for w in windows])
    if (
        scores.shape != (len(windows), 4)
        or not torch.isfinite(scores).all()
        or not ((scores >= 0) & (scores <= 1)).all()
    ):
        raise ValueError("Invalid trajectory scores")
    rank = base.new_tensor(
        [
            [correction(direction, w["features"][int(j > 0)], 1.0) for j in range(3)]
            for w in windows
        ]
    )
    cuts = torch.quantile(
        rank, base.new_tensor([0.25, 0.5, 0.75]), dim=0
    ).T.contiguous()
    options = torch.cat(
        (
            torch.zeros_like(rank[..., None]),
            torch.full_like(rank[..., None], -2),
            torch.full_like(rank[..., None], 2),
            torch.where(rank[..., None] >= cuts, 2.0, -2.0).to(base),
        ),
        dim=-1,
    )
    rules = sorted(
        product(range(6), repeat=3),
        key=lambda v: (sum(i >= 3 for i in v), sum(i != 0 for i in v), v),
    )
    ix = torch.tensor(rules, dtype=torch.long, device=base.device)
    offsets = torch.stack([options[:, j, ix[:, j]] for j in range(3)], dim=-1)
    values = (masses((base[:, None, :] + offsets).sigmoid()) * scores[:, None, :]).sum(
        -1
    )
    ids = sorted({w["dialogue"] for w in windows})
    counts = {sid: sum(w["n"] for w in windows if w["dialogue"] == sid) for sid in ids}
    weights = base.new_tensor(
        [w["n"] / counts[w["dialogue"]] / len(ids) for w in windows]
    )
    utilities = (weights[:, None] * values).sum(0)
    if not torch.isfinite(utilities).all():
        raise ValueError("Nonfinite rank-rule return")

    def best(mask):
        return int(
            torch.where(mask & (utilities >= utilities[mask].max() - 1e-12))[0][0]
        )

    joint = best(torch.ones_like(utilities, dtype=torch.bool))
    shared = best((ix[:, 0] == ix[:, 1]) & (ix[:, 1] == ix[:, 2]))
    context = best((ix < 3).all(dim=1))
    shifts = [
        _shift_for_mean_probability(base[:, j], offsets[:, joint, j]) for j in range(3)
    ]
    if not torch.allclose(
        (base + base.new_tensor(shifts)).sigmoid().mean(0),
        (base + offsets[:, joint]).sigmoid().mean(0),
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("Rate control does not match")
    return {
        "rules": list(rules[joint]),
        "shared_rule": rules[shared][0],
        "context_rules": list(rules[context]),
        "cuts": cuts.tolist(),
        "rate_shifts": shifts,
        "questions": sum(counts.values()),
        "windows": len(windows),
        "dialogues": ids,
        "fit_f1": {
            k: float(utilities[v])
            for k, v in (("joint", joint), ("shared", shared), ("context", context))
        },
    }


def rank_offsets(model, features):
    fit = model["gate"]
    rank = correction(model["direction"], features, 1.0)

    def evaluate(rules):
        return [
            (
                (0.0, -2.0, 2.0)[i]
                if i < 3
                else (2.0 if rank >= thresholds[i - 3] else -2.0)
            )
            for i, thresholds in zip(rules, fit["cuts"], strict=True)
        ]

    joint = evaluate(fit["rules"])
    strength = predicted_offsets(
        dict(direction=model["direction"], gate=model["strength"]), features
    )
    return {
        "prefix": joint,
        "prefix_shared": evaluate([fit["shared_rule"]] * 3),
        "prefix_rate": fit["rate_shifts"],
        "prefix_context": evaluate(fit["context_rules"]),
        "prefix_max": strength["prefix_max"],
        "prefix_strength": strength["prefix"],
        "prefix_swap": [joint[0], joint[2], joint[1]],
    }
