"""Calibrate the strength of verified span credit against complete two-action trajectories."""

from itertools import product

import torch

from experiments.controller.late_pair_policy import STEPS, correction
from experiments.controller.paired_policy import _shift_for_mean_probability


def masses(probabilities):
    p, q0, q1 = probabilities.unbind(dim=-1)
    return torch.stack(((1 - p) * (1 - q0), (1 - p) * q0, p * (1 - q1), p * q1), dim=-1)


def attribute(scores, native, candidate):
    """Exact performance difference with native value-to-go, candidate state occupancy."""
    r00, r01, r10, r11 = scores.unbind(dim=-1)
    p, q0, q1 = candidate.unbind(dim=-1)
    old_p, old_q0, old_q1 = native.unbind(dim=-1)
    a0, a1 = r01 - r00, r11 - r10
    root = (p - old_p) * ((r10 + old_q1 * a1) - (r00 + old_q0 * a0))
    return torch.stack(
        (root, (1 - p) * (q0 - old_q0) * a0, p * (q1 - old_q1) * a1), dim=-1
    )


def fit_strengths(windows, direction):
    if not windows or any(w["n"] <= 0 for w in windows):
        raise ValueError("Need measured four-path training windows")
    ids = sorted({w["dialogue"] for w in windows})
    counts = {sid: sum(w["n"] for w in windows if w["dialogue"] == sid) for sid in ids}
    base = torch.tensor([w["base"] for w in windows], dtype=torch.float64)
    scores = base.new_tensor([w["f1"] for w in windows])
    if (
        scores.shape != (len(windows), 4)
        or not torch.isfinite(scores).all()
        or not ((0 <= scores) & (scores <= 1)).all()
    ):
        raise ValueError("Invalid four-path F1")
    offsets = base.new_tensor(
        [
            [
                [correction(direction, w["features"][int(node > 0)], s) for s in STEPS]
                for node in range(3)
            ]
            for w in windows
        ]
    )
    schedules = sorted(
        product(range(len(STEPS)), repeat=3),
        key=lambda a: (sum(STEPS[i] for i in a), a),
    )
    indices = torch.tensor(schedules, dtype=torch.long, device=base.device)
    adjustments = torch.stack([offsets[:, j, indices[:, j]] for j in range(3)], dim=-1)
    logits = base[:, None, :] + adjustments
    native = base.sigmoid()
    ps = logits.sigmoid()
    dist = masses(ps)
    # Chain rule for KL(path_candidate || path_native), stable even for saturated logits.
    kl = ps * adjustments - (
        torch.nn.functional.softplus(logits)
        - torch.nn.functional.softplus(base[:, None, :])
    )
    kl_path = kl[..., 0] + (1 - ps[..., 0]) * kl[..., 1] + ps[..., 0] * kl[..., 2]
    weights = base.new_tensor(
        [w["n"] / counts[w["dialogue"]] / len(ids) for w in windows]
    )
    values = ((dist * scores[:, None, :]).sum(-1) * weights[:, None]).sum(0)
    penalties = (kl_path * weights[:, None]).sum(0)
    utility = values - 0.02 * penalties
    if not torch.isfinite(utility).all() or penalties.min() < -1e-10:
        raise ValueError("Nonfinite return or invalid path KL")

    def choose(eligible):
        best = utility[eligible].max()
        return int(torch.where(eligible & (utility >= best - 1e-12))[0][0])

    joint = choose(torch.ones_like(utility, dtype=torch.bool))
    shared = choose((indices[:, 0] == indices[:, 1]) & (indices[:, 1] == indices[:, 2]))
    shifts = [
        _shift_for_mean_probability(base[:, j], adjustments[:, joint, j])
        for j in range(3)
    ]
    if not torch.allclose(
        (base + base.new_tensor(shifts)).sigmoid().mean(0),
        ps[:, joint, :].mean(0),
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError(
            "Rate-only control changed mean probability within calibration prefixes"
        )
    return {
        "steps": [STEPS[i] for i in schedules[joint]],
        "shared": STEPS[schedules[shared][0]],
        "rate_shifts": shifts,
        "windows": len(windows),
        "questions": sum(counts.values()),
        "dialogues": ids,
        "native_fit_f1": float(((masses(native) * scores).sum(-1) * weights).sum()),
        "conditional_fit_f1": float(values[joint]),
        "conditional_fit_kl": float(penalties[joint]),
        "shared_fit_f1": float(values[shared]),
        "shared_fit_kl": float(penalties[shared]),
    }


def predicted_offsets(model, features):
    direction, gate = model["direction"], model["gate"]
    offsets = [correction(direction, features, s) for s in gate["steps"]]
    return {
        "prefix": offsets,
        "prefix_shared": [correction(direction, features, gate["shared"])] * 3,
        "prefix_max": [correction(direction, features, 2.0)] * 3,
        "prefix_rate": gate["rate_shifts"],
        "prefix_swap": [offsets[0], offsets[2], offsets[1]],
    }


def budget_envelope(rows, target=0.02):
    """Hindsight bound for reallocating a finite probability budget across measured focal states.

    Rows contain (weight, native probability, proposed probability, mean paired F1 contrast).
    Each state keeps a single decision shared by all its questions.
    """
    import math

    rows = list(rows)
    if not rows or not math.isfinite(target) or target < 0:
        raise ValueError("Need a nonnegative target and measured states")
    for n, old, new, advantage in rows:
        if (
            not all(math.isfinite(v) for v in (n, old, new, advantage))
            or n <= 0
            or not 0 <= old <= 1
            or not 0 <= new <= 1
            or not -1 <= advantage <= 1
        ):
            raise ValueError("Invalid probability budget or focal F1")
    total = sum(n for n, *_ in rows)
    budget = sum(n * abs(new - old) for n, old, new, _ in rows) / total
    observed = sum(n * (new - old) * a for n, old, new, a in rows) / total
    correct_sign = 0.0
    items = []
    for n, old, new, a in rows:
        capacity = 1 - old if a > 0 else old
        correct_sign += n / total * min(abs(new - old), capacity) * abs(a)
        items.append((abs(a), n / total * capacity))
    remaining = budget
    feasible = spent = gain = max_gain = 0.0
    for slope, capacity in sorted(items, reverse=True):
        max_gain += slope * capacity
        use = min(remaining, capacity)
        feasible += use * slope
        remaining -= use
        if slope > 0 and gain < target:
            need = min(capacity, (target - gain) / slope)
            spent += need
            gain += need * slope
    if observed > correct_sign + 1e-10 or correct_sign > feasible + 1e-10:
        raise ValueError("Paired policy gain exceeds its hindsight envelope")
    return {
        "states": len(rows),
        "mean_abs_change": budget,
        "actual_gain": observed,
        "same_magnitudes_best_sign": correct_sign,
        "same_budget_reallocated": feasible,
        "unconstrained_oracle_gain": max_gain,
        "minimum_budget_for_target": spent if gain >= target - 1e-12 else None,
        "target": target,
    }
