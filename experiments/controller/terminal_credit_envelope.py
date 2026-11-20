"""Bound the full two-stage return for every attenuation of the verified terminal credits."""

import argparse
import time
from itertools import product
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.retrieval_competition import learn, probability
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "terminal_credit_envelope.json"


def ridge_weights(x, test):
    """Linear map from donor targets to predictions of the existing centered ridge fit."""
    n = len(x)
    scale = x.std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    z = (x - x.mean(0)) / scale
    target = (test - x.mean(0)) / scale
    penalty = n * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    weights = target @ torch.linalg.solve(z.T @ z + penalty, z.T)
    return weights - weights.mean(1, keepdim=True) + 1 / n


def apply_bounds(weights, bounds):
    positive = weights.clamp_min(0)
    negative = weights.clamp_max(0)
    return torch.einsum("ji,isk->jsk", positive, bounds) + torch.einsum(
        "ji,isk->jsk", negative, bounds.flip([-1])
    )


def continuation_bounds(y, effects, ranges):
    """Extrema on each side of p0=p1, including vertices on the diagonal."""
    p, q = ranges.unbind(0)

    def value(a, b):
        return (
            y[2]
            + b * (y[3] - y[2])
            - y[0]
            - a * (y[1] - y[0])
            + effects[0] * (1 - torch.maximum(a, b))
            + effects[1] * torch.minimum(a, b)
        )

    corners = torch.stack([value(a, b) for a in p.unbind(-1) for b in q.unbind(-1)])
    lo, hi = corners.amin(0), corners.amax(0)
    left, right = torch.maximum(p[:, 0], q[:, 0]), torch.minimum(p[:, 1], q[:, 1])
    for point in (left, right):
        diagonal = value(point, point)
        lo = torch.where(left <= right, torch.minimum(lo, diagonal), lo)
        hi = torch.where(left <= right, torch.maximum(hi, diagonal), hi)
    return torch.stack([lo, hi], dim=-1)


def predicted(states, test, ys):
    x = ys.new_tensor([s["features"] for s in states])
    target = ys.new_tensor([s["features"] for s in test])
    base = ys.new_tensor([s["base"] for s in test])[:, None, None]
    offsets = (apply_bounds(ridge_weights(x, target), ys) / 0.05).clamp(-2, 2)
    return (base + offsets).sigmoid()


def stage_values(windows, held, targets, effects):
    donors = [w for w in windows if w["dialogue"] != held]
    unseen = [w for w in windows if w["dialogue"] == held]

    def terminal(donors, targets_for):
        states = [s for w in donors for s in w["states"][1:]]
        xs = [s for w in targets_for for s in w["states"][1:]]
        ys = torch.cat([targets[w["window"]] for w in donors])
        return predicted(states, xs, ys).reshape(len(targets_for), 2, -1, 2)

    root_targets = {}
    for sid in sorted({w["dialogue"] for w in donors}):
        inner = [w for w in donors if w["dialogue"] != sid]
        omitted = [w for w in donors if w["dialogue"] == sid]
        for w, bounds in zip(omitted, terminal(inner, omitted), strict=True):
            root_targets[w["window"]] = continuation_bounds(
                w["values"], effects[w["window"]], bounds
            )
    root = predicted(
        [w["states"][0] for w in donors],
        [w["states"][0] for w in unseen],
        torch.stack([root_targets[w["window"]] for w in donors]),
    )
    return unseen, torch.cat([root[:, None], terminal(donors, unseen)], dim=1)


def analyze(base):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run the capacity analysis on one leased GPU")
    torch.cuda.set_per_process_memory_fraction(0.05)
    dest = base / "writer-block-repair"
    dest.mkdir(exist_ok=True)
    training = read(base / "retrieval-competition/fit.json")["training"]
    old = read(base / "root-overlap/fit.json")
    previous = read(base / "root-overlap/calibration.json")
    old_windows = {w["window"]: w for w in previous["windows"]}
    keys = []
    for w in training:
        path = base / "factorial-repairs" / (w["window"] + ".json")
        if file_hash(path) != w["repair"]:
            raise ValueError("Different terminal supervision")
        for r in read(path)["rows"]:
            v = r.get("probe", {}).get("signed_effect", 0.0)
            if r["position"] == 1 and v:
                keys.append((w["window"], r["background"], r["original_index"], v))
    if len(keys) != 4 or len(training) != 12:
        raise ValueError("Expected the complete audited four-credit panel")
    vertices = list(product((0, 1), repeat=len(keys)))
    # Each fixed vertex is a point interval. The last scenario is the full continuous box.
    a = torch.tensor(vertices, device="cuda", dtype=torch.float64).T
    box = torch.cat(
        [
            a[:, :, None].expand(-1, -1, 2),
            a.new_tensor([[[0.0, 1.0]]]).expand(len(keys), 1, 2),
        ],
        dim=1,
    )
    target = {}
    for w in training:
        raw = [w["values"][2 * j + 1] - w["values"][2 * j] for j in range(2)]
        coef = a.new_tensor(
            [
                [
                    v / len(w["questions"])
                    if (w["window"], j) == (name, prefix)
                    else 0.0
                    for name, prefix, _, v in keys
                ]
                for j in range(2)
            ]
        )
        torch.testing.assert_close(
            coef.sum(1), a.new_tensor(w["signed"]), rtol=0, atol=1e-12
        )
        target[w["window"]] = a.new_tensor(raw)[:, None, None] + apply_bounds(coef, box)
    receipts = []
    vertex_sum = torch.zeros(len(vertices), device="cuda", dtype=torch.float64)
    for sid in sorted({w["dialogue"] for w in training}):
        unseen, ranges = stage_values(training, sid, target, old["effects"])
        for j, bits in enumerate(vertices):
            # Independent check with the original trainer and exact shared credit-retention choices.
            start = time.monotonic()
            donors = []
            for w in training:
                if w["dialogue"] == sid:
                    continue
                signed = [
                    sum(
                        v * bits[k]
                        for k, (name, prefix, _, v) in enumerate(keys)
                        if (name, prefix) == (w["window"], i)
                    )
                    / len(w["questions"])
                    for i in range(2)
                ]
                donors.append(dict(w, signed=signed))
            with torch.device("cuda"):
                model = learn(donors, True, root_effects=old["effects"])
            for w, r in zip(unseen, ranges, strict=True):
                actual = a.new_tensor(
                    [
                        probability(s, model["root" if i == 0 else "terminal"])
                        for i, s in enumerate(w["states"])
                    ]
                )
                torch.testing.assert_close(
                    r[:, j], actual[:, None].expand(-1, 2), rtol=0, atol=1e-10
                )
                if (
                    (actual < r[:, -1, 0] - 1e-10) | (actual > r[:, -1, 1] + 1e-10)
                ).any():
                    raise ValueError(
                        "An exact vertex policy lies outside the continuous enclosure"
                    )
            torch.cuda.synchronize()
            time.sleep((time.monotonic() - start) * 2)
        for w, r in zip(unseen, ranges, strict=True):
            probs = old_windows[w["window"]]["probabilities"]
            torch.testing.assert_close(
                r[:, -2, 0],
                a.new_tensor([probs[str(j)]["overlap"] for j in range(3)]),
                rtol=0,
                atol=1e-10,
            )
            y = w["values"]

            def f(u, v, t, y=y):
                return (1 - u) * (y[0] + v * (y[1] - y[0])) + u * (
                    y[2] + t * (y[3] - y[2])
                )

            counts = len(w["questions"])
            vertex_sum += counts * f(*r[:, :-1, 0])
            vals = [
                float(f(*(r[j, -1, b] for j, b in enumerate(bits))))
                for bits in product((0, 1), repeat=3)
            ]
            receipts.append(
                {
                    "window": w["window"],
                    "questions": counts,
                    "probability_ranges": r[:, -1].tolist(),
                    "f1_bounds": [min(vals), max(vals)],
                }
            )
    n = sum(w["questions"] for w in receipts)
    vertices_f1 = (vertex_sum / n).tolist()
    scores = previous["summary"]["f1"]
    if n != 91 or abs(vertices_f1[-1] - scores["overlap"]) > 1e-10:
        raise ValueError("All-retained panel F1 failed reproduction")
    report = {
        "plan": file_hash(PLAN),
        "training": file_hash(base / "retrieval-competition/fit.json"),
        "prior": file_hash(base / "root-overlap/fit.json"),
        "labels": keys,
        "questions": n,
        "native": scores["native"],
        "terminal": scores["terminal"],
        "prior_f1": scores["overlap"],
        "continuous_lower": sum(w["questions"] * w["f1_bounds"][0] for w in receipts)
        / n,
        "continuous_upper": sum(w["questions"] * w["f1_bounds"][1] for w in receipts)
        / n,
        "vertices": [
            {"keep": bits, "f1": v}
            for bits, v in zip(vertices, vertices_f1, strict=True)
        ],
        "windows": receipts,
    }
    register(dest / "continuous-envelope.json", report)
    print(
        "91-question continuous upper bound",
        report["continuous_upper"],
        "max measured vertex",
        max(vertices_f1),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    analyze(parser.parse_args().base)
