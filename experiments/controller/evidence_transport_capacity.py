"""Diagnose whether the frozen evidence actor is limited by its signs or its step sizes."""

import argparse
from itertools import product
from pathlib import Path

import torch

from experiments.controller.evidence_transport import ARMS, root_targets
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register


def expected(logits, offsets, y):
    p = (logits + offsets).sigmoid()
    low = y[..., 0] + p[..., 1] * (y[..., 1] - y[..., 0])
    high = y[..., 2] + p[..., 2] * (y[..., 3] - y[..., 2])
    return low + p[..., 0] * (high - low)


def envelope(logits, effects, values, counts):
    """Optimize a bounded sign-preserving offset at every node by enumerating all corners."""
    weights = counts / counts.sum()
    points = logits.new_tensor(list(product((0.0, 2.0), repeat=3)))
    sign = effects.sign()
    per_window = expected(
        logits[:, None], sign[:, None] * points, values[:, None]
    ).amax(1)
    free = expected(logits[:, None], 2 * points - 2, values[:, None]).amax(1)
    scales = logits.new_tensor([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    levels = torch.cat(
        [(effects[:, :, None] * scales / 0.05).clamp(-2, 2), (2 * sign)[:, :, None]],
        dim=2,
    )
    rules = list(product(range(levels.shape[2]), repeat=3))
    indices = torch.tensor(rules, device=logits.device, dtype=torch.long)
    offsets = torch.stack([levels[:, j, indices[:, j]] for j in range(3)], dim=2)
    scores = weights @ expected(logits[:, None], offsets, values[:, None])
    best = int(scores.argmax())

    def value(x):
        return float(weights @ expected(logits, x, values))

    return {
        "native": value(torch.zeros_like(logits)),
        "learned": value((effects / 0.05).clamp(-2, 2)),
        "full_step": value(2 * sign),
        "best_stage_grid": float(scores[best]),
        "best_scale_ids": rules[best],
        "sign_oracle": float(weights @ per_window),
        "unrestricted": float(weights @ free),
    }


def compare_suffixes(exact, draws):
    """Compare two equiprobable randomized training targets with a four-path backup."""
    if not torch.allclose(exact, draws.mean(0), atol=1e-10, rtol=0):
        raise ValueError("Randomized targets failed the exact four-path expectation")
    tol = 1e-8
    stable = exact.abs() > tol
    opposite = stable & (draws.abs() > tol) & (draws.sign() != exact.sign())
    lost = stable & (draws.abs() <= tol)
    return {
        "cases": int(exact.numel()),
        "nonzero": int(stable.sum()),
        "nonzero_draws": 2 * int(stable.sum()),
        "opposite_draws": int(opposite.sum()),
        "zero_draws_from_nonzero": int(lost.sum()),
        "mean_absolute_error": float((draws - exact).abs().mean()),
    }


def backup_diagnostics(src, training, packets, logits, dest):
    by_row, by_window, inputs = [], [], {}
    for i, (w, pred) in enumerate(zip(training, packets, strict=True)):
        name = w["window"]
        path = src / "root-write-repairs" / name / "window.json"
        inputs[name] = file_hash(path)
        bonus = [[0.0] * len(w["questions"]) for _ in range(2)]
        for r in read(path)["rows"]:
            bonus[r["suffix"]][w["questions"].index(r["index"])] = r["signed_effect"]
        offset = logits.new_tensor(
            [[node[k] for k in ARMS] for node in pred["offsets"][1:]]
        )
        continuation = (logits[i, 1:, None] + offset).sigmoid()
        full = {"y": w["cases"], "root": bonus, "pivot": None}
        draws = []
        for b in (0, 1):
            subset = {k: v for k, v in w["cases"].items() if int(k[1]) == b}
            draws.append(
                root_targets(
                    dict(full, pivot=0, bg=b, y=subset), continuation, logits.device
                )
            )
        values = torch.stack([root_targets(full, continuation, logits.device), *draws])
        by_row.append(values)
        by_window.append(values.mean(1, keepdim=True))
    out = {}
    for unit, stats in (("question", by_row), ("window", by_window)):
        joined = torch.cat(stats, 1)
        out[unit] = {
            k: compare_suffixes(
                joined[0, :, ARMS.index(k)], joined[1:, :, ARMS.index(k)]
            )
            for k in ("credit", "raw")
        }
    register(
        dest / "suffix-diagnostic.json",
        {
            "sign_capacity": file_hash(dest / "sign-capacity.json"),
            "root_credit_files": inputs,
            "training": file_hash(src / "retrieval-competition/fit.json"),
            "statistics": out,
        },
    )
    print(out, flush=True)


def run(base):
    src = base / "source-compression"
    dest = src / "evidence-transport/pilot"
    pred = dest / "predictions"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a GPU lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    training = read(src / "retrieval-competition/fit.json")["training"]
    files = [pred / (w["window"] + ".json") for w in training]
    packets = [read(f) for f in files]
    logits = torch.tensor(
        [[s["base"] for s in w["states"]] for w in training],
        device="cuda",
        dtype=torch.float64,
    )
    values = logits.new_tensor([w["values"] for w in training])
    n = logits.new_tensor([len(w["questions"]) for w in training])
    z = logits.new_tensor([[r[0] for r in v["effects"]] for v in packets])
    out = envelope(logits, z, values, n)
    sources = {w["window"]: file_hash(f) for w, f in zip(training, files, strict=True)}
    out.update(
        questions=int(n.sum()),
        predictions=sources,
        scales=[0, 0.25, 0.5, 1, 2, 4, 8, 16, "sign limit"],
    )
    reg = read(dest / "factorial/summary.json")["all"]["f1"]
    if (
        abs(out["native"] - reg["native"]) > 1e-10
        or abs(out["learned"] - reg["credit"]) > 1e-10
    ):
        raise ValueError("Frozen policy failed exact reproduction")
    register(dest / "sign-capacity.json", out)
    print({k: v for k, v in out.items() if k != "predictions"}, flush=True)
    backup_diagnostics(src, training, packets, logits, dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    run(p.parse_args().base)
