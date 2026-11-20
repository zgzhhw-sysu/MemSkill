"""An achievable two-stage oracle and a relaxed upper bound from the same four real paths."""

import argparse
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.terminal_capacity import optimal_rates


def capacity(native, y, n):
    root = native[:, 0]
    reach = torch.stack([1 - root, root], 1)
    targets = (reach * native[:, 1:]).sum(0)
    constants = targets / reach.sum(0)
    lower, upper = (native.logit() - 2).sigmoid(), (native.logit() + 2).sigmoid()
    if ((constants < lower[:, 1:]) | (constants > upper[:, 1:])).any():
        raise ValueError(
            "Fixed conditional usage is outside the allowed terminal logits"
        )

    def values(p):
        return y[:, :, 0] + p * (y[:, :, 1] - y[:, :, 0])

    def score(p0, p1):
        v = values(p1)
        return float(n @ (v[:, 0] + p0 * (v[:, 1] - v[:, 0])) / n.sum())

    fixed = values(constants)
    visits = torch.ones_like(root)
    p0 = optimal_rates(
        visits, root, fixed[:, 1] - fixed[:, 0], n, 2.0, total_uses=visits @ root
    )
    new_reach = torch.stack([1 - p0, p0], 1)
    p1 = torch.stack(
        [
            optimal_rates(
                new_reach[:, a],
                native[:, 1 + a],
                y[:, a, 1] - y[:, a, 0],
                n,
                2.0,
                total_uses=targets[a],
            )
            for a in range(2)
        ],
        1,
    )
    if not torch.allclose(
        p0.sum(), root.sum(), atol=1e-10, rtol=0
    ) or not torch.allclose((new_reach * p1).sum(0), targets, atol=1e-10, rtol=0):
        raise ValueError("The feasible witness changed action usage")
    best1 = torch.where(y[:, :, 1] > y[:, :, 0], upper[:, 1:], lower[:, 1:])
    best = values(best1)
    best0 = torch.where(best[:, 1] > best[:, 0], upper[:, 0], lower[:, 0])
    ans = {
        "native": score(root, native[:, 1:]),
        "root_with_fixed_continuation": score(p0, constants),
        "feasible_both_stages": score(p0, p1),
        "relaxed_upper": score(best0, best1),
        "native_expected_uses": [float(root.sum()), *targets.tolist()],
        "witness_expected_uses": [float(p0.sum()), *(new_reach * p1).sum(0).tolist()],
        "witness_probabilities": torch.cat([p0[:, None], p1], 1).tolist(),
    }
    if ans["feasible_both_stages"] > ans["relaxed_upper"] + 1e-12:
        raise ValueError("Inconsistent oracle bounds")
    return ans


def run(root):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one GPU lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    fit = root / "fit.json"
    tables = read(fit)["training"]
    with torch.device("cuda"):
        result = {}
        for group, rows in (
            ("all", tables),
            (
                "outside_early_pilots",
                [w for w in tables if w["dialogue"] not in ("conv-26", "conv-30")],
            ),
        ):
            n = torch.tensor([len(w["questions"]) for w in rows], dtype=torch.float64)
            y = n.new_tensor([w["values"] for w in rows]).reshape(-1, 2, 2)
            p = n.new_tensor([[s["base"] for s in w["states"]] for w in rows]).sigmoid()
            result[group] = dict(
                capacity(p, y, n),
                windows=[r["window"] for r in rows],
                questions=int(n.sum()),
            )
    register(root / "capacity.json", {"input": file_hash(fit), "results": result})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(**vars(p.parse_args()))
