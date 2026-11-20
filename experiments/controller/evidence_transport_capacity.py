"""Diagnose whether the frozen evidence actor is limited by its signs or its step sizes."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

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


def history_capacity(logits, values, counts, dialogues):
    """Bound the gains within each history and after omitting it, using the same offset cap."""
    baseline = expected(logits, torch.zeros_like(logits), values)
    grids = {
        "both": list(product((-2, 2), repeat=3)),
        "first": [(v, 0, 0) for v in (-2, 2)],
        "last": list(product((0,), (-2, 2), (-2, 2))),
    }
    gains = {
        k: expected(logits[:, None], logits.new_tensor(g), values[:, None]).amax(1)
        - baseline
        for k, g in grids.items()
    }

    def weighted(x, select):
        return float(counts[select] @ x[select] / counts[select].sum())

    result = {}
    for sid in ("all", *sorted(set(dialogues))):
        mask = torch.tensor(
            [sid == "all" or v == sid for v in dialogues], device=logits.device
        )
        result[sid] = {
            "questions": int(counts[mask].sum()),
            "baseline": weighted(baseline, mask),
            "gain": {k: weighted(x, mask) for k, x in gains.items()},
            "gain_other_histories": weighted(gains["both"], ~mask)
            if (~mask).any()
            else None,
        }
    return result


def run(base, *, by_history=False):
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
    if by_history:
        out = src / "assisted-span-backup"
        out.mkdir(exist_ok=True)
        groups = history_capacity(logits, values, n, [w["dialogue"] for w in training])
        register(
            out / "history-capacity.json",
            {
                "fit": file_hash(src / "retrieval-competition/fit.json"),
                "groups": groups,
            },
        )
        print(groups, flush=True)
        return
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


def focal_bound(rows):
    # Moving probability within each pivot preserves the native mean usage there.
    p = {}
    for pos in sorted({r["pivot"] for r in rows}):
        group = [r for r in rows if r["pivot"] == pos]
        budget = sum(r["native_p"] for r in group)
        for r in sorted(group, key=lambda r: -r["n"] * (r["activity"] - r["temporal"])):
            p[r["name"]] = min(1.0, max(0.0, budget))
            budget -= p[r["name"]]
        if abs(budget) > 1e-9:
            raise ValueError("The allocated probability mass differs from native use")
    n = sum(r["n"] for r in rows)

    def value(policy):
        return (
            sum(
                r["n"] * (r["temporal"] + policy(r) * (r["activity"] - r["temporal"]))
                for r in rows
            )
            / n
        )

    native = value(lambda r: r["native_p"])
    matched = value(lambda r: p[r["name"]])
    unrestricted = value(lambda r: float(r["activity"] > r["temporal"]))
    return {
        "windows": len(rows),
        "questions": n,
        "native": native,
        "matched_usage_gain": matched - native,
        "unrestricted_gain": unrestricted - native,
        "matched_probabilities": p,
    }


def donor_bounds(base):
    late = base / "late-pairs"
    labels = late / "selection-labels"
    manifest = labels / "through-12.json"
    ledger = read(manifest)
    split = read(labels / "split.json")["windows"]["donor"]
    if set(ledger["files"]) != set(split) or ledger["questions"] != 88:
        raise ValueError(
            "The complete 88-question randomized training block is required"
        )
    rows = []
    for name, indices in split.items():
        label = labels / (name + ".json")
        pair = late / "outcomes" / name / "pair.json"
        if file_hash(label) != ledger["files"][name] or read(label)["inputs"][
            "pair"
        ] != file_hash(pair):
            raise ValueError("The original paired score or verified labels changed")
        v = read(pair)
        if [q["original_index"] for q in v["questions"]] != indices:
            raise ValueError("The randomized donor population changed")
        probs = v["focal_probabilities"]["native"]
        lo, hi = [mean(q[a] for q in v["questions"]) for a in ("temporal", "activity")]
        native = mean(q["focal"]["native"] for q in v["questions"])
        if abs(lo + probs * (hi - lo) - native) > 1e-10:
            raise ValueError("Failed to reproduce the original focal policy score")
        rows.append(
            {
                "name": name,
                "dialogue": v["dialogue"],
                "pivot": v["pivot"],
                "n": len(indices),
                "temporal": lo,
                "activity": hi,
                "native_p": probs,
            }
        )
    report = {
        "ledger": file_hash(manifest),
        "rows": rows,
        "all": focal_bound(rows),
        "omit_history": {
            s: focal_bound([r for r in rows if r["dialogue"] != s])
            for s in sorted({r["dialogue"] for r in rows})
        },
    }
    register(labels / "donor-focal-capacity.json", report)
    print(report["all"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--by-history", action="store_true")
    mode.add_argument("--donor-window-bound", action="store_true")
    args = p.parse_args()
    if args.donor_window_bound:
        donor_bounds(args.base)
    else:
        run(args.base, by_history=args.by_history)
