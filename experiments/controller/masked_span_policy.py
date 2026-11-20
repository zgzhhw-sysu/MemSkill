"""Enumerate suffix assignments on real trajectories, including the resulting learned policies."""

from itertools import product
from pathlib import Path

import torch

from experiments.controller.assisted_span_backup import (
    cell_table,
    continuation,
    estimates,
    outcome_baseline,
)
from experiments.controller.assisted_trial import (
    EXPERIMENT,
    PLAN,
    verified_data,
    with_types,
)
from experiments.controller.evidence_transport import (
    ARMS,
    TEMP,
    Transport,
    coefficients,
    root_targets,
)
from experiments.controller.evidence_transport_capacity import compare_suffixes
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register

MODES = ("paired", "difficulty", "assisted", "resolved")


def mask(w, b):
    return dict(
        w,
        pivot=0,
        bg=b,
        y={k: v for k, v in w["y"].items() if int(k[1]) == b},
        root=[v if j == b else None for j, v in enumerate(w["root"])],
        last=None,
    )


def target_error(early, parent, device):
    records = []
    for n, w in early.items():
        pred = read(parent / "predictions" / (n + ".json"))
        off = torch.tensor(
            [[r[k] for k in ARMS] for r in pred["offsets"][1:]],
            device=device,
            dtype=torch.float64,
        )
        p = (off + off.new_tensor(w["base"][1:])[:, None]).sigmoid()
        full = root_targets(w, p, device)
        means = outcome_baseline(early, {w["sid"]}, w, device)
        draws = [estimates(mask(w, b), p, means, cell_table(w, device)) for b in (0, 1)]
        records.append(
            {m: torch.stack([full, draws[0][m], draws[1][m]]) for m in MODES}
        )
    summary = {"questions": {}, "windows": {}}
    for mode in MODES:
        rows = [r[mode] for r in records]
        for unit, tables in (
            ("questions", rows),
            ("windows", [r.mean(1, keepdim=True) for r in rows]),
        ):
            merged = torch.cat(tables, 1)
            summary[unit][mode] = {}
            for arm in ("credit", "raw"):
                column = merged[:, :, ARMS.index(arm)]
                summary[unit][mode][arm] = compare_suffixes(column[0], column[1:])
    return summary


def donor_signals(actor, anchors, early, held, device):
    signals = {}
    for n, w in early.items():
        if w["sid"] == held:
            continue
        p = continuation(actor, n, held)
        if n in anchors:
            common = root_targets(w, p, device)[:, 0, None].expand(-1, len(MODES))
            signals[n] = torch.stack([common, common])
            continue
        means = outcome_baseline(anchors, {held, w["sid"]}, w, device)
        actual = cell_table(w, device)
        draws = []
        for b in (0, 1):
            targets = estimates(mask(w, b), p, means, actual)
            draws.append(torch.stack([targets[k][:, 0] for k in MODES], 1))
        signals[n] = torch.stack(draws)
    return signals


def masked_policy(early, turns, device):
    anchors = {n: w for n, w in early.items() if n.endswith("_s5")}
    later = {n: w for n, w in sorted(early.items()) if n.endswith("_s7")}
    if len(anchors) != 6 or len(later) != 6:
        raise ValueError("Expected six full anchors and six masked donors")
    actor = Transport(
        turns, anchors | {n: mask(w, 0) for n, w in later.items()}, device
    )
    coins = torch.tensor(
        list(product((0.0, 1.0), repeat=6)), device=device, dtype=torch.float64
    )
    summaries, details = [], []
    for held in sorted({w["sid"] for w in early.values()}):
        signals = donor_signals(actor, anchors, early, held, device)
        names = sorted(signals)
        groups = [
            (early[n]["sid"], tuple(bool(s) for s in link))
            for n in names
            for link in early[n]["links"]
        ]
        for name, w in early.items():
            if w["sid"] != held:
                continue
            sim = torch.cat([actor.similarity(name, n, 0) for n in names])
            weights, _, _ = coefficients(sim, groups, device)
            offsets = 0
            effects = coins.new_zeros((len(coins), len(MODES)))
            for n in names:
                draws = signals[n]
                size = draws.shape[1]
                vals = weights[offsets : offsets + size] @ draws
                effects += vals[0]
                if n in later:
                    effects += coins[:, list(later).index(n), None] * (
                        vals[1] - vals[0]
                    )
                offsets += size
            logits = effects.new_tensor(w["base"])
            last = (
                logits[1:] + (actor.terminal(name, {held})[:, 0] / TEMP).clamp(-2, 2)
            ).sigmoid()
            root = (logits[0] + (effects / TEMP).clamp(-2, 2)).sigmoid()
            values = cell_table(w, device)
            conditional = values[:, :, 0] + last * (values[:, :, 1] - values[:, :, 0])
            lo, hi = conditional.mean(0)
            delta = hi - lo
            native = logits.sigmoid()
            raw = values.mean(0)
            native_f1 = (1 - native[0]) * (
                (1 - native[1]) * raw[0, 0] + native[1] * raw[0, 1]
            ) + native[0] * ((1 - native[2]) * raw[1, 0] + native[2] * raw[1, 1])
            sums = {
                "f1": lo + root * delta,
                "root_shift": (root - native[0]).abs(),
                "correct_root_sign": (
                    (effects.sign() == delta.sign()) & (delta.abs() > 1e-8)
                ).double(),
            }
            summaries.append((len(w["ids"]), sums))
            details.append(
                {
                    "window": name,
                    "ids": w["ids"],
                    "types": w["types"],
                    "native": float(native_f1),
                    "terminal_only": float(lo + native[0] * delta),
                    "held_advantage": float(delta),
                    "terminal_p": last.tolist(),
                    "native_p": native.tolist(),
                    "root_p": root.tolist(),
                    "per_question_conditional_f1": conditional.tolist(),
                }
            )
    total = sum(n for n, _ in summaries)
    report = {}
    for j, mode in enumerate(MODES):
        stats = {}
        for metric in ("f1", "root_shift", "correct_root_sign"):
            if metric == "correct_root_sign":
                used = [
                    s[metric][:, j]
                    for (_, s), d in zip(summaries, details, strict=True)
                    if abs(d["held_advantage"]) > 1e-8
                ]
                values = torch.stack(used).mean(0)
            else:
                values = sum(n * s[metric][:, j] for n, s in summaries) / total
            stats[metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "max": float(values.max()),
                "by_assignment": values.tolist(),
            }
        report[mode] = stats
    return {
        "questions": total,
        "windows": details,
        "modes": MODES,
        "masked_donors": list(later),
        "assignments": coins.int().tolist(),
        "results": report,
        **{
            k: sum(len(r["ids"]) * r[k] for r in details) / total
            for k in ("native", "terminal_only")
        },
    }


def diagnose(base, study, corpus):
    src = base / "source-compression"
    dest = src / EXPERIMENT
    parent = src / "evidence-transport/pilot"
    turns, data, early = verified_data(base, study, corpus)
    early = with_types(early, data)
    report = masked_policy(early, turns, "cuda")
    report["target_error"] = target_error(early, parent, "cuda")
    report["inputs"] = {
        "protocol": file_hash(PLAN),
        "pilot": file_hash(parent / "donors.json"),
        "predictions": {
            n: file_hash(parent / "predictions" / (n + ".json")) for n in early
        },
        "diagnostic": file_hash(Path(__file__)),
    }
    register(dest / "masked-policy.json", report)
    print(
        "masked F1",
        {k: v["f1"]["mean"] for k, v in report["results"].items()},
        flush=True,
    )
    print(
        "direction",
        {k: v["credit"] for k, v in report["target_error"]["questions"].items()},
        flush=True,
    )
