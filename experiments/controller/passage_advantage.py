"""Measure where a passage-conditioned actor gains and loses F1 on actual skill paths."""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.cross_window_prefix import trajectory_hashes
from experiments.controller.factorial_rollout import CELLS, read
from experiments.controller.late_credit_kernel import support
from experiments.controller.late_pair_trial import paths
from experiments.controller.paired_grounding import window
from experiments.controller.passage_evidence_actor import ARMS, exposure, forecast, gpu
from experiments.controller.retrieval_competition import budgets, mass, summarize
from experiments.controller.source_compression import file_hash, register
from experiments.controller.within_session_opportunity import (
    evaluate as evaluate_halves,
)
from experiments.controller.witness_passage_factorial import inspect_cell
from experiments.controller.witness_passage_policy import (
    anchor_logits,
    limited,
    matched_probabilities,
    report,
)
from src.evaluation_cache import digest


def contributions(table, old, new):
    """Root advantage uses old continuation; suffix visitation uses the new root."""
    first, second = table[:, :, 0], table[:, :, 1]
    advantages = second - first
    future = first + advantages * old[1:]
    root = future[:, 1] - future[:, 0]
    advantage = torch.cat((root[:, None], advantages), dim=1)
    occupancy = torch.stack((torch.ones_like(new[0]), 1 - new[0], new[0]))
    effect = advantage * (new - old) * occupancy
    return advantage, effect


def exact(table, p):
    after = table[:, :, 0] + (table[:, :, 1] - table[:, :, 0]) * p[1:]
    return after[:, 0] + p[0] * (after[:, 1] - after[:, 0])


def rank_correlation(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None

    def ranks(values):
        loc = defaultdict(list)
        for i, v in enumerate(sorted(values)):
            loc[v].append(i)
        return [mean(loc[v]) for v in values]

    a, b = ranks(x), ranks(y)
    u, v = mean(a), mean(b)
    return sum((j - u) * (k - v) for j, k in zip(a, b, strict=True)) / math.sqrt(
        sum((j - u) ** 2 for j in a) * sum((k - v) ** 2 for k in b)
    )


def direction_report(items):
    # Each actual reached predecision state counts once in rank and direction checks.
    shift, adv = ([r[j] for r in items] for j in (0, 1))
    nonzero = [(p, a) for p, a in items if abs(p) > 1e-12 and abs(a) > 1e-12]
    return {
        "states": len(items),
        "nonzero": len(nonzero),
        "correct": sum(p * a > 0 for p, a in nonzero),
        "mean_abs_probability_shift": mean(map(abs, shift)),
        "signed_shift_vs_advantage": rank_correlation(shift, adv),
        "magnitude_vs_directional_advantage": rank_correlation(
            [abs(p) for p, a in nonzero], [a if p > 0 else -a for p, a in nonzero]
        ),
    }


def key(bits):
    return "".join(map(str, bits))


def score_window(
    w, roots, full, packet, folds, terminal, protocol, marker, dst, backend
):
    arms = tuple(packet.get("arms", ARMS))
    sid, n = w["dialogue"], w["start"]
    nodes = [f"{sid}_s{n + i}" for i in (0, 1)]
    sources = [packet["sources"][k]["text"] for k in nodes]
    ps = {}
    contexts = {}
    cells = {}
    costs = {}
    bits_list = CELLS if full else paths(w["pivot"], w["background"])
    hashes = {
        key(bits): trajectory_hashes(roots[bits] / key(bits)) for bits in bits_list
    }
    receipt = {
        "marker": file_hash(marker),
        "forecast": packet["hash"],
        "paths": hashes,
        "backend": backend,
    }
    if dst.exists():
        old = read(dst)
        if old["inputs"] != receipt:
            raise ValueError("Measured window inputs changed")
        return old
    evidence = read(marker)
    if full and backend in ("donor", "replica"):
        for bits, h in hashes.items():
            recorded = evidence["paths"][bits]
            if backend == "donor":
                h = {Path(name).name: v for name, v in h.items()}
            if recorded != h:
                raise ValueError(
                    "Full-path receipt disagrees with underlying measured files"
                )
    if not full and (
        [r["original_index"] for r in evidence["questions"]] != w["questions"]
        or evidence["paths"] != [list(b) for b in bits_list]
    ):
        raise ValueError("Randomized assignment changed")
    for bits in bits_list:
        directory = roots[bits]
        c, d, answers = inspect_cell(directory, sources, bits, w["questions"], protocol)
        for j, node in enumerate((0, 1 + bits[0])):
            z, term, raw = anchor_logits(d[j], folds[sid, j], terminal[sid])
            adj = (
                packet["by_stage"][str(j)][nodes[j]]
                if "by_stage" in packet
                else packet["offsets"][nodes[j]]
            )
            logits = {a: raw + u for a, u in adj.items()}
            logits.update(native=z, terminal=term, raw=raw)
            probs = {a: float(logits[a].sigmoid()) for a in arms}
            if (
                ps.setdefault(node, probs) != probs
                or contexts.setdefault(node, c[j]) != c[j]
            ):
                raise ValueError(
                    "Divergent pre-write states or policy on the same prefix"
                )
        cells[key(bits)] = [r["f1"] for r in answers]
        costs[key(bits)] = budgets(directory, key(bits))
    ordered = [r["f1"] for r in evidence.get("per_question", [])]
    rows = []
    directions = {}
    trial = {
        "window": f"{sid}_s{n}",
        "dialogue": sid,
        "inputs": receipt,
        "contexts": contexts,
        "probabilities": ps,
        "cells": cells,
        "costs": costs,
        "rows": rows,
        "alignment": directions,
    }
    weight = 1 if full else 2
    trial["expected_tokens"] = {
        a: {
            op: sum(weight * mass(b, ps, a) * costs[key(b)][op] for b in bits_list)
            for op in ("reader", "executor")
        }
        for a in arms
    }
    if full:
        tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.float64)
        table = tensor([cells[key(b)] for b in CELLS]).T.reshape(-1, 2, 2)
        props = {a: tensor([ps[i][a] for i in range(3)]) for a in arms}
        values = {a: exact(table, p) for a, p in props.items()}
        effects = {}
        for a in ("credit", "parent"):
            for anchor in ("raw", "native"):
                k = f"{a}-{anchor}"
                advantage, delta = contributions(table, props[anchor], props[a])
                if not torch.allclose(
                    delta.sum(1), values[a] - values[anchor], atol=1e-10, rtol=0
                ):
                    raise ValueError(
                        "Span contributions do not recover the full trajectory gain"
                    )
                directions[k] = list(
                    zip(
                        (props[a] - props[anchor]).tolist(),
                        advantage.mean(0).tolist(),
                        strict=True,
                    )
                )
                effects[k] = delta.tolist()
        uniform = [
            int(digest(["passage-advantage", trial["window"], j])[:13], 16) / 16**13
            for j in (0, 1)
        ]
        draws = {}
        for a in arms:
            bit = int(uniform[0] < ps[0][a])
            draws[a] = key((bit, int(uniform[1] < ps[1 + bit][a])))
        trial["draws"] = draws
        for i, (idx, t) in enumerate(zip(w["questions"], w["types"], strict=True)):
            vals = {a: float(v[i]) for a, v in values.items()}
            if ordered and not math.isclose(
                vals["native"], ordered[i]["native"], abs_tol=1e-10
            ):
                raise ValueError("Native early return did not reproduce")
            if backend == "donor" and not math.isclose(
                vals["native"],
                evidence["rows"][i]["returns"]["native"]["exact"],
                abs_tol=1e-10,
            ):
                raise ValueError("Native completed donor return did not reproduce")
            rows.append(
                {
                    "index": idx,
                    "type": t,
                    "f1": vals,
                    "delta": {k: v[i] for k, v in effects.items()},
                    "draw": {a: cells[b][i] for a, b in draws.items()},
                }
            )
    else:
        pivot = w["pivot"]
        node = 0 if pivot == 0 else 1 + w["background"]
        focal = ps[node]
        trial.update(
            focal_probabilities=focal,
            pivot=pivot,
            base=math.log(focal["native"] / (1 - focal["native"])),
        )
        low, high = [cells[key(b)] for b in bits_list]
        trial["endpoints"] = list(zip(low, high, strict=True))
        for a in ("credit", "parent"):
            for anchor in ("raw", "native"):
                directions[f"{a}-{anchor}"] = [
                    [focal[a] - focal[anchor], mean(high) - mean(low)]
                ]
        for i, r in enumerate(evidence["questions"]):
            if (low[i], high[i]) != (r["temporal"], r["activity"]):
                raise ValueError("Assigned endpoint F1 changed")
            vals = {a: low[i] + focal[a] * (high[i] - low[i]) for a in arms}
            ht = {
                a: 2 * sum(mass(b, ps, a) * cells[key(b)][i] for b in bits_list)
                for a in arms
            }
            if any(
                not math.isclose(actual[k], r[ref][k], abs_tol=1e-10)
                for actual, ref in [(vals, "focal"), (ht, "ht")]
                for k in ("native", "terminal")
            ):
                raise ValueError("Original randomized policy estimates changed")
            rows.append(
                {"index": r["original_index"], "type": r["type"], "f1": vals, "ht": ht}
            )
    register(dst, trial)
    return read(dst)


def equal_rate_exact(panel):
    def t(x):
        return torch.tensor(x, device="cuda", dtype=torch.float64)

    probs = {
        a: t([[w["probabilities"][str(n)][a] for n in range(3)] for w in panel])
        for a in ("native", "credit")
    }
    w = t([len(v["rows"]) for v in panel])
    c = probs["credit"]
    z = probs["native"].logit()
    root = matched_probabilities(z[:, 0], c[:, 0], w)
    target = (w * ((1 - c[:, 0]) * c[:, 1] + c[:, 0] * c[:, 2])).sum() / w.sum()
    weights = (w[:, None] * torch.stack((1 - root, root), dim=1)).flatten()
    suffix = matched_probabilities(
        z[:, 1:].flatten(), torch.full_like(weights, float(target)), weights
    ).reshape(-1, 2)
    conditional = torch.cat((root[:, None], suffix), dim=1)
    if (
        abs(float(w @ (root - c[:, 0]) / w.sum())) > 1e-10
        or abs(float(weights @ suffix.flatten() / weights.sum() - target)) > 1e-10
    ):
        raise ValueError("Path visitation-weighted skill rates do not match")
    vals = []
    for p, v in zip(conditional, panel, strict=True):
        table = t([v["cells"][key(b)] for b in CELLS]).T.reshape(-1, 2, 2)
        vals.extend(exact(table, p).tolist())
    return mean(vals)


def summarize_panel(panel, full, arms=ARMS):
    stats = summarize(panel, arms, "credit", arms[1:]) if full else report(panel, arms)
    if not panel:
        return stats
    rows = [r for w in panel for r in w["rows"]]
    stats["expected_tokens_per_window"] = {
        a: {
            op: mean(w["expected_tokens"][a][op] for w in panel)
            for op in ("reader", "executor")
        }
        for a in arms
    }
    stats["alignment"] = {}
    for k in panel[0]["alignment"]:
        items = defaultdict(list)
        for w in panel:
            for j, v in enumerate(w["alignment"][k]):
                stage = (
                    ("root" if j == 0 else "suffix")
                    if full
                    else ("root" if w["pivot"] == 0 else "suffix")
                )
                items[stage].append(v)
        stats["alignment"][k] = {s: direction_report(v) for s, v in items.items()}
    if full:
        stats["matched"] = equal_rate_exact(panel)
        stats["draws_different_from_raw"] = [
            sum(w["draws"]["credit"] != w["draws"]["raw"] for w in panel),
            len(panel),
        ]
        stats["sampled_f1"] = {a: mean(r["draw"][a] for r in rows) for a in arms}
        stats["stage_gain"] = {
            k: [mean(r["delta"][k][j] for r in rows) for j in range(3)]
            for k in panel[0]["alignment"]
        }
        stats["stage_gain_by_history"] = {
            sid: {
                k: [
                    mean(
                        r["delta"][k][i]
                        for w in panel
                        if w["dialogue"] == sid
                        for r in w["rows"]
                    )
                    for i in range(3)
                ]
                for k in panel[0]["alignment"]
            }
            for sid in stats["by_dialogue"]
        }
    return stats


def evaluate(base, study, corpus, path):
    trial = path.parent
    primary = base / "late-pairs"
    src = base / "source-compression"
    original = src / "witness-passages"
    before = exposure(base, path)
    packet = read(path)
    arms = tuple(packet.get("arms", ARMS))
    packet["hash"] = file_hash(path)
    proto = read(study / "protocol.json")
    weights = read(primary / "frozen.json")["terminal"]
    training = read(src / "capped-source-credit/policy/joint-repaired/fit.json")[
        "observations"
    ]
    folds = {
        (sid, j): support(training, sid, j, "cuda")[0]
        for sid in weights
        for j in (0, 1)
    }
    data = {r["sample_id"]: r for r in read(corpus) if r["sample_id"] in weights}
    cohort = read(primary / "cohort.json")["assignments"]
    split = read(primary / "selection-labels/split.json")["windows"]
    groups = defaultdict(list)
    for sid, n in read(src / "factorial/manifest.json")["windows"]:
        convo = window(data[sid], n, 2)
        ids = [q["original_index"] for q in convo["qa"]]
        w = {
            "dialogue": sid,
            "start": n,
            "questions": ids,
            "types": [q["category"] for q in convo["qa"]],
        }
        root = src / "factorial" / f"{sid}_s{n}"
        marker = root / "measurement.json"
        if ids != [r["original_index"] for r in read(marker)["per_question"]]:
            raise ValueError("Early question mismatch")
        groups["early"].append((w, dict.fromkeys(CELLS, root), marker))
    for w in cohort:
        name = f"{w['dialogue']}_s{w['start']}"
        if (name in split["donor"]) == (name in split["evaluation"]):
            raise ValueError("Ambiguous partition")
        donor = name in split["donor"]
        if w["questions"] != split["donor" if donor else "evaluation"][name]:
            raise ValueError("Different question split")
        orig = primary / "outcomes" / name
        pair = orig / "pair.json"
        b = paths(w["pivot"], w["background"])
        if pair.is_file():
            groups["calibration" if donor else "later"].append(
                (w, dict.fromkeys(b, orig), pair)
            )
        full = (
            primary / "donor-factorial" / name
            if donor
            else original / "four-paths/cells" / name
        )
        marker = full / "complete.json"
        if marker.is_file():
            roots = {k: orig if donor and k in b else full for k in CELLS}
            groups["donor" if donor else "replica"].append((w, roots, marker))
    counts = {
        "early": (12, 91),
        "calibration": (12, 88),
        "donor": (12, 88),
        "later": (40, 378),
        "replica": (40, 378),
    }
    for label, jobs in groups.items():
        full = label not in ("calibration", "later")
        out = trial / label
        out.mkdir(exist_ok=True)
        panels = []
        for w, roots, f in jobs:
            with limited():
                v = score_window(
                    w,
                    roots,
                    full,
                    packet,
                    folds,
                    weights,
                    proto,
                    f,
                    out / f"{w['dialogue']}_s{w['start']}.json",
                    label,
                )
            panels.append(v)
        strict = (
            [w for w in panels if w["window"] not in before]
            if label in ("later", "replica")
            else []
        )
        with limited():
            total = summarize_panel(panels, full, arms)
            fresh = summarize_panel(strict, full, arms)
        result = {"forecast": packet["hash"], "all": total, "strict": fresh}
        register(out / f"through-{len(jobs):02d}.json", result)
        if len(jobs) == counts[label][0]:
            if total["questions"] != counts[label][1]:
                raise ValueError("Final population size changed")
            register(out / "summary.json", result)
    later = trial / "later/summary.json"
    replica = trial / "replica/summary.json"
    if trial.name == "witness" and later.exists() and replica.exists():
        a = read(later)["all"]
        strict = read(later)["strict"]
        f = read(replica)["all"]["f1"]
        ok = (
            all(
                a["f1"]["credit"] >= a["f1"][k] + 0.02
                and a["ht"]["credit"] > a["ht"][k]
                for k in ("native", "terminal")
            )
            and all(a["f1"]["credit"] > a["f1"][k] for k in arms[1:])
            and a["f1"]["credit"] > a["matched"]
            and sum(
                v["f1"]["credit"] > a["matched_by_history"][sid]
                for sid, v in a["by_dialogue"].items()
            )
            >= 4
            and strict["questions"] > 0
            and all(
                strict["f1"]["credit"] > strict["f1"][k] and f["credit"] > f[k]
                for k in ("raw", "native")
            )
        )
        register(
            trial / "decision.json",
            {
                "later": file_hash(later),
                "replica": file_hash(replica),
                "eligible_for_development": ok,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key_arg in ("base", "study", "corpus"):
        parser.add_argument(key_arg, type=Path)
    args = parser.parse_args()
    gpu()
    for stage in ("group", "witness"):
        path = forecast(args.base, args.study, args.corpus, stage)
        if path:
            evaluate(args.base, args.study, args.corpus, path)
            print(path.parent, flush=True)
    evaluate_halves(args.base, args.study, args.corpus)
