"""Allocate a fixed native use rate by out-of-history rankings of physically paired advantages."""

import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller.audit_cross_window import verified_decision
from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.factorial_rollout import read
from experiments.controller.late_credit_kernel import kernel_effect, score, support
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.retrieval_competition import summarize
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "credit_rank_allocation.json"
RANKS = ("credit", "raw", "shuffled")
ARMS = (*RANKS, "soft", "native", "terminal")


def allocate(scores, logits, values, weights):
    """Exact fixed-budget optimum over mixtures of monotone threshold policies."""
    if (
        not all(
            t.ndim == 1 and t.shape == scores.shape and torch.isfinite(t).all()
            for t in (scores, logits, values, weights)
        )
        or not len(scores)
        or not (weights > 0).all()
    ):
        raise ValueError("Expected finite, aligned observations with positive weights")
    w = weights / weights.sum()
    unique = scores.unique(sorted=True)
    mid = unique[:-1] + (unique[1:] - unique[:-1]) / 2
    mid = torch.where(mid <= unique[:-1], unique[1:], mid)
    on = torch.ones_like(scores[:, None], dtype=torch.bool)
    high = torch.cat((on, scores[:, None] >= mid, ~on), dim=1)
    p = (logits[:, None] + torch.where(high, 2.0, -2.0)).sigmoid()
    native = logits.sigmoid()
    budget = w @ native
    rates = w @ p
    gained = (w * values) @ (p - native[:, None])
    width = rates[:, None] - rates[None, :]
    allowed = (
        (width > 1e-13)
        & (rates[:, None] >= budget - 1e-14)
        & (rates[None, :] <= budget + 1e-14)
    )
    mix = ((budget - rates[None, :]) / width.clamp_min(1e-14)).clamp(0, 1)
    gains = mix * gained[:, None] + (1 - mix) * gained[None, :]
    best = gains.masked_fill(~allowed, -torch.inf).max()
    identity = {
        "native": True,
        "budget": float(budget),
        "achieved": float(budget),
        "training_gain": 0.0,
        "movement": 0.0,
    }
    if best <= 1e-12:
        return identity
    delta = (
        p[:, :, None] * mix + p[:, None, :] * (1 - mix) - native[:, None, None]
    ).abs()
    movement = (w[:, None, None] * delta).sum(0)
    selected = (
        movement.masked_fill(~allowed | (gains < best - 1e-12), torch.inf)
        .argmin()
        .item()
    )
    i, j = divmod(selected, len(rates))
    fraction = float(mix[i, j])
    achieved = float(fraction * rates[i] + (1 - fraction) * rates[j])
    if abs(achieved - float(budget)) > 1e-11:
        raise ValueError("The allocation changed the native training rate")
    cuts = ["all", *mid.tolist(), "none"]
    return {
        "native": False,
        "budget": float(budget),
        "achieved": achieved,
        "training_gain": float(gains[i, j]),
        "movement": float(movement[i, j]),
        "cuts": [cuts[i], cuts[j]],
        "fraction": fraction,
    }


def apply_allocation(score, logit, rule):
    if rule["native"]:
        return logit.sigmoid()

    def fraction(cut):
        if cut == "all":
            return torch.ones_like(score)
        if cut == "none":
            return torch.zeros_like(score)
        return (score >= cut).to(logit.dtype)

    a, b = map(fraction, rule["cuts"])
    mass = rule["fraction"] * a + (1 - rule["fraction"]) * b
    low, high = (logit - 2).sigmoid(), (logit + 2).sigmoid()
    return low + mass * (high - low)


def permutation(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["dialogue"], r["stage"], r["weight"]].append(i)
    shuffled = [0.0] * len(rows)
    for ids in groups.values():
        for dst, src in zip(ids, ids[1:] + ids[:1], strict=True):
            shuffled[dst] = rows[src]["credit"]
    return [dict(r, shuffled=s) for r, s in zip(rows, shuffled, strict=True)]


def data_for(rows, stage, device):
    selected = [r for r in rows if r["stage"] == stage]
    # These rows are already filtered by whole history. Reuse the published donor weighting.
    (states, weights, ys), _ = support(selected, None, stage, device)
    additional = ys.new_tensor([r["raw"] + r["shuffled"] for r in selected])
    return states, weights, torch.column_stack((ys, additional))


def fit(rows, held, stage, device):
    selected = [r for r in rows if r["dialogue"] != held and r["stage"] == stage]
    states, weights, ys = data_for(selected, stage, device)
    predictions = torch.empty_like(ys)
    outer = {}
    for history in sorted({r["dialogue"] for r in selected}):
        idx = [i for i, r in enumerate(selected) if r["dialogue"] == history]
        others = [r for r in selected if r["dialogue"] != history]
        data = data_for(others, stage, device)
        predictions[idx] = kernel_effect(states[idx], *data)
        outer[history] = sorted({r["dialogue"] for r in others})
    logits = states.new_tensor([r["base"] for r in selected])
    rules = {
        name: allocate(predictions[:, i], logits, ys[:, 1], weights)
        for i, name in enumerate(RANKS)
    }
    receipt = {
        "inner_histories": outer,
        "calibration": [
            {
                "history": r["dialogue"],
                "window": r["window"],
                "scores": p,
                "weight": wt,
                "return": r["raw"],
                "base": r["base"],
            }
            for r, p, wt in zip(
                selected, predictions.tolist(), weights.tolist(), strict=True
            )
        ],
        "rules": rules,
    }
    return ((states, weights, ys), rules), receipt


def probabilities(decision, model, terminal):
    data, rules = model
    s = data[0].new_tensor(decision["state"])
    z = s.new_tensor(decision["base_logit"])
    scores = kernel_effect(s[None], *data)[0]
    probs = [apply_allocation(scores[i], z, rules[k]) for i, k in enumerate(RANKS)]
    term = s.new_tensor(terminal)
    probs.extend(
        (
            (z + (scores[0] / 0.05).clamp(-2, 2)).sigmoid(),
            z.sigmoid(),
            (z + term[0] + term[1:] @ F.normalize(s, dim=0)).sigmoid(),
        )
    )
    return dict(zip(ARMS, torch.stack(probs).tolist(), strict=True))


def focal_record(decision, model, j):
    (states, weights, ys), _ = model
    s = states.new_tensor(decision["state"])
    k = (F.normalize(states, dim=1) @ F.normalize(s, dim=0)).clamp(0, 1).pow(
        8
    ) * weights
    signal = k @ ys[:, 0]
    absolute = k @ ys[:, 0].abs()
    base = decision["base_logit"]
    prior = float(s.new_tensor(base).sigmoid())
    return {
        "pivot": j,
        "base": base,
        "scores": dict(
            zip(
                RANKS,
                kernel_effect(s[None], states, weights, ys)[0].tolist(),
                strict=True,
            )
        ),
        "prior_slope": prior * (1 - prior),
        "kernel_mass_fraction": float(k.sum() / (1 + k.sum())),
        "noncancelled_fraction": float(signal.abs() / absolute)
        if absolute > 0
        else None,
    }


def observe(late, assignment, models, terminal):
    panel = score(late, assignment, models, terminal, policy=probabilities, arms=ARMS)
    directory = late / "outcomes" / panel["window"]
    pair = read(directory / "pair.json")
    j = pair["pivot"]
    low = "".join(map(str, pair["paths"][0]))
    decision = read(directory / low / "decisions.json")[j]
    panel["focal"] = focal_record(decision, models[pair["dialogue"], j], j)
    panel["observed"] = [
        {
            "index": q["original_index"],
            "low": q["temporal"],
            "delta": q["activity"] - q["temporal"],
        }
        for q in pair["questions"]
    ]
    return panel


def summarize_panel(panels, device, *, diagnostic=False):
    stats = summarize(panels, ARMS, "credit", ARMS[1:])
    if not panels:
        return stats
    matched = []
    ceiling = defaultdict(float)
    for stage in (0, 1):
        group = [p for p in panels if p["focal"]["pivot"] == stage]
        if not group:
            continue

        def vector(values):
            return torch.tensor(values, device=device, dtype=torch.float64)

        base = vector([p["focal"]["base"] for p in group])
        weight = vector([len(p["rows"]) for p in group])
        candidate = vector([p["focal_probabilities"]["credit"] for p in group])
        shift = matched_offset(base, candidate, weight)
        rates = (base + shift).sigmoid()
        if abs(float(weight @ (rates - candidate) / weight.sum())) > 1e-10:
            raise ValueError("Failed realized-rate match")
        for p, rate in zip(group, rates.tolist(), strict=True):
            matched.extend(
                (p["dialogue"], r["low"] + rate * r["delta"]) for r in p["observed"]
            )
        if diagnostic:
            delta = vector([mean(r["delta"] for r in p["observed"]) for p in group])
            for name in ("credit", "raw", "omniscient"):
                order = (
                    delta
                    if name == "omniscient"
                    else vector([p["focal"]["scores"][name] for p in group])
                )
                optimal = allocate(order, base, delta, weight)
                ceiling[name] += optimal["training_gain"] * int(weight.sum())
    stats["matched_rate"] = {
        "f1": mean(v for _, v in matched),
        "by_dialogue": {
            sid: mean(v for s, v in matched if s == sid) for sid in stats["by_dialogue"]
        },
    }
    if diagnostic:
        stats["rank_oracle_gain_vs_native"] = {
            k: v / stats["questions"] for k, v in ceiling.items()
        }
    stats["mean_abs_change_from_raw"] = mean(
        abs(p["focal_probabilities"]["credit"] - p["focal_probabilities"]["raw"])
        for p in panels
    )
    stats["kernel_diagnostics"] = {
        key: mean(v)
        if (v := [p["focal"][key] for p in panels if p["focal"][key] is not None])
        else None
        for key in ("prior_slope", "kernel_mass_fraction", "noncancelled_fraction")
    }
    return stats


def passes(all_stats, strict):
    if (
        all_stats["windows"] != 40
        or all_stats["questions"] != 378
        or not strict["questions"]
    ):
        return False
    v, ht = all_stats["f1"], all_stats["ht"]
    return (
        all(
            v["credit"] >= v[k] + 0.02 and ht["credit"] > ht[k]
            for k in ("native", "terminal")
        )
        and all(v["credit"] > v[k] for k in ("raw", "shuffled", "soft"))
        and v["credit"] > all_stats["matched_rate"]["f1"]
        and sum(
            d["f1"]["credit"] > all_stats["matched_rate"]["by_dialogue"][s]
            for s, d in all_stats["by_dialogue"].items()
        )
        >= 4
        and all(strict["f1"]["credit"] > strict["f1"][k] for k in ("native", "raw"))
    )


def audit_long_prefix(late, dest, models, terminal, splits, device):
    root = late / "cross-window-prefix"
    directory = dest / "long-prefix"
    directory.mkdir(exist_ok=True)
    manifest = root / "manifest.json"
    specs = read(manifest)
    local = {
        (name.rsplit("_s", 1)[0], i)
        for split in splits.values()
        for name, ids in split.items()
        for i in ids
    }
    population = [
        (w["dialogue"], q["original_index"])
        for w in specs["endpoints"]
        for q in w["questions"]
    ]
    if (
        len(local) != 466
        or len(specs["endpoints"]) != 99
        or len(set(population)) != 185
        or len(population) != 185
        or set(population) & local
        or specs["baseline"] != file_hash(late / "frozen.json")
    ):
        raise ValueError("Wrong or overlapping long-prefix cohort")
    path = directory / "registration.json"
    identity = {
        "plan": file_hash(PLAN.with_name("credit_rank_long_prefix.json")),
        "cohort": file_hash(manifest),
        "fit": file_hash(dest / "fit.json"),
    }
    if not path.exists():
        # Include incomplete pairs at the point this evaluation was locked.
        cutoff = time.time_ns()
        old = {
            str(p.relative_to(root)): file_hash(p)
            for p in root.glob("conv-*_to*/*/scores.json")
            if set(p.parent.name) <= {"0", "1"} and p.stat().st_mtime_ns <= cutoff
        }
        register(path, dict(identity, branches=old, cutoff_ns=cutoff))
    saved = read(path)
    if any(saved[k] != v for k, v in identity.items()) or any(
        file_hash(root / k) != v for k, v in saved["branches"].items()
    ):
        raise ValueError("Different long-prefix lock or historical score")
    preexisting = {k.split("/")[0] for k in saved["branches"]}
    panels = []
    for w in specs["endpoints"]:
        sid = w["dialogue"]
        name = f"{sid}_to{w['end']:02d}"
        folder = root / name
        pair = folder / "pair.json"
        if not pair.exists():
            continue
        receipt = directory / (name + ".json")
        inputs = {"pair": file_hash(pair), "registration": file_hash(path)}
        if not receipt.exists():
            r = read(pair)
            if (
                r["manifest"] != identity["cohort"]
                or [
                    {k: q[k] for k in ("original_index", "type", "sessions")}
                    for q in r["questions"]
                ]
                != w["questions"]
            ):
                raise ValueError("Long-prefix evidence changed")
            dec = verified_decision(folder, w, r, specs["skills"])
            state = models[sid, 1]
            ps = probabilities(dec, state, terminal[sid])
            rows = []
            observed = []
            for idx, action in enumerate(("temporal", "activity")):
                bits = "".join(map(str, [*w["background"], idx]))
                answers = read(folder / bits / "scores.json")["answers"]
                if len(answers) != len(w["questions"]) or any(
                    v["qa_index"] != i or v["f1"] != q[action]
                    for i, (v, q) in enumerate(
                        zip(answers, r["questions"], strict=True)
                    )
                ):
                    raise ValueError("Long-prefix paired answers disagree")
            for q in r["questions"]:
                diff = q["activity"] - q["temporal"]
                f1 = {k: q["temporal"] + p * diff for k, p in ps.items()}
                if any(
                    abs(ps[k] - r["probabilities"][k]) > 1e-10
                    or abs(f1[k] - q["focal"][k]) > 1e-10
                    for k in ("native", "terminal")
                ):
                    raise ValueError("Could not reproduce long-prefix baselines")
                rows.append(
                    {
                        "index": q["original_index"],
                        "type": q["type"],
                        "f1": f1,
                        "sessions": q["sessions"],
                    }
                )
                observed.append(
                    {"index": q["original_index"], "low": q["temporal"], "delta": diff}
                )
            register(
                receipt,
                {
                    "inputs": inputs,
                    "panel": {
                        "window": name,
                        "dialogue": sid,
                        "end": w["end"],
                        "cost": r["cost"],
                        "focal_probabilities": ps,
                        "focal": focal_record(dec, state, 1),
                        "rows": rows,
                        "observed": observed,
                    },
                },
            )
        result = read(receipt)
        if result["inputs"] != inputs:
            raise ValueError("Long-prefix paired outcomes changed")
        panels.append(result["panel"])
    if panels:
        fresh = [p for p in panels if p["window"] not in preexisting]
        report = {
            "registration": file_hash(path),
            "registered_questions": 185,
            "all": summarize_panel(panels, device),
            "strict": summarize_panel(fresh, device),
        }
        register(directory / f"through-{len(panels):03d}.json", report)
        if len(panels) == 99:
            register(directory / "summary.json", report)


def run(base):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use one leased GPU for policy learning")
    torch.cuda.set_per_process_memory_fraction(0.06)
    late = base / "late-pairs"
    capped = base / "source-compression/capped-source-credit"
    cohorts = read(late / "cohort.json")["assignments"]
    splits = read(late / "selection-labels/split.json")["windows"]
    terminal = read(late / "frozen.json")["terminal"]
    device = "cuda"
    for stage in ("joint-mask", "joint-repaired"):
        source = capped / "policy" / stage / "fit.json"
        if not source.exists():
            continue
        record = read(source)
        if (
            record["evidence"]["registration"]
            != file_hash(capped / "registration.json")
            or record["evidence"]["writer_blocks"]
            != file_hash(
                base / "source-compression/writer-block-repair/credit/summary.json"
            )
            or record["evidence"]["joint_plan"]
            != file_hash(PLAN.with_name("joint_source_credit.json"))
            or record["evidence"]["split"]
            != file_hash(late / "selection-labels/split.json")
            or record["evidence"]["cohort"] != file_hash(late / "cohort.json")
            or record["evidence"]["terminal"] != file_hash(late / "frozen.json")
        ):
            raise ValueError("Parent policy or measured cohort changed")
        rows = permutation(record["observations"])
        histories = sorted({r["dialogue"] for r in rows})
        if len(histories) != 6 or len(rows) != 60:
            raise ValueError("Incomplete training comparisons")
        dest = capped / "rank-allocation" / stage
        dest.mkdir(exist_ok=True, parents=True)
        frozen = dest / "fit.json"
        identity = {"plan": file_hash(PLAN), "parent": file_hash(source)}
        models = {}
        if frozen.exists():
            saved = read(frozen)
            if any(saved[k] != v for k, v in identity.items()):
                raise ValueError("Frozen allocation inputs changed")
            for sid in histories:
                others = [r for r in rows if r["dialogue"] != sid]
                for j in (0, 1):
                    models[sid, j] = (
                        data_for(others, j, device),
                        saved["folds"][f"{sid}/{j}"]["rules"],
                    )
        else:
            folds = {}
            for sid in histories:
                for j in (0, 1):
                    models[sid, j], folds[f"{sid}/{j}"] = fit(rows, sid, j, device)
            register(frozen, dict(identity, folds=folds))
        old = scored_at(late / "outcomes", frozen.stat().st_mtime_ns)
        register(dest / "preexisting.json", {"fit": file_hash(frozen), "scores": old})
        before = {k.split("/")[0] for k in old}
        audit_long_prefix(late, dest, models, terminal, splits, device)
        for label, split in (("calibration", "donor"), ("later", "evaluation")):
            selected = splits[split]
            panels = []
            for w in cohorts:
                name = f"{w['dialogue']}_s{w['start']}"
                if (
                    name in selected
                    and (late / "outcomes" / name / "pair.json").exists()
                ):
                    if w["questions"] != selected[name]:
                        raise ValueError("Target questions changed")
                    panels.append(observe(late, w, models, terminal))
            if not panels:
                continue
            fresh = [p for p in panels if p["window"] not in before]
            all_stats = summarize_panel(
                panels, device, diagnostic=label == "calibration"
            )
            strict = summarize_panel(fresh, device)
            report = {
                "fit": file_hash(frozen),
                "registered_questions": sum(map(len, selected.values())),
                "all": all_stats,
                "strict": strict,
                "eligible_for_development": stage == "joint-repaired"
                and passes(all_stats, strict),
                "windows": panels,
            }
            directory = dest / label
            directory.mkdir(exist_ok=True)
            register(directory / f"through-{len(panels):02d}.json", report)
            if len(panels) == len(selected):
                register(directory / "summary.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    run(p.parse_args().base)
