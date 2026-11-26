"""Predict a randomized span effect from cause labels in earlier, different dialogues."""

import argparse
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.late_selection_labels import registration
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.retrieval_competition import budgets, mass, summarize

ARMS = ("credit", "early", "raw", "rate", "native", "terminal")


def observation(sid, w, stage, decision, raw, credit, weight, early):
    return {
        "dialogue": sid,
        "window": w,
        "stage": stage,
        "state": decision["state"],
        "base": decision["base_logit"],
        "raw": raw,
        "credit": credit,
        "weight": weight,
        "early": early,
    }


def early_observations(base):
    source = base / "source-compression"
    studies = read(source / "retrieval-competition/fit.json")["training"]
    root = read(source / "root-overlap/fit.json")["effects"]
    rows = []
    for w in studies:
        y = w["values"]
        for b in range(2):
            d = read(source / "factorial" / w["window"] / f"{b}0" / "decisions.json")
            for j, delta, bonus in (
                (0, y[2 + b] - y[b], root[w["window"]][b]),
                (1, y[2 * b + 1] - y[2 * b], w["signed"][b]),
            ):
                rows.append(
                    observation(
                        w["dialogue"], w["window"], j, d[j], delta, bonus, 0.5, True
                    )
                )
    return rows


def donor_observations(late, out, donors):
    rows = []
    for w, ids in donors.items():
        label = read(out / (w + ".json"))
        pair = read(late / "outcomes" / w / "pair.json")
        if (
            label["inputs"]["pair"] != file_hash(late / "outcomes" / w / "pair.json")
            or [q["original_index"] for q in label["rows"]] != ids
        ):
            raise ValueError(
                "Donor observations lost their correspondence to the paired experiment"
            )
        j = label["pivot"]
        bits = "".join(map(str, pair["paths"][0]))
        dec = read(late / "outcomes" / w / bits / "decisions.json")[j]
        rows.append(
            observation(
                label["dialogue"],
                w,
                j,
                dec,
                mean(r["delta"] for r in label["rows"]),
                mean(r["credit"] for r in label["rows"]),
                1.0,
                False,
            )
        )
    return rows


def kernel_effect(query, bank, weights, y):
    similarities = (
        (F.normalize(query, dim=-1) @ F.normalize(bank, dim=-1).T).clamp(0, 1).pow(8)
    )
    k = similarities * weights
    return (k @ y) / (k.sum(1, keepdim=True) + 1.0)


def support(rows, sid, stage, device):
    donors = [r for r in rows if r["dialogue"] != sid and r["stage"] == stage]
    if not donors:
        raise ValueError("No independent donors for this decision")
    weights = torch.tensor(
        [r["weight"] for r in donors], dtype=torch.float64, device=device
    )
    ids = [r["dialogue"] for r in donors]
    counts = {
        name: sum(r["weight"] for r in donors if r["dialogue"] == name)
        for name in set(ids)
    }
    weights = (
        weights
        * weights.sum()
        / len(counts)
        / weights.new_tensor([counts[name] for name in ids])
    )
    states = weights.new_tensor([r["state"] for r in donors])
    ys = weights.new_tensor([[r["raw"] + r["credit"], r["raw"]] for r in donors])
    base = weights.new_tensor([r["base"] for r in donors])
    p = (
        base + (kernel_effect(states, states, weights, ys)[:, 0] / 0.05).clamp(-2, 2)
    ).sigmoid()
    shift = matched_offset(base, p, weights)
    return (states, weights, ys), shift


def freeze(dest, late, evidence, rows):
    path = dest / "fit.json"
    if not path.exists():
        register(path, {"evidence": evidence, "observations": rows})
    if read(path) != {"evidence": evidence, "observations": rows}:
        raise ValueError("Changed donor inputs at numeric freeze")
    archived = scored_at(late / "outcomes", path.stat().st_mtime_ns)
    register(dest / "preexisting.json", {"fit": file_hash(path), "scores": archived})
    return {k.split("/")[0] for k in archived}


def probabilities(d, model, weights):
    data, early, shift = model
    state = data[0].new_tensor(d["state"])
    base = state.new_tensor(d["base_logit"])
    cred, raw = kernel_effect(state[None], *data)[0]
    prev = kernel_effect(state[None], *early)[0, 0]
    w = state.new_tensor(weights)
    terminal = w[0] + w[1:] @ F.normalize(state, dim=0)
    shifts = torch.stack(
        [(t / 0.05).clamp(-2, 2) for t in (cred, prev, raw)]
        + [state.new_tensor(shift), state.new_zeros(()), terminal]
    )
    return dict(zip(ARMS, (base + shifts).sigmoid().tolist(), strict=True))


def score(late, assignment, models, weights, *, policy=probabilities, arms=ARMS):
    sid = assignment["dialogue"]
    name = f"{sid}_s{assignment['start']}"
    folder = late / "outcomes" / name
    pair = read(folder / "pair.json")
    if any(
        pair[k] != assignment[k] for k in ("dialogue", "pivot", "background", "start")
    ) or [(r["original_index"], r["type"]) for r in pair["questions"]] != list(
        zip(assignment["questions"], assignment["types"], strict=True)
    ):
        raise ValueError("Forward panel changed")
    ps = {}
    branches = []
    cost = {}
    for a, b in pair["paths"]:
        bits = f"{a}{b}"
        branches.append(bits)
        decisions = read(folder / bits / "decisions.json")
        if [int(d["use"]) for d in decisions] != [a, b]:
            raise ValueError("Observed actions differ")

        cost[bits] = budgets(folder, bits)
        for stage, (key, d) in enumerate(zip((0, 1 + a), decisions, strict=True)):
            p = policy(d, models[sid, stage], weights[sid])
            if ps.setdefault(key, p) != p:
                raise ValueError("Different probability for same focal state")
    focal = ps[0 if pair["pivot"] == 0 else 1 + pair["background"]]
    rows = []
    for q in pair["questions"]:
        low, high = q["temporal"], q["activity"]
        values = {k: low + p * (high - low) for k, p in focal.items()}
        ht = {
            k: 2 * (mass(branches[0], ps, k) * low + mass(branches[1], ps, k) * high)
            for k in arms
        }
        for k in ("native", "terminal"):
            if not math.isclose(
                values[k], q["focal"][k], abs_tol=1e-10
            ) or not math.isclose(ht[k], q["ht"][k], abs_tol=1e-10):
                raise ValueError("Failed independent return or native check")
        rows.append(
            {"index": q["original_index"], "type": q["type"], "f1": values, "ht": ht}
        )
    return {
        "window": name,
        "dialogue": sid,
        "probabilities": ps,
        "focal_probabilities": focal,
        "rows": rows,
        "expected_tokens": {
            k: {
                op: sum(2 * mass(b, ps, k) * cost[b][op] for b in branches)
                for op in ("reader", "executor")
            }
            for k in arms
        },
        "pair": file_hash(folder / "pair.json"),
    }


def run(base):
    late = base / "late-pairs"
    out = registration(late)
    split = read(out / "split.json")["windows"]
    raw = read(late / "cohort.json")["assignments"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Lease one GPU for fitting and scoring the kernel")
    torch.cuda.set_per_process_memory_fraction(0.12)
    early = early_observations(base)
    donors = split["donor"]
    for stage in ("pilot", "expanded"):
        if stage == "expanded" and not all(
            (out / (w + ".json")).is_file() for w in donors
        ):
            continue
        dest = out / stage
        dest.mkdir(exist_ok=True)
        rows = (
            early if stage == "pilot" else early + donor_observations(late, out, donors)
        )
        evidence = {
            "split": file_hash(out / "split.json"),
            "root": file_hash(base / "source-compression/root-overlap/fit.json"),
            "early": file_hash(
                base / "source-compression/retrieval-competition/fit.json"
            ),
            "later": {}
            if stage == "pilot"
            else {w: file_hash(out / (w + ".json")) for w in donors},
        }
        models = {}
        for sid in sorted({r["dialogue"] for r in rows}):
            for j in (0, 1):
                data, shift = support(rows, sid, j, "cuda")
                first = data if stage == "pilot" else support(early, sid, j, "cuda")[0]
                models[sid, j] = data, first, shift
        evidence["donor_rate_offsets"] = {
            f"{sid}/{j}": fitted[2] for (sid, j), fitted in models.items()
        }
        before = freeze(dest, late, evidence, rows)
        weights = read(late / "frozen.json")["terminal"]
        selected = [w for w in raw if (w["start"] <= 11) == (stage == "pilot")]
        panels = []
        for w in selected:
            name = f"{w['dialogue']}_s{w['start']}"
            if not (late / "outcomes" / name / "pair.json").exists():
                break
            panels.append(score(late, w, models, weights))
        if panels:
            fresh = [w for w in panels if w["window"] not in before]
            stats = summarize(fresh, ARMS, "credit", ARMS[1:])
            passed = False
            if (
                stage == "expanded"
                and len(panels) == 40
                and stats["questions"] >= 0.9 * 378
                and len(stats["by_dialogue"]) == 6
            ):
                v, ht = stats["f1"], stats["ht"]
                passed = (
                    all(
                        v["credit"] >= v[k] + 0.02 and ht["credit"] > ht[k]
                        for k in ("native", "terminal")
                    )
                    and all(v["credit"] > v[k] for k in ("raw", "early", "rate"))
                    and sum(
                        s["f1"]["credit"] > s["f1"]["rate"]
                        for s in stats["by_dialogue"].values()
                    )
                    >= 4
                )
            result = {
                "fit": file_hash(dest / "fit.json"),
                "registered_questions": sum(len(w["questions"]) for w in selected),
                "all": summarize(panels, ARMS, "credit", ARMS[1:]),
                "strict": stats,
                "eligible_for_development": passed,
                "windows": panels,
            }
            register(dest / f"through-{len(panels):02d}.json", result)
            if len(panels) == len(selected):
                register(dest / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    run(parser.parse_args().base)
