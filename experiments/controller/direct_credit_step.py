"""Calibrate the size of a direct credit move on held donor histories."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.direct_skill_allocation import (
    direct_distributions,
    features,
)
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import weights
from experiments.controller.skill_graph_data import training_rows
from experiments.controller.three_skill_margin import fit_contrasts, score
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "direct_credit_step.json"


def scale(gains):
    if not gains:
        raise ValueError("No held training histories")
    return max(0.0, min(1.0, mean(gains) / (mean(map(abs, gains)) + 0.005)))


def endpoint(raw, signal):
    if abs(float(signal - raw)) <= 1e-10:
        return (signal / 0.05).sigmoid()
    return (signal > raw).to(raw.dtype)


def inner_changes(donors, targets):
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float64, device="cuda")

    rows = [
        {
            "history": w["history"],
            "node": w["window"],
            "background": 0,
            "raw": mean(q["scores"][2] - q["scores"][1] for q in w["points"]),
            "credit": mean(q["edges"][2] for q in w["points"]),
        }
        for w in donors
    ]
    ids = {r["history"] for r in rows}
    if ids & {r["history"] for r in targets} or len(ids) != 4:
        raise ValueError("Incorrect inner donor split")
    w = weights(rows, ids, "raw", tensor)
    geom = tensor([r["features"][1] for r in donors])
    center = w @ geom
    sd = (w @ (geom - center).square()).sqrt().clamp_min(0.02)

    def design(data):
        v = tensor([r["features"] for r in data])
        return features(v[:, 1], v[:, 0], torch.zeros_like(v[:, 0]), center, sd, False)

    coef, _ = fit_contrasts(
        rows, design(donors), tensor([r["raw"] for r in rows]), w, ids
    )
    z = {k: design(targets) @ v for k, v in coef.items()}
    result = []
    for i, r in enumerate(targets):
        contrast = mean(q["scores"][2] - q["scores"][1] for q in r["points"])
        gains = {}
        for k in ("credit", "rotated"):
            inc = 0.8 * (endpoint(z["raw"][i], z[k][i]) - (z[k][i] / 0.05).sigmoid())
            gains[k] = float(inc) * contrast
        result.append({"window": r["window"], "gains": gains})
    return result


def allocate(native, spec):
    old = direct_distributions(native, spec)
    temporal = old["raw"][0]
    budget = 1 - temporal
    raw = native.new_tensor(spec["direct"]["raw"])
    prior = native[2] / (native[1] + native[2])

    def mix(q):
        return torch.stack((temporal, budget * (1 - q), budget * q))

    result = dict(old, soft=old["credit"])
    for k in ("credit", "rotated"):
        z = native.new_tensor(spec["direct"][k])
        full = 0.2 * prior + 0.8 * endpoint(raw, z)
        soft = old[k][2] / budget
        result[k] = mix(soft + spec["scale"][k] * (full - soft))
        if k == "credit":
            result["full_step"] = mix(full)
    return result


def freeze(base):
    root = base / "late-pairs/semantic-writer/three-writer"
    parent = root / "direct-allocation/time/forecast.json"
    out = root / "direct-step"
    out.mkdir(exist_ok=True)
    graph = root / "skill-graph/complete/models.json"
    inputs = {
        "plan": file_hash(PLAN),
        "parent": file_hash(parent),
        "graph": file_hash(graph),
    }
    f = out / "forecast.json"
    if f.exists():
        if read(f)["inputs"] != inputs or read(f)["models"] != file_hash(
            out / "models.json"
        ):
            raise ValueError("Changed step calibration")
        return f
    rows, _ = training_rows(base, True)
    if rows != read(graph)["donors"]:
        raise ValueError("Different direct and three-action calibration donors")
    hist = sorted({r["history"] for r in rows})
    models = {}
    for h in hist:
        scores = defaultdict(list)
        folds = {}
        for j in hist:
            if j == h:
                continue
            pool = [r for r in rows if r["history"] not in (h, j)]
            with limited():
                visits = inner_changes(pool, [r for r in rows if r["history"] == j])
            folds[j] = {"train": sorted({r["history"] for r in pool}), "visits": visits}
            for key in ("credit", "rotated"):
                scores[key].append(mean(v["gains"][key] for v in visits))
        models[h] = {"folds": folds, "scale": {k: scale(v) for k, v in scores.items()}}
    forecasts = {}
    for k, v in read(parent)["windows"].items():
        forecasts[k] = {
            **v,
            "scores": {**v["scores"], "scale": models[k.rsplit("_s", 1)[0]]["scale"]},
        }
    register(out / "models.json", {"inputs": inputs, "histories": models})
    register(
        f,
        {
            "inputs": inputs,
            "models": file_hash(out / "models.json"),
            "windows": forecasts,
        },
    )
    return f


def limits(base, path):
    root = path.parent.parent
    panel = read(root.parent / "panel.json")["windows"]
    split = read(base / "late-pairs/selection-labels/split.json")["windows"][
        "evaluation"
    ]
    packet = read(path)
    original = read(root / "direct-allocation/time/forecast.json")
    rows = []
    sources = {}
    for w in panel:
        name = f"{w['dialogue']}_s{w['start']}"
        if name not in split:
            continue
        if split[name] != w["questions"]:
            raise ValueError("Changed later evaluation questions")
        f = root / (name + ".json")
        physical = read(f) if f.exists() else {}
        if physical:
            sources[name] = file_hash(f)
        spec = packet["windows"][name]["scores"]
        n = len(w["questions"])
        with limited():
            if physical.get("matched"):
                if [q["index"] for q in physical["rows"]] != w["questions"]:
                    raise ValueError("Changed matched order")
                native = torch.tensor(
                    physical["probabilities"]["native"],
                    dtype=torch.float64,
                    device="cuda",
                )
                ps = allocate(native, spec)
                old = direct_distributions(native, original["windows"][name]["scores"])
                for arm, key in (("raw", "raw"), ("soft", "credit")):
                    if not torch.equal(ps[arm], old[key]):
                        raise ValueError("Parent allocation changed")
                y = native.new_tensor([q["f1"] for q in physical["rows"]]).mean(0)
                bounds = {
                    key: [float((ps["credit"] - ps[key]) @ y)] * 2
                    for key in ("raw", "soft")
                }
                shift = float(ps["credit"][2] - ps["raw"][2])
            else:
                # Both policies have the same native smoothing. Maximize the unknown
                # non-temporal budget over all native policies: native temporal = 0.
                raw = torch.tensor(
                    spec["two_edge"]["raw"], device="cuda", dtype=torch.float64
                )
                temporal_floor = (
                    0.8 * (torch.cat((raw.new_zeros(1), raw)) / 0.05).softmax(0)[0]
                )
                r, z = (
                    raw.new_tensor(spec["direct"][key]) for key in ("raw", "credit")
                )
                soft = (z / 0.05).sigmoid()
                new = soft + spec["scale"]["credit"] * (endpoint(r, z) - soft)
                bounds = {}
                for key, reference in (("raw", (r / 0.05).sigmoid()), ("soft", soft)):
                    cap = float((1 - temporal_floor) * 0.8 * (new - reference).abs())
                    bounds[key] = [-cap, cap]
                shift = None
        rows.append(
            {"window": name, "n": n, "gain_bounds": bounds, "relation_shift": shift}
        )
    n = sum(v["n"] for v in rows)
    register(
        path.parent / f"evaluation-bounds-{len(sources):02d}.json",
        {
            "forecast": file_hash(path),
            "sources": sources,
            "questions": n,
            "matched": sum(r["n"] for r in rows if r["relation_shift"] is not None),
            "windows": rows,
            "bounds": {
                key: [
                    sum(v["n"] * v["gain_bounds"][key][j] for v in rows) / n
                    for j in (0, 1)
                ]
                for key in ("raw", "soft")
            },
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    a = cli.parse_args()
    gpu()
    p = freeze(a.base)
    score(
        a.base, p, allocate, direct=True, replay_choices=True, extra_controls=("soft",)
    )
    limits(a.base, p)
