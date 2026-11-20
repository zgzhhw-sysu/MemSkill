"""Calibrate verified span credit on out-of-dialogue errors of the raw treatment ranker."""

import argparse
import math
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.matched_span_policy import (
    concordance,
    fit,
    paired_edges,
    predict,
    score_trial,
)
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "matched_span_calibration.json"
ARMS = (
    "credit",
    "raw",
    "raw_rank",
    "rotated",
    "omit",
    "twice",
    "direct",
    "parent",
    "native",
    "terminal",
)


def calibrate(rows, device):
    histories = sorted({r["history"] for r in rows})
    xy = torch.tensor(
        [[r["actual"] - r["raw_rank"], r["direct"], r["placebo"]] for r in rows],
        device=device,
        dtype=torch.float64,
    )
    w = xy.new_tensor(
        [
            1 / (len(histories) * sum(q["history"] == r["history"] for q in rows))
            for r in rows
        ]
    )
    cov = w @ (xy[:, 1:] * xy[:, :1])
    energy = w @ xy[:, 1:].square()
    return (cov / (energy + 0.025**2)).clamp(-1, 1)


def inner_fit(original, units, x, held, j):
    donors = [r for r in original if r["history"] != held and r["stage"] == j]
    pool = [r for r in units if r["history"] != held and r["stage"] == j]
    rows = []
    for h in sorted({r["history"] for r in donors}):
        rm = fit(donors, x, h, j, ("raw",))
        cm = fit(pool, x, h, j, ("credit", "rotated"))
        if (
            held in rm["histories"] + cm["histories"]
            or h in rm["histories"] + cm["histories"]
        ):
            raise ValueError("A validation history entered the inner fit")
        for e in paired_edges([r for r in donors if r["history"] == h]):
            a, b = x[e["a"]], x[e["b"]]
            raw = (predict(rm, b)[0] - predict(rm, a)[0]).item()
            c, sham = (predict(cm, b)[0] - predict(cm, a)[0]).tolist()
            rows.append(
                {
                    "history": h,
                    "a": e["a"],
                    "b": e["b"],
                    "background": e["background"],
                    "actual": e["raw"],
                    "raw_rank": raw,
                    "direct": c,
                    "placebo": sham,
                }
            )
    return calibrate(rows, next(iter(x.values())).device), rows


def unpack(model, device):
    return {
        k: torch.tensor(model[k], dtype=torch.float64, device=device)
        for k in ("coef", "center", "scale", "width")
    }


def corrections(rmodel, cmodel, v, gamma):
    r = predict(rmodel, v)[0][0]
    c, s = predict(cmodel, v)[0]
    combined = torch.stack([r + gamma[0] * c, r, r + gamma[1] * s, r - c, r + c])
    keys = ("credit", "raw_rank", "rotated", "omit", "twice")
    return dict(
        zip(keys, (combined / rmodel["width"][0]).clamp(-2, 2).tolist(), strict=True)
    ), dict(zip(keys, combined.tolist(), strict=True))


def inherit(shifts, prev):
    if not math.isclose(shifts["raw_rank"], prev["raw_rank"], abs_tol=1e-12):
        raise ValueError("Original raw ranker did not reproduce")
    return {**shifts, "direct": prev["credit"], "parent": prev["parent"]}


def freeze(base, stage):
    root = base / "source-compression/matched-span-rank"
    parent = root / stage
    source = parent / "forecast.json"
    long = parent / "long-prefix/forecast.json"
    trial = root / "calibrated" / stage
    local = trial / "forecast.json"
    prefix = trial / "long-prefix/forecast.json"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    a, b = read(source), read(long)
    packet = read(parent / "models.json")
    vals = read(root / "features.json")
    if (
        packet["inputs"] != a["inputs"]
        or b["inputs"] != {**a["inputs"], "local": file_hash(source)}
        or a["inputs"]["features"] != file_hash(root / "features.json")
    ):
        raise ValueError("Different parent model inputs")
    values = vals["features"]
    arms = ARMS
    inputs = {
        "plan": file_hash(PLAN),
        "model": file_hash(parent / "models.json"),
        "local": file_hash(source),
        "prefix": file_hash(long),
        "manifest": b["inputs"]["manifest"],
    }
    if stage == "expanded":
        ep = root / "earlier-sources/features.json"
        if file_hash(ep) != a["inputs"]["earlier_features"]:
            raise ValueError("Changed earlier features")
        values = values | read(ep)["features"]
        old_local = trial.parent / "original/forecast.json"
        old_prefix = trial.parent / "original/long-prefix/forecast.json"
        old = read(old_local)["by_stage"]
        old_ext = read(old_prefix)["offsets"]
        inputs.update(
            original=file_hash(old_local), original_prefix=file_hash(old_prefix)
        )
        arms = (*arms, "original")
    if (trial / "edge-audit.json").exists():
        if read(local)["inputs"] != inputs or read(prefix)["inputs"] != {
            **inputs,
            "local": file_hash(local),
        }:
            raise ValueError("Changed calibration freeze")
        return local, prefix, arms
    x = {
        n: torch.tensor(v, device="cuda", dtype=torch.float64)
        for n, v in values.items()
    }
    units = packet["units"]
    original = [r for r in units if "raw" in r]
    if len(original) != 60:
        raise ValueError("Incomplete raw comparisons")
    models = {}
    off = {"0": {}, "1": {}}
    ext = {}
    forecasts = {"0": {}, "1": {}}
    for h, stages in sorted(packet["folds"].items()):
        models[h] = {}
        for key, trained in stages.items():
            with limited():
                j = int(key)
                gamma, rows = inner_fit(original, units, x, h, j)
                models[h][key] = {"gamma": gamma.tolist(), "inner_edges": rows}
                rm, cm = (unpack(trained[k], "cuda") for k in ("raw", "credit"))
                for n, v in x.items():
                    if n.rsplit("_s", 1)[0] != h:
                        continue
                    shifts, score = corrections(rm, cm, v, gamma)
                    forecasts[key][n] = score
                    if n in a["sources"]:
                        item = inherit(shifts, a["by_stage"][key][n])
                        if stage == "expanded":
                            item["original"] = old[key][n]["credit"]
                        off[key][n] = item
                    if j == 1 and n in b["sources"]:
                        item = inherit(shifts, b["offsets"][n])
                        if stage == "expanded":
                            item["original"] = old_ext[n]["credit"]
                        ext[n] = item
    if any(set(v) != set(a["sources"]) for v in off.values()) or set(ext) != set(
        b["sources"]
    ):
        raise ValueError("Missing target")
    register(trial / "calibration.json", {"inputs": inputs, "models": models})
    register(
        local,
        {"inputs": inputs, "arms": arms, "sources": a["sources"], "by_stage": off},
    )
    register(
        prefix,
        {
            "inputs": {**inputs, "local": file_hash(local)},
            "arms": arms,
            "sources": b["sources"],
            "offsets": ext,
        },
    )
    register(
        trial / "edge-audit.json",
        {
            "forecast": file_hash(local),
            "original": concordance(paired_edges(original), forecasts),
        },
    )
    return local, prefix, arms


def run(base, study, corpus, stage):
    local, prefix, arms = freeze(base, stage)
    controls = ("raw", "raw_rank", "rotated") + (
        ("original",) if "original" in arms else ()
    )
    score_trial(base, study, corpus, stage, local, prefix, controls)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("stage", choices=("original", "expanded"))
    a = p.parse_args()
    gpu()
    run(a.base, a.study, a.corpus, a.stage)
