"""Condition a verified relation-versus-activity contrast on native pre-action odds."""

import argparse
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.direct_skill_allocation import (
    direct_distributions,
    features,
)
from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import weights
from experiments.controller.skill_graph_data import training_rows
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_skill_margin import fit_contrasts, score
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "checkpoint_odds_credit.json"


def design(geometry, offset, odds, centers, scales):
    source = features(
        geometry, offset, torch.zeros_like(offset), centers[0], scales[0], False
    )
    state = ((odds - centers[1]) / scales[1]).clamp(-3, 3)
    return torch.cat([source, state.unsqueeze(-1)], dim=-1)


def fit(donors, held, device):
    rows = [
        {
            "history": w["history"],
            "node": w["window"],
            "background": w.get("background", 0),
            "source": w["features"],
            "native_pair": w["native_pair"],
            "raw": mean(q["scores"][2] - q["scores"][1] for q in w["points"]),
            "credit": mean(q["edges"][2] for q in w["points"]),
        }
        for w in donors
        if w["history"] != held
    ]

    def t(x):
        return torch.as_tensor(x, device=device, dtype=torch.float64)

    hist = {w["history"] for w in rows}
    w = weights(rows, hist, "raw", t)
    odds = t([r["native_pair"] for r in rows]).logit()
    off, g = t([r["source"] for r in rows]).T
    z = odds[:, 1] - odds[:, 0]
    v = torch.stack([g, z], dim=1)
    centers = w @ v
    scales = (w @ (v - centers).square()).sqrt().maximum(t([0.02, 0.25]))
    x = design(g, off, z, centers, scales)
    if not x.isfinite().all():
        raise ValueError("Invalid native donor odds")
    coef, support = fit_contrasts(rows, x, t([r["raw"] for r in rows]), w, hist)
    return {
        "centers": centers.tolist(),
        "scales": scales.tolist(),
        "coef": {k: b.tolist() for k, b in coef.items()},
        "support": support,
        "histories": sorted(hist),
        "windows": [r["node"] for r in rows],
    }


def distribute(native, temporal, source, model):
    t = native.new_tensor
    z = native[2].log() - native[1].log()
    x = design(t(source[1]), t(source[0]), z, t(model["centers"]), t(model["scales"]))
    prior = native[2] / (native[1] + native[2])
    result = {}
    for k, v in model["coef"].items():
        signal = x @ t(v)
        q = 0.2 * prior + 0.8 * (signal / 0.05).sigmoid()
        result[k] = torch.stack(
            (temporal, (1 - temporal) * (1 - q), (1 - temporal) * q)
        )
        if k == "credit":
            q0 = 0.2 * prior + 0.8 * ((x[:-1] @ t(v[:-1])) / 0.05).sigmoid()
            result["masked_state"] = torch.stack(
                (temporal, (1 - temporal) * (1 - q0), (1 - temporal) * q0)
            )
    return result


def allocate(native, scores):
    parent = direct_distributions(native, scores)
    return {
        **distribute(
            native, parent["raw"][0], scores["source_features"], scores["state_model"]
        ),
        "native": native,
        "parent_credit": parent["credit"],
        "parent_raw": parent["raw"],
        "uniform": parent["uniform"],
    }


def freeze(base):
    src = base / "source-compression"
    parent = base / "late-pairs/semantic-writer/three-writer"
    direct = parent / "direct-allocation/time/forecast.json"
    graph = parent / "skill-graph/complete/models.json"
    early = src / "earliest-three-skills/forecast.json"
    gp = read(parent / "skill-graph/complete/forecast.json")
    direct_rows = read(direct)["windows"]
    out = parent / "checkpoint-odds"
    out.mkdir(exist_ok=True)
    path = out / "forecast.json"
    target = early.parent / "checkpoint-odds"
    target.mkdir(exist_ok=True)
    identity = {
        "plan": file_hash(PLAN),
        "direct": file_hash(direct),
        "graph": file_hash(graph),
        "early": file_hash(early),
        "source": file_hash(parent / "skill-graph/complete/forecast.json"),
    }
    if gp["models"] != identity["graph"]:
        raise ValueError("Unbound complete graph")
    if path.exists() and (target / "forecast.json").exists():
        if (
            read(path)["inputs"] != identity
            or read(path)["models"] != file_hash(out / "models.json")
            or read(target / "forecast.json")["parent"] != file_hash(path)
        ):
            raise ValueError("Changed state-conditional policy")
        return path
    donors, _ = training_rows(base, True)
    if read(graph)["donors"] != donors:
        raise ValueError("Changed original three-skill donors")
    reg = parent / "skill-graph/registration.json"
    registered = read(reg)["inputs"]
    geometry = src / "matched-span-rank/features.json"
    affinity = parent.parent / "forecast.json"
    if (
        gp["inputs"]["registration"] != file_hash(reg)
        or registered["geometry"] != file_hash(geometry)
        or registered["affinity"] != file_hash(affinity)
    ):
        raise ValueError("Changed pre-action source features")
    geo = read(geometry)["features"]
    source = {(v["dialogue"], v["session"]): v for v in read(affinity)["targets"]}
    original = read(early)
    models = {}
    predictions = {}
    original_states = {}
    for h in sorted({w["history"] for w in original["windows"].values()}):
        with limited():
            model = fit(donors, h, "cuda")
            models[h] = model
            for k, w in direct_rows.items():
                if k.rsplit("_s", 1)[0] != h:
                    continue
                ref = gp["windows"][k]
                if ref["source"] != w["source"] or ref["background"] != w["background"]:
                    raise ValueError("Different later decision")
                n = int(k.rsplit("_s", 1)[1]) + 1
                row = source[h, n]
                if row["source"] != w["source"]:
                    raise ValueError("Wrong target affinity")
                predictions[k] = {
                    **w,
                    "scores": {
                        **w["scores"],
                        "state_model": model,
                        "source_features": [row["offset"], geo[f"{h}_s{n}"][5]],
                    },
                }
            for k, w in original["windows"].items():
                if w["history"] != h:
                    continue
                p = w["probabilities"]
                native = torch.tensor(p["native"], device="cuda", dtype=torch.float64)
                state = distribute(
                    native, native.new_tensor(p["direct_raw"][0]), w["features"], model
                )
                original_states[k] = {
                    **w,
                    "probabilities": {
                        **p,
                        **{"state_" + a: v.tolist() for a, v in state.items()},
                    },
                }
    if len(predictions) != 25 or len(original_states) != 12:
        raise ValueError("Incomplete frozen cohorts")
    register(out / "models.json", {"inputs": identity, "folds": models})
    register(
        path,
        {
            "inputs": identity,
            "models": file_hash(out / "models.json"),
            "windows": predictions,
        },
    )
    exposure = target / "exposure.json"
    if not exposure.exists():
        register(
            exposure,
            {
                k: {
                    p.stem: file_hash(p) for p in (early.parent / k).glob("conv-*.json")
                }
                for k in ("writers", "readers", "pairs", "judged")
            },
        )
    register(
        target / "forecast.json",
        {
            "parent": file_hash(path),
            "inputs": original["inputs"],
            "exposure": file_hash(exposure),
            "windows": original_states,
        },
    )
    return path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    a = cli.parse_args()
    gpu()
    score(
        a.base,
        freeze(a.base),
        allocate,
        direct=True,
        extra_controls=("masked_state",),
        replay_choices=True,
    )
