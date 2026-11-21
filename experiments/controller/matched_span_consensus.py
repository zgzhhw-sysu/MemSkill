"""Vote on a span's skill preference after deleting individual training histories."""

import argparse
import math
from pathlib import Path

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.matched_span_calibration import (
    corrections,
    inner_fit,
    unpack,
)
from experiments.controller.matched_span_policy import fit, fit_receipt, score_trial
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "matched_span_consensus.json"
ARMS = (
    "credit",
    "raw",
    "raw_rank",
    "rotated",
    "calibrated",
    "single",
    "unanimous",
    "smooth",
    "parent",
    "native",
    "terminal",
)


def deleted_models(original, units, x, held, stage):
    raw = [u for u in original if u["history"] != held]
    credit = [u for u in units if u["history"] != held]
    models = []
    for h in sorted({u["history"] for u in raw}):
        rm = fit(raw, x, h, stage, ("raw",))
        cm = fit(credit, x, h, stage, ("credit", "rotated"))
        gamma, rows = inner_fit(raw, credit, x, h, stage)
        models.append((h, rm, cm, gamma, rows))
    return models


def hard(scores):
    return torch.where(scores.abs() >= 0.01, scores.sign(), 0.0)


def voting(scores, whole):
    if scores.ndim != 2 or scores.shape[1] != 3 or len(scores) == 0:
        raise ValueError("Missing donor-deletion scores")
    ballot = hard(scores)
    used = 2 * ballot.mean(0)
    primary = used[0]
    return {
        "credit": float(primary),
        "raw_rank": float(used[1]),
        "rotated": float(used[2]),
        "single": float(2 * hard(whole)),
        "unanimous": float(torch.where(primary.abs() == 2, primary, 0)),
        "smooth": float(2 * torch.tanh(scores[:, 0] / 0.025).mean()),
    }


def physical(rmodel, cmodel, v, gamma):
    raw = ((v - rmodel["center"]) / rmodel["scale"]) @ rmodel["coef"]
    cs = ((v - cmodel["center"]) / cmodel["scale"]) @ cmodel["coef"]
    return torch.stack((raw[0] + gamma[0] * cs[0], raw[0], raw[0] + gamma[1] * cs[1]))


def freeze(base, stage):
    root = base / "source-compression/matched-span-rank"
    original = root / stage
    parent = root / "calibrated" / stage
    trial = root / "consensus" / stage
    local = trial / "forecast.json"
    prefix = trial / "long-prefix/forecast.json"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    a = read(parent / "forecast.json")
    b = read(parent / "long-prefix/forecast.json")
    fitted = read(original / "models.json")
    calibrated = read(parent / "calibration.json")
    inputs = {
        "plan": file_hash(PLAN),
        "local_parent": file_hash(parent / "forecast.json"),
        "prefix_parent": file_hash(parent / "long-prefix/forecast.json"),
        "calibration": file_hash(parent / "calibration.json"),
        "models": file_hash(original / "models.json"),
        "manifest": b["inputs"]["manifest"],
    }
    if (
        a["inputs"] != calibrated["inputs"]
        or a["inputs"]["model"] != inputs["models"]
        or b["inputs"]["local"] != inputs["local_parent"]
    ):
        raise ValueError("Different calibrated parent")
    fp = root / "features.json"
    if fitted["inputs"]["features"] != file_hash(fp):
        raise ValueError("Changed current sources")
    vals = read(fp)["features"]
    arms = ARMS
    if stage == "expanded":
        extra = root / "earlier-sources/features.json"
        if fitted["inputs"]["earlier_features"] != file_hash(extra):
            raise ValueError("Changed earlier sources")
        vals = vals | read(extra)["features"]
        earlier = trial.parent / "original/forecast.json"
        earlier_ext = trial.parent / "original/long-prefix/forecast.json"
        old = read(earlier)["by_stage"]
        old_ext = read(earlier_ext)["offsets"]
        inputs.update(
            original=file_hash(earlier), original_prefix=file_hash(earlier_ext)
        )
        arms = (*arms, "original")
    if local.is_file() and prefix.is_file() and (trial / "votes.json").is_file():
        if read(local)["inputs"] != inputs or read(prefix)["inputs"] != {
            **inputs,
            "local": file_hash(local),
        }:
            raise ValueError("Changed consensus freeze")
        return local, prefix, arms
    x = {
        n: torch.tensor(v, device="cuda", dtype=torch.float64) for n, v in vals.items()
    }
    units = fitted["units"]
    raw = [u for u in units if "raw" in u]
    if len(raw) != 60:
        raise ValueError("Different raw observation population")
    models = {}
    votes = {}
    off = {"0": {}, "1": {}}
    ext = {}
    for h, stages in fitted["folds"].items():
        models[h] = {}
        for key, outer in stages.items():
            with limited():
                j = int(key)
                deleted = deleted_models(raw, units, x, h, j)
                models[h][key] = {
                    d: {
                        "raw": fit_receipt(rm),
                        "credit": fit_receipt(cm),
                        "gamma": g.tolist(),
                        "inner_edges": rows,
                    }
                    for d, rm, cm, g, rows in deleted
                }
                rm, cm = (unpack(outer[k], "cuda") for k in ("raw", "credit"))
                gamma = rm["coef"].new_tensor(calibrated["models"][h][key]["gamma"])
                for node, v in x.items():
                    if node.rsplit("_s", 1)[0] != h or (
                        node not in a["sources"]
                        and (j != 1 or node not in b["sources"])
                    ):
                        continue
                    whole, values = corrections(rm, cm, v, gamma)
                    scores = torch.stack(
                        [physical(m, c, v, g) for _, m, c, g, _ in deleted]
                    )
                    shifts = voting(scores, v.new_tensor(values["credit"]))
                    votes.setdefault(node, {})[key] = {
                        "deleted": [d for d, *_ in deleted],
                        "f1_scores": scores.tolist(),
                        "whole": values["credit"],
                    }
                    for dest, prev, earlier_shift in (
                        (
                            off[key],
                            a["by_stage"][key].get(node),
                            old[key].get(node) if stage == "expanded" else None,
                        ),
                        (
                            ext,
                            b["offsets"].get(node) if j == 1 else None,
                            old_ext.get(node)
                            if stage == "expanded" and j == 1
                            else None,
                        ),
                    ):
                        if prev is None:
                            continue
                        if not math.isclose(
                            whole["credit"], prev["credit"], abs_tol=1e-12
                        ):
                            raise ValueError("Calibrated parent did not reproduce")
                        item = {
                            **shifts,
                            "parent": prev["parent"],
                            "calibrated": prev["credit"],
                        }
                        if stage == "expanded":
                            item["original"] = earlier_shift["credit"]
                        dest[node] = item
    if any(set(v) != set(a["sources"]) for v in off.values()) or set(ext) != set(
        b["sources"]
    ):
        raise ValueError("Missing consensus target")
    register(trial / "models.json", {"inputs": inputs, "folds": models})
    register(
        local,
        {"inputs": inputs, "sources": a["sources"], "by_stage": off, "arms": arms},
    )
    register(
        prefix,
        {
            "inputs": {**inputs, "local": file_hash(local)},
            "sources": b["sources"],
            "offsets": ext,
            "arms": arms,
        },
    )
    register(
        trial / "votes.json",
        {
            "forecast": file_hash(local),
            "prefix": file_hash(prefix),
            "columns": ["credit", "raw_rank", "rotated"],
            "nodes": votes,
        },
    )
    return local, prefix, arms


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("stage", choices=("original", "expanded"))
    a = p.parse_args()
    gpu()
    local, prefix, arms = freeze(a.base, a.stage)
    controls = ("raw", "raw_rank", "rotated", "calibrated") + (
        ("original",) if "original" in arms else ()
    )
    score_trial(a.base, a.study, a.corpus, a.stage, local, prefix, controls)
