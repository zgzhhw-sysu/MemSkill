"""Use a directly measured non-temporal skill comparison at a fixed temporal probability."""

import argparse
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import weights
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_skill_margin import (
    distributions,
    fit_contrasts,
    score,
)
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "direct_skill_allocation.json"


def features(x, affinity, background, center, sd, both):
    columns = [torch.ones_like(x), affinity / 2, ((x - center) / sd).clamp(-3, 3)]
    if both:
        columns.append(2 * background - 1)
    return torch.stack(columns, dim=-1)


def donor_rows(src, both, *, raw_only=False):
    rows = []
    inputs = {}
    archive = src / "activity-relation-repairs"
    offsets = {
        (j["dialogue"], j["session"]): j["offset"]
        for j in read(src / "semantic-writer/source-forecast.json")["rows"]
        if "offset" in j
    }
    for bit in range(2 if both else 1):
        folder = archive / str(bit)
        phase = "prepared" if raw_only else "outcomes"
        marker = folder / (phase + "-complete.json")
        reg = folder / "registration.json"
        hashes = read(marker)
        if len(hashes) != 12:
            raise ValueError("Incomplete direct comparison")
        inputs[str(bit)] = {
            phase: file_hash(marker),
            "registration": file_hash(reg),
        }
        for name, h in hashes.items():
            f = folder / phase / (name + ".json")
            d = read(f)
            prep = folder / "prepared" / f.name
            upstream = (
                src / "semantic-writer" / name / "pair.json" if raw_only else prep
            )
            if (
                file_hash(f) != h
                or d["inputs"]["registration"] != file_hash(reg)
                or d["inputs"]["parent"] != file_hash(upstream)
            ):
                raise ValueError("Changed two-action repair")
            if not d["matched"]:
                continue
            a = d["raw_pairs"]
            n = d["questions"]
            if len(a) != n or len({q["original_index"] for q in a}) != n:
                raise ValueError("Incomplete direct contrast")
            delta = {q["original_index"]: q["relation"] - q["activity"] for q in a}
            jobs = [] if raw_only else d["jobs"]
            if len({j["original_index"] for j in jobs}) != len(jobs) or any(
                j["raw"] != delta[j["original_index"]]
                or j["credit"] != (j["raw"] if j["status"] == "confirmed" else 0.0)
                for j in jobs
            ):
                raise ValueError("Different signed credit")
            sid, num = name.rsplit("_s", 1)
            session = int(num) + 1
            rows.append(
                {
                    "history": sid,
                    "node": f"{sid}_s{session}",
                    "background": bit,
                    "raw": mean(delta.values()),
                    "credit": sum(j["credit"] for j in jobs) / n,
                    "offset": offsets[sid, session],
                }
            )
    return rows, inputs


def direct_distributions(native, packet):
    first = distributions(native, packet["two_edge"])
    temporal = first["raw"][0]
    budget = 1 - temporal
    prior = native[2] / (native[1] + native[2])

    def allocate(q):
        return torch.stack((temporal, budget * (1 - q), budget * q))

    def relationship(v):
        return 0.20 * prior + 0.80 * (native.new_tensor(v) / 0.05).sigmoid()

    result = {k: allocate(relationship(v)) for k, v in packet["direct"].items()}
    result.update(
        native=native,
        parent_raw=first["raw"],
        parent_credit=first["credit"],
        uniform=allocate(native.new_tensor(0.5)),
    )
    return result


def register_trial(base):
    parent = base / "late-pairs/semantic-writer/three-writer"
    dest = parent / "direct-allocation"
    dest.mkdir(exist_ok=True)
    path = dest / "registration.json"
    archive = base / "source-compression/activity-relation-repairs"
    identity = {
        "plan": file_hash(PLAN),
        "baseline": file_hash(parent / "margin/forecast.json"),
    }
    if not path.exists():
        register(
            path,
            {
                **identity,
                "earlier_results": {
                    str(f.relative_to(archive)): file_hash(f)
                    for f in archive.glob("*/outcomes/conv-*.json")
                },
            },
        )
    locked = read(path)
    if any(locked[k] != v for k, v in identity.items()) or any(
        file_hash(archive / f) != h for f, h in locked["earlier_results"].items()
    ):
        raise ValueError("Changed direct-edge registration")
    return path


def freeze(base, mode):
    both = mode == "both"
    src = base / "source-compression"
    semantic = base / "late-pairs/semantic-writer"
    parent = semantic / "three-writer"
    root = parent / "direct-allocation"
    out = root / mode
    out.mkdir(parents=True, exist_ok=True)
    f = out / "forecast.json"
    prior = parent / "margin/forecast.json"
    anchor = read(prior)
    rows, inputs = donor_rows(src, both, raw_only=mode == "raw-reference")
    geometry = src / "matched-span-rank/features.json"
    inputs.update(
        registration=file_hash(register_trial(base)),
        plan=file_hash(PLAN),
        geometry=file_hash(geometry),
        baseline=file_hash(prior),
        affinity=file_hash(semantic / "forecast.json"),
        early_affinity=file_hash(src / "semantic-writer/source-forecast.json"),
        panel=file_hash(semantic / "panel.json"),
    )
    reference = {}
    if mode == "time":
        ref = freeze(base, "raw-reference")
        inputs["raw_reference"] = file_hash(ref)
        reference = read(ref)["windows"]
    previous = {}
    if both:
        other = root / "time/forecast.json"
        inputs["time_only"] = file_hash(other)
        previous = read(other)["windows"]
    if f.exists():
        if read(f)["inputs"] != inputs:
            raise ValueError("Changed frozen direct allocation")
        return f
    vals = read(geometry)["features"]
    panel = read(semantic / "panel.json")["windows"]
    late = {
        (r["dialogue"], r["session"]): r
        for r in read(semantic / "forecast.json")["targets"]
    }
    hist = sorted({w["dialogue"] for w in panel})

    def tensor(v):
        return torch.as_tensor(v, dtype=torch.float64, device="cuda")

    models = {}
    preds = {}
    for h in hist:
        a = [r for r in rows if r["history"] != h]
        ids = {r["history"] for r in a}
        if not ids:
            raise ValueError("Empty out-of-history fit")
        with limited():
            w = weights(a, ids, "raw", tensor)
            g = tensor([vals[r["node"]][5] for r in a])
            center = w @ g
            sd = (w @ (g - center).square()).sqrt().clamp_min(0.02)
            design = features(
                g,
                tensor([r["offset"] for r in a]),
                tensor([r["background"] for r in a]),
                center,
                sd,
                both,
            )
            coef, counts = fit_contrasts(
                a, design, tensor([r["raw"] for r in a]), w, ids
            )
            models[h] = {
                "center": float(center),
                "sd": float(sd),
                "coef": {k: v.tolist() for k, v in coef.items()},
                "support": counts,
                "histories": sorted(ids),
            }
            for spec in panel:
                if spec["dialogue"] != h:
                    continue
                n = spec["start"] + 1
                k = f"{h}_s{n - 1}"
                bg = spec["background"]
                parent_row = anchor["windows"][k]
                aff = late[h, n]
                if (
                    parent_row["source"] != aff["source"]
                    or parent_row["background"] != bg
                ):
                    raise ValueError("Different pre-action source")
                x = features(
                    tensor(vals[f"{h}_s{n}"][5]),
                    tensor(aff["offset"]),
                    tensor(bg),
                    center,
                    sd,
                    both,
                )
                direct = {k: float(x @ v) for k, v in coef.items()}
                if (
                    reference
                    and abs(direct["raw"] - reference[k]["scores"]["direct"]["raw"])
                    > 1e-12
                ):
                    raise ValueError("Raw edge forecast changed after unmasking credit")
                if both:
                    direct["time_only"] = previous[k]["scores"]["direct"]["credit"]
                preds[k] = {
                    **parent_row,
                    "scores": {"two_edge": parent_row["scores"], "direct": direct},
                }
    register(out / "models.json", {"inputs": inputs, "folds": models})
    register(
        f,
        {"inputs": inputs, "models": file_hash(out / "models.json"), "windows": preds},
    )
    return f


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("mode", choices=("time", "both", "raw-reference"))
    p.add_argument("--register-only", action="store_true")
    a = p.parse_args()
    if a.register_only:
        register_trial(a.base)
        raise SystemExit(0)
    gpu()
    score(a.base, freeze(a.base, a.mode), direct_distributions, direct=True)
