"""Issue only span-utility corrections supported after removing any one donor dialogue."""

import argparse
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.shared_outlier_policy import (
    allocate,
    corrected_utilities,
    corrections,
    fit_correction,
    skill_report,
)
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_action_bounds import report as population_bounds
from experiments.controller.three_skill_margin import score
from experiments.controller.three_skill_shared_credit import SKILLS
from experiments.controller.three_skill_temporal_gate import simplex, utilities
from experiments.controller.witness_passage_policy import limited
from experiments.controller.writer_format_policy import scores_at

PLAN = Path(__file__).with_name("protocols") / "replicated_outlier_credit.json"


def registration(base):
    parent = base / "late-pairs/semantic-writer/three-writer"
    early = base / "source-compression/earliest-three-skills/shared-weak-credit"
    dest = early / "replication/registration.json"
    dest.parent.mkdir(exist_ok=True)
    identity = {
        "plan": file_hash(PLAN),
        "early": file_hash(early / "registration.json"),
        "previous": file_hash(parent / "outlier-components/forecast.json"),
    }
    if dest.exists():
        saved = read(dest)
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError("Changed cross-dialogue protocol")
    else:
        register(
            dest,
            {
                **identity,
                "exposure": {
                    str(f.relative_to(base)): file_hash(f)
                    for d in ("outlier-initial", "outlier-final", "outlier-components")
                    for f in (parent / d).glob("through-*.json")
                },
                "six_at_registration": {
                    f.name: file_hash(f)
                    for f in (early / "outcomes").glob("conv-*.json")
                },
            },
        )
    return dest


def support(rows):
    return {
        s: sorted({r["history"] for r in rows if any(r["credit"][j])})
        for j, s in enumerate(SKILLS)
    }


def ensemble(rows, full, held):
    rows = [r for r in rows if r["history"] != held]
    history_set = sorted({r["history"] for r in rows})
    if full["windows"] != [r["node"] for r in rows] or held in full["histories"]:
        raise ValueError("Unexpected target-dialogue exposure")
    sources = support(rows)
    folds = {}
    if any(len(hs) >= 2 for hs in sources.values()):
        for h in history_set:
            with limited():
                m = fit_correction(rows, h, "cuda")
            if h in m["histories"] or held in m["histories"]:
                raise ValueError("Deletion did not remove the full donor dialogue")
            folds[h] = m
    return {"support": sources, "full": full, "deletions": folds}


def intersect(updates, counts):
    """Farthest commonly supported endpoint of signed intervals from zero."""
    mask = updates.new_tensor(counts) >= 2
    return (updates.amin(0).clamp_min(0) + updates.amax(0).clamp_max(0)) * mask


def predict(model, x, device):
    all_models = {"full": model["full"], **model["deletions"]}
    issued = {h: corrections(m, x, device, False) for h, m in all_models.items()}
    counts = [len(model["support"][s]) for s in SKILLS]
    robust = {
        arm: intersect(
            torch.tensor(
                [p[arm] for p in issued.values()], device=device, dtype=torch.float64
            ),
            counts,
        ).tolist()
        for arm in issued["full"]
    }
    return robust, {h: p["credit"] for h, p in issued.items()}


def issue(saved, x, old):
    up, probes = predict(saved["model"], x, "cuda")
    cleared, _ = predict(saved["zero_new"], x, "cuda")
    if not all(
        abs(a - b) < 1e-12 for a, b in zip(probes["full"], old["credit"], strict=True)
    ):
        raise ValueError("Full model and original policy disagree")
    up.update(
        zero_new=cleared["credit"],
        unreplicated=old["credit"],
        previous=old["previous"],
    )
    return up, probes


def freeze(base, wave):
    reg = registration(base)
    parent = base / "late-pairs/semantic-writer/three-writer"
    early = base / "source-compression/earliest-three-skills"
    src = "outlier-" + wave
    mode = "outlier-replicated-" + wave
    out = parent / mode
    eo = early / mode
    for folder in (out, eo, out / "by-history"):
        folder.mkdir(exist_ok=True)
    model_path = parent / src / "models.json"
    source = parent / src / "forecast.json"
    earlier = early / src / "forecast.json"
    fg = parent / "temporal-gate/models.json"
    fr = parent / "writer-format-verified/models.json"
    inputs = {
        k: file_hash(v)
        for k, v in {
            "plan": reg,
            "models": model_path,
            "later": source,
            "early": earlier,
            "gate": fg,
            "route": fr,
        }.items()
    }
    f = out / "forecast.json"
    ef = eo / "forecast.json"
    if f.exists() and ef.exists():
        issued = read(f)
        if (
            issued["inputs"] != inputs
            or issued["models"] != file_hash(out / "models.json")
            or read(ef)["parent"] != file_hash(f)
        ):
            raise ValueError("Different replicated policy")
        return f
    parent_pred, early_pred = read(source), read(earlier)
    fit = read(model_path)
    if (
        parent_pred["models"] != inputs["models"]
        or early_pred["parent"] != inputs["later"]
        or fit["inputs"]["gate"] != file_hash(parent / "temporal-gate/forecast.json")
    ):
        raise ValueError("Unbound unconstrained component fit")
    rows = fit["donors"]
    archived_path = parent / "outlier-components/models.json"
    if fit["inputs"]["previous"] != file_hash(archived_path):
        raise ValueError("Changed reference trained before the early wave")
    archived = read(archived_path)
    old_names = {r["node"] for r in archived["donors"]}
    zeroed = [
        r if r["node"] in old_names else dict(r, credit=[[0.0, 0.0] for _ in SKILLS])
        for r in rows
    ]
    gm, rm = read(fg)["models"], read(fr)["models"]
    models, later, prior = {}, {}, {}
    for h, full in fit["models"].items():
        mf = out / "by-history" / (h + ".json")
        if not mf.exists():
            m = ensemble(rows, {k: v for k, v in full.items() if k != "zero_new"}, h)
            z = ensemble(zeroed, full["zero_new"], h)
            register(mf, {"inputs": inputs, "model": m, "zero_new": z})
        saved = read(mf)
        if saved["inputs"] != inputs:
            raise ValueError("Changed history-excluded ensemble")
        models[h] = file_hash(mf)

        with limited():
            for n, w in parent_pred["windows"].items():
                if n.rsplit("_s", 1)[0] != h:
                    continue
                s = w["scores"]
                up, probes = issue(saved, s["source_features"], s["outlier"])
                later[n] = {
                    **w,
                    "scores": {**s, "outlier": up},
                    "deletion_utilities": probes,
                }
            for n, w in early_pred["windows"].items():
                if w["history"] != h:
                    continue
                x = w["features"]
                native = torch.tensor(
                    w["probabilities"]["native"], device="cuda", dtype=torch.float64
                )
                raw = utilities(gm[h], scores_at(rm[h], x, "cuda"), x, "cuda")["raw"]
                ordinary = corrections(full, x, "cuda", False)
                historic = archived["models"][h]
                ordinary["previous"] = corrections(historic, x, "cuda", False)["credit"]
                up, probes = issue(saved, x, ordinary)
                ps = simplex(native, corrected_utilities(raw, up))
                baseline = w["probabilities"]
                for arm, prev in (
                    ("raw", "outlier_raw"),
                    ("unreplicated", "outlier_credit"),
                    ("previous", "outlier_previous"),
                ):
                    if not torch.allclose(
                        ps[arm], native.new_tensor(baseline[prev]), atol=1e-12, rtol=0
                    ):
                        raise ValueError("Different early benchmarks")
                prior[n] = {
                    **w,
                    "probabilities": {
                        **baseline,
                        **{"outlier_" + k: v.tolist() for k, v in ps.items()},
                    },
                    "deletion_utilities": probes,
                }
    if len(later) != 25 or len(prior) != 12:
        raise ValueError("Incomplete replicated policy")
    register(out / "models.json", {"inputs": inputs, "histories": models})
    register(
        f,
        {"inputs": inputs, "models": file_hash(out / "models.json"), "windows": later},
    )
    exposure = eo / "exposure.json"
    if not exposure.exists():
        register(
            exposure,
            {
                k: {v.stem: file_hash(v) for v in (early / k).glob("conv-*.json")}
                for k in ("writers", "readers", "pairs", "judged")
            },
        )
    register(
        ef,
        {
            "inputs": early_pred["inputs"],
            "parent": file_hash(f),
            "exposure": file_hash(exposure),
            "windows": prior,
        },
    )
    return f


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("wave", choices=("initial", "final"))
    cli.add_argument("--register-only", action="store_true")
    a = cli.parse_args()
    if a.register_only:
        registration(a.base)
        raise SystemExit(0)
    gpu()
    f = freeze(a.base, a.wave)
    score(
        a.base,
        f,
        allocate,
        extra_controls=(
            "zero_new",
            "previous",
            "unreplicated",
            "rotated",
            "undivided",
            "swapped",
            "equal",
            "old_gate",
            "fixed_t_credit",
            "joint_graph",
        ),
        replay_choices=True,
    )
    skill_report(f)
    population_bounds(a.base, f)
