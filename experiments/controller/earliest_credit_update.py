"""Refit the state-conditional policy once the complete early third-skill audit returns."""

import argparse
from pathlib import Path

import torch

from experiments.controller.checkpoint_odds_credit import allocate as previous_policy
from experiments.controller.checkpoint_odds_credit import distribute, fit
from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.skill_graph_data import training_rows
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_skill_margin import score
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "earliest_credit_update.json"


def registration(base):
    early = base / "source-compression/earliest-three-skills"
    parent = base / "late-pairs/semantic-writer/three-writer"
    dest = parent / "earliest-credit-update"
    dest.mkdir(exist_ok=True)
    path = dest / "registration.json"
    deps = {"plan": file_hash(PLAN), "cohort": file_hash(early / "forecast.json")}
    if not path.exists():
        register(
            path,
            {
                "inputs": deps,
                "existing": {
                    k: {p.stem: file_hash(p) for p in (early / k).glob("conv-*.json")}
                    for k in ("readers", "pairs", "judged")
                },
            },
        )
    if read(path)["inputs"] != deps:
        raise ValueError("Changed early credit registration")
    return path


def points(pair, audit):
    if len({r["original_index"] for r in pair["rows"]}) != len(pair["rows"]):
        raise ValueError("Repeated original question")
    lookup = {}
    for v in audit["cases"]:
        if v["baseline"] != "activity":
            continue
        idx = v["index"]
        if idx in lookup or v["credit"] != (
            v["raw"] if v["status"] == "confirmed" else 0
        ):
            raise ValueError("Ambiguous direct action credit")
        lookup[idx] = v
    expected = {r["qa"]["original_index"] for r in pair["jobs"]["activity"]}
    if expected != set(lookup):
        raise ValueError("Incomplete adjudication")
    data = []
    for r, scores in zip(pair["rows"], pair["f1"], strict=True):
        idx = r["original_index"]
        case = lookup.get(idx)
        if case and abs(scores[2] - scores[1] - case["raw"]) > 1e-10:
            raise ValueError("Credited a different question or skill direction")
        data.append(
            {
                "index": idx,
                "category": r["category"],
                "scores": scores,
                "edges": [0, 0, case["credit"] if case else 0],
            }
        )
    return data


def extra_rows(early):
    orig = read(early / "forecast.json")["windows"]
    finals = read(early / "judged-complete.json")
    if set(orig) != set(finals) or len(orig) != 12:
        raise ValueError("Incomplete earliest cohort")
    rows = []
    excluded = {}
    for k, w in orig.items():
        parent = early / "pairs" / (k + ".json")
        judge = early / "judged" / parent.name
        repair = early / "repairs" / parent.name
        pair = read(parent)
        audit = read(judge)
        if (
            finals[k] != file_hash(judge)
            or audit["inputs"]["parent"] != file_hash(repair)
            or read(repair)["inputs"]["parent"] != file_hash(parent)
            or pair["inputs"]["forecast"] != file_hash(early / "forecast.json")
        ):
            raise ValueError("Unbound original-state triplet")
        if audit["status"] != "judged":
            if pair["status"] == "matched":
                raise ValueError("Matched triplet lacks adjudication")
            excluded[k] = audit["status"]
            continue
        if (
            pair["status"] != "matched"
            or [v["original_index"] for v in pair["rows"]] != w["questions"]
        ):
            raise ValueError("Different physical branch questions")
        t, a, r = w["probabilities"]["native"]
        rows.append(
            {
                "window": k,
                "history": w["history"],
                "background": w["background"],
                "features": w["features"],
                "native_pair": [a / (a + t), r / (r + t)],
                "points": points(pair, audit),
            }
        )
    return rows, excluded


def allocate(native, scores):
    prior = previous_policy(native, scores)
    temporal = prior["raw"][0]
    ans = distribute(
        native, temporal, scores["source_features"], scores["augmented_model"]
    )
    zero = distribute(
        native, temporal, scores["source_features"], scores["zero_new_model"]
    )
    if not torch.allclose(ans["raw"], zero["raw"], rtol=0, atol=1e-12):
        raise ValueError("Removing new span credit changed the raw policy")
    return {**prior, **ans, "zero_new": zero["credit"], "previous": prior["credit"]}


def freeze(base):
    reg = registration(base)
    out = reg.parent
    early = base / "source-compression/earliest-three-skills"
    prior = out.parent / "checkpoint-odds/forecast.json"
    identity = {
        "registration": file_hash(reg),
        "early_complete": file_hash(early / "judged-complete.json"),
        "previous": file_hash(prior),
    }
    f = out / "forecast.json"
    if f.exists():
        if read(f)["inputs"] != identity or read(f)["models"] != file_hash(
            out / "models.json"
        ):
            raise ValueError("Changed completed early update")
        return f
    donors, _ = training_rows(base, True)
    extra, excluded = extra_rows(early)
    if {v["window"] for v in donors} & {v["window"] for v in extra}:
        raise ValueError("Duplicate windows across training cohorts")
    masked = [
        {**w, "points": [{**q, "edges": [0, 0, 0]} for q in w["points"]]} for w in extra
    ]
    forecasts = {}
    models = {}
    windows = read(prior)["windows"]
    for h in sorted({k.rsplit("_s", 1)[0] for k in windows}):
        with limited():
            full = fit(donors + extra, h, "cuda")
            zero = fit(donors + masked, h, "cuda")
        if (
            full["coef"]["raw"] != zero["coef"]["raw"]
            or full["centers"] != zero["centers"]
            or full["scales"] != zero["scales"]
            or h in full["histories"]
        ):
            raise ValueError("Wrong raw or held-history control")
        models[h] = {"full": full, "zero_new": zero}
        for k, w in windows.items():
            if k.rsplit("_s", 1)[0] == h:
                forecasts[k] = {
                    **w,
                    "scores": {
                        **w["scores"],
                        "augmented_model": full,
                        "zero_new_model": zero,
                    },
                }
    register(
        out / "models.json",
        {
            "inputs": identity,
            "folds": models,
            "included": {w["window"]: len(w["points"]) for w in extra},
            "excluded": excluded,
        },
    )
    register(
        f,
        {
            "inputs": identity,
            "models": file_hash(out / "models.json"),
            "windows": forecasts,
        },
    )
    return f


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("--register-only", action="store_true")
    a = cli.parse_args()
    if a.register_only:
        registration(a.base)
    else:
        gpu()
        score(
            a.base,
            freeze(a.base),
            allocate,
            direct=True,
            extra_controls=("previous", "zero_new", "masked_state"),
            replay_choices=True,
        )
