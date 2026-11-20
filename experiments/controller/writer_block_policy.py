"""Freeze and score a policy using only terminal credits robust to the parser repair."""

import argparse
import math
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.factorial_rollout import read
from experiments.controller.retrieval_competition import (
    PATHS,
    learn,
    mass,
    probability,
    summarize,
)
from experiments.controller.source_compression import file_hash, register

ARMS = ("robust", "overlap", "rate", "return_only", "native", "terminal")


def revise(windows, folder):
    summary = read(folder / "summary.json")
    if summary["matched"] != 4 or len(summary["files"]) != 12:
        raise ValueError("Reproduce all four factual credit controls before fitting")
    updated = []
    for w in windows:
        path = folder / (w["window"] + ".json")
        receipt = read(path)
        if (
            file_hash(path) != summary["files"][w["window"]]
            or receipt["repairs"] != w["repair"]
        ):
            raise ValueError("Training windows or prior repair evidence differ")
        old, new = [0.0, 0.0], [0.0, 0.0]
        for g in receipt["groups"]:
            a = g["prefix"]
            for q in g["questions"]:
                original, credit = q["original_credit"], q["credit"]
                if (
                    not q["matched"]
                    or q["original_index"] != w["questions"][q["qa_index"]]
                    or abs(credit) > abs(original)
                    or credit * original < 0
                ):
                    raise ValueError("Increased or misaligned local credit")
                old[a] += original / len(w["questions"])
                new[a] += credit / len(w["questions"])
        if not all(
            math.isclose(a, b, abs_tol=1e-12)
            for a, b in zip(old, w["signed"], strict=True)
        ):
            raise ValueError("Previous terminal supervision did not reproduce")
        updated.append(dict(w, signed=new))
    return updated


def table(original, states, model, cases, costs, *, pivot=None, background=None):
    ps = {}
    for j, s in states.items():
        prior = original["probabilities"][str(j)]
        stage = "root" if j == 0 else "terminal"
        candidate = probability(s, model[stage])
        rate = float(
            torch.tensor(
                s["base"] + model["shifts"][j], device="cuda", dtype=torch.float64
            ).sigmoid()
        )
        ps[j] = dict(
            robust=candidate,
            rate=rate,
            **{k: prior[k] for k in ("overlap", "return_only", "native", "terminal")},
        )
    focus = ps[0 if pivot == 0 else 1 + background] if pivot is not None else None
    factor = 2 if focus else 1
    expected_tokens = {
        k: {
            op: sum(factor * mass(bits, ps, k) * use[op] for bits, use in costs.items())
            for op in ("reader", "executor")
        }
        for k in ARMS
    }
    rows = []
    for i, r in enumerate(original["rows"]):
        integrated = {
            k: sum(factor * mass(bits, ps, k) * y[i] for bits, y in cases.items())
            for k in ARMS
        }
        if focus:
            low, high = (v[i] for v in cases.values())
            metrics = {
                "ht": integrated,
                "f1": {k: low + p * (high - low) for k, p in focus.items()},
            }
        else:
            metrics = {"f1": integrated}
        for metric, values in metrics.items():
            for k in ("overlap", "return_only", "native", "terminal"):
                if not math.isclose(values[k], r[metric][k], abs_tol=1e-10):
                    raise ValueError(
                        "An unchanged policy or trajectory failed reproduction"
                    )
        rows.append(dict(index=r["index"], type=r["type"], **metrics))
    result = {
        "dialogue": original["dialogue"],
        "window": original["window"],
        "probabilities": ps,
        "expected_tokens": expected_tokens,
        "rows": rows,
    }
    if focus:
        result["focal_probabilities"] = focus
    return result


def report(windows):
    v = summarize(windows, ARMS, "robust", ARMS[1:])
    if windows:
        v["mean_shift_from_previous"] = mean(
            abs(p["robust"] - p["overlap"])
            for w in windows
            for p in (
                [w["focal_probabilities"]]
                if "focal_probabilities" in w
                else w["probabilities"].values()
            )
        )
    return v


def population(dest, training, calibration):
    out = dest / "population"
    available = [w for w in training if (out / (w["window"] + ".json")).exists()]
    if not available:
        return
    windows = []
    for w in available:
        stored = read(out / (w["window"] + ".json"))
        sides = {g["prefix"]: g for g in stored["groups"]}
        if set(sides) != {0, 1} or stored["repairs"] != w["repair"]:
            raise ValueError("Missing committed prefix or changed original data")
        ps = {int(k): p for k, p in calibration[w["window"]]["probabilities"].items()}
        rows = []
        for i, idx in enumerate(w["questions"]):
            branches = {
                b: sides[int(b[0])]["ends"][int(b[1])]["rows"][str(i)] for b in PATHS
            }
            if not all(v["matched"] for v in branches.values()):
                continue
            if any(v["raw"]["f1"] != w["cases"][b][i] for b, v in branches.items()):
                raise ValueError("Raw four-path F1 differs after replay")
            scores = {
                label: {
                    arm: sum(
                        mass(b, ps, arm) * v[label]["f1"] for b, v in branches.items()
                    )
                    for arm in ARMS
                }
                for label in ("raw", "fixed")
            }
            rows.append(dict(index=idx, type=w["types"][i], **scores))
        windows.append(
            {
                "window": w["window"],
                "questions": len(w["questions"]),
                "rows": rows,
                "measurement": file_hash(out / (w["window"] + ".json")),
            }
        )
    rows = [r for w in windows for r in w["rows"]]
    result = {
        "windows": windows,
        "questions": sum(w["questions"] for w in windows),
        "matched": len(rows),
        "fit": file_hash(dest / "fit.json"),
    }
    if rows:
        result["f1"] = {
            label: {arm: mean(r[label][arm] for r in rows) for arm in ARMS}
            for label in ("raw", "fixed")
        }
    register(out / f"through-{len(windows):02d}.json", result)


def run(base):
    dest, comp, older = (
        base / "writer-block-repair",
        base / "retrieval-competition",
        base / "root-overlap",
    )
    late = base.parent / "late-pairs/outcomes"
    old = read(older / "fit.json")
    src = read(comp / "fit.json")
    fit_path = dest / "fit.json"
    inputs = {
        "old": file_hash(older / "fit.json"),
        "features": file_hash(comp / "fit.json"),
        "evidence": file_hash(dest / "credit/summary.json"),
        "manifest": file_hash(dest / "manifest.json"),
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A single leased GPU is required to fit the credit model")
    torch.cuda.set_per_process_memory_fraction(0.06)
    with torch.device("cuda"):
        training = revise(src["training"], dest / "credit")
        if not fit_path.exists():
            models = {
                sid: learn(
                    [w for w in training if w["dialogue"] != sid],
                    True,
                    root_effects=old["effects"],
                )
                for sid in src["models"]
            }
            register(
                fit_path,
                {
                    "inputs": inputs,
                    "models": models,
                    "revised": {w["window"]: w["signed"] for w in training},
                },
            )
        fit = read(fit_path)
        if fit["inputs"] != inputs:
            raise ValueError("Changed frozen model inputs")
        freeze = dest / "preexisting.json"
        register(
            freeze,
            {
                "fit": file_hash(fit_path),
                "scores": scored_at(late, fit_path.stat().st_mtime_ns),
            },
        )
        pre = {p.split("/")[0] for p in read(freeze)["scores"]}
        original = {w["window"]: w for w in read(older / "calibration.json")["windows"]}
        calibration = []
        for w in training:
            m = fit["models"][w["dialogue"]]
            if w["dialogue"] in m["donors"]:
                raise ValueError("Target dialogue leaked into donors")
            calibration.append(
                table(
                    original[w["window"]],
                    dict(enumerate(w["states"])),
                    m,
                    w["cases"],
                    w["tokens"],
                )
            )
        register(
            dest / "calibration.json",
            {
                "fit": file_hash(fit_path),
                "summary": report(calibration),
                "windows": calibration,
            },
        )
        population(dest, training, {w["window"]: w for w in calibration})
        latest = max(
            older.glob("through-*.json"),
            key=lambda p: int(p.stem.split("-")[-1]),
            default=None,
        )
        if latest is None:
            return
        original = read(latest)
        if original["fit"] != inputs["old"]:
            raise ValueError("Mismatched forward baseline")
        windows = []
        for o in original["windows"]:
            name = o["window"]
            a = read(late / name / "pair.json")
            encoded = read(comp / "pairs" / (name + ".json"))
            if encoded["inputs"] != {
                "fit": inputs["features"],
                "pair": file_hash(late / name / "pair.json"),
            }:
                raise ValueError("Frozen feature source changed")
            states = {}
            cases = {}
            for bits, label in zip(a["paths"], ("temporal", "activity"), strict=True):
                b = "".join(map(str, bits))
                dec = read(late / name / b / "decisions.json")
                for j, d in zip((0, 1 + bits[0]), dec, strict=True):
                    state = read(
                        comp / "states" / (encoded["state_hashes"][str(j)] + ".json")
                    )
                    if state["state"] != encoded["state_hashes"][str(j)]:
                        raise ValueError("Different predecision state features")
                    value = dict(state, base=d["base_logit"])
                    if states.setdefault(j, value) != value:
                        raise ValueError(
                            "The unchanged state differs between interventions"
                        )
                cases[b] = [q[label] for q in a["questions"]]
            windows.append(
                table(
                    o,
                    states,
                    fit["models"][a["dialogue"]],
                    cases,
                    encoded["tokens"],
                    pivot=a["pivot"],
                    background=a["background"],
                )
            )
        strict = report([w for w in windows if w["window"] not in pre])
        result = {
            "fit": file_hash(fit_path),
            "registered_questions": 466,
            "all": report(windows),
            "prospective": strict,
            "windows": windows,
        }
        register(dest / f"through-{len(windows):02d}.json", result)
        if len(windows) == 52:
            register(dest / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    run(parser.parse_args().base)
