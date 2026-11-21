"""Identify factor effects from equivalent requests and test held-history grouping predictions."""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.witness_passage_policy import limited
from experiments.controller.writer_format_trial import CELLS, NATIVE
from src.evaluation_cache import digest

METRICS = (
    "template",
    "grouping",
    "activity_grouping",
    "relation_grouping",
    "interaction",
)


def interval_map(groups, known):
    # Each repeated prompt consumes one common unknown answer, including when its two coefficients cancel.
    incidence = F.one_hot(groups, num_classes=4).to(known.dtype)
    coefficients = (
        known.new_tensor(
            [
                [-0.5, -0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5, 0.5],
                [-1, 1, 0, 0],
                [0, 0, -1, 1],
                [1, -1, -1, 1],
            ]
        )
        @ incidence
    )
    unknown = known.isnan().unsqueeze(1)
    estimate = (coefficients * known.nan_to_num().unsqueeze(1)).sum(-1)
    return torch.stack(
        [
            estimate + (unknown * part).sum(-1)
            for part in (coefficients.clamp_max(0), coefficients.clamp_min(0))
        ],
        dim=-1,
    )


def parameters(prompts, observed):
    classes = {}
    assign = []
    known = [float("nan")] * 4
    for k in CELLS:
        j = classes.setdefault(prompts[k], len(classes))
        assign.append(j)
        if k in observed:
            v = observed[k]
            if not 0 <= v <= 1 or (
                not math.isnan(known[j]) and abs(known[j] - v) > 1e-10
            ):
                raise ValueError("An identical reader request has conflicting outcomes")
            known[j] = v
    return assign, known


def reader_inputs(archived, prepared_status, packet, physical):
    diag = dict(archived)
    keys = {k: k for k in CELLS}
    if prepared_status == "different_pure_bank":
        diag.clear()
    if packet:
        for k in packet["mismatch"]:
            diag.pop(k, None)
        keys = {k: v["hash"] for k, v in packet["arms"].items()}
        if any(abs(diag[k] - packet["arms"][k]["old"]["f1"]) > 1e-10 for k in diag):
            raise ValueError("Changed original endpoint")
    if physical.get("status") == "different_pure_answer":
        # A failed repeat invalidates extrapolation from unverified historical answers.
        diag.clear()
    diag.update({k: v["f1"] for k, v in physical.get("reads", {}).items()})
    return parameters(keys, diag)


def run(base):
    s = base / "source-compression/writer-format-credit"
    p = base / "late-pairs/semantic-writer/three-writer/writer-format-history"
    study = read(s / "registration.json")
    dest = s / "identification"
    dest.mkdir(exist_ok=True)
    # A completion marker covers immutable child receipts. Capture it before listing them.
    marker = s / "readers-complete.json"
    completion = file_hash(marker) if marker.is_file() else None
    answers = {
        str(f.relative_to(s)): file_hash(f)
        for f in sorted((s / "questions").glob("*/*.json"))
    }
    inputs = {
        "audit_version": 2,
        "completion": completion,
        "registration": file_hash(s / "registration.json"),
        "history": file_hash(s / "history.json"),
        "forecast": file_hash(p / "forecast.json"),
        "models": file_hash(p / "models.json"),
        "prepared": {
            f.stem: file_hash(f) for f in sorted((s / "prepared").glob("conv-*.json"))
        },
        "answers": answers,
    }
    out = dest / ("snapshot-" + digest(inputs) + ".json")
    if out.exists():
        if completion:
            register(dest / "summary.json", {"file": out.name, "hash": file_hash(out)})
        return
    model = read(p / "models.json")
    forecast = read(p / "forecast.json")
    if (
        forecast["models"] != inputs["models"]
        or forecast["inputs"]["historical"] != inputs["history"]
    ):
        raise ValueError("Different frozen transport model")
    earlier = read(s / "history.json")["rows"]
    original = read(
        base
        / "late-pairs/semantic-writer/three-writer/skill-graph/complete/models.json"
    )["donors"]
    windows = {}
    groups = []
    known = []
    ids = []
    done = {}
    labels = {}
    predictions = {}
    for w in original:
        name = w["window"]
        h = w["history"]
        m = model["models"][h]
        if h in m["histories"] or any(
            n.startswith(h + "_s") for n in m["historical_windows"]
        ):
            raise ValueError("Target history entered grouping forecast")
        if [r["index"] for r in w["points"]] != study["questions"][name]:
            raise ValueError("Changed source questions")
        prep = (
            read(s / "prepared" / (name + ".json"))
            if name in inputs["prepared"]
            else {}
        )
        if prep and prep["inputs"]["registration"] != inputs["registration"]:
            raise ValueError("Unbound ranking")
        by = defaultdict(list)
        for r in earlier:
            if r["history"] != h:
                by[r["history"]].append(r["raw"])
        g = min(3.0, max(-3.0, (w["features"][1] - m["center"]) / m["sd"]))
        predictions[name] = {
            "constant": mean(mean(v) for v in by.values()),
            **{k: m["coef"][k][2] + g * m["coef"][k][3] for k in ("raw", "credit")},
        }
        done[name] = []
        labels[name] = []
        old = []
        for r in w["points"]:
            i = r["index"]
            diag = {NATIVE[0]: r["scores"][1], NATIVE[1]: r["scores"][2]}
            file = s / "questions" / name / (str(i) + ".json")
            rel = str(file.relative_to(s))
            physical = read(file) if rel in inputs["answers"] else {}
            if physical and (
                physical["inputs"]["parent"] != inputs["prepared"][name]
                or physical["inputs"]["registration"] != inputs["registration"]
            ):
                raise ValueError("Unbound crossed response")
            if (
                physical
                and file.stat().st_mtime_ns <= (p / "forecast.json").stat().st_mtime_ns
            ):
                old.append(i)
            packet = prep.get("questions", {}).get(str(i), {})
            a, y = reader_inputs(diag, prep.get("status"), packet, physical)
            if physical.get("status") == "matched":
                done[name].append(i)
                labels[name].append(physical["effects"]["grouping_at_activity"])
            ids.append((name, i))
            groups.append(a)
            known.append(y)
        windows[name] = {
            "history": h,
            "n": len(w["points"]),
            "prepared": prep.get("status"),
            "measured": done[name],
            "crossed_answers_at_freeze": old,
            "predictions": predictions[name],
            "scores": {},
        }
    with limited():
        bounds = interval_map(
            torch.tensor(groups, device="cuda"),
            torch.tensor(known, device="cuda", dtype=torch.float64),
        )
        global_bounds = bounds.mean(0).tolist()
        point_bounds = bounds.tolist()
    for (name, idx), v in zip(ids, point_bounds, strict=True):
        windows[name]["scores"][str(idx)] = dict(zip(METRICS, v, strict=True))
    by_window = {}
    for name, r in windows.items():
        if len(labels[name]) == r["n"]:
            actual = mean(labels[name])
            by_window[name] = {
                "activity_grouping": actual,
                "error": {k: v - actual for k, v in predictions[name].items()},
            }

    def errors(names):
        if not names:
            return {"windows": 0}
        result = {}
        for k in ("constant", "raw", "credit"):
            es = [by_window[n]["error"][k] for n in names]
            result[k] = {
                "mae": mean(map(abs, es)),
                "rmse": math.sqrt(mean(e * e for e in es)),
                "right_direction": sum(
                    predictions[n][k] * by_window[n]["activity_grouping"] > 0
                    for n in names
                ),
            }
        return {
            "windows": len(names),
            "questions": sum(windows[n]["n"] for n in names),
            "errors": result,
        }

    register(
        out,
        {
            "inputs": inputs,
            "windows": windows,
            "questions": len(ids),
            "measured": sum(map(len, done.values())),
            "conditional_population_bounds": dict(
                zip(METRICS, global_bounds, strict=True)
            ),
            "identified_questions": {
                k: sum(abs(row[j][1] - row[j][0]) < 1e-12 for row in point_bounds)
                for j, k in enumerate(METRICS)
            },
            "complete": by_window,
            "calibration": errors(list(by_window)),
            "prospective_calibration": errors(
                [n for n in by_window if not windows[n]["crossed_answers_at_freeze"]]
            ),
        },
    )
    if completion:
        register(dest / "summary.json", {"file": out.name, "hash": file_hash(out)})
    print(out.name, len(by_window), sum(map(len, done.values())), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    a = cli.parse_args()
    gpu()
    run(a.base)
