"""Validate a frozen later-panel policy against complete four-path training windows."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch

from experiments.controller.factorial_rollout import CELLS, load_cell, record_scored_paths, summarize_window
from experiments.controller.late_pair_policy import read
from experiments.controller.late_pair_trial import aggregate, path_policies, paths, summarize_pair
from experiments.controller.paired_grounding import window
from experiments.controller.source_compression import file_hash, register
from src.data_processing.locomo import LoCoMoProcessor
from src.eval import get_evaluator


def check_design(w, cells, preds, weights):
    raw = {k: path_policies(cells[k][0], preds, weights, k)[1] for k in CELLS}
    arms = list(raw[0, 0])
    if any(abs(sum(raw[k][arm] for k in CELLS) - 1) > 1e-10 for arm in arms):
        raise ValueError("Policy path masses do not sum to one")
    pairs = []
    for pivot, bg in product(range(2), repeat=2):
        pairs.append(
            summarize_pair(
                dict(w, pivot=pivot, background=bg), {k: cells[k] for k in paths(pivot, bg)}, preds, weights
            )
        )
    exact = []
    for i, orig in enumerate(w["questions"]):
        returns = {arm: sum(raw[k][arm] * cells[k][2][i] for k in CELLS) for arm in arms}
        if any(
            abs(v - mean(r["questions"][i]["ht"][arm] for r in pairs)) > 1e-10 for arm, v in returns.items()
        ):
            raise ValueError("Two-path randomization average differs from the exact policy value")
        exact.append({"original_index": orig, "returns": returns})
    return {
        "questions": exact,
        "steps": [
            {"pivot": p["pivot"], "background": p["background"], "probabilities": p["focal_probabilities"]}
            for p in pairs
        ],
    }


def add_mixture(dest, target):
    mixture = read(dest / "step-mixture/forecast.json")
    if mixture["inputs"] != {
        "base": file_hash(dest / "frozen.json"),
        "plan": file_hash(dest / "step-mixture/plan.json"),
    }:
        raise ValueError("The step mixture no longer belongs to the original forecast")
    offsets = {(r["dialogue"], r["session"]): r["offsets"] for r in mixture["rows"]}
    if offsets.keys() != target.keys() or len(offsets) != len(mixture["rows"]):
        raise ValueError("Incomplete step-mixture forecasts")
    return {
        key: dict(r, offsets=offsets[key], weights=mixture["weights"][key[0]]) for key, r in target.items()
    }


def audit_randomized(dest, trainer, data, versions, base, base_target, target):
    archive = dest / "outcomes"
    specs = read(archive / "manifest.json")
    if specs["forecast"] != file_hash(dest / "frozen.json") or specs["skill_versions"] != versions:
        raise ValueError("Randomized paths use different inputs")
    output = dest / "step-mixture"
    registration = {"base": specs, "mixture": file_hash(output / "forecast.json")}
    reports = []
    for w in read(dest / "cohort.json")["assignments"]:
        sid, start = w["dialogue"], w["start"]
        root = archive / f"{sid}_s{start}"
        if not (root / "pair.json").exists():
            continue
        c = window(data[sid], start, 2)
        if [q["original_index"] for q in c["qa"]] != w["questions"]:
            raise ValueError("The registered questions changed")
        cells = {
            k: load_cell(trainer, c, root / f"{k[0]}{k[1]}", k, versions)
            for k in paths(w["pivot"], w["background"])
        }
        keys = [(sid, start), (sid, start + 1)]
        old = summarize_pair(w, cells, [base_target[k] for k in keys], base["terminal"][sid])
        if old != read(root / "pair.json"):
            raise ValueError("Could not reproduce the original randomized comparison")
        report = summarize_pair(w, cells, [target[k] for k in keys], base["terminal"][sid])
        register(output / (root.name + ".json"), {"inputs": registration, "result": report})
        reports.append(report)
    if reports:
        complete = len(reports) == len(read(dest / "cohort.json")["assignments"])
        summary = aggregate(reports, complete=complete)
        register(output / f"randomized-{len(reports):02d}.json", {"inputs": registration, **summary})
        if complete:
            register(output / "summary.json", {"inputs": registration, **summary})
        arms = summary["all"]["arms"]
        print(
            len(reports),
            summary["all"]["questions"],
            {k: arms[k]["focal"] for k in ("mixture", "calibrated", "native", "terminal")},
            flush=True,
        )


def run(current, corpus, source, dest, randomized=False):
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("Use one CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    model = read(dest / "frozen.json")
    parent = read(source / "frozen.json")
    protocol = read(current / "protocol.json")
    if (
        model["inputs"]["parent"] != file_hash(source / "frozen.json")
        or file_hash(corpus) != protocol["input_sha256"]
    ):
        raise ValueError("Different source actor or corpus")
    archive = source / "factorial"
    manifest = read(archive / "manifest.json")
    if manifest["forecast"] != model["inputs"]["parent"] or manifest["protocol"] != file_hash(
        current / "protocol.json"
    ):
        raise ValueError("Four-path targets do not match frozen study")
    output = dest / "step-mixture"
    early = record_scored_paths(archive, output / "forecast.json")
    target = {(r["dialogue"], r["session"]): r for r in model["forecasts"]}
    data = {c["sample_id"]: c for c in read(corpus) if c["sample_id"] in protocol["train"]}
    trainer = SimpleNamespace(
        data_processor=LoCoMoProcessor(chunk_mode="full-session"),
        evaluator=get_evaluator("locomo", SimpleNamespace()),
    )
    versions = [r["version"] for r in protocol["skills"]]
    raw_target = target
    target = add_mixture(dest, target)
    result = {}
    with torch.device("cuda"):
        if randomized:
            return audit_randomized(dest, trainer, data, versions, model, raw_target, target)
        for sid, start in manifest["windows"]:
            name = f"{sid}_s{start}"
            root = archive / name
            if not (root / "measurement.json").exists():
                continue
            c = window(data[sid], start, 2)
            ids = [r["original_index"] for r in c["qa"]]
            pred = [target[sid, i] for i in (start, start + 1)]
            cells = {k: load_cell(trainer, c, root / f"{k[0]}{k[1]}", k, versions) for k in CELLS}
            original = {(r["dialogue"], r["session"]): r["p_activity"] for r in parent["forecasts"]}
            old = summarize_window(
                cells, [original[sid, i] for i in (start, start + 1)], parent["fits"][sid], ids
            )
            if old != read(root / "measurement.json"):
                raise ValueError("Published four-path window did not reproduce")
            w = dict(dialogue=sid, start=start, questions=ids, types=[q["category"] for q in c["qa"]])
            report = check_design(w, cells, pred, model["terminal"][sid])
            for i, item in enumerate(report["questions"]):
                for a, b in (("native", "native"), ("terminal", "terminal"), ("source", "credit")):
                    if abs(item["returns"][a] - old["per_question"][i]["f1"][b]) > 1e-10:
                        raise ValueError("Parent or native path policy value differs")
            result[name] = report

    def summary(group):
        rows = [r["returns"] for name in group for r in result[name]["questions"]]
        return {
            "questions": len(rows),
            "mean": {k: mean(r[k] for r in rows) for k in rows[0]} if rows else {},
        }

    if result:
        fresh = sorted(set(result) - early)
        register(
            output / ("four-path-audit-" + "-".join(result) + ".json"),
            {
                "forecast": file_hash(output / "forecast.json"),
                "any_early_path": sorted(early),
                "windows": result,
                "all": summary(result),
                "strictly_later": summary(fresh),
            },
        )
        print(
            len(result),
            summary(result)["questions"],
            {
                k: round(v, 5)
                for k, v in summary(result)["mean"].items()
                if k in ("calibrated", "unit", "native", "terminal", "source", "component", "mixture")
            },
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--randomized", action="store_true")
    for n in ("current", "corpus", "source", "dest"):
        p.add_argument(n, type=Path)
    run(**vars(p.parse_args()))
