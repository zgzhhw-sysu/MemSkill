"""Fit and audit the preregistered source-component credit policy after training reader interventions."""

import argparse
import math
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.audit_query_prototypes import compare
from experiments.controller.audit_source_compression import check_forecasts, forecasts, panel, summarize
from experiments.controller.conditional_credit import read, writer_outcomes
from experiments.controller.factorial_rollout import record_scored_paths
from experiments.controller.source_compression import file_hash, register
from experiments.controller.write_components import contributions, retain_credit, score


def relabel(training, results, dest):
    by_key = {(r["episode"], r["original_index"]): r for r in results["rows"]}
    if len(by_key) != len(results["rows"]):
        raise ValueError("Repeated component measurement")
    support = set()
    filtered, first_only = [], []
    for span in training:
        if span["span"] == 0:
            filtered.append(span)
            first_only.append(span)
            continue
        values = []
        for orig, original in span["credits"].items():
            key = (span["episode"], orig)
            retained = 0.0
            if key in by_key:
                row = by_key[key]
                saved = read(dest / "answers" / f"{span['episode']}-q{orig}.json")
                f1 = {k: v["f1"] for k, v in saved["answers"].items()}
                effect = contributions(f1)
                if (
                    saved["manifest"] != results["manifest"]
                    or f1 != row["f1"]
                    or row["strict_credit"] != original
                    or row["dialogue"] != span["dialogue"]
                    or any(effect[k] != row[k] for k in effect)
                ):
                    raise ValueError("Source effect differs from archived reader scores or rule-2 label")
                retained = retain_credit(original, effect["source"], row["endpoints_reproduced"])
                if not math.isclose(retained, row["retained"], abs_tol=1e-12):
                    raise ValueError("Filtered training credit did not reproduce")
            if retained:
                support.add(span["dialogue"])
            values.append(retained)
        filtered.append(dict(span, signed=mean(values) if values else 0.0))
        first_only.append(dict(span, signed=0.0))
    return filtered, first_only, sorted(support)


def run(phase, previous, current, corpus, source, dest):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Fit and evaluate only under a CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    features = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    parent = read(source / "frozen.json")
    if parent["inputs"]["features.json"] != file_hash(source / "features.json"):
        raise ValueError("Source features changed")
    if phase == "freeze":
        score(dest, credit_only=True)
    results = read(dest / "eligible-labels.json")
    jobs = read(dest / "jobs.json")
    expected = {(r["episode"], r["original_index"]) for r in jobs["jobs"] if r["strict_credit"]}
    if expected != {(r["episode"], r["original_index"]) for r in results["rows"]}:
        raise ValueError("Missing eligible historical questions with nonzero final-span credit")
    if (
        results["manifest"] != read(dest / "reader.json")
        or results["manifest"]["jobs"] != file_hash(dest / "jobs.json")
        or jobs["plan"] != file_hash(dest / "plan.json")
    ):
        raise ValueError("The mechanism plan, questions or response backend changed")
    inputs = {
        k: file_hash(path)
        for k, path in {
            "parent": source / "frozen.json",
            "credit_rows": dest / "eligible-labels.json",
            "plan": dest / "plan.json",
            "corpus": corpus,
        }.items()
    }
    with torch.device("cuda"):
        training, hashes, backend = panel(previous, corpus, features, (1, 3), supplement=True)
        if hashes != parent["prior_hashes"] or backend != parent["backend"]:
            raise ValueError("Original training evidence changed")
        rows, ablated, support = relabel(training, results, dest)
        if len(support) < 2:
            register(
                dest / "screened-out.json",
                {
                    "inputs": inputs,
                    "support": support,
                    "reason": "retained final-span credit in fewer than two dialogues",
                },
            )
            return
        if phase == "freeze":
            existing = dest / "frozen.json"
            if existing.exists():
                record_scored_paths(source / "factorial", existing)
                if read(existing)["inputs"] != inputs:
                    raise ValueError("Frozen component policy inputs changed")
                return
            models = {}
            for k, t in (("filtered", rows), ("first_only", ablated)):
                fits, pred = forecasts(t, list(features.values()))
                model = {"fits": fits, "forecasts": pred}
                check_forecasts(features, model)
                models[k] = dict(model, matched_historical=summarize(training, fits, pred))
            already = sorted(f.parent.name for f in (source / "factorial").glob("*/measurement.json"))
            register(
                existing,
                {
                    "inputs": inputs,
                    "models": models,
                    "support": support,
                    "already_published_windows": already,
                },
            )
            record_scored_paths(source / "factorial", existing)
        else:
            frozen = read(dest / "frozen.json")
            if frozen["inputs"] != inputs:
                raise ValueError("Wrong policy or reader result")
            already = frozen["already_published_windows"]
            partial = record_scored_paths(source / "factorial", dest / "frozen.json")
            sections = {}
            for k, model in dict(original=parent, **frozen["models"]).items():
                check_forecasts(features, model)
                windows = writer_outcomes(current, corpus, source, parent, candidate=model)
                fresh = [
                    w for w in windows if f"{w['dialogue']}_s{w['start']}" not in partial | set(already)
                ]
                sections[k] = {
                    "all": compare(windows, model),
                    "strictly_later": compare(fresh, model) if fresh else None,
                }

            def beats_controls(scope):
                output = sections["filtered"][scope]
                return bool(
                    output
                    and output["passes"]
                    and all(
                        output["aggregate"][arm] > sections[control][scope]["aggregate"][arm] + 1e-12
                        for arm in ("credit", "greedy")
                        for control in ("original", "first_only")
                    )
                )

            register(
                dest / "evaluation.json",
                {
                    "forecast": file_hash(dest / "frozen.json"),
                    "already_published_windows": already,
                    "policies": sections,
                    "windows_with_any_early_scores": sorted(partial),
                    "passes_all": beats_controls("all"),
                    "passes_strictly_later": beats_controls("strictly_later"),
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate"))
    for name in ("previous", "current", "corpus", "source", "dest"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
