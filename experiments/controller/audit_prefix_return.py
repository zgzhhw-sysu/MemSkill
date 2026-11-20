"""Fit prefix-specific credit strength on training trajectories and evaluate frozen policies."""

import argparse
import hashlib
from itertools import product
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.controller.audit_late_pairs import add_mixture, check_design
from experiments.controller.audit_source_compression import panel
from experiments.controller.factorial_rollout import CELLS, load_cell, summarize_window
from experiments.controller.late_pair_policy import actor, read
from experiments.controller.late_pair_trial import (
    aggregate,
    path_policies,
    paths,
    summarize_pair,
)
from experiments.controller.paired_grounding import window
from experiments.controller.prefix_return import (
    attribute,
    budget_envelope,
    fit_strengths,
    predicted_offsets,
)
from experiments.controller.source_compression import file_hash, register
from experiments.controller.rank_return import fit_rules, rank_offsets
from src.data_processing.locomo import LoCoMoProcessor
from src.eval import get_evaluator

PLAN = Path(__file__).with_name("protocols") / "prefix_return.json"
PILOT = {"conv-26", "conv-30"}
BENCHMARKS = (
    "native",
    "terminal",
    "prefix_shared",
    "prefix_max",
    "prefix_rate",
    "mixture",
    "temporal",
    "activity",
)


def scored_at(archive, cutoff):
    return {
        str(p.relative_to(archive)): file_hash(p)
        for p in sorted(archive.glob("*/*/scores.json"))
        if p.parent.name in ("00", "01", "10", "11") and p.stat().st_mtime_ns <= cutoff
    }


def verify_target(trainer, data, sid, start, versions, parent, features, archive):
    c = window(data[sid], start, 2)
    for n in (start, start + 1):
        key = f"session_{n}"
        (source,) = trainer.data_processor._extract_session_chunks(
            c["conversation"][key], c["conversation"][key + "_date_time"]
        )
        if (
            hashlib.sha256(source.encode()).hexdigest()
            != features[sid, n]["source_sha256"]
        ):
            raise ValueError(
                "Four-path state differs from the registered source features"
            )
    root = archive / f"{sid}_s{start}"
    cells = {
        k: load_cell(trainer, c, root / f"{k[0]}{k[1]}", k, versions) for k in CELLS
    }
    ids = [q["original_index"] for q in c["qa"]]
    predictions = {
        (r["dialogue"], r["session"]): r["p_activity"] for r in parent["forecasts"]
    }
    ref = summarize_window(
        cells,
        [predictions[sid, i] for i in (start, start + 1)],
        parent["fits"][sid],
        ids,
    )
    if ref != read(root / "measurement.json"):
        raise ValueError("Original four-path results do not reproduce")
    return (
        dict(
            dialogue=sid,
            start=start,
            questions=ids,
            types=[q["category"] for q in c["qa"]],
        ),
        cells,
        dict(
            dialogue=sid,
            n=len(ids),
            f1=[mean(cells[k][2].values()) for k in CELLS],
            base=[
                cells[0, 0][0][0]["base_logit"],
                cells[0, 0][0][1]["base_logit"],
                cells[1, 0][0][1]["base_logit"],
            ],
            features=[features[sid, i]["features"] for i in (start, start + 1)],
        ),
    )


def freeze(
    output,
    stage,
    previous,
    corpus,
    data,
    versions,
    trainer,
    source,
    dest,
    early,
    parent,
    identity,
    strength_fits=None,
):
    archive = source / "factorial"
    locked = output / "forecast.json"
    ids = set(data)
    train_ids = PILOT if stage == "pilot" else ids
    if not locked.exists():
        training, receipts, _ = panel(previous, corpus, early, (1, 3), supplement=True)
        if receipts != read(dest / "frozen.json")["training_evidence"]:
            raise ValueError("Historical rule-2 evidence changed")
        tables, hashes = [], {}
        for sid, start in product(sorted(train_ids), (5, 7)):
            name = f"{sid}_s{start}"
            tables.append(
                verify_target(
                    trainer, data, sid, start, versions, parent, early, archive
                )[2]
            )
            hashes[name] = file_hash(archive / name / "measurement.json")
        fits = {}
        folds = (
            {"pilot": train_ids}
            if stage == "pilot"
            else {**{d: ids - {d} for d in ids}, "deployment": ids}
        )
        for key, keep in sorted(folds.items()):
            begin = time.monotonic()
            calibration = [w for w in tables if w["dialogue"] in keep]
            if strength_fits is None:
                direction = actor([r for r in training if r["dialogue"] in keep])
                fits[key] = {
                    "direction": direction,
                    "gate": fit_strengths(calibration, direction),
                }
            else:
                strength = strength_fits[key]
                if set(strength["gate"]["dialogues"]) != keep:
                    raise ValueError("Rank and strength training histories differ")
                direction = strength["direction"]
                fits[key] = {
                    "direction": direction,
                    "gate": fit_rules(calibration, direction),
                    "strength": strength["gate"],
                }
            torch.cuda.synchronize()
            time.sleep((time.monotonic() - begin) * (1 / 0.30 - 1))
        register(
            locked,
            {
                "inputs": identity,
                "stage": stage,
                "training_measurements": hashes,
                "fits": fits,
            },
        )
    model = read(locked)
    if (
        model["inputs"] != identity
        or model["stage"] != stage
        or any(
            file_hash(archive / n / "measurement.json") != h
            for n, h in model["training_measurements"].items()
        )
    ):
        raise ValueError("Frozen policy inputs changed")
    receipt = output / "preexisting.json"
    fingerprint = file_hash(locked)
    if not receipt.exists():
        cutoff = locked.stat().st_mtime_ns
        register(
            receipt,
            {
                "forecast": fingerprint,
                "utc_epoch_ns": cutoff,
                "factorial": scored_at(archive, cutoff),
                "paired": scored_at(dest / "outcomes", cutoff),
            },
        )
    prior = read(receipt)
    if prior["forecast"] != fingerprint or any(
        file_hash(folder / path) != digest
        for kind, folder in (("factorial", archive), ("paired", dest / "outcomes"))
        for path, digest in prior[kind].items()
    ):
        raise ValueError("Published outcomes at freeze changed")
    return model["fits"]


def exact_report(w, cells, preds, terminal):
    audit = check_design(w, cells, preds, terminal)
    snapshots = [
        path_policies(cells[k][0], preds, terminal, k)[0] for k in [(0, 0), (1, 0)]
    ]

    def states(arm):
        return torch.tensor(
            [snapshots[0][arm][0], snapshots[0][arm][1], snapshots[1][arm][1]],
            dtype=torch.float64,
        )

    scores = torch.tensor(
        [[cells[k][2][i] for k in CELLS] for i in range(len(w["questions"]))],
        dtype=torch.float64,
    )
    parts = attribute(scores, states("native"), states("prefix"))
    diffs = parts.new_tensor(
        [r["returns"]["prefix"] - r["returns"]["native"] for r in audit["questions"]]
    )
    if not torch.allclose(parts.sum(1), diffs, rtol=0, atol=1e-10):
        raise ValueError(
            "Per-question span contributions failed to reconstruct the policy gain"
        )
    return {
        "dialogue": w["dialogue"],
        "start": w["start"],
        "states": {k: states(k).tolist() for k in ("native", "prefix", "prefix_rate")},
        "questions": [
            dict(r, contributions=v)
            for r, v in zip(audit["questions"], parts.tolist(), strict=True)
        ],
    }


def summarize_full(reports):
    rows = [r for w in reports for r in w["questions"]]
    if not rows:
        return {"windows": 0, "questions": 0}
    return {
        "windows": len(reports),
        "questions": len(rows),
        "mean": {k: mean(r["returns"][k] for r in rows) for k in rows[0]["returns"]},
        "contributions": [mean(r["contributions"][i] for r in rows) for i in range(3)],
    }


def verdict(reports, total_questions, finished):
    if not reports:
        return {"windows": 0, "eligible_for_development": False}
    base = aggregate(reports)
    result = {k: base[k] for k in ("windows", "all", "dialogue", "type", "pivot")}
    result["focal_budget"] = budget_envelope(
        (
            len(w["questions"]),
            w["focal_probabilities"]["native"],
            w["focal_probabilities"]["prefix"],
            mean(q["activity"] - q["temporal"] for q in w["questions"]),
        )
        for w in reports
    )
    buckets = list(base["dialogue"].values())
    arms = base["all"]["arms"]
    controls = BENCHMARKS + (
        ("prefix_context", "prefix_strength") if "prefix_context" in arms else ()
    )
    count = sum(
        g["arms"]["prefix"]["focal"] - g["arms"]["prefix_rate"]["focal"] > 1e-12
        for g in buckets
    )
    result["positive_histories_vs_rate"] = count
    if len(buckets) == 6:
        n = np.array([g["questions"] for g in buckets])
        draws = np.array(list(product(range(6), repeat=6)))
        intervals = {}
        for arm in controls:
            intervals[arm] = {}
            for metric in ("focal", "ht"):
                deltas = np.array(
                    [
                        g["arms"]["prefix"][metric] - g["arms"][arm][metric]
                        for g in buckets
                    ]
                )
                intervals[arm][metric] = np.quantile(
                    (n * deltas)[draws].sum(1) / n[draws].sum(1), [0.025, 0.975]
                ).tolist()
        result["dialogue_bootstrap"] = intervals
    result["eligible_for_development"] = bool(
        finished
        and len(buckets) == 6
        and count >= 4
        and base["all"]["questions"] >= 0.9 * total_questions
        and all(
            arms["prefix"]["focal"] - arms[k]["focal"] >= 0.02
            for k in ("native", "terminal")
        )
        and all(
            arms["prefix"]["focal"] > arms[k]["focal"] + 1e-12 for k in controls[2:]
        )
        and all(arms["prefix"]["ht"] > arms[k]["ht"] for k in ("native", "terminal"))
    )
    return result


def run(previous, current, corpus, base, stage, rank=False):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Use one leased CUDA device for span policy fitting and scoring"
        )
    torch.cuda.set_per_process_memory_fraction(0.06)
    source, dest = base / "source-compression", base / "late-pairs"
    archive = source / "factorial"
    randomized = dest / "outcomes"
    plan = PLAN.with_name("rank_return.json") if rank else PLAN
    output = dest / ("rank-return" if rank else "prefix-return") / stage
    output.mkdir(parents=True, exist_ok=True)
    register(output / "plan.json", read(plan))
    protocol = read(current / "protocol.json")
    if file_hash(corpus) != protocol["input_sha256"]:
        raise ValueError("Changed corpus")
    data = {
        r["sample_id"]: r for r in read(corpus) if r["sample_id"] in protocol["train"]
    }
    if (
        len(data) != 6
        or not PILOT < set(data)
        or set(data) & set(protocol["validation"] + protocol["test"])
    ):
        raise ValueError("Invalid development/test separation")
    parent = read(source / "frozen.json")
    frozen = read(dest / "frozen.json")
    paired_specs = read(randomized / "manifest.json")
    full_specs = read(archive / "manifest.json")
    versions = [s["version"] for s in protocol["skills"]]
    early = {
        (r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]
    }
    features = dict(early)
    features.update(
        {(r["dialogue"], r["session"]): r for r in read(dest / "features.json")["rows"]}
    )
    table = {(r["dialogue"], r["session"]): r for r in frozen["forecasts"]}
    target = add_mixture(dest, table)
    assignment = read(dest / "cohort.json")["assignments"]
    if (
        frozen["inputs"]["cohort"] != file_hash(dest / "cohort.json")
        or frozen["inputs"]["early_features"] != file_hash(source / "features.json")
        or frozen["inputs"]["features"] != file_hash(dest / "features.json")
        or full_specs["forecast"] != frozen["inputs"]["parent"]
        or full_specs["forecast"] != file_hash(source / "frozen.json")
        or paired_specs["forecast"] != file_hash(dest / "frozen.json")
        or paired_specs["skill_versions"] != versions
        or paired_specs["protocol"] != full_specs["protocol"]
        or full_specs["protocol"] != file_hash(current / "protocol.json")
    ):
        raise ValueError("Forecasts and trajectory manifests disagree")
    identity = {
        "plan": file_hash(plan),
        "history": paired_specs["forecast"],
        "mixture": file_hash(dest / "step-mixture/forecast.json"),
        "full_manifest": file_hash(archive / "manifest.json"),
        "paired_manifest": file_hash(randomized / "manifest.json"),
    }
    strength_fits = None
    if rank:
        earlier = dest / "prefix-return" / stage / "forecast.json"
        strength = read(earlier)
        if strength["stage"] != stage or any(
            strength["inputs"][k] != v for k, v in identity.items() if k != "plan"
        ):
            raise ValueError("Different strength calibration panel")
        identity["strength_forecast"] = file_hash(earlier)
        strength_fits = strength["fits"]
    trainer = SimpleNamespace(
        data_processor=LoCoMoProcessor(chunk_mode="full-session"),
        evaluator=get_evaluator("locomo", SimpleNamespace()),
    )
    with torch.device("cuda"):
        fits = freeze(
            output,
            stage,
            previous,
            corpus,
            data,
            versions,
            trainer,
            source,
            dest,
            early,
            parent,
            identity,
            strength_fits,
        )
        prior = read(output / "preexisting.json")
        early_names = {
            mode: {Path(n).parts[0] for n in prior[mode]}
            for mode in ("factorial", "paired")
        }
        eligible = set(data) - PILOT if stage == "pilot" else set(data)
        registry = {
            "forecast": file_hash(output / "forecast.json"),
            "preexisting": file_hash(output / "preexisting.json"),
        }

        def forecasts(sid, start):
            fit = fits["pilot" if stage == "pilot" else sid]
            offsets = rank_offsets if rank else predicted_offsets
            return [
                dict(
                    target[sid, i],
                    prefix_offsets=offsets(fit, features[sid, i]["features"]),
                )
                for i in (start, start + 1)
            ]

        full, paired = {}, {}
        for sid, start in full_specs["windows"]:
            name = f"{sid}_s{start}"
            root = archive / name
            if sid not in eligible or not (root / "measurement.json").is_file():
                continue
            path = output / f"factorial-{name}.json"
            inputs = dict(registry, outcome=file_hash(root / "measurement.json"))
            if not path.exists():
                w, cells, _ = verify_target(
                    trainer, data, sid, start, versions, parent, early, archive
                )
                report = exact_report(
                    w, cells, forecasts(sid, start), frozen["terminal"][sid]
                )
                register(path, {"inputs": inputs, "report": report})
            saved = read(path)
            if saved["inputs"] != inputs:
                raise ValueError("Changed full trajectory")
            full[name] = saved["report"]

        for w in assignment:
            sid, start = w["dialogue"], w["start"]
            name = f"{sid}_s{start}"
            root = randomized / name
            if sid not in eligible or not (root / "pair.json").is_file():
                continue
            path = output / f"paired-{name}.json"
            inputs = dict(registry, outcome=file_hash(root / "pair.json"))
            if not path.exists():
                c = window(data[sid], start, 2)
                if [q["original_index"] for q in c["qa"]] != w["questions"] or [
                    int(q["category"]) for q in c["qa"]
                ] != w["types"]:
                    raise ValueError("Question cohort differs from frozen assignment")
                cells = {
                    k: load_cell(trainer, c, root / f"{k[0]}{k[1]}", k, versions)
                    for k in paths(w["pivot"], w["background"])
                }
                weights = frozen["terminal"][sid]
                if summarize_pair(
                    w, cells, [table[sid, n] for n in (start, start + 1)], weights
                ) != read(root / "pair.json"):
                    raise ValueError("Original paired estimate did not reproduce")
                report = summarize_pair(w, cells, forecasts(sid, start), weights)
                register(path, {"inputs": inputs, "report": report})
            saved = read(path)
            if saved["inputs"] != inputs:
                raise ValueError("Changed randomized trajectory")
            paired[name] = saved["report"]

    windows = [w for w in assignment if w["dialogue"] in eligible]
    done = len(full) == 2 * len(eligible) and len(paired) == len(windows)
    prospect = [r for n, r in paired.items() if n not in early_names["paired"]]
    total = sum(len(w["questions"]) for w in assignment)
    report = {
        "inputs": registry,
        "stage": stage,
        "finished": done,
        "four_path_all": summarize_full(full.values()),
        "four_path_prospective": summarize_full(
            [r for n, r in full.items() if n not in early_names["factorial"]]
        ),
        "paired_all": verdict(list(paired.values()), total, False),
        "paired_prospective": verdict(prospect, total, done and stage == "main"),
    }
    register(output / f"progress-{len(full):02d}-{len(paired):02d}.json", report)
    if done:
        register(output / "summary.json", report)
    print(
        stage,
        len(full),
        "four-path windows;",
        len(paired),
        "randomized windows;",
        len(prospect),
        "prospective",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", action="store_true")
    p.add_argument("stage", choices=("pilot", "main"))
    for name in ("previous", "current", "corpus", "base"):
        p.add_argument(name, type=Path)
    run(**vars(p.parse_args()))
