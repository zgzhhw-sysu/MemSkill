"""Fit and evaluate span credit from naturally occurring question-to-reply retrieval."""

import argparse
import json
import math
from pathlib import Path

import torch

from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.factorial_rollout import read
from experiments.controller.late_pair_trial import paths as assigned_paths
from experiments.controller.reply_probes import PLAN
from experiments.controller.retrieval_competition import (
    PATHS,
    budgets,
    learn,
    probability,
    summarize,
)
from experiments.controller.source_compression import file_hash, register
from experiments.controller.writer_block_policy import trajectory_table
from src.evaluation_cache import digest

BASELINES = ("robust", "native", "terminal")
ARMS = ("reply", "raw", "sham", "rate", *BASELINES)
SKILLS = ("capture_temporal_context", "capture_activity_details")


def states(folder, paths, sources):
    sid, start = folder.name.rsplit("_s", 1)
    receipts = [read(sources / f"{sid}_s{int(start) + j}.json") for j in range(2)]
    result = {}
    versions = {}
    for bits in paths:
        (trace,) = (folder / bits / "traces").glob("*.jsonl")
        spans = [
            s
            for line in trace.read_bytes().splitlines()
            if (s := json.loads(line))["kind"] == "span"
        ]
        decisions = read(folder / bits / "decisions.json")
        for j, (bit, span, decision, receipt) in enumerate(
            zip(bits, spans, decisions, receipts, strict=True)
        ):
            ordered = span["selected_skills"]
            names = [s["name"] for s in ordered]
            chosen = int(bit)
            if (
                span["span_id"] != j
                or decision["use"] != chosen
                or receipt["source"] != digest(span["source"])
                or names != ["insert", "noop", SKILLS[chosen]]
                or ordered[-1]["source_rule"]
                != ("dated_single_turns", "dated_adjacent_turns")[chosen]
            ):
                raise ValueError(
                    "Recorded source or selected skill disagrees with intervention"
                )
            for skill in ordered:
                if (
                    versions.setdefault(skill["name"], skill["version"])
                    != skill["version"]
                ):
                    raise ValueError("Fixed skill version changed between branches")
            key = 0 if j == 0 else 1 + int(bits[0])
            observed = {
                "base": decision["base_logit"],
                "actual": receipt["actual"],
                "sham": receipt["sham"],
                "state": digest(decision["state"]),
                "source": receipt["source"],
            }
            if result.setdefault(key, observed) != observed:
                raise ValueError(
                    "An unchanged predecision state differs between branches"
                )
    return result


def training_models(windows, evidence, sid):
    donors = [w for w in windows if w["dialogue"] != sid]
    models = {}
    for arm in ("reply", "raw", "sham"):
        view = [
            dict(
                w,
                states=[
                    dict(s, features=s["sham" if arm == "sham" else "actual"])
                    for s in w["states"]
                ],
            )
            for w in donors
        ]
        credit = arm != "raw"
        models[arm] = learn(view, credit, root_effects=evidence if credit else None)
        if sid in models[arm]["donors"]:
            raise ValueError("Held dialogue entered policy training")
    return models


def evaluate(original, nodes, models, cases, costs, *, pivot=None, background=None):
    ps = {}
    for j, node in nodes.items():
        prior = original["probabilities"][str(j)]
        stage = "root" if j == 0 else "terminal"
        p = {
            arm: probability(
                dict(node, features=node["sham" if arm == "sham" else "actual"]),
                model[stage],
            )
            for arm, model in models.items()
        }
        p["rate"] = 1 / (1 + math.exp(-node["base"] - models["reply"]["shifts"][j]))
        native = 1 / (1 + math.exp(-node["base"]))
        if not math.isclose(native, prior["native"], abs_tol=1e-10):
            raise ValueError("Archived probability and native logit disagree")
        ps[j] = dict(p, **{k: prior[k] for k in BASELINES})
    result = trajectory_table(
        original, ps, cases, costs, ARMS, BASELINES, pivot=pivot, background=background
    )
    result["source_hashes"] = {j: n["source"] for j, n in nodes.items()}
    return result


def report(windows):
    return summarize(windows, ARMS, "reply", ARMS[1:])


def development_gate(temporal, prospective, total):
    covered = temporal["questions"] / total
    f1, joint = temporal.get("f1", {}), temporal.get("ht", {})
    checks = {
        "coverage": covered >= 0.90,
        "histories": len(temporal.get("by_dialogue", {})) >= 4,
        "prospective": prospective["questions"] > 0,
        "native_and_terminal": bool(f1)
        and all(f1["reply"] >= f1[k] + 0.02 for k in ("native", "terminal")),
        "placement_and_credit_controls": bool(f1)
        and all(f1["reply"] > f1[k] for k in ("rate", "raw", "sham", "robust")),
        "joint": bool(joint) and joint["reply"] > joint["native"],
    }
    return {
        "registered_questions": total,
        "coverage": covered,
        "checks": checks,
        "eligible_for_independent_development": all(checks.values()),
    }


def run(base):
    comp = base / "source-compression"
    dest, previous = comp / "reply-placement", comp / "writer-block-repair"
    src, late = dest / "sources", base / "late-pairs"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one leased CUDA device for fitting")
    torch.cuda.set_per_process_memory_fraction(0.06)
    inputs = {
        "plan": file_hash(PLAN),
        "learner": file_hash(Path(__file__)),
        "features": file_hash(dest / "manifest.json"),
        "factorial": file_hash(comp / "retrieval-competition/fit.json"),
        "root": file_hash(comp / "root-overlap/fit.json"),
        "verified": file_hash(previous / "fit.json"),
        "cohort": file_hash(late / "cohort.json"),
    }
    training = read(comp / "retrieval-competition/fit.json")["training"]
    effects = read(comp / "root-overlap/fit.json")["effects"]
    signed = read(previous / "fit.json")["revised"]
    for w in training:
        nodes = states(comp / "factorial" / w["window"], PATHS, src)
        if any(n["base"] != w["states"][j]["base"] for j, n in nodes.items()):
            raise ValueError("Wrong factorial calibration state")
        w.update(states=[nodes[j] for j in range(3)], signed=signed[w["window"]])
    inputs["calibration_sources"] = digest([w["states"] for w in training])
    fp = dest / "fit.json"
    if not fp.exists():
        with torch.device("cuda"):
            models = {
                sid: training_models(training, effects, sid)
                for sid in sorted({w["dialogue"] for w in training})
            }
        register(fp, {"inputs": inputs, "models": models})
    frozen = read(fp)
    if frozen["inputs"] != inputs:
        raise ValueError("Frozen inputs changed")
    fit_id = file_hash(fp)
    pre = scored_at(late / "outcomes", fp.stat().st_mtime_ns)
    register(dest / "preexisting.json", {"fit": fit_id, "scores": pre})
    calibration = read(previous / "calibration.json")
    if calibration["fit"] != inputs["verified"]:
        raise ValueError("The calibration comparator uses different control evidence")
    original = {w["window"]: w for w in calibration["windows"]}
    if not (dest / "calibration.json").exists():
        rows = [
            evaluate(
                original[w["window"]],
                dict(enumerate(w["states"])),
                frozen["models"][w["dialogue"]],
                w["cases"],
                w["tokens"],
            )
            for w in training
        ]
        register(
            dest / "calibration.json",
            {"fit": fit_id, "summary": report(rows), "windows": rows},
        )
    assignments = {
        f"{v['dialogue']}_s{v['start']}": v
        for v in read(late / "cohort.json")["assignments"]
    }
    latest = max(previous.glob("through-*.json"), default=None)
    if latest is None:
        return
    prior = read(latest)
    if prior["fit"] != inputs["verified"]:
        raise ValueError("The randomized comparator uses a different policy")
    windows = []
    for o in prior["windows"]:
        name = o["window"]
        spec = assignments[name]
        if not all(
            (src / f"{spec['dialogue']}_s{spec['start'] + j}.json").exists()
            for j in (0, 1)
        ):
            continue
        folder = late / "outcomes" / name
        pair = read(folder / "pair.json")
        if (
            pair["paths"]
            != [list(b) for b in assigned_paths(spec["pivot"], spec["background"])]
            or [(r["index"], r["type"]) for r in o["rows"]]
            != list(zip(spec["questions"], spec["types"], strict=True))
            or [q["original_index"] for q in pair["questions"]] != spec["questions"]
            or any(
                spec[k] != pair[k] for k in ("dialogue", "pivot", "background", "start")
            )
        ):
            raise ValueError("Wrong randomized pair or question order")
        paths = ["".join(map(str, bits)) for bits in pair["paths"]]
        cases = {
            b: [q[key] for q in pair["questions"]]
            for b, key in zip(paths, ("temporal", "activity"), strict=True)
        }
        nodes = states(folder, paths, src)
        windows.append(
            evaluate(
                o,
                nodes,
                frozen["models"][o["dialogue"]],
                cases,
                {b: budgets(folder, b) for b in paths},
                pivot=spec["pivot"],
                background=spec["background"],
            )
        )
    if not windows:
        return
    # Wait for one complete archived reporting prefix so a later retroactive file
    # cannot replace a different set of the same size under the same receipt name.
    if len(windows) != len(prior["windows"]):
        return
    seen = {k.split("/")[0] for k in pre}
    fresh = [w for w in windows if w["window"] not in seen]
    after = [w for w in windows if assignments[w["window"]]["start"] >= 13]
    result = {
        "fit": fit_id,
        "all": report(windows),
        "prospective": report(fresh),
        "session13_plus": report(after),
        "prospective_session13_plus": report(
            [w for w in after if w["window"] not in seen]
        ),
        "windows": windows,
    }
    total = sum(len(s["questions"]) for s in assignments.values() if s["start"] >= 13)
    result["development_gate"] = development_gate(
        result["session13_plus"], result["prospective_session13_plus"], total
    )
    register(dest / f"through-{len(windows):02d}.json", result)
    if len(windows) == len(assignments):
        register(dest / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    run(parser.parse_args().base)
