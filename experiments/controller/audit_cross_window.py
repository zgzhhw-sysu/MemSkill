"""Evaluate the two six-way cross-fits on real long-prefix outcomes as they arrive."""

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.controller.cross_window_prefix import (
    compare_prefixes,
    probabilities,
    summarize,
    trajectory_hashes,
)
from experiments.controller.late_pair_policy import read
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "cross_window_main.json"


def held_models(records, subject, population):
    selected = {}
    for name, payload in records.items():
        if payload["stage"] != "main" or set(payload["fits"]) != population | {
            "deployment"
        }:
            raise ValueError("Incomplete main cross-fits")
        fit = payload["fits"][subject]
        if set(fit["gate"]["dialogues"]) != population - {subject}:
            raise ValueError("Held history contaminated the numerical calibration")
        selected[name] = fit
    return selected


def register_models(dest, root, output):
    sources = {
        k: dest / f / "main/forecast.json"
        for k, f in (("strength", "prefix-return"), ("rank", "rank-return"))
    }
    raw = read(root / "manifest.json")
    identity = {
        "plan": file_hash(PLAN),
        "cohort": file_hash(root / "manifest.json"),
        "fits": {k: file_hash(p) for k, p in sources.items()},
    }
    records = {k: read(p) for k, p in sources.items()}
    pilot = {
        k: read(dest / f / "pilot/forecast.json")
        for k, f in (("strength", "prefix-return"), ("rank", "rank-return"))
    }
    if (
        raw["baseline"] != file_hash(dest / "frozen.json")
        or any(
            v["inputs"]["history"] != raw["baseline"]
            for v in [*records.values(), *pilot.values()]
        )
        or records["rank"]["inputs"]["strength_forecast"]
        != identity["fits"]["strength"]
        or any(
            raw["pilots"][k] != file_hash(dest / f / "pilot/forecast.json")
            for k, f in (("strength", "prefix-return"), ("rank", "rank-return"))
        )
    ):
        raise ValueError("Inconsistent frozen actor inputs")
    written_plan = root / "main-plan.json"
    if read(written_plan) != read(PLAN):
        raise ValueError("The prospective main crossfit protocol changed")
    register(output / "plan.json", read(PLAN))
    locked = output / "registration.json"
    if not locked.is_file():
        cutoff = time.time_ns()
        scores = {
            str(p.relative_to(root)): file_hash(p)
            for p in root.glob("conv-*_to*/*/scores.json")
            if p.parent.name
            and set(p.parent.name) <= {"0", "1"}
            and p.stat().st_mtime_ns <= cutoff
        }
        register(
            locked,
            {"inputs": identity, "cutoff_ns": cutoff, "published_branches": scores},
        )
    receipt = read(locked)
    if receipt["inputs"] != identity or any(
        file_hash(root / n) != h for n, h in receipt["published_branches"].items()
    ):
        raise ValueError("Long-prefix numerical lock or earlier outcomes changed")
    return records, {k: v["fits"]["pilot"] for k, v in pilot.items()}, receipt


def verified_decision(root, w, r, versions=None):
    if r["dialogue"] != w["dialogue"] or r["end"] != w["end"] or r["held"] != w["held"]:
        raise ValueError("Mismatched long-prefix window")
    decisions = []
    events = []
    for a in (0, 1):
        path = root / "".join(map(str, [*w["background"], a]))
        if trajectory_hashes(path) != r["trajectories"][path.name]:
            raise ValueError("Changed physical trajectory")
        decisions.append((read(path / "decisions.json"), None, None))
        (trace,) = (path / "traces").glob("*.jsonl")
        events.append(
            [
                e
                for line in trace.read_bytes().splitlines()
                if (e := json.loads(line))["kind"] == "span"
            ]
        )
    if versions is not None:
        for (actions, _, _), spans in zip(decisions, events, strict=True):
            for action, span in zip(actions, spans, strict=True):
                low, high = action["plain"], action["grounded"]
                picked = high if action["use"] else low
                if (len(low) != 3 or len(high) != 3 or low[:2] != high[:2]
                        or len(set(low)) != 3 or len(set(high)) != 3 or low[-1] == high[-1]
                        or [versions[i] for i in picked] != [s["version"] for s in span["selected_skills"]]):
                    raise ValueError("Long-prefix action differs from its fixed three ordered slots")
    return compare_prefixes(decisions, events, w["background"])


def score_pair(root, w, r, weights, old, new):
    current = verified_decision(root, w, r)

    def probs(fits):
        return probabilities(current, w["background"][-1], r["features"], weights, fits)

    before = probs(old)
    if before != r["probabilities"]:
        raise ValueError("Earlier pilot probabilities did not reproduce")
    after = probs(new)
    questions = []
    for q in r["questions"]:

        def focus(ps):
            return {
                k: q["temporal"] + p * (q["activity"] - q["temporal"])
                for k, p in ps.items()
            }

        if focus(before) != q["focal"]:
            raise ValueError("Earlier focal F1 did not reproduce")
        questions.append(dict(q, focal=focus(after)))
    return dict(
        dialogue=w["dialogue"], end=w["end"], probabilities=after, questions=questions
    )


def run(dest):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    root = dest / "cross-window-prefix"
    output = root / "main-crossfit"
    output.mkdir(exist_ok=True)
    records, pilot, receipt = register_models(dest, root, output)
    weights = read(dest / "frozen.json")["terminal"]
    specs = read(root / "manifest.json")
    subject_ids = {w["dialogue"] for w in specs["endpoints"]}
    if len(subject_ids) != 6:
        raise ValueError("Expected six training histories")
    models = {d: held_models(records, d, subject_ids) for d in subject_ids}
    registered = file_hash(output / "registration.json")
    early = {Path(p).parts[0] for p in receipt["published_branches"]}
    reports = {}
    with torch.device("cuda"):
        for w in specs["endpoints"]:
            sid, n = w["dialogue"], w["end"]
            name = f"{sid}_to{n:02d}"
            folder = root / name
            outcome = folder / "pair.json"
            if not outcome.is_file():
                continue
            path = output / (name + ".json")
            inputs = {"registration": registered, "pair": file_hash(outcome)}
            if not path.is_file():
                r = read(outcome)
                if r["manifest"] != receipt["inputs"]["cohort"]:
                    raise ValueError("Different long-prefix cohort")
                report = score_pair(folder, w, r, weights[sid], pilot, models[sid])
                register(path, {"inputs": inputs, "report": report})
            saved = read(path)
            if saved["inputs"] != inputs:
                raise ValueError("Archived pair changed since scoring")
            reports[name] = saved["report"]
    complete = len(reports) == len(specs["endpoints"])
    summary = {
        "registration": registered,
        "complete": complete,
        "all": summarize(list(reports.values()), complete),
        "strictly_later": summarize(
            [r for k, r in reports.items() if k not in early], complete
        ),
    }
    register(output / f"through-{len(reports):03d}.json", summary)
    if complete:
        register(output / "summary.json", summary)
    print(len(reports), "long-prefix states under both six-way cross-fits", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dest", type=Path)
    run(**vars(p.parse_args()))
