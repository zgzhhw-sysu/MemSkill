"""Fit and prospectively score span policies from separate old and new memory effects."""

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.evidence_transport_data import source_keys, sources
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.temporal_update_audit import parent_cell
from experiments.controller.temporal_update_forecast import CUE, continuity
from experiments.controller.turn_key_credit import paired_interval
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "temporal_partition_policy.json"
LEARNED = ("credit", "split_return", "pooled_credit", "pooled_return")


def features(previous, current, turns):
    if not len(turns) or len(turns) != len(current) or not len(previous):
        raise ValueError("A policy state requires both visible sessions")
    match = continuity(previous, current, turns)
    cross = (current @ previous.T).amax(dim=1).mean().item()
    words = sum(len(t["text"].split()) for t in turns)
    within = 0.0
    if len(turns) > 1:
        sims = current @ current.T
        sims.fill_diagonal_(-torch.inf)
        within = sims.amax(dim=1).mean().item()
    return [
        cross,
        0.0 if match is None else match,
        mean(bool(CUE.search(t["text"])) for t in turns),
        math.log1p(words),
        within,
        mean(t["text"].rstrip().endswith("?") for t in turns),
    ]


def state_table(source, out, locked, study, corpus):
    path = out / "states.json"
    manifest = source.parent / "evidence-transport"
    protocol = read(study / "protocol.json")
    identity = {
        "forecast": file_hash(source / "forecast.json"),
        "study": file_hash(study / "protocol.json"),
        "corpus": file_hash(corpus),
        "vectors": file_hash(manifest / "sources.json"),
    }
    if (
        identity["study"] != locked["study"]
        or identity["corpus"] != protocol["input_sha256"]
        or identity["vectors"] != locked["source_manifest"]
    ):
        raise ValueError("Physical source vectors or benchmark split changed")
    if not path.exists():
        vectors = sources(manifest, "cuda")
        data = {
            x["sample_id"]: x["conversation"]
            for x in read(corpus)
            if x["sample_id"] in protocol["train"]
        }
        states = {}
        for w in locked["windows"]:
            a, b = (vectors[k] for k in source_keys(w["name"]))
            if [a[3], b[3]] != w["vectors"][:2]:
                raise ValueError(
                    "An input vector file changed since the action forecast"
                )
            turns = data[w["dialogue"]][f"session_{w['start'] + 1}"]
            if b[0] != [t["dia_id"] for t in turns]:
                raise ValueError("Source turns do not match the physical Qwen vectors")
            states[w["name"]] = features(a[1], b[1], turns)
        register(path, dict(identity, states=states))
    snapshot = read(path)
    if any(snapshot[k] != v for k, v in identity.items()):
        raise ValueError("Source state identity changed")
    return snapshot["states"]


def predict(target, early, states, examples, device):
    held = target["dialogue"]
    calibration = [states[w["name"]] for w in early if w["dialogue"] != held]
    others = [e for e in examples if e["dialogue"] != held]
    if not others:
        return dict.fromkeys(LEARNED, 0.5)
    x = torch.tensor(calibration, device=device, dtype=torch.float64)
    scale = (x.quantile(0.75, dim=0) - x.quantile(0.25, dim=0)).clamp_min(0.1)
    target_x = x.new_tensor(states[target["name"]])
    diff = (
        (x.new_tensor([states[e["name"]] for e in others]) - target_x) / scale
    ).square()
    # Centering both states by the same calibration median cancels in the distance.
    counts = Counter(e["dialogue"] for e in others)
    mass = x.new_tensor([1 / counts[e["dialogue"]] for e in others])

    def smooth(distance, key):
        weights = mass * (-0.5 * distance.mean(dim=1)).exp()
        y = x.new_tensor([e[key] for e in others])
        return (weights * y).sum() / (1 + weights.sum())

    components = smooth(diff[:, :3], "old") + smooth(diff[:, 3:], "new")
    pooled = smooth(diff, "total")
    bonus = smooth(diff, "bonus")
    logits = torch.stack([components + bonus, components, pooled + bonus, pooled])
    return dict(
        zip(LEARNED, (25 * logits).clamp(-2, 2).sigmoid().tolist(), strict=True)
    )


def donor_example(w, audit):
    if (
        audit["questions"] != w["questions"]
        or audit["dialogue"] != w["dialogue"]
        or audit["name"] != w["name"]
    ):
        raise ValueError("Wrong training window")
    if audit["status"] != "matched":
        return None
    if [r["index"] for r in audit["rows"]] != w["questions"]:
        raise ValueError("Donor question order changed")
    bonus = []
    for r in audit["rows"]:
        p = r["probe"]
        if (
            p
            and p["signed_effect"]
            and (
                p["status"] != "survived_probes"
                or abs(p["signed_effect"] - r["effects"]["total"]) > 1e-10
            )
        ):
            raise ValueError("Invalid donor selection credit")
        bonus.append(p["signed_effect"] if p else 0.0)
        if (
            abs(r["effects"]["old"] + r["effects"]["new"] - r["effects"]["total"])
            > 1e-10
        ):
            raise ValueError("Partition effects do not reconstruct the paired gain")
    return dict(
        name=w["name"],
        dialogue=w["dialogue"],
        bonus=mean(bonus),
        **{
            k: mean(r["effects"][k] for r in audit["rows"])
            for k in ("old", "new", "total")
        },
    )


def fit(source, out, locked, states):
    early = [w for w in locked["windows"] if w["panel"] == "early"]
    later = [w for w in locked["windows"] if w["panel"] == "later"]
    training = {
        "pilot": [w for w in early if w["name"] == "conv-26_s5"],
        "six": [w for w in early if w["start"] == 5],
        "twelve": early,
    }
    if [len(v) for v in training.values()] != [1, 6, 12] or len(later) != 25:
        raise ValueError("Different temporal training or later cohorts")
    for stage, donors in training.items():
        stage_dir = out / stage
        stage_dir.mkdir(exist_ok=True)
        path = stage_dir / "forecast.json"
        paths = {w["name"]: out.parent / w["name"] / "result.json" for w in donors}
        if not all(p.is_file() for p in paths.values()):
            continue
        identity = {
            "donors": {n: file_hash(p) for n, p in paths.items()},
            "states": file_hash(out / "states.json"),
            "registration": file_hash(out / "registration.json"),
        }
        if path.exists():
            saved = read(path)
            if any(saved[k] != v for k, v in identity.items()):
                raise ValueError("Frozen policy input changed")
            continue
        examples = [
            e
            for w in donors
            if (e := donor_example(w, read(paths[w["name"]]))) is not None
        ]
        probs = {w["name"]: predict(w, early, states, examples, "cuda") for w in later}
        register(
            path,
            dict(
                identity,
                implementation=file_hash(Path(__file__)),
                training=examples,
                probabilities=probs,
            ),
        )
        print(
            "frozen", stage, "donors", len(examples), "targets", len(probs), flush=True
        )


def panel(group):
    if not group:
        return {"windows": 0, "questions": 0}
    rate = mean(w["p"]["credit"] for w in group)
    rows = [(w, r) for w in group for r in w["rows"]]

    def summary(subset):
        metrics = {}
        for arm in (*group[0]["p"], "rate"):
            metrics[arm] = mean(
                r["temporal"]["f1"]
                + (rate if arm == "rate" else w["p"][arm])
                * (r["update"]["f1"] - r["temporal"]["f1"])
                for w, r in subset
            )
        return {"questions": len(subset), "f1": metrics}

    def expected_cost(w, arm, pair):
        prob = rate if arm == "rate" else w["p"][arm]
        return (1 - prob) * pair[0] + prob * pair[1]

    arms = (*group[0]["p"], "rate")
    histories = {
        s: summary([(w, r) for w, r in rows if w["dialogue"] == s])
        for s in sorted({w["dialogue"] for w in group})
    }
    intervals = {}
    if len(histories) >= 2:
        for a in arms:
            if a != "credit":
                intervals[a] = paired_interval(
                    [h["questions"] for h in histories.values()],
                    [h["f1"]["credit"] - h["f1"][a] for h in histories.values()],
                )
    return dict(
        **summary(rows),
        windows=len(group),
        matched_rate=rate,
        mean_abs_probability_shift=mean(
            abs(w["p"]["credit"] - w["p"]["native"]) for w in group
        ),
        mean_abs_credit_adjustment=mean(
            abs(w["p"]["credit"] - w["p"]["split_return"]) for w in group
        ),
        writer_tokens_per_window={
            arm: mean(expected_cost(w, arm, w["writer_tokens"]) for w in group)
            for arm in arms
        },
        reader_tokens_per_question={
            arm: mean(
                expected_cost(
                    w, arm, [sum(r[a]["tokens"]) for a in ("temporal", "update")]
                )
                for w, r in rows
            )
            for arm in arms
        },
        by_history=histories,
        paired_history_intervals=intervals,
        by_type={
            str(t): summary([(w, r) for w, r in rows if r["type"] == t])
            for t in sorted({r["type"] for _, r in rows})
        },
    )


def score(source, out, locked):
    later = [w for w in locked["windows"] if w["panel"] == "later"]
    for path in sorted(out.glob("*/forecast.json")):
        forecast = read(path)
        results, files = [], {}
        eligible_arrived = 0
        for w in later:
            name = w["name"]
            pair = source / name / "pair.json"
            if not pair.exists():
                continue
            observed = read(pair)
            if observed["questions"] != w["questions"]:
                raise ValueError("Scored window population changed")
            files[name] = file_hash(pair)
            if observed["status"] != "ineligible_fixed_slots":
                eligible_arrived += len(w["questions"])
            if observed["status"] != "matched":
                continue
            writes = source / name / "writes.json"
            if file_hash(writes) != observed["writes"]:
                raise ValueError("Scored write receipt changed")
            data = read(writes)
            costs = [
                sum(
                    c["input_tokens"] + c["output_tokens"]
                    for c in data[a]["executor_responses"]
                )
                for a in ("control", "update")
            ]
            at = path.stat().st_mtime_ns
            before = at < min(
                writes.stat().st_mtime_ns,
                (writes.parent / "choice.json").stat().st_mtime_ns,
            )
            strict = (
                before
                and at < (parent_cell(source, w) / "scores.json").stat().st_mtime_ns
            )
            probabilities = dict(
                forecast["probabilities"][name],
                native=observed["choice"]["native"],
                continuity=observed["choice"]["source"],
                fixed_update=1.0,
                fixed_temporal=0.0,
                equal=0.5,
            )
            results.append(
                {
                    "name": name,
                    "dialogue": w["dialogue"],
                    "before": before,
                    "strict": strict,
                    "writer_tokens": costs,
                    "p": probabilities,
                    "rows": observed["rows"],
                }
            )
        if not files:
            continue
        overall = panel(results)
        fresh = panel([r for r in results if r["strict"]])
        passed = False
        if len(files) == len(later) and overall["questions"] and fresh["questions"]:
            f = overall["f1"]
            passed = (
                overall["questions"] >= 0.9 * eligible_arrived
                and f["credit"] >= max(f["native"], f["pooled_return"]) + 0.02
                and f["credit"] > max(f["pooled_credit"], f["split_return"], f["rate"])
                and sum(
                    g["f1"]["credit"] > g["f1"]["rate"]
                    for g in overall["by_history"].values()
                )
                >= 4
                and fresh["f1"]["credit"] > fresh["f1"]["native"]
            )
        report = {
            "forecast": file_hash(path),
            "files": files,
            "eligible_arrived_questions": eligible_arrived,
            "registered_questions": 209,
            "all": overall,
            "before_update": panel([r for r in results if r["before"]]),
            "strict": fresh,
            "passed": passed,
        }
        register(
            path.parent / f"through-{len(files):02d}-{digest(report)[:12]}.json", report
        )
        if len(files) == len(later):
            register(path.parent / "summary.json", report)


def run(forecast, study, corpus):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one bounded GPU for policy fitting")
    torch.cuda.set_per_process_memory_fraction(0.06)
    source = forecast.parent
    out = source.parent / "temporal-update-audit/controller"
    out.mkdir(exist_ok=True, parents=True)
    reg = out / "registration.json"
    if not reg.exists():
        register(
            reg,
            {
                "plan": file_hash(PLAN),
                "forecast": file_hash(forecast),
                "pairs": {
                    p.parent.name: file_hash(p)
                    for p in sorted(source.glob("conv-*/pair.json"))
                },
                "audits": {
                    p.parent.name: file_hash(p)
                    for p in sorted(out.parent.glob("conv-*/result.json"))
                },
            },
        )
    saved = read(reg)
    if (
        saved["plan"] != file_hash(PLAN)
        or saved["forecast"] != file_hash(forecast)
        or any(
            file_hash(source / n / "pair.json") != h for n, h in saved["pairs"].items()
        )
        or any(
            file_hash(out.parent / n / "result.json") != h
            for n, h in saved["audits"].items()
        )
    ):
        raise ValueError("Changed controller registration")
    locked = read(forecast)
    states = state_table(source, out, locked, study, corpus)
    fit(source, out, locked, states)
    score(source, out, locked)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("forecast", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    run(a.forecast, a.study, a.corpus)
