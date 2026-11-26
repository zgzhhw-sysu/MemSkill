"""Pair the same relation-versus-temporal contrast under two measured prefix actions."""

import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.paired_grounding import window
from src.credit_assignment import _digest, recovered
from src.eval.locomo import LoCoMoEvaluator
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "relation_prefix_interaction.json"
PREFIX_NAMES = ("capture_temporal_context", "capture_activity_details")


def preregister(dest, parent, manifest, identity, source_hash):
    keys = [f"{sid}_s{n}" for sid, n in manifest["windows"]]
    if len(set(keys)) != 12:
        raise ValueError("Unexpected number of unique training windows")
    reg = dest / "registration.json"
    register(
        reg,
        {
            "plan": file_hash(PLAN),
            "identity": identity,
            "source_forecast": source_hash,
            "temporal_prefix_summary": file_hash(parent / "summary.json"),
            "temporal_prefix_pairs": {
                k: file_hash(parent / k / "pair.json") for k in keys
            },
        },
    )
    return file_hash(reg)


def metrics(rows):
    # Both prefix observations belong to the same question. Choose once per window.
    count = len(rows)
    scores = [
        [mean(r[i][a] for r in rows) for a in ("temporal", "relation")]
        for i in range(2)
    ]
    first, second = [r - t for t, r in scores]
    shared = max(map(mean, zip(*scores, strict=True)))
    conditional = mean(max(a) for a in scores)
    return {
        "questions": count,
        "temporal_prefix_delta": first,
        "activity_prefix_delta": second,
        "interaction": second - first,
        "shared_oracle": shared,
        "conditional_oracle": conditional,
        "oracle_gain": conditional - shared,
        "strict_flip": min(first, second) < -1e-12 and max(first, second) > 1e-12,
    }


def combine(cases):
    matched = [c for c in cases if c["matched"]]
    count = sum(c["questions"] for c in matched)
    result = {"attempted": len(cases), "matched": len(matched), "questions": count}
    if count:
        for k in (
            "temporal_prefix_delta",
            "activity_prefix_delta",
            "interaction",
            "shared_oracle",
            "conditional_oracle",
            "oracle_gain",
        ):
            result[k] = sum(c[k] * c["questions"] for c in matched) / count
        result.update(
            strict_flips=sum(c["strict_flip"] for c in matched),
            identical_banks=sum(c["same_prefix"] for c in matched),
        )
    return result


def one(parent, dest, archive, data, sid, n, registered):
    key = f"{sid}_s{n}"
    out = [(p / key) for p in (parent, dest)]
    pairs = [read(d / "pair.json") for d in out]
    hashes = [file_hash(d / "pair.json") for d in out]
    if registered["temporal_prefix_pairs"][key] != hashes[0] or pairs[1][
        "registration"
    ] != file_hash(dest / "registration.json"):
        raise ValueError("Changed registered pair")
    spans = []
    for i, (d, p) in enumerate(zip(out, pairs, strict=True)):
        (trace,) = (archive / key / f"{i}0" / "traces").glob("*.jsonl")
        if (
            p["source_trace"] != file_hash(trace)
            or p["dialogue"] != sid
            or p["forecast"] != registered["source_forecast"]
        ):
            raise ValueError("Changed source trace or forecast")
        prefix, end = [
            r
            for r in map(json.loads, trace.read_bytes().splitlines())
            if r["kind"] == "span"
        ]
        if prefix["selected_skills"][-1]["name"] != PREFIX_NAMES[i]:
            raise ValueError("Unexpected actual prefix action")
        if p["matched"]:
            choice = read(d / "choice.json")
            if (
                choice != p["choice"]
                or read(d / "write.json")["choice"] != file_hash(d / "choice.json")
                or choice["fixed_prefix"]
                != _digest(list(restore_text([prefix]).items()))
            ):
                raise ValueError("Changed measured write or prefix")
        spans.append((prefix, end))
    base = {"window": key, "history": sid, "pairs": hashes, "matched": False}
    if not all(p["matched"] for p in pairs):
        return dict(base, reasons=[p.get("reason") for p in pairs])
    if (
        any(
            a["source"] != b["source"] or a["source_turns"] != b["source_turns"]
            for a, b in zip(*spans, strict=True)
        )
        or pairs[0]["source_forecast"] != pairs[1]["source_forecast"]
    ):
        raise ValueError("Different source text or evidence turns")
    if (
        spans[0][0]["selected_skills"][:-1] != spans[1][0]["selected_skills"][:-1]
        or spans[0][1]["selected_skills"] != spans[1][1]["selected_skills"]
        or pairs[0]["choice"]["selected"] != pairs[1]["choice"]["selected"]
    ):
        return dict(base, reasons=["different_fixed_skills"])
    ev = LoCoMoEvaluator(SimpleNamespace())
    turns = set(spans[0][1]["source_turns"])
    qa = window(data[sid], n, 2)["qa"]
    values = []
    candidates = []
    for q, a, b in zip(qa, pairs[0]["rows"], pairs[1]["rows"], strict=True):
        if any(
            (r["original_index"], r["category"]) != (q["original_index"], q["category"])
            for r in (a, b)
        ):
            raise ValueError("Mismatched question")
        values.append(
            [{k: r[k]["f1"] for k in ("temporal", "relation")} for r in (a, b)]
        )
        if turns.intersection(q["evidence"]):
            for i, r in enumerate((a, b)):
                for weak, strong in (
                    ("temporal", "relation"),
                    ("relation", "temporal"),
                ):
                    if recovered(r[weak], r[strong], ev.get_ground_truth(q)):
                        candidates.append(
                            {"prefix": i, "question": q["original_index"], "weak": weak}
                        )
    if not values:
        return dict(base, reasons=["no_shared_questions"])
    return dict(
        base,
        matched=True,
        same_prefix=list(restore_text([spans[0][0]]).values())
        == list(restore_text([spans[1][0]]).values()),
        candidates=candidates,
        **metrics(values),
    )


def report(parent, dest, archive, manifest, data, count):
    reg = read(dest / "registration.json")
    cases = [
        one(parent, dest, archive, data, sid, n, reg)
        for sid, n in manifest["windows"][:count]
    ]
    out = dest / "contrasts"
    out.mkdir(exist_ok=True)
    register(
        out / f"through-{count:02d}.json",
        {
            "registration": file_hash(dest / "registration.json"),
            "cases": cases,
            "overall": combine(cases),
            "by_history": {
                sid: combine([c for c in cases if c["history"] == sid])
                for sid in sorted({c["history"] for c in cases})
            },
        },
    )
