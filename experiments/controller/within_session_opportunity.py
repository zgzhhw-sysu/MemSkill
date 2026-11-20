"""Quantify opposing whole-session treatment effects by local evidence position."""

import argparse
from collections import Counter
from pathlib import Path

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "within_session_opportunity.json"


def group_questions(history, session, observations, qa):
    turns = history[f"session_{session}"]
    ids = {t["dia_id"]: i for i, t in enumerate(turns)}
    if len(ids) != len(turns):
        raise ValueError("Duplicate turn")
    cut = min(
        range(2, len(ids), 2), key=lambda i: (abs(2 * i - len(ids)), i), default=None
    )
    rows = []
    for idx, low, high in observations:
        evidence = set(qa[idx]["evidence"])
        if not evidence:
            raise ValueError("Invalid evidence question")
        inside = evidence & ids.keys()
        sides = (
            set()
            if cut is None
            else {"first" if ids[k] < cut else "second" for k in inside}
        )
        group = (
            "short"
            if cut is None
            else "elsewhere"
            if not sides
            else next(iter(sides))
            if len(sides) == 1
            else "both"
        )
        rows.append(
            {
                "index": idx,
                "group": group,
                "wholly_in_session": inside == evidence,
                "temporal": low,
                "activity": high,
                "evidence": sorted(evidence),
            }
        )
    return cut, rows


def mass(rows, require_whole=False):
    gains = {
        k: [
            r["activity"] - r["temporal"]
            for r in rows
            if r["group"] == k and (not require_whole or r["wholly_in_session"])
        ]
        for k in ("first", "second")
    }
    a, b = map(sum, gains.values())
    return {
        "first": len(gains["first"]),
        "second": len(gains["second"]),
        "supported": all(gains.values()),
        "opposite": a * b < -1e-12,
        "sum_differences": [a, b],
        "extra": max(0, a) + max(0, b) - max(0, a + b),
    }


def summarize(rows):
    total = sum(len(row["rows"]) for row in rows)
    metric = {}
    for k in ("all_evidence_links", "all_links_in_pivot"):
        metric[k] = {
            "both_groups": sum(v[k]["supported"] for v in rows),
            "opposite": sum(v[k]["opposite"] for v in rows),
            "opposite_windows": sorted({v["window"] for v in rows if v[k]["opposite"]}),
            "question_aware_extra_f1": sum(v[k]["extra"] for v in rows) / total,
        }
    return {
        "comparisons": len(rows),
        "questions_with_repeats": total,
        "distinct_questions": len(
            {(v["dialogue"], r["index"]) for v in rows for r in v["rows"]}
        ),
        "placement_counts": dict(Counter(r["group"] for v in rows for r in v["rows"])),
        **metric,
    }


def evaluate(base, study, corpus):
    train = read(study / "protocol.json")
    late = base / "late-pairs"
    data = {c["sample_id"]: c for c in read(corpus) if c["sample_id"] in train["train"]}
    cohort = base / "source-compression/reader-slot-credit/cohort.json"
    split = read(late / "selection-labels/split.json")["windows"]
    if train["input_sha256"] != file_hash(corpus) or len(data) != 6:
        raise ValueError("Wrong training split")
    src = read(cohort)["contrasts"]
    if len(src) != 60 or sum(len(v["rows"]) for v in src) != 452:
        raise ValueError("Changed original contrasts")
    groups = {"early": [], "donor": [], "later": []}

    def append(label, sid, name, start, j, obs, receipt, bits):
        cut, rows = group_questions(
            data[sid]["conversation"], start + j, obs, data[sid]["qa"]
        )
        groups[label].append(
            {
                "dialogue": sid,
                "window": name,
                "bits": bits,
                "pivot": j,
                "cut_after_turns": cut,
                "rows": rows,
                "receipt": receipt,
                "all_evidence_links": mass(rows),
                "all_links_in_pivot": mass(rows, True),
            }
        )

    for v in src:
        sid, n = v["window"].rsplit("_s", 1)
        bits = v["bits"]
        (j,) = [i for i, (a, b) in enumerate(zip(*bits, strict=True)) if a != b]
        if j != v["stage"] or bits[0][j] != "0" or bits[1][j] != "1":
            raise ValueError("Wrong pivot")
        append(
            "donor" if v["window"] in split["donor"] else "early",
            sid,
            v["window"],
            int(n),
            j,
            [(r["index"], r["low"]["f1"], r["high"]["f1"]) for r in v["rows"]],
            v["observation"],
            bits,
        )
    for spec in read(late / "cohort.json")["assignments"]:
        sid = spec["dialogue"]
        name = f"{sid}_s{spec['start']}"
        marker = late / "outcomes" / name / "pair.json"
        if name not in split["evaluation"] or not marker.exists():
            continue
        v = read(marker)
        obs = [
            (r["original_index"], r["temporal"], r["activity"]) for r in v["questions"]
        ]
        if [r[0] for r in obs] != split["evaluation"][name]:
            raise ValueError("Changed later questions")
        append(
            "later",
            sid,
            name,
            spec["start"],
            spec["pivot"],
            obs,
            file_hash(marker),
            v["paths"],
        )
    out = base / "source-compression/witness-passages/evidence-advantage/within-session"
    out.mkdir(parents=True, exist_ok=True)
    for label, rows in groups.items():
        if not rows:
            continue
        report = {
            "plan": file_hash(PLAN),
            "cohort": file_hash(cohort),
            "corpus": train["input_sha256"],
            "rows": rows,
            "all": summarize(rows),
            "by_history": {
                sid: summarize([r for r in rows if r["dialogue"] == sid])
                for sid in sorted({v["dialogue"] for v in rows})
            },
        }
        register(out / f"{label}-{len(rows):02d}.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    args = p.parse_args()
    evaluate(args.base, args.study, args.corpus)
