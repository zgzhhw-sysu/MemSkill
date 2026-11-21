"""Bind earlier root feedback to its registered window and complete suffix replay."""

from collections import defaultdict
from pathlib import Path

from experiments.controller.factorial_rollout import read
from experiments.controller.matched_span_data import early_sources
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "matched_root_credit.json"


def root_units(base, study, corpus):
    earlier, nodes = early_sources(base, study, corpus, pivot=0)
    out = earlier.parent.parent / "legacy-root-symmetry"
    reg = out / "registration.json"
    summary = read(out / "summary.json")
    jobs = {j["id"]: j for j in read(reg)["jobs"]}
    if (
        read(reg)["terminal_registration"] != file_hash(earlier)
        or summary["registration"] != file_hash(reg)
        or summary["outcomes"] != read(out / "outcomes-complete.json")
        or set(summary["cases"]) != set(jobs)
        or set(summary["outcomes"]) != set(jobs)
    ):
        raise ValueError("Root outcomes do not cover the registered population")
    values = defaultdict(float)
    for key, job in jobs.items():
        f = out / "outcomes" / (key + ".json")
        if file_hash(f) != summary["outcomes"][key]:
            raise ValueError("Changed root outcome")
        row = read(f)
        expected = job["signed_credit"] if row["status"] == "confirmed" else 0.0
        if (
            row.get("credit", 0.0) != expected
            or job["pivot"] != 0
            or job["window"] not in nodes
        ):
            raise ValueError("Changed root outcome or training source")
        values[job["window"]] += expected
    root = out.parent / "matched-span-rank"
    prereg = root / "root-augmented/registration.json"
    register(
        prereg,
        {
            "plan": file_hash(PLAN),
            "root_registration": file_hash(reg),
            "parent_forecast": file_hash(root / "expanded/forecast.json"),
        },
    )
    return [
        {
            "history": n.rsplit("_s", 1)[0],
            "stage": 0,
            "background": other,
            "node": n,
            "questions": count,
            "credit": values[name] / count,
        }
        for n, (_, _, _, name, other, count) in sorted(nodes.items())
    ]
