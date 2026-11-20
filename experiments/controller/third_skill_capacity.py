"""Compare the attainable returns of two and three actions at matched terminal decisions."""

import argparse
import math
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import mean

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "matched_third_skill_capacity.json"


def bounds(returns, probabilities):
    if (
        len(returns) != 3
        or any(not 0 <= v <= 1 for v in returns)
        or any(p <= 0 for p in probabilities)
        or not math.isclose(sum(probabilities), 1)
    ):
        raise ValueError(
            "Expected three measured bounded skill returns and positive propensities"
        )
    t, a, r = returns
    pt, pa, pr = probabilities
    act = [pa / pt * math.exp(d) for d in (-2, 2)]
    rel = [pr / pt * math.exp(d) for d in (-2, 2)]
    two = max((t + x * a) / (1 + x) for x in act)
    three = max((t + x * a + y * r) / (1 + x + y) for x, y in product(act, rel))
    return {
        "free_two": max(t, a),
        "free_three": max(returns),
        "bounded_two": two,
        "bounded_three": three,
        "bounded_optional_relation": max(two, three),
    }


def aggregate(rows):
    if not rows:
        return {"questions": 0, "windows": 0}
    n = sum(w["n"] for w in rows)

    def weighted(fn):
        return sum(fn(w) * w["n"] for w in rows) / n

    fixed = [weighted(lambda w, j=j: w["means"][j]) for j in range(3)]
    return {
        "questions": n,
        "windows": len(rows),
        "fixed": fixed,
        "best_fixed_two": max(fixed[:2]),
        "best_fixed_three": max(fixed),
        "strict_relation_wins": sum(w["means"][2] > max(w["means"][:2]) for w in rows),
        **{k: weighted(lambda w, k=k: w["bounds"][k]) for k in rows[0]["bounds"]},
    }


def run(base):
    late = base / "late-pairs"
    parent = late / "semantic-writer/three-writer"
    out = parent / "capacity"
    out.mkdir(exist_ok=True)
    reg = parent / "registration.json"
    panel = late / "semantic-writer/panel.json"
    split = late / "selection-labels/split.json"
    specs = read(panel)["windows"]
    ids = {f"{w['dialogue']}_s{w['start']}": w for w in specs}
    if (
        len(ids) != 25
        or sum(len(w["questions"]) for w in specs) != 209
        or read(reg)["panel"] != file_hash(panel)
    ):
        raise ValueError("Different three-action population")
    partition = read(split)["windows"]
    files = {p.stem: p for p in parent.glob("conv-*.json")}
    if not files.keys() <= ids.keys():
        raise ValueError("Unregistered terminal window")
    rows = []
    reasons = Counter()
    for name, path in sorted(files.items()):
        v = read(path)
        spec = ids[name]
        semantic = late / "semantic-writer" / name / "pair.json"
        if (
            file_hash(semantic) != v["semantic"]
            or v["name"] != name
            or v["dialogue"] != spec["dialogue"]
        ):
            raise ValueError("Physical source changed")
        if not v["matched"]:
            reasons[v["reason"]] += 1
            continue
        activity = (
            base / "source-compression/retrieval-competition/pairs" / (name + ".json")
        )
        if (
            file_hash(activity) != v["activity"]
            or [q["index"] for q in v["rows"]] != spec["questions"]
        ):
            raise ValueError("Mismatched questions or third skill")
        (partition_name,) = [k for k in ("donor", "evaluation") if name in partition[k]]
        means = list(map(mean, zip(*[q["f1"] for q in v["rows"]], strict=True)))
        rows.append(
            {
                "window": name,
                "history": v["dialogue"],
                "panel": partition_name,
                "n": len(v["rows"]),
                "means": means,
                "bounds": bounds(means, v["probabilities"]["native"]),
            }
        )
    register(
        out / f"through-{len(files):02d}.json",
        {
            "plan": file_hash(PLAN),
            "registration": file_hash(reg),
            "split": file_hash(split),
            "sources": {n: file_hash(p) for n, p in files.items()},
            "unmatched": dict(reasons),
            "rows": rows,
            "all": aggregate(rows),
            "by_partition": {
                k: aggregate([w for w in rows if w["panel"] == k])
                for k in ("donor", "evaluation")
            },
            "by_history": {
                h: aggregate([w for w in rows if w["history"] == h])
                for h in sorted({w["dialogue"] for w in specs})
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    run(p.parse_args().base)
