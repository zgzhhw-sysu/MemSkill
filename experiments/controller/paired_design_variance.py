"""Enumerate the four randomized path pairs on complete windows before using a control variate."""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance

from experiments.controller import file_hash, read, register
from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.late_pair_trial import paths
from experiments.controller.retrieval_competition import mass

METRICS = ("credit", "credit-native", "credit-rate", "credit-return_only")


def estimates(probabilities, scores, constant):
    def weight(bits, name):
        first, sep, second = name.partition("-")
        return mass(bits, probabilities, first) - (
            mass(bits, probabilities, second) if sep else 0.0
        )

    result = {}
    for name in METRICS:
        target_mass = 0.0 if "-" in name else 1.0
        exact = sum(weight(b, name) * y for b, y in scores.items())
        different = []
        for pivot, background in ((0, 0), (0, 1), (1, 0), (1, 1)):
            sampled = ["".join(map(str, b)) for b in paths(pivot, background)]
            total = 2 * sum(weight(b, name) for b in sampled)
            ht = 2 * sum(weight(b, name) * scores[b] for b in sampled)
            different.append((ht, ht - constant * (total - target_mass)))
        if not all(
            math.isclose(mean(v), exact, abs_tol=1e-11)
            for v in zip(*different, strict=True)
        ):
            raise ValueError(
                "Estimator does not have the exact four-path design expectation"
            )
        result[name] = different
    return result


def constants(tables, sid):
    by_type = defaultdict(list)
    for donor in tables:
        if donor["dialogue"] == sid:
            continue
        for i, cat in enumerate(donor["types"]):
            by_type[str(cat)].append(mean(x[i] for x in donor["cases"].values()))
    result = {cat: mean(rows) for cat, rows in by_type.items()}
    result["all"] = mean(v for group in by_type.values() for v in group)
    return result


def audit(root):
    fit = read(root / "fit.json")
    tables = fit["training"]
    cal = read(root / "calibration.json")
    if cal["fit"] != file_hash(root / "fit.json"):
        raise ValueError("Different complete tables")
    all_windows = []
    individual = defaultdict(list)
    for t, w in zip(tables, cal["windows"], strict=True):
        if t["window"] != w["window"]:
            raise ValueError("Mismatched window")
        refs = constants(tables, t["dialogue"])
        ps = {int(k): v for k, v in w["probabilities"].items()}
        drawn = []
        for i, cat in enumerate(t["types"]):
            ref = refs.get(str(cat), refs["all"])
            results = estimates(ps, {k: v[i] for k, v in t["cases"].items()}, ref)
            drawn.append(results)
            for name, assignments in results.items():
                individual[name].append(
                    [pvariance(vals) for vals in zip(*assignments, strict=True)]
                )
        group = {}
        for name in METRICS:
            group[name] = [
                pvariance([mean(q[name][a][j] for q in drawn) for a in range(4)])
                for j in (0, 1)
            ]
        all_windows.append(
            {
                "dialogue": t["dialogue"],
                "window": t["window"],
                "questions": len(drawn),
                "variance": group,
            }
        )
    n = sum(w["questions"] for w in all_windows)
    result = {}
    for name in METRICS:
        pooled = [
            sum(w["questions"] * w["variance"][name][j] for w in all_windows) / n
            for j in (0, 1)
        ]
        result[name] = {
            "mean_question_variance": [
                mean(v) for v in zip(*individual[name], strict=True)
            ],
            "question_weighted_mean_window_variance": pooled,
            "windows_with_lower_variance": sum(
                w["variance"][name][1] < w["variance"][name][0] for w in all_windows
            ),
        }
    register(
        root / "design-variance.json",
        {
            "calibration": file_hash(root / "calibration.json"),
            "questions": n,
            "summary": result,
            "windows": all_windows,
        },
    )


def evaluate_pairs(root):
    dest = root / "return-control"
    dest.mkdir(exist_ok=True)
    plan = Path(__file__).with_name("protocols") / "randomized_return_control.json"
    training = read(root / "fit.json")["training"]
    lock = dest / "constants.json"
    register(
        lock,
        {
            "early_fit": file_hash(root / "fit.json"),
            "design_check": file_hash(root / "design-variance.json"),
            "plan": file_hash(plan),
            "constants": {
                d: constants(training, d)
                for d in sorted({r["dialogue"] for r in training})
            },
        },
    )
    anchors = read(lock)["constants"]
    late = root.parent.parent / "late-pairs/outcomes"
    old = scored_at(late, lock.stat().st_mtime_ns)
    register(dest / "preexisting.json", {"constants": file_hash(lock), "scores": old})
    previous = {key.split("/")[0] for key in old}
    rows = []
    for p in sorted((root / "pairs").glob("*.json")):
        w = read(p)
        name = w["window"]
        assigned = read(late / name / "pair.json")
        ps = {int(k): v for k, v in w["probabilities"].items()}
        selected = assigned["paths"]
        total = {k: 2 * sum(mass(bits, ps, k) for bits in selected) for k in ps[0]}
        if any(
            not math.isclose(total[k], assigned["sampling_mass"][k], abs_tol=1e-10)
            for k in ("native", "terminal")
        ):
            raise ValueError("Physical sampling probabilities changed")
        for r in w["rows"]:
            c = anchors[w["dialogue"]].get(
                str(r["type"]), anchors[w["dialogue"]]["all"]
            )
            rows.append(
                {
                    "window": name,
                    "dialogue": w["dialogue"],
                    "index": r["index"],
                    "type": r["type"],
                    "c": c,
                    "focal": r["f1"],
                    "ht": r["ht"],
                    "adjusted": {
                        arm: v - c * (total[arm] - 1) for arm, v in r["ht"].items()
                    },
                }
            )

    def summary(items):
        if not items:
            return {"questions": 0, "windows": 0}

        def means(group):
            return {
                "questions": len(group),
                **{
                    metric: {
                        arm: mean(r[metric][arm] for r in group)
                        for arm in group[0][metric]
                    }
                    for metric in ("ht", "adjusted", "focal")
                },
            }

        return dict(
            **means(items),
            windows=len({r["window"] for r in items}),
            by_dialogue={
                sid: means([r for r in items if r["dialogue"] == sid])
                for sid in sorted({r["dialogue"] for r in items})
            },
            by_type={
                cat: means([r for r in items if r["type"] == cat])
                for cat in sorted({r["type"] for r in items})
            },
        )

    n = len({r["window"] for r in rows})
    register(
        dest / f"through-{n:02d}.json",
        {
            "constants": file_hash(lock),
            "all": summary(rows),
            "prospective": summary([r for r in rows if r["window"] not in previous]),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    audit(root)
    evaluate_pairs(root)
