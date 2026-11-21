"""Measure each possible direct label's effect and bound the registered-panel gain."""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.skill_graph_policy import project, ridge
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited


def credit_shift(fold, donors, case, targets, tensor):
    window, idx, value = case
    if window not in fold["windows"]:
        return tensor([[0.0, 0.0, 0.0]] * len(targets))
    pool = [donors[k] for k in fold["windows"]]

    def design(v):
        return ((tensor(v) - tensor(fold["center"])) / tensor(fold["sd"])).clamp(
            -3, 3
        ) - tensor(fold["shift"])

    x = design([r["features"] for r in pool])
    at = fold["windows"].index(window)
    d = pool[at]
    (point,) = [q for q in d["points"] if q["index"] == idx]
    y = tensor([point["scores"]])
    edges = tensor([point["edges"]])
    old = project(y, edges)
    edges[0, 2] = value
    delta = x.new_zeros((len(x), 3))
    delta[at] = ((project(y, edges) - old) / len(d["points"]))[0]
    w = tensor(
        [
            1
            / (len(fold["histories"]) * sum(k["history"] == r["history"] for k in pool))
            for r in pool
        ]
    )
    return 0.5 * design(targets) @ ridge(x, delta, w)


def response_surface(base, path, directory, cases):
    root = path.parent.parent
    model = path.parent / "models.json"
    receipt = directory / "surface.json"
    identity = {"forecast": file_hash(path), "models": file_hash(model), "cases": cases}
    if receipt.exists():
        if read(receipt)["inputs"] != identity:
            raise ValueError("Changed attribution surface")
        return read(receipt)
    packet = read(path)
    models = read(model)
    donors = {w["window"]: w for w in models["donors"]}
    semantic = root.parent.parent
    rows = read(semantic / "panel.json")["windows"]
    xs = {
        (r["dialogue"], r["session"]): r
        for r in read(semantic / "forecast.json")["targets"]
    }
    geo = read(base / "source-compression/matched-span-rank/features.json")["features"]
    later = {}
    for h, fold in models["folds"].items():
        names = []
        features = []
        for w in rows:
            if w["dialogue"] == h:
                n = w["start"] + 1
                names.append(f"{h}_s{n - 1}")
                features.append([xs[h, n]["offset"], geo[f"{h}_s{n}"][5]])
        with limited():

            def tensor(v):
                return torch.as_tensor(v, device="cuda", dtype=torch.float64)

            shifts = torch.stack(
                [credit_shift(fold, donors, case, features, tensor) for case in cases]
            )
            for i, name in enumerate(names):
                anchor = packet["windows"][name]["scores"]["graph"]
                later[name] = {
                    "anchor": anchor["credit"],
                    "raw": anchor["raw"],
                    "shifts": shifts[:, i].tolist(),
                }
    register(receipt, {"inputs": identity, "windows": later})
    return read(receipt)


def gain_range(delta, values):
    """Sharp interval for the weighted difference over actions with unknown F1 in [0,1]."""
    if abs(float(delta.sum())) > 1e-10:
        raise ValueError("Contrasts must conserve probability")
    if values is None:
        half = float(delta.abs().sum()) / 2
        return [-half, half]
    val = float(delta @ values)
    return [val, val]


def summarize(windows, masks, actual, base_arm):
    n = sum(w["n"] for w in windows)
    means = {}
    for mask in masks:
        bounds = [
            sum(w["n"] * w["gains"][mask][base_arm][j] for w in windows) / n
            for j in (0, 1)
        ]
        means[mask] = bounds
    feasible = [
        m
        for m in masks
        if all(v is None or int(k) == v for k, v in zip(m, actual, strict=True))
    ]
    return {
        "questions": n,
        "matched_questions": sum(w["n"] for w in windows if w["matched"]),
        "matched_windows": sum(w["matched"] for w in windows),
        "windows": len(windows),
        "bounds_by_mask": means,
        "feasible_masks": feasible,
        "feasible_bounds": [
            min(means[m][0] for m in feasible),
            max(means[m][1] for m in feasible),
        ],
    }


def run(base):
    late = base / "late-pairs"
    semantic = late / "semantic-writer"
    parent = semantic / "three-writer"
    root = parent / "skill-graph"
    out = root / "attribution"
    out.mkdir(exist_ok=True)
    audit = base / "source-compression/activity-relation-repairs/0"
    prep = read(audit / "prepared-complete.json")
    cases = []
    actual = []
    for name, sha in sorted(prep.items()):
        f = audit / "prepared" / (name + ".json")
        if file_hash(f) != sha:
            raise ValueError("Changed candidate manifest")
        ready = read(f)
        done = audit / "outcomes" / f.name
        jobs = {}
        if done.exists():
            o = read(done)
            if o["inputs"]["parent"] != sha:
                raise ValueError("Unbound adjudication")
            jobs = {j["original_index"]: j for j in o["jobs"]}
        for item in ready["jobs"]:
            if item["result"]["status"] != "prepared":
                continue
            j = item["job"]
            idx = j["qa"]["original_index"]
            signed = j["signed_credit"]
            cases.append([name, idx, signed])
            result = jobs.get(idx)
            if result and (
                result["raw"] != signed
                or result["credit"]
                != (signed if result["status"] == "confirmed" else 0)
            ):
                raise ValueError("Wrong sign of adjudicated effect")
            actual.append(int(result["status"] == "confirmed") if result else None)
    if len(cases) != 3:
        raise ValueError("Changed registered direct candidates")
    surface = response_surface(base, root / "masked/forecast.json", out, cases)
    complete = root / "complete/forecast.json"
    final = read(complete)["windows"] if complete.exists() else {}
    if final and None in actual:
        raise ValueError("Final forecast before completed adjudication")
    masks = ["".join(p) for p in product("01", repeat=len(cases))]
    rows = []
    hashes = {}
    split = read(late / "selection-labels/split.json")["windows"]
    coverage = defaultdict(Counter)
    for w in read(semantic / "panel.json")["windows"]:
        k = f"{w['dialogue']}_s{w['start']}"
        f = parent / (k + ".json")
        result = None
        state = "pending"
        if f.exists():
            p = read(f)
            hashes[k] = file_hash(f)
            if p["questions"] != len(w["questions"]) or p["semantic"] != file_hash(
                semantic / k / "pair.json"
            ):
                raise ValueError("Changed intervention")
            state = "matched" if p["matched"] else p["reason"]
            if p["matched"]:
                if [r["index"] for r in p["rows"]] != w["questions"]:
                    raise ValueError("Changed question order")
                if [r["category"] for r in p["rows"]] != w["types"]:
                    raise ValueError("Changed registered question categories")
                result = [r["f1"] for r in p["rows"]]
        g = surface["windows"][k]
        gains = {}
        with limited():
            t = torch.tensor(g["anchor"], dtype=torch.float64, device="cuda")
            raw = t.new_tensor(g["raw"])
            changes = t.new_tensor(g["shifts"])
            values = t.new_tensor(result).mean(0) if result is not None else None
            for mask in masks:
                u = t + t.new_tensor(list(map(int, mask))) @ changes
                if (
                    final
                    and mask == "".join(map(str, actual))
                    and not torch.allclose(
                        u,
                        t.new_tensor(final[k]["scores"]["graph"]["credit"]),
                        atol=1e-11,
                        rtol=0,
                    )
                ):
                    raise ValueError("Realized model disagrees with label attribution")
                gains[mask] = {
                    a: gain_range(
                        0.8 * ((u / 0.05).softmax(0) - (v / 0.05).softmax(0)), values
                    )
                    for a, v in (("masked", t), ("raw", raw))
                }
        part = "donor" if k in split["donor"] else "evaluation"
        for category in w["types"]:
            coverage[f"{part}/{w['dialogue']}"][state] += 1
            coverage[f"{part}/category-{category}"][state] += 1
        if split[part][k] != w["questions"]:
            raise ValueError("Unexpected split")
        rows.append(
            {
                "window": k,
                "n": len(w["questions"]),
                "partition": part,
                "matched": result is not None,
                "gains": gains,
            }
        )
    register(
        out / f"coverage-{len(hashes):02d}.json",
        {"sources": hashes, "question_counts": dict(coverage)},
    )
    pattern = "".join("x" if v is None else str(v) for v in actual)
    stats = {
        k: {a: summarize(ws, masks, actual, a) for a in ("masked", "raw")}
        for k, ws in (
            ("all", rows),
            ("evaluation", [r for r in rows if r["partition"] == "evaluation"]),
            (
                "matched_evaluation",
                [r for r in rows if r["partition"] == "evaluation" and r["matched"]],
            ),
        )
        if ws
    }
    register(
        out / f"through-{len(hashes):02d}-{pattern}.json",
        {
            "forecast": file_hash(out / "surface.json"),
            "sources": hashes,
            "cases": cases,
            "observed": actual,
            "panels": stats,
            "windows": rows,
        },
    )
    if final:
        register(
            out / "verified-complete.json",
            {
                "surface": file_hash(out / "surface.json"),
                "realized_mask": pattern,
                "forecast": file_hash(complete),
                "windows": len(rows),
            },
        )

    if final and len(hashes) == len(rows):
        register(
            out / "summary.json",
            {
                "complete": file_hash(out / "verified-complete.json"),
                "bounds": file_hash(out / f"through-{len(hashes):02d}-{pattern}.json"),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    a = p.parse_args()
    gpu()
    run(a.base)
