"""Compute window-level hindsight bounds on the actually measured action space."""

import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.controller import file_hash, read, register
from src.evaluation_cache import digest


def window_capacity(window, *, directional=False):
    rows = window["rows"]
    if not rows:
        raise ValueError("Empty measured window")
    if "focal_probabilities" in window:
        mode = "focal"
        ps = [window["focal_probabilities"]]
        means = list(map(mean, zip(*window["endpoints"], strict=True)))
        if len(means) != 2 or len(window["endpoints"]) != len(rows):
            raise ValueError("Invalid pair")

        def mix(probs):
            return (1 - probs[0]) * means[0] + probs[0] * means[1]
    else:
        mode = "joint"
        ps = [window["probabilities"][str(i)] for i in range(3)]
        if set(window["cells"]) != {"00", "01", "10", "11"} or any(
            len(v) != len(rows) for v in window["cells"].values()
        ):
            raise ValueError("Expected all four measured paths")
        means = [mean(window["cells"][k]) for k in ("00", "01", "10", "11")]

        def mix(probs):
            p, l, r = probs
            return (1 - p) * ((1 - l) * means[0] + l * means[1]) + p * (
                (1 - r) * means[2] + r * means[3]
            )

    scores = {
        arm: mix([node[arm] for node in ps]) for arm in ("raw", "native", "credit")
    }
    if any(
        not math.isclose(v, mean(r["f1"][arm] for r in rows), abs_tol=1e-10)
        for arm, v in scores.items()
    ):
        raise ValueError("Measured F1 disagrees with shared-window probabilities")

    def interval(p):
        if not 0 < p < 1:
            raise ValueError("Invalid raw probability")
        return [1 / (1 + (1 - p) * math.exp(-z) / p) for z in (-2.0, 2.0)]

    corners = itertools.product(*(interval(node["raw"]) for node in ps))
    best = max(mix(p) for p in corners)
    if scores["credit"] > best + 1e-10 or best > max(means) + 1e-10:
        raise ValueError("Actor outside its bounded logit family")
    limits = {}
    if directional:
        support = []
        for node in ps:
            raw, credit = node["raw"], node["credit"]
            lower, upper = interval(raw)
            support.append(
                (raw, upper if credit > raw else lower if credit < raw else raw)
            )
        limits["same_direction"] = max(mix(x) for x in itertools.product(*support))
        limits["shrink_only"] = max(
            mix(x)
            for x in itertools.product(*[(node["raw"], node["credit"]) for node in ps])
        )
        if not (
            scores["credit"] <= limits["shrink_only"] + 1e-10
            and limits["shrink_only"] <= limits["same_direction"] + 1e-10
            and limits["same_direction"] <= best + 1e-10
        ):
            raise ValueError("Nested directional bounds disagree")
    return {
        "window": window["window"],
        "history": window["dialogue"],
        "mode": mode,
        "questions": len(rows),
        **scores,
        **limits,
        "bounded": best,
        "unbounded": max(means),
    }


def summarize_capacity(trial, out=None, *, directional=False):
    out = trial / "capacity" if out is None else out
    out.mkdir(exist_ok=True)
    if directional:
        plan = Path(__file__).with_name("protocols") / "query_direction_capacity.json"
        register(out / "protocol.json", {"plan": file_hash(plan)})
    for label in ("early", "calibration", "donor", "later", "replica", "long-prefix"):
        files = sorted((trial / label).glob("conv-*.json"))
        if not files:
            continue
        hashes = {f.name: file_hash(f) for f in files}
        windows = [window_capacity(read(f), directional=directional) for f in files]
        grouped = defaultdict(list)
        for w in windows:
            grouped[w["history"]].append(w)

        def aggregate(ws):
            n = sum(w["questions"] for w in ws)
            return {
                "n": n,
                **{
                    k: sum(w[k] * w["questions"] for w in ws) / n
                    for k in (
                        ("native", "raw", "credit", "bounded", "unbounded")
                        + (("shrink_only", "same_direction") if directional else ())
                    )
                },
            }

        register(
            out / f"{label}-{len(files):02d}.json",
            {
                "sources": hashes,
                "fingerprint": digest(hashes),
                "windows": windows,
                "all": aggregate(windows),
                "by_history": {h: aggregate(ws) for h, ws in grouped.items()},
            },
        )
