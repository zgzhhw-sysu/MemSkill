"""Bound the value of taking or declining individual frozen credit proposals."""

from itertools import product
from statistics import mean

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited


def state_ranges(x, native, folds, recorded):
    from experiments.controller.selective_span_policy import MASS, matches, prob

    x = x[None]
    empty = torch.zeros(len(folds), device=x.device, dtype=x.dtype)

    def actions(rules, otherwise):
        active = matches(x, [r["region"] for r in rules])[:, 0]
        return torch.where(
            active, x.new_tensor([r["action"] for r in rules]), otherwise
        )

    raw = actions([m["base"] for m in folds], empty)
    rules = [m["offers"]["credit"] for m in folds]
    offer = actions(rules, raw)
    keep = x.new_tensor([r["accepted"] for r in rules]).bool()
    p = x.new_tensor(native)
    base = prob(p, raw)
    unrestricted = prob(p, offer)
    checked = torch.where(keep, unrestricted, base)
    for arm, v in (
        ("credit", checked),
        ("base_rule", base),
        ("unscreened", unrestricted),
    ):
        if abs(float(v.mean()) - recorded[arm]) > 1e-10:
            raise ValueError("Failed to reproduce frozen policy")

    def span(a):
        return [
            float(torch.minimum(a, base).mean()),
            float(torch.maximum(a, base).mean()),
        ]

    return {
        "approved": span(checked),
        "discovered": span(unrestricted),
        "unrestricted": [(1 - MASS) * native, (1 - MASS) * native + MASS],
    }


def measure(base):
    root = base / "source-compression/matched-span-rank"
    trial = root / "selective-rules"
    out = trial / "opportunity"
    out.mkdir(exist_ok=True)
    models = read(trial / "models.json")["folds"]
    geometry = read(root / "features.json")["features"]
    prior = read(base / "late-pairs/cross-window-prefix/manifest.json")["endpoints"]
    histories = {(a["dialogue"], a["end"]): a for a in prior}
    local = read(base / "late-pairs/cohort.json")["assignments"]
    assignment = {f"{a['dialogue']}_s{a['start']}": a for a in local}
    names = ("approved", "discovered", "unrestricted")
    for label in ("early", "calibration", "donor", "later", "replica", "long-prefix"):
        files = sorted((trial / label).glob("conv-*.json"))
        dest = out / f"{label}-{len(files):02d}.json"
        if not files:
            continue
        inputs = {
            "models": file_hash(trial / "models.json"),
            "features": file_hash(root / "features.json"),
            "sources": {f.name: file_hash(f) for f in files},
        }
        if dest.exists():
            if read(dest)["inputs"] != inputs:
                raise ValueError("Changed comparison window")
            continue
        panel = []
        for f in files:
            r = read(f)
            h = r["dialogue"]
            focus = "focal_probabilities" in r
            if label == "long-prefix":
                n = int(f.stem.rsplit("_to", 1)[1])
                bg = histories[h, n]["background"]
                states = [(n, int(bool(bg)), bg[-1] if bg else 0)]
            else:
                n = int(f.stem.rsplit("_s", 1)[1])
                if focus:
                    a = assignment[f.stem]
                    j = a["pivot"]
                    states = [(n + j, j, a["background"] if j else 0)]
                else:
                    states = [(n, 0, 0), (n + 1, 1, 0), (n + 1, 1, 1)]
            ranges = []
            with limited():
                for i, (session, j, b) in enumerate(states):
                    resp = (
                        r["focal_probabilities"]
                        if focus
                        else r["probabilities"][str(i)]
                    )
                    x = torch.tensor(
                        [*geometry[f"{h}_s{session}"], b],
                        dtype=torch.float64,
                        device="cuda",
                    )
                    ranges.append(
                        state_ranges(x, resp["native"], models[h][str(j)], resp)
                    )
            if focus:
                lo, hi = map(mean, zip(*r["endpoints"], strict=True))

                def value(policy, lo=lo, hi=hi):
                    return lo + policy[0] * (hi - lo)
            else:
                table = [mean(r["cells"][k]) for k in ("00", "01", "10", "11")]

                def value(policy, c=table):
                    first, left, right = policy
                    return (1 - first) * ((1 - left) * c[0] + left * c[1]) + first * (
                        (1 - right) * c[2] + right * c[3]
                    )

            bounds = {
                k: max(value(policy) for policy in product(*(d[k] for d in ranges)))
                for k in names
            }
            actual = {
                k: mean(q["f1"][k] for q in r["rows"])
                for k in ("credit", "base_rule", "native", "terminal")
            }
            if (
                not actual["credit"] <= bounds["approved"] + 1e-10
                or not bounds["approved"] <= bounds["discovered"] + 1e-10
                or not bounds["discovered"] <= bounds["unrestricted"] + 1e-10
            ):
                raise ValueError("Invalid nested opportunity bounds")
            panel.append(
                {
                    "window": f.stem,
                    "history": h,
                    "n": len(r["rows"]),
                    **bounds,
                    **actual,
                    "ranges": ranges,
                }
            )
        count = sum(r["n"] for r in panel)
        register(
            dest,
            {
                "inputs": inputs,
                "windows": panel,
                "questions": count,
                "means": {
                    k: sum(r["n"] * r[k] for r in panel) / count
                    for k in (*names, "credit", "base_rule", "native", "terminal")
                },
            },
        )
