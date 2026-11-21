"""Fit and evaluate a three-skill controller from question-level verified differences."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.skill_graph_data import registration, training_rows
from experiments.controller.skill_graph_extension import extra_rows
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_skill_margin import aggregate, distributions, score
from experiments.controller.three_writer_credit import three_probabilities
from experiments.controller.turn_key_credit import paired_interval
from experiments.controller.witness_passage_policy import limited


def project(scores, edges):
    """Minimum-norm utilities with the observed differences on each connected component."""
    verified = edges != 0
    oriented = scores[:, [1, 2, 2]] - scores[:, [0, 0, 1]]
    if ((edges - oriented).abs()[verified] > 1e-10).any():
        raise ValueError("Verified comparison disagrees with measured outcomes")
    incidence = scores.new_tensor([[-1, 1, 0], [-1, 0, 1], [0, -1, 1]])
    return torch.where(
        (verified.sum(1) >= 2)[:, None],
        scores - scores.mean(1, keepdim=True),
        (edges @ incidence) / 2,
    )


def ridge(x, y, weight):
    gram = x.T @ (weight[:, None] * x)
    return torch.linalg.solve(
        gram + 0.25 * torch.eye(len(gram), device=x.device, dtype=x.dtype),
        x.T @ (weight[:, None] * (y - weight @ y)),
    )


def fit(donors):
    histories = [v["history"] for v in donors]
    counts = Counter(histories)
    raw, credit, masked = [], [], []

    def tensor(value):
        return torch.as_tensor(value, device="cuda", dtype=torch.float64)

    for window in donors:
        points = window["points"]
        y = tensor([p["scores"] for p in points])
        c = tensor([p["edges"] for p in points])
        raw.append((y - y.mean(1, keepdim=True)).mean(0))
        credit.append(project(y, c).mean(0))
        c[:, 2] = 0
        masked.append(project(y, c).mean(0))
    y = torch.stack(raw)
    c = torch.stack(credit)
    w = tensor([1 / (len(counts) * counts[h]) for h in histories])
    features = tensor([d["features"] for d in donors])
    center = w @ features
    sd = (w @ (features - center).square()).sqrt().clamp_min(0.02)
    z = ((features - center) / sd).clamp(-3, 3)
    shift = w @ z
    x = z - shift
    raw = ridge(x, y, w)
    verified = ridge(x, c, w)
    rotation = list(range(len(donors)))
    for h in counts:
        group = sorted(
            (i for i, d in enumerate(donors) if d["history"] == h),
            key=lambda i: int(donors[i]["window"].rsplit("_s", 1)[1]),
        )
        for i, j in zip(group, group[1:] + group[:1], strict=True):
            rotation[i] = j
    model = {
        "histories": sorted(counts),
        "windows": [v["window"] for v in donors],
        "center": center,
        "sd": sd,
        "shift": shift,
        "anchor": w @ y,
        "coef": {
            "credit": (raw + verified) / 2,
            "raw": raw / 2,
            "full_raw": raw,
            "global": torch.zeros_like(raw),
            "rotated": (raw + ridge(x, c[rotation], w)) / 2,
            "no_direct": (raw + ridge(x, torch.stack(masked), w)) / 2,
        },
    }
    if any(d.get("additional") for d in donors):
        old = tensor([0.0 if d.get("additional") else 1.0 for d in donors])
        model["coef"]["zero_new"] = (raw + ridge(x, c * old[:, None], w)) / 2
    return model


def predict(fit, features):
    z = ((fit["center"].new_tensor(features) - fit["center"]) / fit["sd"]).clamp(
        -3, 3
    ) - fit["shift"]
    return {k: (fit["anchor"] + z @ v).tolist() for k, v in fit["coef"].items()}


def allocate(native, packet):
    ps = {
        k: 0.20 * native + 0.80 * (native.new_tensor(v) / 0.05).softmax(0)
        for k, v in packet["graph"].items()
    }
    ps.update(native=native, uniform=0.20 * native + 0.80 / 3)
    if "parent" in packet:
        parent = distributions(native, packet["parent"])
        ps.update(parent_credit=parent["credit"], parent_raw=parent["raw"])
    return ps


def freeze(base, mode):
    expanded = mode == "expanded"
    complete = mode != "masked"
    reg = registration(base)
    out = reg.parent / mode
    out.mkdir(exist_ok=True)
    path = out / "forecast.json"
    source = base / "source-compression"
    later = reg.parent.parent.parent
    parent = read(reg.parent.parent / "margin/forecast.json")["windows"]
    inputs = {"registration": file_hash(reg)}
    if complete:
        inputs.update(
            direct=file_hash(
                source / "activity-relation-repairs/0/outcomes-complete.json"
            ),
            masked=file_hash(reg.parent / "masked/forecast.json"),
        )
    if expanded:
        extra = reg.parent / "extra-donors"
        original = reg.parent / "complete/forecast.json"
        inputs.update(
            extra_registration=file_hash(extra / "registration.json"),
            extra=file_hash(extra / "outcomes-complete.json"),
            original=file_hash(original),
        )
    if path.exists():
        packet = read(path)
        if packet["inputs"] != inputs or packet["models"] != file_hash(
            out / "models.json"
        ):
            raise ValueError("Changed frozen graph model")
        return path
    donors, excluded = training_rows(base, complete)
    if expanded:
        new, rejected = extra_rows(base)
        if {d["window"] for d in donors} & {d["window"] for d in new}:
            raise ValueError("Repeated training decision")
        donors += new
        excluded.update(rejected)
    geo = read(source / "matched-span-rank/features.json")["features"]
    panel = read(later / "panel.json")["windows"]
    target = {
        (a["dialogue"], a["session"]): a
        for a in read(later / "forecast.json")["targets"]
    }
    folds = {}
    early = {}
    predictions = {}
    for h in sorted({d["history"] for d in donors}):
        with limited():
            m = fit([d for d in donors if d["history"] != h])
            if h in m["histories"] or len(m["histories"]) != 5:
                raise ValueError("Target history entered training")
            folds[h] = {
                k: v.tolist() if torch.is_tensor(v) else v
                for k, v in m.items()
                if k != "coef"
            }
            folds[h]["coef"] = {k: v.tolist() for k, v in m["coef"].items()}
            for d in donors:
                if d["history"] == h:
                    early[d["window"]] = predict(m, d["features"])
            for w in panel:
                if w["dialogue"] != h:
                    continue
                n = w["start"] + 1
                name = f"{h}_s{n - 1}"
                before = target[h, n]
                a = parent[name]
                if (
                    before["source"] != a["source"]
                    or w["background"] != a["background"]
                ):
                    raise ValueError(
                        "Source differs from registered three-skill action"
                    )
                predictions[name] = {
                    **a,
                    "scores": {
                        "parent": a["scores"],
                        "graph": predict(m, [before["offset"], geo[f"{h}_s{n}"][5]]),
                    },
                }
    if len(predictions) != 25 or sum(len(w["questions"]) for w in panel) != 209:
        raise ValueError("Incomplete later forecast")
    if expanded:
        previous = read(original)
        for k, r in predictions.items():
            r["scores"]["graph"]["original"] = previous["windows"][k]["scores"][
                "graph"
            ]["credit"]
        for k, r in early.items():
            source_scores = (
                previous["early"][k]
                if k in previous["early"]
                else previous["windows"][k]["scores"]["graph"]
            )
            r["original"] = source_scores["credit"]
    elif complete:
        reference = read(reg.parent / "masked/forecast.json")
        for name, r in predictions.items():
            for arm in ("raw", "full_raw", "global", "no_direct"):
                if (
                    reference["windows"][name]["scores"]["graph"][arm]
                    != r["scores"]["graph"][arm]
                ):
                    raise ValueError("Masked control changed after third-edge audit")
        if any(
            v["no_direct"] != reference["early"][k]["credit"] for k, v in early.items()
        ):
            raise ValueError("Changed earlier held-history control")
    register(
        out / "models.json",
        {"inputs": inputs, "folds": folds, "donors": donors, "excluded": excluded},
    )
    register(
        path,
        {
            "inputs": inputs,
            "models": file_hash(out / "models.json"),
            "early": early,
            "windows": predictions,
        },
    )
    return path


def early_report(path):
    dest = path.parent / "early-held-histories.json"
    packet = read(path)
    data = read(path.parent / "models.json")
    results = []
    questions = {}
    for d in data["donors"]:
        with limited():
            pair = torch.tensor(d["native_pair"], device="cuda", dtype=torch.float64)
            native = three_probabilities(*pair)
            ps = allocate(native, {"graph": packet["early"][d["window"]]})
            y = native.new_tensor([r["scores"] for r in d["points"]])
            means = y.mean(0)
            q = (y @ torch.stack(list(ps.values()), dim=1)).tolist()
            questions[d["window"]] = [
                dict(point, expected=dict(zip(ps, values, strict=True)))
                for point, values in zip(d["points"], q, strict=True)
            ]
            results.append(
                {
                    "window": d["window"],
                    "history": d["history"],
                    "n": len(y),
                    "means": means.tolist(),
                    "p": {k: v.tolist() for k, v in ps.items()},
                    "f1": {k: float(means @ v) for k, v in ps.items()},
                }
            )
    by_hist = {
        h: aggregate([r for r in results if r["history"] == h])
        for h in sorted(data["folds"])
    }
    intervals = {
        k: paired_interval(
            [r["questions"] for r in by_hist.values()],
            [r["f1"]["credit"] - r["f1"][k] for r in by_hist.values()],
        )
        for k in ("raw", "full_raw", "native", "rotated", "no_direct")
    }
    register(
        dest,
        {
            "forecast": file_hash(path),
            "all": aggregate(results),
            "by_history": by_hist,
            "dialogue_95pct": intervals,
            "windows": results,
            "questions": questions,
            "excluded": data["excluded"],
        },
    )


def run(base, mode):
    path = freeze(base, mode)
    early_report(path)
    score(
        base,
        path,
        allocate,
        eligible=mode != "masked",
        replay_choices=True,
        extra_controls=(
            "full_raw",
            "global",
            "parent_raw",
            "parent_credit",
            "no_direct",
            *(("original", "zero_new") if mode == "expanded" else ()),
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("mode", choices=("masked", "complete", "expanded"))
    args = cli.parse_args()
    gpu()
    run(args.base, args.mode)
