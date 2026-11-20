"""Fit a cross-history skill policy with completed, same-skill writer counterfactuals."""

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller.capped_source_repair import affected, archive, census
from experiments.controller.factorial_rollout import read
from experiments.controller.late_credit_kernel import (
    donor_observations,
    early_observations,
    freeze,
    kernel_effect,
    score,
    support,
)
from experiments.controller.late_pair_trial import paths
from experiments.controller.retrieval_competition import summarize
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import recovered
from src.eval.locomo import LoCoMoEvaluator

ARMS = (
    "credit",
    "raw",
    "uncorrected",
    "uncorrected_raw",
    "mask",
    "mask_raw",
    "rate",
    "native",
    "terminal",
)


def surviving_credit(old, low, high, truth):
    if old == 0:
        return 0.0
    before, after = (low, high) if old > 0 else (high, low)
    if recovered(before, after, truth):
        return math.copysign(min(abs(old), after["f1"] - before["f1"]), old)
    return 0.0


def baseline(base, registered):
    source = base / "source-compression"
    late = base / "late-pairs"
    labels = late / "selection-labels"
    donor = read(labels / "split.json")["windows"]["donor"]
    ledger = read(labels / "through-12.json")["files"]
    previous = labels / "expanded/fit.json"
    early = early_observations(base)
    all_rows = early + donor_observations(late, labels, donor)
    if read(previous)["observations"] != all_rows:
        raise ValueError("Observations differ from the published cross-history fit")
    evidence = {
        "original_fit": file_hash(previous),
        "split": file_hash(labels / "split.json"),
        "terminal": file_hash(late / "frozen.json"),
        "cohort": file_hash(late / "cohort.json"),
    }
    train = {
        w["window"]: w
        for w in read(source / "retrieval-competition/fit.json")["training"]
    }
    counts = Counter()
    contrasts = []
    for r in all_rows:
        name, j = r["window"], r["stage"]
        w = registered["windows"][name]
        ids = w["questions"]
        if r["early"]:
            b = counts[name, j]
            counts[name, j] += 1
            root = source / "root-write-repairs" / name / "window.json"
            term = source / "factorial-repairs" / (name + ".json")
            if train[name]["repair"] != file_hash(term):
                raise ValueError("Training probe receipt changed")
            if j == 0:
                by_index = {
                    v["index"]: v["signed_effect"]
                    for v in read(root)["rows"]
                    if v["suffix"] == b
                }
                if set(by_index) - set(ids):
                    raise ValueError("Unregistered root question")
                credit = [by_index.get(i, 0.0) for i in ids]
            else:
                records = [
                    v
                    for v in read(term)["rows"]
                    if v["position"] == 1 and v["background"] == b
                ]
                if [v["original_index"] for v in records] != ids:
                    raise ValueError("Terminal probe order changed")
                credit = [v.get("probe", {}).get("signed_effect", 0.0) for v in records]
        else:
            measured = labels / (name + ".json")
            label = read(measured)
            b = label["background"]
            if ledger[name] != file_hash(measured) or label["inputs"]["scores"] != {
                k: v["scores"] for k, v in w["cells"].items()
            }:
                raise ValueError("Donor cell lost its verified score correspondence")
            credit = [v["credit"] for v in label["rows"]]
        cells = ["".join(map(str, p)) for p in paths(j, b)]
        if not math.isclose(mean(credit), r["credit"], abs_tol=1e-12):
            raise ValueError("Question credits differ from the published target")
        contrasts.append((w, cells, credit))
    if set(counts.values()) != {2} or len(counts) != 24:
        raise ValueError("Wrong factorial background coverage")
    return all_rows, contrasts, evidence


def cell_scores(base, out, w, bits, corrected):
    physical = archive(base, w["panel"]) / w["name"] / bits / "scores.json"
    if file_hash(physical) != w["cells"][bits]["scores"]:
        raise ValueError("Different factual trajectory")
    original = read(physical)["answers"]
    if [v["qa_index"] for v in original] != list(range(len(w["questions"]))):
        raise ValueError("Factual answers out of order")
    control = [
        {
            "answer": v["prediction"],
            "ok": min(v["tokens"]) > 0,
            "f1": v["f1"],
            "tokens": v["tokens"],
        }
        for v in original
    ]
    if not any(w["cells"][bits]["triggers"]):
        return control, control
    path = out / w["name"] / bits / "scores.json"
    if file_hash(path) != corrected[w["name"] + "/" + bits]:
        raise ValueError("Changed repaired trajectory")
    rows = read(path)["rows"]
    if [v["index"] for v in rows] != w["questions"]:
        raise ValueError("Repaired scores lost their original question identifiers")
    fixed = []
    for raw, item in zip(control, rows, strict=True):
        if any(item["factual"][k] != v for k, v in raw.items()):
            raise ValueError("Different factual answer or token count")
        v = item["fixed"]
        if min(v["tokens"]) <= 0:
            raise ValueError("No physical answer")
        fixed.append(dict(v, ok=True))
    return control, fixed


def targets(base, out, original, contrasts, truths, corrected=None):
    result = []
    for row, (w, bits, credits) in zip(original, contrasts, strict=True):
        if not any(any(w["cells"][k]["triggers"]) for k in bits):
            result.append(row)
            continue
        if corrected is None:
            result.append(dict(row, raw=0.0, credit=0.0))
            continue
        (a, low), (b, high) = [cell_scores(base, out, w, k, corrected) for k in bits]
        if not math.isclose(
            mean(v["f1"] - u["f1"] for u, v in zip(a, b, strict=True)),
            row["raw"],
            abs_tol=1e-12,
        ):
            raise ValueError("Original question returns failed to reproduce")
        bonus = [
            surviving_credit(c, l, h, truths[w["dialogue"], idx])
            for c, l, h, idx in zip(credits, low, high, w["questions"], strict=True)
        ]
        result.append(
            dict(
                row,
                raw=mean(h["f1"] - l["f1"] for l, h in zip(low, high, strict=True)),
                credit=mean(bonus),
            )
        )
    return result


def probabilities(d, models, weights):
    chosen, old, mask = models
    state = chosen[0][0].new_tensor(d["state"])
    shifts = []
    for data, _ in (chosen, old, mask):
        shifts.extend(
            (kernel_effect(state[None], *data)[0] / 0.05).clamp(-2, 2).unbind()
        )
    w = state.new_tensor(weights)
    shifts.extend(
        [
            state.new_tensor(chosen[1]),
            state.new_zeros(()),
            w[0] + w[1:] @ F.normalize(state, dim=0),
        ]
    )
    return dict(
        zip(
            ARMS,
            (state.new_tensor(d["base_logit"]) + torch.stack(shifts))
            .sigmoid()
            .tolist(),
            strict=True,
        )
    )


def metrics(panels):
    stats = summarize(panels, ARMS, "credit", ARMS[1:])
    if panels:
        stats["mean_abs_focal_change"] = {
            arm: mean(
                abs(p["focal_probabilities"]["credit"] - p["focal_probabilities"][arm])
                for p in panels
            )
            for arm in ("raw", "uncorrected", "mask")
        }
    return stats


def evaluate(dest, late, assignments, models, weights, before, stage):
    for label, donor in (("calibration", True), ("later", False)):
        chosen = [w for w in assignments if (w["start"] <= 11) == donor]
        panels = [
            score(late, w, models, weights, policy=probabilities, arms=ARMS)
            for w in chosen
            if (
                late / "outcomes" / f"{w['dialogue']}_s{w['start']}" / "pair.json"
            ).is_file()
        ]
        fresh = [p for p in panels if p["window"] not in before]
        if not panels:
            continue
        all_stats, strict = metrics(panels), metrics(fresh)
        passed = False
        if (
            stage == "repaired"
            and not donor
            and len(panels) == 40
            and all_stats["questions"] == 378
            and strict["questions"]
            and len(strict["by_dialogue"]) == 6
        ):
            v, ht, sv = all_stats["f1"], all_stats["ht"], strict["f1"]
            passed = (
                all(
                    v["credit"] >= v[k] + 0.02 and ht["credit"] > ht[k]
                    for k in ("native", "terminal")
                )
                and all(v["credit"] > v[k] for k in ("rate", "raw", "uncorrected"))
                and sv["credit"] > sv["native"]
                and sum(
                    x["f1"]["credit"] > x["f1"]["rate"]
                    for x in all_stats["by_dialogue"].values()
                )
                >= 4
            )
        folder = dest / label
        folder.mkdir(exist_ok=True)
        report = {
            "fit": file_hash(dest / "fit.json"),
            "all": all_stats,
            "strict": strict,
            "windows": panels,
            "registered_questions": sum(len(w["questions"]) for w in chosen),
            "eligible_for_development": passed,
        }
        register(folder / f"through-{len(panels):02d}.json", report)
        if len(panels) == len(chosen):
            register(folder / "summary.json", report)


def run(base, study, corpus):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Fit this policy under a single GPU lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    reg = census(base, study)
    out, manifest = reg.parent, read(reg)
    late = base / "late-pairs"
    spec = read(study / "protocol.json")
    if file_hash(corpus) != spec["input_sha256"]:
        raise ValueError("Unexpected benchmark")
    evaluator = LoCoMoEvaluator(None)
    truths = {
        (d["sample_id"], i): evaluator.get_ground_truth(q)
        for d in read(corpus)
        if d["sample_id"] in spec["train"]
        for i, q in enumerate(d["qa"])
    }
    original, contrasts, evidence = baseline(base, manifest)
    evidence = dict(evidence, registration=file_hash(reg))
    mask = targets(base, out, original, contrasts, truths)
    weights = read(late / "frozen.json")["terminal"]
    assignments = read(late / "cohort.json")["assignments"]
    for stage in ("mask", "repaired"):
        chosen = mask
        identity = dict(evidence)
        if stage == "repaired":
            path = out / "complete.json"
            if not path.exists():
                continue
            checked = read(path)
            if checked["registration"] != file_hash(reg) or set(checked["scores"]) != {
                w["name"] + "/" + b for w, b in affected(manifest)
            }:
                raise ValueError("Incomplete physical repair coverage")
            chosen = targets(base, out, original, contrasts, truths, checked["scores"])
            identity["measured"] = file_hash(path)
        dest = out / "policy" / stage
        dest.mkdir(exist_ok=True, parents=True)
        models = {
            (sid, j): tuple(
                support(rows, sid, j, "cuda") for rows in (chosen, original, mask)
            )
            for sid in sorted(spec["train"])
            for j in (0, 1)
        }
        identity["rate_offsets"] = {
            f"{sid}/{j}": m[0][1] for (sid, j), m in models.items()
        }
        before = freeze(dest, late, identity, chosen)
        evaluate(dest, late, assignments, models, weights, before, stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    run(args.base, args.study, args.corpus)
