"""Score frozen verified-question prototypes, including a separate report for still-unpublished windows."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean
import torch

from experiments.controller.audit_source_compression import panel, summarize
from experiments.controller.conditional_credit import read, writer_outcomes
from experiments.controller.factorial_rollout import CELLS, path_mass, record_scored_paths
from experiments.controller.granularity_placement import load_panel
from experiments.controller.query_prototypes import prototype_logits
from experiments.controller.source_compression import file_hash, register


def validate(report, training):
    witness = report["witnesses"]
    expected = sorted(
        (r["dialogue"], r["session"], i, v) for r in training for i, v in r["credits"].items() if v
    )
    archived = sorted((w["dialogue"], w["session"], w["original_index"], w["effect"]) for w in witness)
    if expected != archived:
        raise ValueError("Prototype labels differ from verified credits")
    matrix = {(r["dialogue"], r["session"]): r["values"] for r in report["similarities"]}
    preds = {(r["dialogue"], r["session"]): r["p_activity"] for r in report["forecasts"]}
    ids = {r["dialogue"] for r in training}
    if (
        set(matrix) != {(sid, n) for sid in ids for n in range(1, 9)}
        or len(matrix) != 48
        or preds.keys() != matrix.keys()
        or len(report["forecasts"]) != 48
    ):
        raise ValueError("Incomplete prototype forecast panel")
    for held in ids:
        fit = report["fits"][held]
        cols = [j for j, w in enumerate(witness) if w["dialogue"] != held]
        if cols != fit["training_witnesses"] or set(fit["training_dialogues"]) != ids - {held}:
            raise ValueError("Held-dialogue labels in prototype weights")
        xs = torch.tensor(
            [matrix[r["dialogue"], r["session"]] for r in training if r["dialogue"] != held],
            dtype=torch.float64,
        )[:, cols]
        mu, sd = xs.mean(0), xs.std(0, unbiased=False)
        sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
        if not torch.allclose(mu, mu.new_tensor(fit["center"]), atol=1e-12, rtol=0) or not torch.allclose(
            sd, sd.new_tensor(fit["scale"]), atol=1e-12, rtol=0
        ):
            raise ValueError("Feature normalization did not reproduce")
        effects = mu.new_tensor([witness[j]["effect"] for j in cols])
        for n in range(1, 9):
            v = mu.new_tensor(matrix[held, n])[cols]
            prob = float(prototype_logits(v, mu, sd, effects).sigmoid())
            if not math.isclose(prob, preds[held, n], abs_tol=1e-12):
                raise ValueError("Question prototype probability changed")


def compare(windows, model):
    probs = {(r["dialogue"], r["session"]): r["p_activity"] for r in model["forecasts"]}
    scores, by_sid, by_type = [], defaultdict(list), defaultdict(list)
    for w in windows:
        sid, start = w["dialogue"], w["start"]
        p, q = (probs[sid, i] for i in (start, start + 1))
        const = model["fits"][sid]["training_mean"]
        for item in w["questions"].values():

            def expected(u, v):
                return sum(path_mass(k, u, v) * item["f1"][f"{k[0]}{k[1]}"] for k in CELLS)

            hard = (float(p > 0.5), float(q > 0.5))
            r = {
                "credit": expected(p, q),
                "greedy": expected(*hard),
                "activity": expected(1, 1),
                "temporal": expected(0, 0),
                "training_constant": expected(const, const),
                "shuffled": (expected(p, q) + expected(q, p)) / 2,
                "greedy_shuffled": (expected(*hard) + expected(*reversed(hard))) / 2,
            }
            if "controls" in item:
                for arm in ("credit", "greedy", "activity", "temporal", "shuffled", "greedy_shuffled"):
                    if not math.isclose(r[arm], item["controls"][arm], abs_tol=1e-10):
                        raise ValueError("Prototype path F1 did not reproduce")
                r.update(item["controls"])
            scores.append(r)
            by_sid[sid].append(r)
            by_type[str(item["question"]["category"])].append(r)

    def summary(group):
        return {k: mean(r[k] for r in group) for k in group[0]}

    overall = summary(scores)
    folds = {sid: summary(group) for sid, group in by_sid.items()}
    positive = sum(v["credit"] > v["shuffled"] + 1e-12 for v in folds.values())
    return {
        "questions": len(scores),
        "windows": len(windows),
        "aggregate": overall,
        "per_dialogue": folds,
        "per_type": {k: {"questions": len(group), "means": summary(group)} for k, group in by_type.items()},
        "positive_dialogues_vs_shuffled": positive,
        "passes": positive >= 4
        and all(k in overall and overall["credit"] - overall[k] >= 0.02 for k in ("native", "terminal"))
        and all(
            overall["credit"] > overall[k] + 1e-12
            for k in ("activity", "temporal", "training_constant", "rate", "shuffled")
        )
        and all(
            overall["greedy"] > overall[k] + 1e-12 for k in ("activity", "temporal", "greedy_shuffled")
        ),
    }


def run(phase, previous, current, corpus, proxy, source):
    d = source / "query-prototypes"
    model = read(d / "frozen.json")
    parent = read(source / "frozen.json")
    if (
        model["inputs"]
        != {
            "parent": file_hash(source / "frozen.json"),
            "plan": file_hash(d / "plan.json"),
            "corpus": file_hash(corpus),
        }
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_available()
    ):
        raise ValueError("Use one GPU and the originally registered inputs")
    torch.cuda.set_per_process_memory_fraction(0.06)
    features = {(r["dialogue"], r["session"]): r for r in read(source / "features.json")["rows"]}
    with torch.device("cuda"):
        rows, hashes, backend = panel(previous, corpus, features, (1, 3), supplement=True)
        if hashes != parent["prior_hashes"] or backend != parent["backend"]:
            raise ValueError("Training outcomes changed")
        validate(model, rows)
        if phase == "historical":
            rep = {
                "source_only": compare(load_panel(previous, corpus, proxy, include_questions=True), model),
                "matched_writer": summarize(rows, model["fits"], model["forecasts"]),
            }
        else:
            windows = writer_outcomes(current, corpus, source, parent, candidate=model)
            early = record_scored_paths(source / "factorial", d / "frozen.json") | set(
                model["already_published_windows"]
            )
            fresh = [w for w in windows if f"{w['dialogue']}_s{w['start']}" not in early]
            rep = {
                "complete_panel": compare(windows, model),
                "windows_with_any_early_scores": sorted(early),
                "already_published_windows": model["already_published_windows"],
                "strictly_later": compare(fresh, model) if fresh else None,
            }
        register(d / (phase + ".json"), {"forecast_sha256": file_hash(d / "frozen.json"), **rep})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("historical", "prospective"))
    for n in ("previous", "current", "corpus", "proxy", "source"):
        parser.add_argument(n, type=Path)
    run(**vars(parser.parse_args()))
