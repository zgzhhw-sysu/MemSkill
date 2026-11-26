"""Check the frozen two-dialogue pilot on earlier matched swaps from the other four histories."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.audit_source_compression import panel
from experiments.controller.late_pair_policy import logistic, read
from experiments.controller.rank_return import rank_offsets
from experiments.controller.prefix_return import budget_envelope, predicted_offsets
from experiments.controller.source_compression import file_hash, register


def measure(rows, model, mixture, terminal):
    from torch.nn.functional import normalize

    offsets_for = rank_offsets if "strength" in model else predicted_offsets
    grouped = defaultdict(list)
    index = {(r["dialogue"], r["session"]): r for r in rows}
    offsets = {(r["dialogue"], r["session"]): r["offsets"] for r in mixture["rows"]}
    for (sid, session), r in index.items():
        if sid in model["gate"]["dialogues"] or not r["scores"]:
            continue
        node = 0 if session % 2 else 1 + int(index[sid, session - 1]["use"])
        base = r["base_logit"]
        prob = {
            key: logistic(base + vals[node])
            for key, vals in offsets_for(model, r["features"]).items()
        }
        w = torch.tensor(terminal[sid], dtype=torch.float64)
        term = float(w[0] + w[1:] @ normalize(w.new_tensor(r["state"]), dim=0))
        mixp = [logistic(base + s) for s in offsets[sid, session]]
        prob.update(
            native=logistic(base),
            terminal=logistic(base + term),
            mixture_max=mixp[-1],
            mixture=sum(
                a * b for a, b in zip(mixp, mixture["weights"][sid], strict=True)
            ),
            temporal=0.0,
            activity=1.0,
        )
        off, on = [mean(q[k] for q in r["scores"]) for k in ("temporal", "activity")]
        grouped[sid].append(
            {
                "session": session,
                "context": node,
                "focal_probabilities": prob,
                "links": r["scores"],
                "returns": {k: off + p * (on - off) for k, p in prob.items()},
                "oracle": max(off, on),
            }
        )
    return grouped


def summarize(group):
    def summary(by_link):
        weights = [len(r["links"]) if by_link else 1 for r in group]
        count = sum(weights)

        def weighted(vals):
            return sum(w * v for w, v in zip(weights, vals, strict=True)) / count

        return {
            **{
                k: weighted([r["returns"][k] for r in group])
                for k in group[0]["returns"]
            },
            "oracle": weighted([r["oracle"] for r in group]),
            "absolute_probability_change": weighted(
                [
                    abs(
                        r["focal_probabilities"]["prefix"]
                        - r["focal_probabilities"]["native"]
                    )
                    for r in group
                ]
            ),
        }

    return {
        "spans": len(group),
        "links": sum(len(r["links"]) for r in group),
        "by_link": summary(True),
        "by_span": summary(False),
    }


def run(previous, corpus, base, rank=False):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    dest = base / "late-pairs"
    root = dest / ("rank-return" if rank else "prefix-return") / "pilot"
    saved = read(root / "forecast.json")
    fit = saved["fits"]["pilot"]
    frozen = read(dest / "frozen.json")
    if saved["inputs"]["history"] != file_hash(dest / "frozen.json") or saved["inputs"][
        "mixture"
    ] != file_hash(dest / "step-mixture/forecast.json"):
        raise ValueError("Frozen actor inputs changed")
    raw = base / "source-compression/features.json"
    if frozen["inputs"]["early_features"] != file_hash(raw):
        raise ValueError("Early source features changed")
    features = {(r["dialogue"], r["session"]): r for r in read(raw)["rows"]}
    with torch.device("cuda"):
        rows, receipts, _ = panel(previous, corpus, features, (1, 3), supplement=True)
        if receipts != frozen["training_evidence"]:
            raise ValueError("Historical response pairs changed")
        groups = measure(
            rows, fit, read(dest / "step-mixture/forecast.json"), frozen["terminal"]
        )
    records = [dict(r, dialogue=d) for d, items in groups.items() for r in items]
    if len(groups) != 4:
        raise ValueError("Expected four pilot-excluded histories")
    result = {
        "forecast": file_hash(root / "forecast.json"),
        "historical_evidence": receipts,
        "spans": records,
        "unique_questions": len(
            {(r["dialogue"], q["original_index"]) for r in records for q in r["links"]}
        ),
        "all": summarize(records),
        "dialogues": {k: summarize(v) for k, v in groups.items()},
        "contexts": {
            str(i): summarize(g)
            for i in range(3)
            if (g := [r for r in records if r["context"] == i])
        },
    }
    register(root / "earlier-held-histories.json", result)

    def inputs(r, count):
        return (
            count,
            r["focal_probabilities"]["native"],
            r["focal_probabilities"]["prefix"],
            mean(q["activity"] - q["temporal"] for q in r["links"]),
        )

    register(
        root / "historical-budget.json",
        {
            "data": file_hash(root / "earlier-held-histories.json"),
            "by_link": budget_envelope(inputs(r, len(r["links"])) for r in records),
            "by_span": budget_envelope(inputs(r, 1) for r in records),
        },
    )
    print(result["all"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", action="store_true")
    for key in ("previous", "corpus", "base"):
        p.add_argument(key, type=Path)
    run(**vars(p.parse_args()))
