"""Score frozen source affinities under a balanced prior for the rare relation skill."""

import argparse
import time
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.semantic_writer import summarize
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "semantic_writer_balanced.json"


def registration(root):
    path = root / "balanced-registration.json"
    existing = {
        str(p.relative_to(root)): file_hash(p)
        for kind in ("write.json", "pair.json")
        for p in root.glob("conv-*/" + kind)
    }
    identity = {
        "plan": file_hash(PLAN),
        "source_forecast": file_hash(root / "source-forecast.json"),
    }
    if not path.exists():
        register(path, dict(identity, preexisting=existing))
    saved = read(path)
    if any(saved[k] != v for k, v in identity.items()) or any(
        existing.get(k) != v for k, v in saved["preexisting"].items()
    ):
        raise ValueError("Published relation writer, forecast or plan changed")
    return path, saved


def freeze(root):
    path, saved = registration(root)
    rows = [r for r in read(root / "source-forecast.json")["rows"] if "offset" in r]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one leased GPU to freeze relation probabilities")
    torch.cuda.set_per_process_memory_fraction(0.06)
    probabilities = (
        torch.tensor([r["offset"] for r in rows], device="cuda", dtype=torch.float64)
        .sigmoid()
        .tolist()
    )
    forecasts = {
        f"{r['dialogue']}_s{r['session'] - 1}": p
        for r, p in zip(rows, probabilities, strict=True)
    }
    if len(forecasts) != 12:
        raise ValueError("Incomplete prospective source panel")
    # A numeric freeze that follows a completed pair is disclosed separately from the protocol registration.
    receipt = root / "balanced-forecast.json"
    if not receipt.exists():
        scored = {
            str(p.relative_to(root)): file_hash(p)
            for p in root.glob("conv-*/pair.json")
        }
        register(
            receipt,
            {
                "registration": file_hash(path),
                "probabilities": forecasts,
                "scored_at_freeze": scored,
            },
        )
    measure = read(receipt)
    if (
        measure["registration"] != file_hash(path)
        or measure["probabilities"] != forecasts
        or any(file_hash(root / k) != v for k, v in measure["scored_at_freeze"].items())
    ):
        raise ValueError("Changed balanced probability forecast")
    return measure, saved


def panels(reports, forecasts):
    checkpoint = summarize([report for _, report in reports])
    adjusted = []
    for name, report in reports:
        if report["matched"]:
            report = dict(report, choice={"native": 0.5, "proposed": forecasts[name]})
        adjusted.append(report)
    balanced = summarize(adjusted)
    if balanced["matched"]:
        for group in [balanced, *balanced["per_dialogue"].values()]:
            scores = group["f1"]
            scores["uniform"] = scores.pop("native")
            scores["source_only"] = scores.pop("proposed")
    return {"checkpoint": checkpoint, "balanced": balanced}


def score(root):
    path, saved = registration(root)
    forecast_path = root / "balanced-forecast.json"
    forecast = read(forecast_path)
    if forecast["registration"] != file_hash(path):
        raise ValueError("Numerical freeze has another registration")
    probabilities = forecast["probabilities"]
    names = sorted(probabilities)
    reports = [
        (n, read(root / n / "pair.json"))
        for n in names
        if (root / n / "pair.json").exists()
    ]
    if any(r["forecast"] != saved["source_forecast"] for _, r in reports):
        raise ValueError("Writer used another source forecast")
    if not reports:
        return
    excluded = {
        Path(k).parts[0] for k in (*saved["preexisting"], *forecast["scored_at_freeze"])
    } | {"conv-26_s5"}
    result = {
        "forecast": file_hash(forecast_path),
        "all": panels(reports, probabilities),
        "strict": panels(
            [(n, r) for n, r in reports if n not in excluded], probabilities
        ),
    }
    register(root / f"balanced-through-{len(reports)}.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Aggregate published GPU measurements until all windows have a receipt",
    )
    args = parser.parse_args()
    if args.watch:
        while True:
            score(args.root)
            if (args.root / "balanced-through-12.json").is_file():
                break
            time.sleep(45)
    else:
        freeze(args.root)
        score(args.root)
