"""Upper bound on terminal-span placement F1 at the native activity rate within each prefix."""

import argparse
import json
import math
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import CELLS, read
from experiments.controller.source_compression import file_hash, register


def optimal_rates(reach, native, advantage, question_count, radius, *, total_uses=None):
    """Allocate activity mass by per-decision return, with a fixed expected number of uses."""
    if radius is None:
        lower, upper = torch.zeros_like(native), torch.ones_like(native)
    else:
        lower, upper = (
            (native.logit() - radius).sigmoid(),
            (native.logit() + radius).sigmoid(),
        )
    order = torch.argsort(question_count * advantage, descending=True, stable=True)
    capacity = (reach * (upper - lower))[order]
    if total_uses is None:
        target = reach @ native
        # Keep the arithmetic of previously published native-budget receipts.
        budget = reach @ (native - lower)
    else:
        target = torch.as_tensor(total_uses, dtype=native.dtype, device=native.device)
        budget = target - reach @ lower
    if budget < -1e-10 or budget > capacity.sum() + 1e-10:
        raise ValueError("Requested use count is outside the probability bounds")
    previous = capacity.cumsum(0) - capacity
    added = torch.minimum((budget - previous).clamp_min(0), capacity)
    result = lower.clone()
    result[order] += added / reach[order]
    if not torch.allclose(reach @ result, target, rtol=0, atol=1e-10):
        raise ValueError("Oracle changed the number of activity uses within one prefix")
    return result


def table(folder, repair):
    m = read(folder / "measurement.json")
    ids = [r["original_index"] for r in m["per_question"]]
    native = {k: m["path_mass"][f"{k[0]}{k[1]}"]["native"] for k in CELLS}
    scores = {}
    for a, b in CELLS:
        path = folder / f"{a}{b}"
        rows = read(path / "scores.json")["answers"]
        if {r["qa_index"] for r in rows} != set(range(len(ids))) or len(rows) != len(
            ids
        ):
            raise ValueError("Question sets differ between schedules")
        scores[a, b] = {r["qa_index"]: r["f1"] for r in rows}
    for i, q in enumerate(m["per_question"]):
        if not math.isclose(
            sum(native[k] * scores[k][i] for k in CELLS),
            q["f1"]["native"],
            abs_tol=1e-12,
        ):
            raise ValueError("Measured native expectation did not reproduce")
    r = read(repair) if repair.is_file() else None
    if r and r["inputs"]["window"] != file_hash(folder / "measurement.json"):
        raise ValueError("Repair probes belong to a different trajectory table")
    rows = []
    for a in (0, 1):
        reach = native[a, 0] + native[a, 1]
        deltas = [scores[a, 1][i] - scores[a, 0][i] for i in range(len(ids))]
        row = {
            "name": folder.name,
            "prefix": a,
            "reach": reach,
            "native": native[a, 1] / reach,
            "n": len(ids),
            "off": sum(scores[a, 0].values()) / len(ids),
            "raw": sum(deltas) / len(ids),
        }
        if r:
            terms = [
                q for q in r["rows"] if q["position"] == 1 and q["background"] == a
            ]
            if [q["original_index"] for q in terms] != ids or [
                q["delta"] for q in terms
            ] != deltas:
                raise ValueError("Signed probe IDs or measured differences changed")
            row["screened"] = sum(
                q.get("probe", {}).get("signed_effect", 0.0) for q in terms
            ) / len(ids)
        rows.append(row)
    return rows


def estimate(rows, score):
    # Each history makes one decision here regardless of its number of questions.
    n = sum(r["n"] for r in rows if r["prefix"] == 0)
    totals = {}
    receipts = []
    for a in (0, 1):
        group = [r for r in rows if r["prefix"] == a]
        reach, native, gain, count, off = [
            torch.tensor([r[k] for r in group], dtype=torch.float64)
            for k in ("reach", "native", score, "n", "off")
        ]
        rate = (reach @ native) / reach.sum()
        estimates = {
            "native": native,
            "prefix_rate": rate.expand_as(native),
            "bounded": optimal_rates(reach, native, gain, count, 2.0),
            "unrestricted": optimal_rates(reach, native, gain, count, None),
        }
        for arm, probs in estimates.items():
            # For screened effects only differences from native have an interpretation.
            result = off + probs * gain if score == "raw" else (probs - native) * gain
            totals[arm] = totals.get(arm, 0.0) + float((reach * count) @ result) / n
        receipts.append(
            {
                "prefix": a,
                "expected_visits": float(reach.sum()),
                "expected_uses": float(reach @ native),
                "windows": [
                    dict(
                        name=r["name"], **{k: float(v[i]) for k, v in estimates.items()}
                    )
                    for i, r in enumerate(group)
                ],
            }
        )
    return {
        "windows": len(rows) // 2,
        "questions": n,
        "scores": totals,
        "bounded_over_native": totals["bounded"] - totals["native"],
        "bounded_over_prefix_rate": totals["bounded"] - totals["prefix_rate"],
        "allocations": receipts,
    }


def audit(root):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    archive = root / "factorial"
    repair = root / "factorial-repairs"
    windows = read(archive / "manifest.json")["windows"]
    all_rows, hashes = [], {}
    for sid, start in windows:
        folder = archive / f"{sid}_s{start}"
        if (folder / "measurement.json").is_file():
            probe = repair / (folder.name + ".json")
            all_rows.extend(table(folder, probe))
            hashes[folder.name] = {
                "measurement": file_hash(folder / "measurement.json"),
                "repair": file_hash(probe) if probe.is_file() else None,
            }
    if not all_rows:
        raise ValueError("No complete factorial training windows")
    with torch.device("cuda"):
        subsets = {
            "all": all_rows,
            "outside_two_pilot_dialogues": [
                r
                for r in all_rows
                if not r["name"].startswith(("conv-26_", "conv-30_"))
            ],
        }
        panels = {k: estimate(v, "raw") for k, v in subsets.items() if v}
        checked = [r for r in all_rows if "screened" in r]
        if checked:
            panels["probed_raw"] = estimate(checked, "raw")
            panels["probed_signed_component"] = estimate(checked, "screened")
    output = root / "terminal-capacity"
    output.mkdir(exist_ok=True)
    register(
        output / f"measured-{len(all_rows) // 2}-probed-{len(checked) // 2}.json",
        {
            "source": file_hash(archive / "manifest.json"),
            "inputs": hashes,
            "panels": panels,
        },
    )
    print(
        json.dumps(
            {
                k: {
                    z: v[z]
                    for z in (
                        "windows",
                        "questions",
                        "bounded_over_native",
                        "bounded_over_prefix_rate",
                    )
                }
                for k, v in panels.items()
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(**vars(parser.parse_args()))
