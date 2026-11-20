"""Score evidence-conditioned decisions on the complete factorial and randomized panels."""

import argparse
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.audit_case_credit_joint import collect, evaluate_offsets
from experiments.controller.evidence_transport import ARMS
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from src.evaluation_cache import digest

ALL = (*ARMS, "rate", "return_only", "native", "terminal")
CHECKS = ("native", "terminal", "raw", "sham", "blind", "rate", "equal")


def report(group, device):
    summary, rows = evaluate_offsets(group, device, ALL, ARMS, CHECKS)
    if group:
        p = (
            [r["focal_probabilities"] for r in rows]
            if "focal_probabilities" in rows[0]
            else [n for r in rows for n in r["probabilities"].values()]
        )
        summary["mean_abs_change_from_raw"] = mean(
            abs(v["credit"] - v["raw"]) for v in p
        )
        summary["mean_abs_change_from_equal_backup"] = mean(
            abs(v["credit"] - v["equal"]) for v in p
        )
    return summary, rows


def gate(overall, fresh):
    if overall["questions"] < 0.9 * 378 or not fresh["questions"]:
        return False
    f = overall["f1"]
    return (
        all(f["credit"] >= f[k] + 0.02 for k in ("native", "terminal"))
        and all(f["credit"] > f[k] for k in ("raw", "sham", "blind", "rate"))
        and overall["ht"]["credit"] > overall["ht"]["native"]
        and fresh["f1"]["credit"] > fresh["f1"]["native"]
        and sum(
            g["f1"]["credit"] > g["f1"]["rate"] for g in overall["by_dialogue"].values()
        )
        >= 4
    )


def audit(base):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select one capped CUDA device for evaluation")
    torch.cuda.set_per_process_memory_fraction(0.06)
    src = base / "source-compression"
    cal = {
        w["window"]: w
        for w in read(src / "retrieval-competition/calibration.json")["windows"]
    }
    dest = src / "evidence-transport"
    for stage in ("pilot", "expanded"):
        out = dest / stage
        donors = out / "donors.json"
        if not donors.exists():
            continue
        fingerprint = file_hash(donors)
        groups = {"factorial": [], "later": []}
        for path in sorted((out / "predictions").glob("*.json")):
            n = path.stem
            sid, start = n.rsplit("_s", 1)
            panel = "factorial" if n in cal else "later"
            v = read(path)
            if (
                v["inputs"]["donors"] != fingerprint
                or sid in v["donors"]
                or len(v["donors"]) != 5
            ):
                raise ValueError("Wrong prediction fold or donor registration")
            spec = {"dialogue": sid, "start": int(start), "panel": panel}
            packets = [
                {
                    "stamp": path.stat().st_mtime_ns,
                    "source": s,
                    "fingerprint": file_hash(path),
                }
                for s in v["sources"]
            ]
            w = collect(base, n, spec, packets, cal)
            if w is not None:
                w.update(
                    node_offsets=v["offsets"],
                    policy_details={
                        "source_hashes": v["sources"],
                        "donors": v["donors"],
                    },
                )
                groups[panel].append(w)
        for panel, group in groups.items():
            if not group:
                continue
            folder = out / panel
            folder.mkdir(exist_ok=True)
            stamp = digest([w["inputs"] for w in group])[:16]
            path = folder / f"through-{len(group):02d}-{stamp}.json"
            total = 12 if panel == "factorial" else (52 if stage == "pilot" else 40)
            if path.exists():
                if len(group) == total and not (folder / "summary.json").exists():
                    register(folder / "summary.json", read(path))
                continue
            summary, rows = report(group, "cuda")
            result = {"donors": fingerprint, "all": summary, "windows": rows}
            if panel == "later":
                later = [w for w in group if w["start"] >= 13]
                for name, subset in (
                    ("strict", [w for w in group if w["strict"]]),
                    ("session13_plus", later),
                    ("strict_session13_plus", [w for w in later if w["strict"]]),
                ):
                    result[name], _ = report(subset, "cuda")
                result["eligible_for_development"] = stage == "expanded" and gate(
                    result["session13_plus"], result["strict_session13_plus"]
                )
            register(path, result)
            if len(group) == total:
                register(folder / "summary.json", result)
            print(
                stage,
                panel,
                len(group),
                summary["questions"],
                summary["f1"],
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    audit(p.parse_args().base)
