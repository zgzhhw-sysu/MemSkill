"""Measure whether source-only offsets at both decisions can improve a real trajectory."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.controller.case_credit import possible_offsets
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_writer_credit import question_links
from src.evaluation_cache import digest

GRID = tuple(i / 2 for i in range(-4, 5))


def value_table(logits, y, root_offsets, last_offsets):
    root = (logits[0] + logits.new_tensor(root_offsets)).sigmoid()
    last = (logits[1:] + logits.new_tensor(last_offsets)[:, None]).sigmoid()
    conditional = y[None, :, 0] + last[:, :, None] * (y[:, 1] - y[:, 0])[None]
    return conditional[None, :, 0] + root[:, None, None] * (
        conditional[None, :, 1] - conditional[None, :, 0]
    )


def question_changes(y):
    root = y[1] - y[0]
    last = y[:, 1] - y[:, 0]
    return {
        "affected_first": root.abs().amax(0) > 1e-12,
        "affected_last": last.abs().amax(0) > 1e-12,
        "opposite_last_by_prefix": last[0] * last[1] < -1e-12,
    }


def run(base, corpus=None):
    comp = base / "source-compression"
    dest = comp / "case-credit"
    path = comp / "retrieval-competition/fit.json"
    targets = read(dest / "prompts.json")["targets"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one leased CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.06)
    windows = []
    training = read(path)["training"]
    panel = [
        {
            "dialogue": w["dialogue"],
            "start": int(w["window"].rsplit("_s", 1)[1]),
            "questions": w["questions"],
        }
        for w in training
    ]
    links = question_links(base, corpus, panel) if corpus is not None else {}
    counts = defaultdict(Counter)
    joint = dest / "joint"
    root_targets = (
        read(joint / "prompts.json")["targets"]
        if (joint / "prompts.json").exists()
        else {}
    )
    for w in training:
        observed, votes = possible_offsets(targets[w["window"]], dest)
        z = torch.tensor(
            [s["base"] for s in w["states"]], device="cuda", dtype=torch.float64
        )
        y = z.new_tensor([w["cases"][a + b] for a in "01" for b in "01"]).reshape(
            2, 2, -1
        )
        if links:
            changes = {k: v.tolist() for k, v in question_changes(y).items()}
            for i, link in enumerate(links[w["window"]]):
                counts[link]["questions"] += 1
                for k, flags in changes.items():
                    counts[link][k] += int(flags[i])
        root_key = w["window"] + "_0"
        roots_available, root_votes = (
            possible_offsets(root_targets[root_key], joint)
            if root_targets
            else (GRID, {})
        )
        limits = {}
        for key, roots, lasts in (
            ("native", [0], [0]),
            ("last", [0], GRID),
            ("first", GRID, [0]),
            ("joint", GRID, GRID),
            ("joint_pending_terminal", GRID, observed),
            ("joint_pending_both", roots_available, observed),
        ):
            values = value_table(z, y, roots, lasts)
            scores = values.mean(-1)
            i, j = divmod(int(scores.argmax()), len(lasts))
            limits[key] = {
                "f1": float(scores[i, j]),
                "source_offsets": [roots[i], lasts[j]],
            }
        windows.append(
            {
                "window": w["window"],
                "questions": len(w["questions"]),
                "votes": votes,
                "root_votes": root_votes,
                "limits": limits,
            }
        )
    n = sum(w["questions"] for w in windows)
    scores = {
        k: sum(w["questions"] * w["limits"][k]["f1"] for w in windows) / n
        for k in windows[0]["limits"]
    }
    report = {
        "fit": file_hash(path),
        "prompts": file_hash(dest / "prompts.json"),
        "questions": n,
        "windows": windows,
        "f1": scores,
    }
    folder = dest / "joint"
    folder.mkdir(exist_ok=True)
    if links:
        report.update(corpus=file_hash(corpus), evidence_linkage=dict(counts))
        print(dict(counts), flush=True)

    register(folder / ("capacity-" + digest(report)[:16] + ".json"), report)
    print(scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    run(args.base, args.corpus)
