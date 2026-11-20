"""Register source-only probabilities for an existing time-refinement skill."""

import argparse
import re
from pathlib import Path

import torch

from experiments.controller.evidence_transport_data import source_keys, sources
from experiments.controller.factorial_rollout import read
from experiments.controller.semantic_writer import source_offsets
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "temporal_update_writer.json"
UPDATE = "refine_temporal_details_with_context"
CUE = re.compile(
    r"\b(?:yesterday|today|tomorrow|last|next|before|after|since|ago|recently|previously|actually|instead|rescheduled|postponed|changed|week(?:end)?s?|months?|years?|days?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may(?=\s+\d)|june|july|august|september|october|november|december|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def continuity(previous, current, turns):
    ids = [j for j, t in enumerate(turns) if CUE.search(t["text"])]
    if not ids:
        return None
    maxima = (current[ids] @ previous.T).amax(1)
    return float(maxima.topk(min(3, len(ids))).values.mean())


def freeze(base, study, corpus):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select a bounded GPU lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    src = base / "source-compression"
    late = base / "late-pairs"
    out = src / "temporal-update-writer"
    out.mkdir(exist_ok=True)
    source = src / "evidence-transport"
    protocol = read(study / "protocol.json")
    if (
        file_hash(corpus) != protocol["input_sha256"]
        or file_hash(study / "protocol.json") != read(source / "sources.json")["study"]
    ):
        raise ValueError("Changed corpus or study")
    h = {
        r["sample_id"]: r["conversation"]
        for r in read(corpus)
        if r["sample_id"] in protocol["train"]
    }
    early = []
    for w in read(src / "retrieval-competition/fit.json")["training"]:
        n = int(w["window"].rsplit("_s", 1)[1])
        early.append(
            {
                "dialogue": w["dialogue"],
                "start": n,
                "background": 0,
                "panel": "early",
                "questions": w["questions"],
            }
        )
    later = [
        dict(w, panel="later")
        for w in read(late / "cohort.json")["assignments"]
        if w["pivot"] == 1
    ]
    if (
        len(early),
        len(later),
        sum(len(w["questions"]) for w in early),
        sum(len(w["questions"]) for w in later),
    ) != (12, 25, 91, 209):
        raise ValueError("Different registered temporal cohort")
    turns = sources(source, "cuda")
    rows = []
    for w in early + later:
        sid = w["dialogue"]
        n = w["start"]
        name = f"{sid}_s{n}"
        first, last = source_keys(name)
        other = min(set(h) - {sid}) + "_s5"
        visible = h[sid][f"session_{n + 1}"]
        if turns[last][0] != [t["dia_id"] for t in visible]:
            raise ValueError("Source turn misalignment")
        values = [
            continuity(turns[k][1], turns[last][1], visible) for k in (first, other)
        ]
        rows.append(
            {
                "name": name,
                "dialogue": sid,
                "start": n,
                "prefix": w["background"],
                "panel": w["panel"],
                "questions": w["questions"],
                "sources": [turns[k][2] for k in (first, last)],
                "vectors": [turns[k][3] for k in (first, last, other)],
                "sham_prefix": other,
                "scores": values,
            }
        )
    for row in rows:
        calibration = [
            r["scores"][0]
            for r in rows
            if r["panel"] == "early"
            and r["dialogue"] != row["dialogue"]
            and r["scores"][0] is not None
        ]
        if not calibration:
            raise ValueError("No independent calibration histories")
        x = torch.tensor(calibration, device="cuda", dtype=torch.float64)
        offsets = [
            0.0 if s is None else float(source_offsets(x, x.new_tensor([s]))[0])
            for s in row["scores"]
        ]
        row.update(
            offset=offsets[0],
            source_p=float(x.new_tensor(offsets[0]).sigmoid()),
            sham_p=float(x.new_tensor(offsets[1]).sigmoid()),
        )
    register(
        out / "forecast.json",
        {
            "plan": file_hash(PLAN),
            "implementation": file_hash(Path(__file__)),
            "study": file_hash(study / "protocol.json"),
            "source_manifest": file_hash(source / "sources.json"),
            "windows": rows,
        },
    )
    print(
        "frozen",
        len(rows),
        "windows;",
        sum(r["scores"][0] is not None for r in rows),
        "with cues",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "study", "corpus"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    freeze(args.base, args.study, args.corpus)
