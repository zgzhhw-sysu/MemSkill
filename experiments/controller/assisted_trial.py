"""Freeze and score chronological controllers trained with model-assisted span credit."""

import argparse
from pathlib import Path

import torch

from experiments.controller import file_hash, read, register
from experiments.controller import (
    assisted_span_backup,
    evidence_transport,
    evidence_transport_data,
)
from experiments.controller.assisted_span_backup import CELLS, AssistedTransport
from experiments.controller.audit_evidence_transport import CHECKS, audit, gate
from experiments.controller.evidence_transport import ARMS
from experiments.controller.evidence_transport_data import (
    factorial,
    randomized,
    source_keys,
    sources,
)
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "assisted_span_backup.json"
STAGES = {
    "interim": (12, 46, False),
    "expanded": (0, 40, True),
    "resolved": (0, 40, True),
}
EXPERIMENT = "assisted-span-backup"


def promote(overall, fresh):
    return gate(overall, fresh) and overall["f1"]["credit"] > overall["f1"]["paired"]


def verified_data(base, study, corpus):
    src = base / "source-compression"
    parent = src / "evidence-transport"
    protocol = read(study / "protocol.json")
    manifest = read(parent / "sources.json")
    if (
        manifest["study"] != file_hash(study / "protocol.json")
        or manifest["corpus"] != file_hash(corpus)
        or len(protocol["train"]) != 6
        or set(protocol["train"]) & set(protocol["validation"] + protocol["test"])
    ):
        raise ValueError("Changed source, frozen split or corpus")
    turns = sources(parent, "cuda")
    data = {
        r["sample_id"]: r for r in read(corpus) if r["sample_id"] in protocol["train"]
    }
    early = factorial(base, turns, data)
    if len(early) != 12 or sum(len(w["ids"]) for w in early.values()) != 91:
        raise ValueError("Different factorial anchors")
    return turns, data, early


def with_types(donors, data):
    return {
        n: dict(w, types=[int(data[w["sid"]]["qa"][i]["category"]) for i in w["ids"]])
        for n, w in donors.items()
    }


def resolved_values(base, donors, device):
    late = base / "late-pairs"
    dest = late / "donor-factorial"
    measured, hashes = {}, {}
    if not all((dest / n / "complete.json").exists() for n in donors):
        return None, None
    for n, w in donors.items():
        receipt = dest / n / "complete.json"
        v = read(receipt)
        folder = late / "outcomes" / n
        if (
            v["dialogue"] != w["sid"]
            or [r["index"] for r in v["rows"]] != w["ids"]
            or v["inputs"]["pair"] != file_hash(folder / "pair.json")
            or v["inputs"]["forecast"]
            != file_hash(
                base
                / "source-compression/evidence-transport/pilot/predictions"
                / (n + ".json")
            )
        ):
            raise ValueError("Unaligned complementary trajectory receipt")
        hashes[n] = file_hash(receipt)
        values = []
        for k in CELLS:
            cell = (folder if k in w["y"] else dest / n) / k
            scores = cell / "scores.json"
            if file_hash(scores) != v["paths"][k]["scores.json"]:
                raise ValueError("Changed physical path scores")
            rows = read(scores)["answers"]
            if [r["qa_index"] for r in rows] != list(range(len(w["ids"]))):
                raise ValueError("Invalid measured question order")
            vals = [r["f1"] for r in rows]
            if k in w["y"] and vals != w["y"][k]:
                raise ValueError("Original assigned outcomes changed")
            values.append(vals)
        measured[n] = torch.tensor(
            values, device=device, dtype=torch.float64
        ).T.reshape(-1, 2, 2)
    return measured, hashes


def fit(base, study, corpus):
    src = base / "source-compression"
    parent = src / "evidence-transport"
    turns, data, early = verified_data(base, study, corpus)
    early = with_types(early, data)
    cohort = read(base / "late-pairs/cohort.json")["assignments"]
    for stage in STAGES:
        extra = randomized(base, turns, data, first_session_only=stage == "interim")
        if extra is None:
            continue
        extra = with_types(extra, data)
        physical, receipts = (
            resolved_values(base, extra, "cuda")
            if stage == "resolved"
            else (None, None)
        )
        if stage == "resolved" and physical is None:
            continue
        dest = src / EXPERIMENT / stage
        preds = dest / "predictions"
        preds.mkdir(parents=True, exist_ok=True)
        names = (list(early) if stage == "interim" else []) + [
            f"{w['dialogue']}_s{w['start']}"
            for w in cohort
            if w["start"] >= (11 if stage == "interim" else 13)
        ]
        if len(names) != sum(STAGES[stage][:2]) or set(names) & extra.keys():
            raise ValueError("Unexpected target/extra donor overlap or cohort size")
        lock = dest / "donors.json"
        register(
            lock,
            {
                "protocol": file_hash(PLAN),
                "source_manifest": file_hash(parent / "sources.json"),
                "cohort": file_hash(base / "late-pairs/cohort.json"),
                "implementation": {
                    m.__name__: file_hash(Path(m.__file__))
                    for m in (
                        evidence_transport,
                        evidence_transport_data,
                        assisted_span_backup,
                    )
                },
                "training_rows": digest(early | extra),
                "training_features": {
                    k: turns[k][3] for n in (early | extra) for k in source_keys(n)
                },
                "resolved": receipts,
                "ids": {k: v["ids"] for k, v in (early | extra).items()},
                "targets": names,
            },
        )
        learner = AssistedTransport(turns, early, extra, "cuda", physical=physical)
        for n in names:
            path = preds / (n + ".json")
            keys = source_keys(n)
            inputs = {
                "donors": file_hash(lock),
                "features": [turns[k][3] for k in keys],
            }
            if path.exists():
                if read(path)["inputs"] != inputs:
                    raise ValueError("Frozen inputs changed")
                continue
            register(
                path,
                dict(
                    learner.forecast(n),
                    sources=[turns[k][2] for k in keys],
                    inputs=inputs,
                ),
            )
        print(stage, len(names), "frozen windows", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "audit", "diagnose"))
    for key in ("base", "study", "corpus"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one bounded CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    if a.mode == "fit":
        fit(a.base, a.study, a.corpus)
    elif a.mode == "audit":
        audit(
            a.base,
            experiment=EXPERIMENT,
            stages=STAGES,
            learned=(*ARMS, "paired", "difficulty", "assisted"),
            checks=(*CHECKS, "paired", "difficulty", "assisted"),
            promotion=promote,
        )
    else:
        from experiments.controller.masked_span_policy import diagnose

        diagnose(a.base, a.study, a.corpus)


if __name__ == "__main__":
    main()
