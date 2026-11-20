"""Register and collect the remaining source decisions for the two-write case policy."""

import argparse
import os
from pathlib import Path

from experiments.controller.case_credit import KINDS, examples, excerpt, measure, prompt
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "case_credit_joint.json"


def run(base, study, corpus):
    src = base / "source-compression"
    parent = src / "case-credit"
    joint = parent / "joint"
    joint.mkdir(exist_ok=True)
    parent_file = parent / "prompts.json"
    original = read(parent_file)
    protocol = read(study / "protocol.json")
    train = set(protocol["train"])
    if (
        file_hash(corpus) != original["corpus"]
        or file_hash(study / "protocol.json") != original["study"]
        or train & set(protocol["validation"] + protocol["test"])
        or len(train) != 6
        or original["backend"] != os.environ["MEMSKILL_BACKEND_ID"]
    ):
        raise ValueError(
            "Expected the original disjoint training cohort and critic backend"
        )
    data = {d["sample_id"]: d for d in read(corpus) if d["sample_id"] in train}
    source = src / "query-prototypes/frozen.json"
    if original["witnesses"] != file_hash(source):
        raise ValueError("Training witnesses changed")
    witnesses = read(source)["witnesses"]
    early = [t for t in original["targets"].values() if t["panel"] == "factorial"]
    late = [
        dict(w, panel="later")
        for w in read(base / "late-pairs/cohort.json")["assignments"]
    ]
    processor = LoCoMoProcessor(chunk_mode="full-session")
    targets, windows = {}, {}
    for i in range(max(len(early), len(late))):
        for panel in (early, late):
            if i >= len(panel):
                continue
            w = panel[i]
            sid, start = w["dialogue"], w["start"]
            name = f"{sid}_s{start}"
            history = data[sid]["conversation"]
            donors = examples(data, witnesses, sid)
            refs = []
            for stage in range(2):
                n = start + stage
                k = f"session_{n}"
                (raw,) = processor._extract_session_chunks(
                    history[k], history[k + "_date_time"]
                )
                text = excerpt(history, n)
                prompts = {
                    kind: [prompt(text, donors, kind, r) for r in (False, True)]
                    for kind in KINDS
                }
                if stage and name in original["targets"]:
                    old = original["targets"][name]
                    if old["prompts"] != prompts or old["source"] != digest(raw):
                        raise ValueError(
                            "A reused terminal forecast differs from its original source"
                        )
                    refs.append(["parent", name])
                else:
                    key = f"{name}_{stage}"
                    targets[key] = {
                        "dialogue": sid,
                        "session": n,
                        "source": digest(raw),
                        "prompts": prompts,
                        "donors": [v["donor"] for v in donors],
                    }
                    refs.append(["joint", key])
            windows[name] = {
                "dialogue": sid,
                "start": start,
                "panel": w["panel"],
                "packets": refs,
            }
    if len(windows) != 64 or len(targets) != 91:
        raise ValueError("Wrong scope of the registered two-write policy")
    frozen = joint / "prompts.json"
    register(
        frozen,
        {
            "plan": file_hash(PLAN),
            "code": file_hash(Path(__file__)),
            "parent": file_hash(parent_file),
            "backend": original["backend"],
            "windows": windows,
            "targets": targets,
        },
    )
    cutoff = frozen.stat().st_mtime_ns
    receipt = {}
    for name, folder, pat in (
        ("scores", base / "late-pairs/outcomes", "*/*/scores.json"),
        ("terminal", parent / "choices", "*.json"),
    ):
        receipt[name] = {
            str(f.relative_to(folder)): file_hash(f)
            for f in sorted(folder.glob(pat))
            if f.stat().st_mtime_ns <= cutoff
        }
    register(joint / "preexisting.json", dict(receipt, registration=file_hash(frozen)))
    measure(joint, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "study", "corpus"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    run(args.base, args.study, args.corpus)
