"""Reissue the root write before inspecting weaker-skill repairs with a fixed suffix."""

import argparse
from pathlib import Path

from experiments.controller import file_hash, read, register
from experiments.controller.legacy_terminal_symmetry import (
    census,
    source,
    verify_writer,
)

PLAN = Path(__file__).with_name("protocols") / "legacy_root_pivots.json"


def run(base, study, corpus):
    prior = census(base, study, corpus)
    packet = read(prior)
    out = prior.parent.parent / "legacy-root-symmetry"
    writers = out / "writers"
    writers.mkdir(parents=True, exist_ok=True)
    reg = out / "registration.json"
    jobs = packet["root_separate"]
    register(
        reg,
        {
            "terminal_registration": file_hash(prior),
            "plan": file_hash(PLAN),
            "jobs": jobs,
        },
    )
    for job in jobs:
        f = writers / (job["id"] + ".json")
        if f.exists():
            if read(f)["registration"] != file_hash(reg):
                raise ValueError("Different root pivot registration")
            continue
        spans, d = source(job, packet, study)
        low, high = d["control"], d["swap"]
        repeat = d["swap_repeat"]
        if (
            job["pivot"] != 0
            or spans[0]["retrieved"]
            or any(
                high[k] != repeat[k]
                for k in (
                    "actions",
                    "qa",
                    "suffix",
                    "bank_hash",
                    "contract_fault",
                    "writer_prompt_hashes",
                )
            )
        ):
            raise ValueError("Different historical root or repeated alternative")
        (a,) = low["suffix"]
        (b,) = high["suffix"]
        names = [v["name"] for v in spans[1]["selected_skills"]]
        if not a["matches_rollout"] or a["selected"] != names or b["selected"] != names:
            raise ValueError("Historical suffix actions differ")
        idx = str(job["question_index"])
        writer = verify_writer(out, job, spans, d, read(study / "protocol.json"))
        register(
            f,
            {
                "registration": file_hash(reg),
                "writer": writer,
                "suffix": {"control": a, "swap": b},
                "original_f1": [low["qa"][idx]["f1"], high["qa"][idx]["f1"]],
            },
        )
        print(job["id"], writer["status"], flush=True)
    register(
        out / "writers-complete.json",
        {j["id"]: file_hash(writers / (j["id"] + ".json")) for j in jobs},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    run(a.base, a.study, a.corpus)
