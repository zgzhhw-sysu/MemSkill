"""Audit the registered later donors for additional same-question skill comparisons."""

import argparse
from pathlib import Path

from experiments.controller import file_hash, read, register
from experiments.controller.exchange_paths import Ranking
from experiments.controller.legacy_terminal_symmetry import measure
from experiments.controller.relation_skill_repairs import forecast_window
from experiments.controller.skill_graph_data import credits
from src.evaluation_cache import history_embeddings

PLAN = Path(__file__).with_name("protocols") / "skill_graph_extension.json"


def register_donors(base):
    late = base / "late-pairs"
    sem = late / "semantic-writer"
    dest = sem / "three-writer/skill-graph/extra-donors"
    dest.mkdir(exist_ok=True)
    split = late / "selection-labels/split.json"
    groups = read(split)["windows"]
    donors = [
        w
        for w in read(sem / "panel.json")["windows"]
        if f"{w['dialogue']}_s{w['start']}" in groups["donor"]
    ]
    if (
        len(donors) != 7
        or sum(len(w["questions"]) for w in donors) != 48
        or set(groups["donor"]) & set(groups["evaluation"])
    ):
        raise ValueError("Changed later donor partition")
    sources = {}
    for w in donors:
        k = f"{w['dialogue']}_s{w['start']}"
        triple = sem / "three-writer" / (k + ".json")
        pair = sem / k / "pair.json"
        if groups["donor"][k] != w["questions"] or read(triple)[
            "semantic"
        ] != file_hash(pair):
            raise ValueError("Donor did not reproduce registered questions or triplet")
        sources[k] = {
            "triplet": file_hash(triple),
            "pair": file_hash(pair),
            "activity": file_hash(late / "selection-labels" / (k + ".json")),
        }
    lock = dest / "registration.json"
    register(
        lock,
        {
            "plan": file_hash(PLAN),
            "split": file_hash(split),
            "panel": file_hash(sem / "panel.json"),
            "sources": sources,
        },
    )
    return lock, donors


def collect(base, study, corpus, phase):
    lock, donors = register_donors(base)
    late = base / "late-pairs"
    sem = late / "semantic-writer"
    out = lock.parent
    proto = read(study / "protocol.json")
    if (
        file_hash(corpus) != proto["input_sha256"]
        or read(late / "cohort.json")["corpus"] != file_hash(corpus)
        or set(proto["train"]) & set(proto["validation"] + proto["test"])
        or proto["rule"] != 2
    ):
        raise ValueError("Changed training study")
    data = {d["sample_id"]: d for d in read(corpus) if d["sample_id"] in proto["train"]}
    phase_dir = out / phase
    phase_dir.mkdir(exist_ok=True)
    rank = None

    def encoder():
        nonlocal rank
        if rank is None:
            rank = Ranking(proto["retriever"])
        return rank

    hashes = {}
    for w in donors:
        h = w["dialogue"]
        n = w["start"]
        k = f"{h}_s{n}"
        f = phase_dir / (k + ".json")
        prev = out / "prepared" / f.name
        parent = lock if phase == "prepared" else prev
        inputs = {"registration": file_hash(lock), "parent": file_hash(parent)}
        if not f.exists():
            if phase == "prepared":
                with history_embeddings():
                    arms = {
                        kind: forecast_window(
                            sem,
                            late / "outcomes",
                            data[h],
                            h,
                            n,
                            w["background"],
                            proto,
                            encoder,
                            baseline=kind,
                        )
                        for kind in ("activity", "temporal")
                    }
                matched = all(a["matched"] for a in arms.values())
                result = {
                    "matched": matched,
                    "reasons": {
                        k: a.get("reason") for k, a in arms.items() if not a["matched"]
                    },
                    "arms": arms,
                }
            else:
                prepared = read(prev)
                arms = {}
                for k0, arm in prepared["arms"].items():
                    jobs = []
                    for item in arm["jobs"] if prepared["matched"] else []:
                        j = item["job"]
                        v = item["result"]
                        receipt = (
                            measure(out, j, v)
                            if v["status"] == "prepared"
                            else {"status": v["status"], "credit": 0.0}
                        )
                        jobs.append(
                            {
                                "original_index": j["qa"]["original_index"],
                                "raw": j["signed_credit"],
                                "weak": j["weak"],
                                **receipt,
                            }
                        )
                    arms[k0] = {"jobs": jobs}
                result = {
                    "matched": prepared["matched"],
                    "reasons": prepared["reasons"],
                    "arms": arms,
                }
            register(f, dict(result, inputs=inputs))
            print(phase, k, result["matched"], flush=True)
        if read(f)["inputs"] != inputs:
            raise ValueError("Changed donor repair stage")
        hashes[k] = file_hash(f)
    register(out / (phase + "-complete.json"), hashes)


def extra_rows(base):
    lock, donors = register_donors(base)
    out = lock.parent
    late = base / "late-pairs"
    sem = late / "semantic-writer"
    src = base / "source-compression"
    geom = read(src / "matched-span-rank/features.json")["features"]
    marks = {p: read(out / (p + "-complete.json")) for p in ("prepared", "outcomes")}
    rows = []
    excluded = {}
    for w in donors:
        sid = w["dialogue"]
        n = w["start"]
        k = f"{sid}_s{n}"
        name = k + ".json"
        prep = out / "prepared" / name
        f = out / "outcomes" / name
        done = read(f)
        if (
            file_hash(f) != marks["outcomes"][k]
            or file_hash(prep) != marks["prepared"][k]
            or done["inputs"]["parent"] != file_hash(prep)
            or read(prep)["inputs"]["registration"] != file_hash(lock)
        ):
            raise ValueError("Unbound later physical result")
        if not done["matched"]:
            excluded[k] = done["reasons"]
            continue
        triplet = read(sem / "three-writer" / name)
        label = read(late / "selection-labels" / name)
        if (
            not triplet["matched"]
            or label["inputs"]["pair"] != file_hash(late / "outcomes" / k / "pair.json")
            or label["inputs"]["registration"]
            != file_hash(late / "selection-labels/registration.json")
            or label["background"] != w["background"]
            or label["pivot"] != 1
        ):
            raise ValueError("Different later terminal comparison")
        activity, temporal = [
            credits(done["arms"][key]) for key in ("activity", "temporal")
        ]
        points = []
        for q, r, idx in zip(
            triplet["rows"], label["rows"], w["questions"], strict=True
        ):
            y = q["f1"]
            if (
                idx != r["original_index"]
                or idx != q["index"]
                or abs(r["delta"] - (y[1] - y[0])) > 1e-10
                or (
                    r["credit"]
                    and (r["status"] != "survived_probes" or r["credit"] != r["delta"])
                )
            ):
                raise ValueError("Changed activity credit")
            points.append(
                {
                    "index": idx,
                    "category": q["category"],
                    "scores": y,
                    "edges": [r["credit"], temporal.get(idx, 0), activity.get(idx, 0)],
                }
            )
        p = triplet["probabilities"]["native"]
        sf = read(sem / k / "pair.json")["source_forecast"]
        rows.append(
            {
                "window": k,
                "history": sid,
                "additional": True,
                "features": [sf["offset"], geom[f"{sid}_s{n + 1}"][5]],
                "native_pair": [p[1] / (p[0] + p[1]), p[2] / (p[0] + p[2])],
                "points": points,
            }
        )
    return rows, excluded


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        cli.add_argument(k, type=Path)
    cli.add_argument("phase", choices=("prepared", "outcomes"))
    a = cli.parse_args()
    collect(a.base, a.study, a.corpus, a.phase)
