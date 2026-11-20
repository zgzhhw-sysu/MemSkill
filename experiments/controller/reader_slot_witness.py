"""Test whether changing one retrieved rank can remove or recover a verified answer."""

import argparse
import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.reader_slot_credit import (
    archive,
    endpoint,
    measure_response,
    read,
    register,
    sha,
    slots,
)
from experiments.controller.source_compression import file_hash
from src.credit_assignment import recovered
from src.eval.locomo import LoCoMoEvaluator
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "reader_slot_witness.json"


def at_rank(ctx, other, i):
    if not 0 <= i < max(len(ctx), len(other)):
        raise ValueError("Rank outside both observed readers")
    replacement = [other[i][1]] if i < len(other) else []
    return [v[1] for v in ctx[:i]] + replacement + [v[1] for v in ctx[i + 1 :]]


def run(base, corpus, measure):
    root = base / "source-compression/reader-slot-credit"
    out = root / "single-rank"
    out.mkdir(exist_ok=True)
    manifests = root.parent / "capped-source-credit/registration.json"
    inputs = {
        "plan": file_hash(PLAN),
        "cases": file_hash(root / "verified-cases.json"),
        "cohort": file_hash(root / "cohort.json"),
        "corpus": file_hash(corpus),
        "manifest": file_hash(manifests),
    }
    if any(
        read(root / "registration.json")[k] != inputs[k] for k in ("corpus", "manifest")
    ):
        raise ValueError("Unverified data or paths for policy witnesses")
    manifest = read(manifests)["windows"]
    parent = read(root / "cohort.json")
    evidence = read(root / "verified-cases.json")
    if evidence["cohort"] != inputs["cohort"]:
        raise ValueError("Verified cases came from a different rank grouping")
    cases = {(c["window"], tuple(c["bits"]), c["index"]): c for c in evidence["rows"]}
    selected = [(c, r) for c in parent["contrasts"] for r in c["rows"] if r["credit"]]
    selected.sort(key=lambda cr: "other" not in cr[1]["slots"])
    if len(cases) != len(selected) or set(cases) != {
        (c["window"], tuple(c["bits"]), r["index"]) for c, r in selected
    }:
        raise ValueError("Lost a verified intervention")
    subjects = {c["dialogue"] for c, _ in selected}
    data = {c["sample_id"]: c["qa"] for c in read(corpus) if c["sample_id"] in subjects}
    ev = LoCoMoEvaluator(SimpleNamespace())
    path = out / "requests.json"
    if not path.exists():
        jobs = []
        for c, r in selected:
            proof = cases[c["window"], tuple(c["bits"]), r["index"]]
            sid, j = c["dialogue"], c["stage"]
            w = manifest[c["window"]]
            folder = archive(base, w["panel"])
            qs = [data[sid][i] for i in w["questions"]]
            i = w["questions"].index(r["index"])
            q = qs[i]
            os.environ["MEMSKILL_BACKEND_ID"] = c["backend"]
            os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
            paths = [
                endpoint(
                    folder / w["name"] / k,
                    folder / "responses",
                    w["cells"][k],
                    qs,
                    ev,
                    parent["model"],
                )
                for k in c["bits"]
            ]
            contexts = [x[1][i] for x in paths]
            buckets = [set(restore_text(x[0][: j + 1]).values()) for x in paths]
            tags, _ = slots(*contexts, buckets, j)
            original = [x[2][i] for x in paths]
            if (
                tags != r["slots"]
                or original != [r["low"], r["high"]]
                or proof["credit"] != r["credit"]
            ):
                raise ValueError("Original reader or physical credit did not reproduce")
            win = int(r["credit"] > 0)
            lose = 1 - win
            if not recovered(
                dict(original[lose], ok=True),
                dict(original[win], ok=True),
                ev.get_ground_truth(q),
            ):
                raise ValueError("Registered selection effect lacks a strict recovery")
            for rank, t in enumerate(tags):
                if t != "direct":
                    continue
                prompts = [
                    ev.build_prompt(
                        q["question"], at_rank(contexts[a], contexts[1 - a], rank), q
                    )
                    for a in (win, lose)
                ]
                texts = [x[rank] if rank < len(x) else None for x in contexts]
                jobs.append(
                    {
                        "dialogue": sid,
                        "window": c["window"],
                        "bits": c["bits"],
                        "index": r["index"],
                        "rank": rank + 1,
                        "backend": c["backend"],
                        "credit": r["credit"],
                        "winner": win,
                        "texts": texts,
                        "exclusive_focal": [
                            v is not None and v[0] == j and v[1] not in buckets[1 - a]
                            for a, v in enumerate(texts)
                        ],
                        "original": original,
                        "prompts": prompts,
                    }
                )
        register(path, {"inputs": inputs, "model": parent["model"], "jobs": jobs})
    plan = read(path)
    if plan["inputs"] != inputs:
        raise ValueError("Different one-rank requests")
    if not measure:
        return
    with (out / "measure.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = []
        for i, job in enumerate(plan["jobs"]):
            f = out / f"rank-{i:02d}.json"
            if not f.exists():
                q = data[job["dialogue"]][job["index"]]
                truth = ev.get_ground_truth(q)
                answers = []
                for prompt in job["prompts"]:
                    v = measure_response(
                        base,
                        out,
                        job["backend"],
                        plan["model"],
                        q,
                        sha(prompt),
                        prompt,
                        ev,
                    )
                    answers.append(
                        dict(v, ok=True, f1=ev.compute_f1(v["answer"], truth, q))
                    )
                w = job["winner"]
                winner, loser = (dict(job["original"][k], ok=True) for k in (w, 1 - w))
                report_row = {
                    "requests": file_hash(path),
                    "job": i,
                    "answers": answers,
                    "necessary_in_winner": recovered(answers[0], winner, truth),
                    "sufficient_in_loser": recovered(loser, answers[1], truth),
                    "winner_drop": winner["f1"] - answers[0]["f1"],
                    "loser_gain": answers[1]["f1"] - loser["f1"],
                }
                register(f, report_row)
            res = read(f)
            if res["requests"] != file_hash(path) or res["job"] != i:
                raise ValueError("Different rank result")
            report.append(res)
        register(
            out / "summary.json",
            {
                "requests": file_hash(path),
                "ranks": len(report),
                "necessary": sum(r["necessary_in_winner"] for r in report),
                "sufficient": sum(r["sufficient_in_loser"] for r in report),
                "files": {
                    str(r["job"]): file_hash(out / f"rank-{r['job']:02d}.json")
                    for r in report
                },
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("--read", action="store_true")
    a = p.parse_args()
    run(a.base, a.corpus, a.read)
