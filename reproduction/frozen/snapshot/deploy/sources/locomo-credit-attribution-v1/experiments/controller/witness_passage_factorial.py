"""A physical four-path replication of the complete later training-history panel."""

import argparse
import fcntl
import json
import math
import os
from collections import defaultdict
from itertools import permutations, product
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch

from experiments.controller.audit_paired import reader_inputs
from experiments.controller.complete_donor_paths import decision_context, returns
from experiments.controller.cross_window_prefix import trajectory_hashes
from experiments.controller.factorial_repairs import checked_cell
from experiments.controller.factorial_rollout import CELLS, measure_cell, read
from experiments.controller.late_credit_kernel import support
from experiments.controller.late_pair_trial import paths
from experiments.controller.paired_grounding import window
from experiments.controller.retrieval_competition import budgets, mass, summarize
from experiments.controller.semantic_writer import measured_readers
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import ARMS, probabilities
from src.data_processing.locomo import LoCoMoProcessor
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest, history_embeddings

PLAN = Path(__file__).with_name("protocols") / "witness_passage_four_paths.json"
BACKEND = "251-hf-llama3-bf16-gpu56-duty28-qwen4-b2-duty30"


def setup(base, study):
    late = base / "late-pairs"
    trial = base / "source-compression/witness-passages"
    out = trial / "four-paths"
    out.mkdir(exist_ok=True)
    cohorts = late / "cohort.json"
    split = late / "selection-labels/split.json"
    template = read(split)["windows"]["evaluation"]
    panel = [
        w
        for w in read(cohorts)["assignments"]
        if f"{w['dialogue']}_s{w['start']}" in template
    ]
    unique = {(w["dialogue"], i) for w in panel for i in w["questions"]}
    if (
        len(panel) != 40
        or len(unique) != 378
        or sum(len(w["questions"]) for w in panel) != 378
        or any(
            w["questions"] != template[f"{w['dialogue']}_s{w['start']}"] for w in panel
        )
    ):
        raise ValueError("Wrong four-path training panel")
    reg = out / "registration.json"
    register(
        reg,
        {
            "plan": file_hash(PLAN),
            "cohort": file_hash(cohorts),
            "split": file_hash(split),
            "starts": panel,
            "group": file_hash(trial / "group/forecast.json"),
            "study": file_hash(study / "protocol.json"),
            "backend": BACKEND,
        },
    )
    return reg


def ready(base, trial, w):
    name = f"{w['dialogue']}_s{w['start']}"
    return (base / "late-pairs/outcomes" / name / "pair.json").exists() or (
        trial / "witness/forecast.json"
    ).exists()


def inspect_cell(folder, expected, bits, questions, protocol):
    physical = folder / "".join(map(str, bits))
    (trace,) = (physical / "traces").glob("*.jsonl")
    events = [json.loads(l) for l in trace.read_bytes().splitlines()]
    spans = [e for e in events if e["kind"] == "span"]
    decisions = read(physical / "decisions.json")
    scored = read(physical / "scores.json")["answers"]
    if [v["qa_index"] for v in scored] != list(range(len(questions))):
        raise ValueError("Replica answer order changed")
    contexts = []
    for s, d, h, a in zip(spans, decisions, expected, bits, strict=True):
        spec = d["grounded"] if a else d["plain"]
        if int(d["use"]) != a or [protocol["skills"][i]["version"] for i in spec] != [
            sk["version"] for sk in s["selected_skills"]
        ]:
            raise ValueError("Wrong skill order in replication")
        contexts.append(decision_context(s, d, h, a))
    return contexts, decisions, scored


def verify_responses(trainer, convo, cell, bits, specs, cache):
    checked_cell(trainer, convo, cell, bits, specs, cache)
    (trace,) = (cell / "traces").glob("*.jsonl")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    rows = read(cell / "scores.json")["answers"]
    prompts = reader_inputs(events, rows, convo["qa"], trainer.evaluator)
    responses = measured_readers(cell, cache)
    if responses.keys() != {h for h, _ in prompts.values()}:
        raise ValueError("Replica reader prompt set differs from the response archive")
    for row in rows:
        key, _ = prompts[row["qa_index"]]
        payload = responses[key]
        if (
            payload["tokens"] != row["tokens"]
            or payload["response"].strip() != row["prediction"]
        ):
            raise ValueError(
                "Replica reader tokens or answer differ from the archived response"
            )


def summarize_window(folder, w, packet, folds, terminal, protocol):
    name = folder.name
    sid = w["dialogue"]
    starts = [f"{sid}_s{w['start'] + i}" for i in (0, 1)]
    sources = [packet["sources"][k]["text"] for k in starts]
    states = {}
    ps = {}
    cases = {}
    cost = {}
    for bits in CELLS:
        c, d, sc = inspect_cell(folder, sources, bits, w["questions"], protocol)
        for j, node in enumerate((0, 1 + bits[0])):
            model = folds[sid, j], packet["offsets"][starts[j]]
            p = probabilities(d[j], model, terminal[sid])
            if states.setdefault(node, c[j]) != c[j] or ps.setdefault(node, p) != p:
                raise ValueError(
                    "Different reached context across equivalent decision prefixes"
                )
        cases[bits] = [r["f1"] for r in sc]
        cost[bits] = budgets(folder, "".join(map(str, bits)))
    if any(abs(sum(mass(b, ps, k) for b in CELLS) - 1) > 1e-10 for k in ARMS):
        raise ValueError("Incomplete trajectory probability")
    chosen = paths(w["pivot"], w["background"]) if "pivot" in w else None
    u = [
        int(digest(["witness-four-path-draw", name, j])[:13], 16) / 16**13
        for j in (0, 1)
    ]

    def sample(k):
        a = int(u[0] < ps[0][k])
        return a, int(u[1] < ps[1 + a][k])

    draws = {k: sample(k) for k in ARMS}
    rows = []
    for index, (original, category) in enumerate(
        zip(w["questions"], w["types"], strict=True)
    ):
        row = {
            "index": original,
            "type": category,
            "f1": {
                k: sum(mass(b, ps, k) * cases[b][index] for b in CELLS) for k in ARMS
            },
            "draw": {k: cases[b][index] for k, b in draws.items()},
        }
        if chosen is not None:
            vals = returns(cases, ps, chosen, w["pivot"], w["background"], index)
            row.update(
                ht={k: v["ht"] for k, v in vals.items()},
                focal={k: v["focal"] for k, v in vals.items()},
            )
        rows.append(row)
    result = {
        "window": name,
        "dialogue": sid,
        "rows": rows,
        "probabilities": ps,
        "decision_contexts": states,
        "draws": draws,
        "expected_tokens": {
            k: {
                op: sum(mass(b, ps, k) * cost[b][op] for b in CELLS)
                for op in ("reader", "executor")
            }
            for k in ARMS
        },
    }
    if chosen is not None:
        result["focal_probabilities"] = ps[
            0 if w["pivot"] == 0 else 1 + w["background"]
        ]
    return result


def backend_check(base, folder, w):
    parent = base / "late-pairs/outcomes" / folder.name
    if not (parent / "pair.json").exists():
        return None
    pair = read(parent / "pair.json")
    f1 = []
    same = []
    for bits in paths(w["pivot"], w["background"]):
        k = "".join(map(str, bits))
        a, b = [read(p / k / "scores.json")["answers"] for p in (parent, folder)]
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if (
                x["qa_index"] != i
                or y["qa_index"] != i
                or x["f1"]
                != pair["questions"][i][("temporal", "activity")[bits[w["pivot"]]]]
            ):
                raise ValueError("Cannot align replicas with original randomized arm")
            same.append(x["prediction"] == y["prediction"])
            f1.append(y["f1"] - x["f1"])
    result = {
        "primary_pair": file_hash(parent / "pair.json"),
        "responses": len(f1),
        "identical": sum(same),
        "mean_f1_difference": mean(f1),
    }
    register(folder / "backend-comparison.json", result)
    return result


def summarize_set(panels):
    stats = summarize(panels, ARMS, "credit", ARMS[1:])
    if not panels:
        return stats
    rows = [r for w in panels for r in w["rows"]]
    stats["sampled_f1"] = {k: mean(r["draw"][k] for r in rows) for k in ARMS}
    stats["windows_with_sampled_action_change_vs_raw"] = sum(
        w["draws"]["credit"] != w["draws"]["raw"] for w in panels
    )
    return stats


def early_archive(base, study, corpus, reg):
    """Read the separate 12 early factorial windows, never used as prospective results."""
    trial = reg.parent.parent
    root = trial.parent / "factorial"
    protocol = read(study / "protocol.json")
    manifest = read(root / "manifest.json")
    panel = [(sid, n) for sid in protocol["train"] for n in (5, 7)]
    if manifest["protocol"] != file_hash(study / "protocol.json") or manifest[
        "windows"
    ] != list(map(list, panel)):
        raise ValueError("Unexpected early archive")
    data = {
        r["sample_id"]: r for r in read(corpus) if r["sample_id"] in protocol["train"]
    }
    train = read(trial.parent / "capped-source-credit/policy/joint-repaired/fit.json")[
        "observations"
    ]
    terminal = read(base / "late-pairs/frozen.json")["terminal"]
    folds = {
        (sid, j): support(train, sid, j, "cuda")[0] for sid in terminal for j in (0, 1)
    }
    audit = SimpleNamespace(
        evaluator=LoCoMoEvaluator(SimpleNamespace()),
        data_processor=LoCoMoProcessor(chunk_mode="full-session"),
    )
    for stage in ("group", "witness"):
        forecast = trial / stage / "forecast.json"
        dest = trial / stage / "early-factorial.json"
        if not forecast.is_file() or dest.is_file():
            continue
        packet = read(forecast)
        observed = []
        originals = {}
        unique = set()
        for sid, n in panel:
            convo = window(data[sid], n, 2)
            folder = root / f"{sid}_s{n}"
            measured = folder / "measurement.json"
            previous = read(measured)["per_question"]
            ids = [q["original_index"] for q in convo["qa"]]
            if ids != [r["original_index"] for r in previous]:
                raise ValueError("Early archive question order changed")
            for bits in CELLS:
                cell = folder / "".join(map(str, bits))
                verify_responses(
                    audit, convo, cell, bits, protocol["skills"], root / "responses"
                )
            w = {
                "questions": ids,
                "types": [q["category"] for q in convo["qa"]],
                "dialogue": sid,
                "start": n,
            }
            result = summarize_window(folder, w, packet, folds, terminal, protocol)
            if any(
                not math.isclose(r["f1"]["native"], p["f1"]["native"], abs_tol=1e-10)
                for r, p in zip(result["rows"], previous, strict=True)
            ):
                raise ValueError(
                    "Published early native trajectory value did not reproduce"
                )
            observed.append(result)
            unique.update((sid, i) for i in ids)
            originals[folder.name] = file_hash(measured)
        stats = summarize_set(observed)
        if stats["questions"] != 91 or len(unique) != 91 or stats["windows"] != 12:
            raise ValueError("Different retrospective training population")
        register(
            dest,
            {
                "forecast": file_hash(forecast),
                "archive": file_hash(root / "manifest.json"),
                "measurements": originals,
                "all": stats,
                "windows": observed,
            },
        )


def assignments(matches):
    """Preserve each history's labels and the one estimate per distinct question."""
    bank = defaultdict(dict)
    for rows in matches.values():
        for row in rows:
            sid, idx = row["donor"]
            if bank[sid].setdefault(idx, row["credit"]) != row["credit"]:
                raise ValueError("Donor signed credit changed across held histories")
    groups = [sorted(bank[sid].items()) for sid in sorted(bank)]
    variants = [sorted(set(permutations([v for _, v in g]))) for g in groups]
    keys = [
        (sid, idx) for sid, g in zip(sorted(bank), groups, strict=True) for idx, _ in g
    ]
    observed = [v for g in groups for _, v in g]
    settings = [[x for g in order for x in g] for order in product(*variants)]
    return keys, settings, settings.index(observed)


def audit_labels(trial, root, stage):
    dest = trial / stage / "early-label-assignments.json"
    early = trial / stage / "early-factorial.json"
    if dest.exists() or not early.exists():
        return
    forecast = trial / stage / "forecast.json"
    pack = read(forecast)
    panels = read(early)
    if panels["forecast"] != file_hash(forecast):
        raise ValueError("Different early forecast")
    keys, variants, original = assignments(pack["matches"])
    labels = torch.tensor(variants, device="cuda", dtype=torch.float64)
    offsets = {}
    for name, rows in pack["matches"].items():
        if not rows:
            offsets[name] = labels.new_zeros(len(variants))
            continue
        ids = [keys.index(tuple(r["donor"])) for r in rows]
        if any(r["donor"][0] == name.rsplit("_s", 1)[0] for r in rows):
            raise ValueError("Held-history label entered forecast")
        terms = labels[:, ids] * labels.new_tensor([r["strength"][0] for r in rows])
        dz = 2 * (terms.amax(1).clamp_min(0) + terms.amin(1).clamp_max(0))
        if abs(float(dz[original]) - pack["offsets"][name]["credit"]) > 1e-10:
            raise ValueError(
                "Observed label alignment did not reproduce the registered offset"
            )
        offsets[name] = dz
    values = []
    for w in panels["windows"]:
        probs = {}
        sid, n = w["window"].rsplit("_s", 1)
        for node in (0, 1, 2):
            raw = labels.new_tensor(w["probabilities"][str(node)]["raw"])
            probs[node] = (
                raw.logit() + offsets[f"{sid}_s{int(n) + (node != 0)}"]
            ).sigmoid()
        returns = labels.new_zeros((len(variants), len(w["rows"])))
        for a, b in CELLS:
            prob = (probs[0] if a else 1 - probs[0]) * (
                probs[1 + a] if b else 1 - probs[1 + a]
            )
            raw = read(root / w["window"] / f"{a}{b}" / "scores.json")["answers"]
            if [r["qa_index"] for r in raw] != list(range(len(w["rows"]))):
                raise ValueError("Incomplete question group")
            returns += prob[:, None] * labels.new_tensor([r["f1"] for r in raw])
        if not torch.allclose(
            returns[original],
            labels.new_tensor([r["f1"]["credit"] for r in w["rows"]]),
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                "Actual assignment did not reproduce every expected answer score"
            )
        values.append(returns)
    f1 = torch.cat(values, dim=1).mean(1)
    register(
        dest,
        {
            "early": file_hash(early),
            "forecast": file_hash(forecast),
            "keys": keys,
            "assignments": variants,
            "observed": original,
            "questions": sum(v.shape[1] for v in values),
            "f1": f1.tolist(),
            "strictly_better": int((f1 > f1[original] + 1e-12).sum()),
            "tied": int(((f1 - f1[original]).abs() <= 1e-12).sum()),
        },
    )


def score_ready(base, reg, protocol):
    out = reg.parent
    trial = out.parent
    late = base / "late-pairs"
    train = read(trial.parent / "capped-source-credit/policy/joint-repaired/fit.json")[
        "observations"
    ]
    weights = read(late / "frozen.json")["terminal"]
    completed = [
        w
        for w in read(reg)["starts"]
        if (
            out / "cells" / f"{w['dialogue']}_s{w['start']}" / "complete.json"
        ).is_file()
    ]
    if not completed:
        return
    folds = {
        (sid, j): support(train, sid, j, "cuda")[0] for sid in weights for j in (0, 1)
    }
    for stage in ("group", "witness"):
        forecast = trial / stage / "forecast.json"
        if not forecast.exists():
            continue
        dest = out / stage
        dest.mkdir(exist_ok=True)
        packet = read(forecast)
        cutoff = forecast.stat().st_mtime_ns
        prior = {
            f.parent.parent.name
            for directory in (late / "outcomes", out / "cells")
            for f in directory.glob("conv-*/*/scores.json")
            if f.stat().st_mtime_ns <= cutoff
        }
        lock = dest / "preexisting.json"
        register(
            lock,
            {
                "forecast": file_hash(forecast),
                "registration": file_hash(reg),
                "previously_scored": sorted(prior),
            },
        )
        panels = []
        for w in completed:
            name = f"{w['dialogue']}_s{w['start']}"
            folder = out / "cells" / name
            complete = read(folder / "complete.json")
            if complete["registration"] != file_hash(reg) or any(
                trajectory_hashes(folder / k) != h for k, h in complete["paths"].items()
            ):
                raise ValueError("Original replication files changed")
            f = dest / (name + ".json")
            if not f.exists():
                v = summarize_window(folder, w, packet, folds, weights, protocol)
                register(
                    f,
                    dict(
                        v,
                        complete=file_hash(folder / "complete.json"),
                        forecast=file_hash(forecast),
                    ),
                )
            v = read(f)
            if v["complete"] != file_hash(folder / "complete.json") or v[
                "forecast"
            ] != file_hash(forecast):
                raise ValueError("Different exact policy values")
            panels.append(v)
            backend_check(base, folder, w)
        result = {
            "lock": file_hash(lock),
            "all": summarize_set(panels),
            "strict": summarize_set([v for v in panels if v["window"] not in prior]),
        }
        register(dest / f"through-{len(panels):02d}.json", result)
        if len(panels) == 40:
            register(dest / "summary.json", result)


def measure(trainer, study, reg, train, specs):
    base = reg.parents[3]
    if reg != setup(base, study) or os.environ["MEMSKILL_BACKEND_ID"] != BACKEND:
        raise ValueError("Wrong trial or physical backend")
    out = reg.parent
    trial = out.parent
    protocol = read(study / "protocol.json")
    versions = [s["version"] for s in specs]
    if specs != protocol["skills"]:
        raise ValueError("Study skill bank changed")
    data = {v["sample_id"]: v for v in train}
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    with (out / "measure.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for w in read(reg)["starts"]:
            name = f"{w['dialogue']}_s{w['start']}"
            folder = out / "cells" / name
            if (folder / "complete.json").exists() or not ready(base, trial, w):
                continue
            convo = window(data[w["dialogue"]], w["start"], 2)
            if [q["original_index"] for q in convo["qa"]] != w["questions"]:
                raise ValueError("Replica corpus mapping changed")
            folder.mkdir(parents=True, exist_ok=True)
            missing = [
                b
                for b in CELLS
                if not (folder / "".join(map(str, b)) / "scores.json").exists()
            ]
            if missing:
                with history_embeddings():
                    measure_cell(trainer, convo, folder, missing[0], versions)
                print("measured", name, missing[0], flush=True)
            if len(missing) <= 1:
                for bits in CELLS:
                    cell = folder / "".join(map(str, bits))
                    verify_responses(
                        trainer, convo, cell, bits, specs, out / "responses"
                    )
                register(
                    folder / "complete.json",
                    {
                        "registration": file_hash(reg),
                        "paths": {
                            "".join(map(str, b)): trajectory_hashes(
                                folder / "".join(map(str, b))
                            )
                            for b in CELLS
                        },
                    },
                )
            break
        score_ready(base, reg, protocol)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("study", type=Path)
    p.add_argument("--score", action="store_true")
    p.add_argument("--early-archive", type=Path, metavar="LOCOMO_DATA")
    a = p.parse_args()
    reg = setup(a.base, a.study)
    if a.score or a.early_archive is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected one guarded CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.06)
        if a.early_archive is not None:
            early_archive(a.base, a.study, a.early_archive, reg)
            for stage in ("group", "witness"):
                audit_labels(
                    reg.parent.parent, a.base / "source-compression/factorial", stage
                )
        if a.score:
            with (reg.parent / "measure.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                score_ready(a.base, reg, read(a.study / "protocol.json"))
    print(reg)
