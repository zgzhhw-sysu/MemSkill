"""Evaluate the exact two-decision return on all four memory-building trajectories."""
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from experiments.controller.audit_source_compression import check_forecasts
from experiments.controller.paired_grounding import window
from experiments.controller.replay_validation_choices import verify_rollout
from experiments.controller.source_compression import file_hash, register
from src.revision_validation import evaluate_bank


CELLS = tuple(itertools.product((0, 1), repeat=2))


def read(path):
    return json.loads(path.read_bytes())


def record_scored_paths(archive, forecast):
    """Bind any public path scores present by the forecast's creation to immutable content hashes."""
    receipt = forecast.with_name("scored-at-freeze.json")
    fingerprint = file_hash(forecast)
    if not receipt.exists():
        cutoff = forecast.stat().st_mtime_ns
        paths = {str(p.relative_to(archive)): file_hash(p) for p in archive.glob("*/*/scores.json")
                 if p.parent.name in {"00", "01", "10", "11"} and p.stat().st_mtime_ns <= cutoff}
        register(receipt, {"forecast": fingerprint, "cutoff_ns": cutoff, "scored_paths": paths})
    saved = read(receipt)
    if saved["forecast"] != fingerprint or any(file_hash(archive / path) != digest
                                              for path, digest in saved["scored_paths"].items()):
        raise ValueError("Published path scores changed after registration")
    return {Path(path).parts[0] for path in saved["scored_paths"]}


def path_mass(bits, p, q):
    a, b = bits
    return (p if a else 1-p) * (q if b else 1-q)


def trajectory_masses(probs, base, terminal, shift):
    """Conditional second-step probabilities use the state reached by the actual first choice."""
    p, q = probs
    hard = [float(v > .5) for v in probs]
    out = {}
    for bits in CELLS:
        a, b = bits
        out[bits] = {"credit": path_mass(bits,p,q), "greedy": path_mass(bits,*hard),
                     "temporal": float(bits == (0,0)), "activity": float(bits == (1,1)),
                     "shuffled": (path_mass(bits,p,q) + path_mass(bits,q,p)) / 2,
                     "greedy_shuffled": (path_mass(bits,*hard) + path_mass(bits,*reversed(hard))) / 2}
        for key, logits, offset in (("native",base,0.), ("terminal",terminal,0.), ("rate",terminal,shift)):
            # Root, then one of the two distinct second-step states.
            u, v = [float(torch.sigmoid(torch.as_tensor(logits[i] + offset, dtype=torch.float64))) for i in (0,1+a)]
            out[bits][key] = path_mass(bits,u,v)
            out[bits][key+"_greedy"] = path_mass(bits,float(u >= .5),float(v >= .5))
    if any(abs(sum(v[arm] for v in out.values())-1.) > 1e-10 for arm in out[(0,0)]):
        raise ValueError("Trajectory probabilities do not sum to one")
    return out


def measure_cell(trainer, convo, root, bits, specs):
    name = "".join(map(str,bits))
    path = root / name
    schedule = list(map(bool,bits))
    if not path.exists():
        with TemporaryDirectory(dir=root) as tmp:
            folder = Path(tmp) / name
            policy = trainer.controller
            policy.mode, policy.schedule, policy.weights = "count", schedule, None
            policy.decisions.clear()
            usage = Path(tmp) / "usage.jsonl"
            usage.touch()
            os.environ["MEMSKILL_USAGE_LOG"] = str(usage)
            evaluate_bank(trainer, trainer.operation_bank, [convo], folder)
            if [r["use"] for r in policy.decisions] != schedule:
                raise ValueError("The memory builder did not execute both assigned actions")
            verify_rollout(trainer, folder, convo, policy.decisions, specs)
            (folder / "decisions.json").write_text(json.dumps(policy.decisions))
            usage.rename(folder / "usage.jsonl")
            folder.rename(path)
    return load_cell(trainer, convo, path, bits, specs)


def load_cell(trainer, convo, path, bits, specs):
    """Verify an archived rollout without invoking the writer or reader."""
    decisions = read(path / "decisions.json")
    if [r["use"] for r in decisions] != list(map(bool, bits)):
        raise ValueError("Saved path differs from its assigned pair of actions")
    verify_rollout(trainer, path, convo, decisions, specs)
    trace, = (path / "traces").glob("*.jsonl")
    spans = [e for line in trace.read_bytes().splitlines() if (e:=json.loads(line))["kind"] == "span"]
    prefix = {k:spans[0][k] for k in ("source","retrieved","selected_skills","changes")}
    scores = {r["qa_index"]:r["f1"] for r in read(path / "scores.json")["answers"]}
    return decisions, prefix, scores


def summarize_window(cells, probs, fit, original_ids):
    def state(row):
        return {k:row[k] for k in ("state","base_logit","plain","grounded")}
    first = cells[(0,0)][0][0]
    for bits, (decisions, prefix, scores) in cells.items():
        if (state(decisions[0]) != state(first) or prefix != cells[bits[0],0][1]
                or state(decisions[1]) != state(cells[bits[0],0][0][1]) or set(scores) != set(range(len(original_ids)))):
            raise ValueError("Inconsistent factual prefix or question set in the four measured trajectories")
    states = [first, cells[(0,0)][0][1], cells[(1,0)][0][1]]
    base = [r["base_logit"] for r in states]
    w = torch.tensor(fit["terminal"],dtype=torch.float64)
    terminal = [r["base_logit"] + float(w[0] + w[1:] @ F.normalize(w.new_tensor(r["state"]),dim=0)) for r in states]
    masses = trajectory_masses(probs,base,terminal,fit["matched_logit_shift"])
    per_question = [{"original_index":orig, "f1":{arm:sum(masses[k][arm]*cells[k][2][i] for k in CELLS)
                       for arm in masses[(0,0)]}} for i,orig in enumerate(original_ids)]
    return {"source_probabilities":probs, "path_mass":{"".join(map(str,k)):v for k,v in masses.items()},
            "per_question":per_question}


def run(trainer, study, forecast, train, specs):
    base = forecast.parent
    frozen = read(forecast)
    feature = read(base / "features.json")
    protocol = read(study / "protocol.json")
    if (frozen["inputs"] != {name:file_hash(base/name) for name in ("features.json","plan.json")}
            or feature["corpus"] != protocol["input_sha256"] or set(protocol["train"]) != {c["sample_id"] for c in train}
            or not protocol.get("all_linked_questions") or not protocol.get("equal_sources")
            or protocol["train_sessions"] != [[1,2],[3,4],[5,6],[7,8]]):
        raise ValueError("The factorial panel differs from the frozen study")
    features = {(r["dialogue"], r["session"]):r for r in feature["rows"]}
    with torch.device(trainer.args.device):
        check_forecasts(features, frozen)
    probs = {(r["dialogue"],r["session"]):r["p_activity"] for r in frozen["forecasts"]}
    output = base / "factorial"
    output.mkdir(exist_ok=True)
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(output / "responses")
    versions = [s["version"] for s in specs]
    writer = trainer.executor.writer_context
    manifest = {"forecast":file_hash(forecast), "protocol":file_hash(study/"protocol.json"),
                "backend":os.environ["MEMSKILL_BACKEND_ID"], "writer_context":writer.signature if writer else None,
                "windows":[[sid,start] for sid in protocol["train"] for start in (5,7)], "cells": ["00","01","10","11"]}
    register(output/"manifest.json", manifest)
    measurements = []
    for c in train:
        sid = c["sample_id"]
        for start in (5,7):
            convo = window(c,start,2)
            for n in (start,start+1):
                k = f"session_{n}"
                source, = trainer.data_processor._extract_session_chunks(convo["conversation"][k],convo["conversation"][k+"_date_time"])
                if hashlib.sha256(source.encode()).hexdigest() != features[sid,n]["source_sha256"]:
                    raise ValueError("Forecast source differs from the actual memory-builder input")
            root = output/f"{sid}_s{start}"
            root.mkdir(exist_ok=True)
            with (root/"measure.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                cells = {bits:measure_cell(trainer,convo,root,bits,versions) for bits in CELLS}
                with torch.device(trainer.args.device):
                    report = summarize_window(cells,[probs[sid,n] for n in (start,start+1)],frozen["fits"][sid],
                                              [q["original_index"] for q in convo["qa"]])
                register(root/"measurement.json",report)
            measurements.append((sid,report))
            print(root.name, len(report["per_question"]), "questions verified on four paths", flush=True)
    per_dialogue = {}
    all_questions = []
    for sid in protocol["train"]:
        group = [q["f1"] for key,r in measurements if key==sid for q in r["per_question"]]
        per_dialogue[sid] = {k:mean(x[k] for x in group) for k in group[0]}
        all_questions.extend(group)
    register(output/"summary.json", {"questions":len(all_questions),"windows":len(measurements),
             "aggregate":{k:mean(x[k] for x in all_questions) for k in all_questions[0]},"per_dialogue":per_dialogue})
