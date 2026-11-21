"""Freeze source-only placement forecasts and evaluate later paired GPU training interventions."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.paired_policy import fit_policy, _training_states, _shift_for_mean_probability
from experiments.controller.probe_omitted import check_responses
from experiments.controller.resume_status import load_episode
from experiments.controller.source_compression import FEATURES, fit_actor
from src.eval import get_evaluator


def panel(study, corpus, features, starts, *, supplement=False):
    protocol = read(study / "protocol.json")
    data = {c["sample_id"]: c for c in json.loads(corpus.read_bytes()) if c["sample_id"] in protocol["train"]}
    if (protocol["input_sha256"] != file_hash(corpus) or set(data) != set(protocol["train"]) or len(data) != 6
            or set(data) & set(protocol["test"] + protocol["validation"]) or not protocol.get("equal_sources")
            or protocol["rule"] != 2 or not protocol.get("deployment_writer")):
        raise ValueError("Unexpected training corpus, split, writer or credit rule")
    evaluator = get_evaluator("locomo", SimpleNamespace())
    rows, receipts, backends = [], {}, set()
    for sid in sorted(data):
        for start in starts:
            ep = study / "train" / f"{sid}_s{start}"
            backend = read(ep / "backend.json")["inference"]
            backends.add(backend)
            ids = read(ep / "questions.json")
            archive = load_episode(ep, sid, 2, ids, deployment_writer=True)
            path, = (ep / "traces").glob("*.jsonl")
            raw = path.read_bytes()
            events = [json.loads(line) for line in raw.splitlines()]
            spans = [e for e in events if e["kind"] == "span"]
            diagnoses = [e["diagnosis"] for e in events if e["kind"] == "span_credit"]
            fingerprint = hashlib.sha256(raw).hexdigest()
            receipts[ep.name] = {"trace": fingerprint, "rows": file_hash(ep / "rows.json")}
            if supplement:
                folder = hashlib.sha256(backend.encode()).hexdigest()[:16]
                probe = study / "diagnostics/omitted-final" / folder / (ep.name + ".json")
                tmp = read(probe)
                receipts[ep.name]["supplement"] = file_hash(probe)
                diag = tmp["diagnosis"]
                extra = [q["original_index"] for q in diag["questions"]] if diag else []
                if tmp["inputs"] != {"episode": ep.name, "backend": backend, "trace_sha256": fingerprint,
                                     "rule": 2, "span": 1, "original_question_ids": extra}:
                    raise ValueError(f"Mismatched supplemental feedback at {ep}")
            for i, (row, span, diagnosis) in enumerate(zip(archive, spans, diagnoses, strict=True)):
                f = features[sid, start+i]
                if f["source_sha256"] != hashlib.sha256(span["source"].encode()).hexdigest():
                    raise ValueError(f"Different pre-action source at {ep}, span {i}")
                effects, scores = {}, []
                branches = [diagnosis]
                if supplement and i == 1 and diag:
                    branches.append(diag)
                for d in branches:
                    if (d["evidence_rule_version"] != 2 or d["span_id"] != i
                            or d["sampled_skills"] != diagnosis["sampled_skills"]
                            or d.get("alternative") != diagnosis.get("alternative")):
                        raise ValueError(f"Different intervention at {ep}")
                    check_responses(d)
                    for q in d["questions"]:
                        qi, k = q["qa_index"], str(q["qa_index"])
                        if ids[qi] != q["original_index"] or qi in effects or (q["controller_delta"] and q["cause"] != "selection"):
                            raise ValueError(f"Invalid labeled training question at {ep}")
                        effects[qi] = q["controller_delta"]
                        control, swap = d.get("control", {}), d.get("swap", {})
                        if (not q["control_matched"] or d.get("execution_fault") or control.get("contract_fault")
                                or swap.get("contract_fault")):
                            continue
                        a, b = [v.get("qa", {}).get(k) for v in (control, swap)]
                        if not a or not b or not a["ok"] or not b["ok"]:
                            continue
                        qa = data[sid]["qa"][ids[qi]]
                        on, off = (a, b) if row["use"] else (b, a)
                        if (a["answer"] != q["original"] or a["f1"] != q["original_f1"]
                                or any(not math.isclose(v["f1"], evaluator.compute_f1(v["answer"], evaluator.get_ground_truth(qa), qa))
                                       for v in (a, b))):
                            raise ValueError(f"Answer and score differ at {ep}")
                        scores.append({"original_index": ids[qi], "activity": on["f1"], "temporal": off["f1"]})
                if not supplement or i == 1:
                    src = set(span["source_turns"])
                    expected = {j for j, orig in enumerate(ids) if src.intersection(data[sid]["qa"][orig].get("evidence") or ())}
                    if expected != effects.keys():
                        raise ValueError(f"Incomplete linked panel at {ep}, span {i}")
                rows.append(dict(row, dialogue=sid, session=start+i, features=f["features"], links=len(effects), scores=scores,
                                 credits={ids[j]: (1 if row["use"] else -1)*v for j,v in effects.items()},
                                 signed=(1 if row["use"] else -1) * (mean(effects.values()) if effects else 0.)))
    if len(backends) != 1:
        raise ValueError("The counterfactual measurements must share one backend")
    return rows, receipts, backends.pop()


def forecasts(train, input_rows):
    fits, predictions = {}, []
    for held in sorted({r["dialogue"] for r in input_rows}):
        started = time.monotonic()
        fold = [r for r in train if r["dialogue"] != held]
        x = torch.tensor([r["features"] for r in fold], dtype=torch.float64)
        center, scale, actor = fit_actor(x, x.new_tensor([r["signed"] for r in fold]))
        logits = actor[0] + (x - center) / scale @ actor[1:]
        terminal = fit_policy(fold, 0)
        design, offset = _training_states(fold)
        base = offset + design @ terminal.double()
        shift = _shift_for_mean_probability(base, logits - base)
        fits[held] = {"training_dialogues": sorted({r["dialogue"] for r in fold}), "center": center.tolist(),
                       "scale": scale.tolist(), "actor": actor.tolist(), "terminal": terminal.tolist(),
                       "matched_logit_shift": shift, "training_mean": float(logits.sigmoid().mean())}
        for r in input_rows:
            if r["dialogue"] == held:
                p = (actor[0] + (actor.new_tensor(r["features"]) - center) / scale @ actor[1:]).sigmoid()
                predictions.append({"dialogue": held, "session": r["session"], "p_activity": float(p)})
        if x.is_cuda:
            torch.cuda.synchronize()
            time.sleep((time.monotonic()-started) * (1/.35 - 1))
    return fits, predictions


def check_forecasts(features, report):
    records = report["forecasts"]
    data = {(r["dialogue"], r["session"]): r for r in records}
    groups = {sid for sid, _ in features}
    if len(data) != len(records) or data.keys() != features.keys() or report["fits"].keys() != groups:
        raise ValueError("Incomplete or duplicate held-dialogue probabilities")
    for (sid, n), r in features.items():
        model = report["fits"][sid]
        if set(model["training_dialogues"]) != groups - {sid}:
            raise ValueError("Registered actor includes an incorrect training dialogue")
        actor = torch.tensor(model["actor"], dtype=torch.float64)
        center, scale = (actor.new_tensor(model[k]) for k in ("center", "scale"))
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("Invalid feature scale")
        logit = actor[0] + ((actor.new_tensor(r["features"]) - center) / scale) @ actor[1:]
        if not math.isclose(float(logit.sigmoid()), data[sid, n]["p_activity"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Saved probability disagrees with the actor")


def summarize(rows, fit, predictions):
    by_key = {(r["dialogue"], r["session"]): r["p_activity"] for r in predictions}
    groups = defaultdict(list)
    for r in rows:
        if not r["scores"]:
            continue
        model = fit[r["dialogue"]]
        p = by_key[r["dialogue"], r["session"]]
        weights = torch.tensor(model["terminal"], dtype=torch.float64)
        base = r["base_logit"] + weights[0] + F.normalize(weights.new_tensor(r["state"]), dim=0) @ weights[1:]
        probs = {"activity": 1., "temporal": 0., "credit": p, "greedy": float(p > .5),
                 "terminal": float(base.sigmoid()), "rate": float((base + model["matched_logit_shift"]).sigmoid())}
        groups[r["dialogue"]].append((r, probs))
    if not groups:
        raise ValueError("No measured two-branch outcomes")
    def aggregate(by_link):
        totals, mass = defaultdict(float), 0
        per_dialogue = {}
        for sid, group in groups.items():
            shuffled = mean(p["credit"] for _, p in group)
            hard = mean(p["greedy"] for _, p in group)
            sums, count = defaultdict(float), 0
            for r, p in group:
                n = len(r["scores"]) if by_link else 1
                a, b = [mean(s[k] for s in r["scores"]) for k in ("activity","temporal")]
                for arm, v in dict(p, shuffled=shuffled, greedy_shuffled=hard).items():
                    sums[arm] += n*(b + v*(a-b))
                count += n
            for arm, value in sums.items():
                totals[arm] += value
            per_dialogue[sid] = {k: v/count for k,v in sums.items()}
            mass += count
        return {"mean": {k: v/mass for k,v in totals.items()}, "dialogues": per_dialogue}
    by_span = aggregate(False)
    s = by_span["mean"]
    passed = (s["credit"] - s["terminal"] >= .02 and s["credit"] > max(s[k] for k in ("rate","activity","temporal","shuffled"))
              and s["greedy"] > max(s[k] for k in ("activity","temporal","greedy_shuffled"))
              and sum(v["credit"] > v["shuffled"] + 1e-9 for v in by_span["dialogues"].values()) >= 4)
    return {"spans": len(rows), "measured_spans": sum(map(len, groups.values())), "links": sum(r["links"] for r in rows),
            "measured_links": sum(len(r["scores"]) for r in rows), "by_span": by_span, "by_link": aggregate(True), "passes": passed}


def run(phase, previous, prospective, corpus, work):
    features = read(work / "features.json")
    if features["features"] != list(FEATURES) or features["corpus"] != file_hash(corpus):
        raise ValueError("Wrong source feature inputs")
    data = {(r["dialogue"], r["session"]): r for r in features["rows"]}
    if len(data) != 48 or len(features["rows"]) != 48:
        raise ValueError("Incomplete or duplicate source features")
    old_protocol, new_protocol = (read(p / "protocol.json") for p in (previous, prospective))
    if (old_protocol["train_sessions"] != [[1,2],[3,4]] or new_protocol["train_sessions"] != [[1,2],[3,4],[5,6],[7,8]]
            or not new_protocol.get("all_linked_questions")
            or any(old_protocol[k] != new_protocol[k] for k in ("train","validation","test","input_sha256","checkpoint_sha256","skills","retriever"))):
        raise ValueError("Expected matching source conditions and disjoint later training windows")
    locked = work / "frozen.json"
    immutable = {k: file_hash(work / k) for k in ("features.json","plan.json")}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select a single CUDA GPU for actor calculations")
    torch.cuda.set_per_process_memory_fraction(.06)
    with torch.device("cuda"):
        if phase == "freeze":
            if locked.exists():
                registered = read(locked)
                check_forecasts(data, registered)
                if registered["inputs"] != immutable:
                    raise ValueError("A different forecast is already registered")
                return
            rows, prior_hashes, backend = panel(previous, corpus, data, (1,3), supplement=True)
            fits, probs = forecasts(rows, features["rows"])
            report = {"inputs": immutable, "backend": backend, "prior_hashes": prior_hashes, "fits": fits, "forecasts": probs,
                      "historical": summarize(rows, fits, probs)}
            check_forecasts(data, report)
            if any((prospective / "train" / f"{sid}_s{n}/rows.json").exists() for sid, n in data if n in (5,7)):
                raise RuntimeError("A prospective target was published before forecast registration")
            register(locked, report)
        else:
            frozen = read(locked)
            check_forecasts(data, frozen)
            if frozen["inputs"] != immutable:
                raise ValueError("Frozen input identity changed")
            rows, hashes, backend = panel(prospective, corpus, data, (5,7))
            if backend != frozen["backend"]:
                raise ValueError("Prospective backend differs from training measurements")
            register(work / "prospective.json", {"forecast_sha256": file_hash(locked), "target_hashes": hashes,
                     "measurement": summarize(rows, frozen["fits"], frozen["forecasts"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    for name in ("previous", "prospective", "corpus", "work"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
