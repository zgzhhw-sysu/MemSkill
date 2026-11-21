"""Bound the issued credit-minus-raw return on every registered later evaluation question."""

import json
import os
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.controller.earliest_skill_evaluation import contrast_bounds
from experiments.controller.factorial_rollout import read
from experiments.controller.semantic_writer import measured_readers, sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited
from src.credit_assignment import execution_fault
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest
from src.replay import restore_text


def binary_scores(base, spec, prediction, history):
    late = base / "late-pairs"
    name = f"{spec['dialogue']}_s{spec['start']}"
    folder = late / "outcomes" / name
    source = base / "source-compression/retrieval-competition/pairs" / (name + ".json")
    n = len(spec["questions"])
    values = [[None, None, None] for _ in range(n)]
    pf = folder / "pair.json"
    if not pf.is_file():
        return values, {}, "waiting_for_binary_pair"
    p = read(pf)
    c = read(source) if source.is_file() else {}
    if c and (
        c["window"] != name or c["pivot"] != 1 or c["inputs"]["pair"] != file_hash(pf)
    ):
        raise ValueError("Unbound binary state archive")
    if (
        (p["dialogue"], p["start"], p["pivot"]) != (spec["dialogue"], spec["start"], 1)
        or p["background"] != prediction["background"]
        or p["paths"] != [[p["background"], j] for j in range(2)]
        or [r["original_index"] for r in p["questions"]] != spec["questions"]
    ):
        raise ValueError("Changed binary decision or questions")
    parent = None
    hashes = {"pair": file_hash(pf)}
    if c:
        hashes["state_pair"] = file_hash(source)
    ev = LoCoMoEvaluator(SimpleNamespace())
    bad = []
    for j, skill in enumerate(("capture_temporal_context", "capture_activity_details")):
        bits = str(spec["background"]) + str(j)
        branch = folder / bits
        (trace,) = (branch / "traces").glob("*.jsonl")
        sf = branch / "scores.json"
        hashes[bits] = [file_hash(trace), file_hash(sf)]
        if c and hashes[bits][1] != c["paths"][bits]:
            raise ValueError("Changed binary physical scores")
        events = list(map(json.loads, trace.read_bytes().splitlines()))
        a, b = (v for v in events if v["kind"] == "span")
        before = [
            list(restore_text([a]).items()),
            a["selected_skills"],
            b["selected_skills"][:-1],
            b["source"],
            b["source_turns"],
            b["refs"],
            b["retrieved"],
        ]
        if (
            (parent is not None and before != parent)
            or b["selected_skills"][-1]["name"] != skill
            or sha(b["source"]) != prediction["source"]
        ):
            raise ValueError("Different state at intervention")
        parent = before
        if any(
            execution_fault(v["parsed_actions"], v["applied"], v["skills"])
            for v in (a, b)
        ):
            bad.append(skill)
            continue
        qs = [v for v in events if v["kind"] == "qa"]
        scores = read(sf)["answers"]
        live = list(restore_text([a, b]).items())
        versions = {z["memory_id"]: z["version"] for e in (a, b) for z in e["changes"]}
        cache = measured_readers(branch, late / "outcomes/responses")
        hashes[bits].append(file_hash(branch / "usage.jsonl"))
        for i, (r, q, row, idx, cat) in enumerate(
            zip(
                scores,
                qs,
                p["questions"],
                spec["questions"],
                spec["types"],
                strict=True,
            )
        ):
            if (
                r["qa_index"] != i
                or row["type"] != cat
                or row["original_index"] != idx
                or [q[k] for k in ("prediction", "tokens", "f1")]
                != [r[k] for k in ("prediction", "tokens", "f1")]
                or row[("temporal", "activity")[j]] != q["f1"]
            ):
                raise ValueError("Changed archived question")
            if q["success"]:
                qa = history["qa"][idx]
                ix = r["retrieved_indices"]
                refs = [[live[t][0], versions[live[t][0]]] for t in ix]
                prompt = ev.build_prompt(qa["question"], [live[t][1] for t in ix], qa)
                saved = cache.get(sha(prompt))
                if not saved or (
                    refs != q["refs"]
                    or saved["response"].strip() != r["prediction"]
                    or saved["tokens"] != r["tokens"]
                    or qa["question"] != q["question"]
                    or int(qa["category"]) != cat
                    or abs(
                        ev.compute_f1(r["prediction"], ev.get_ground_truth(qa), qa)
                        - r["f1"]
                    )
                    > 1e-10
                ):
                    raise ValueError(
                        "Reader QA, retrieved memory or authenticated answer changed"
                    )
                values[i][j] = q["f1"]
    return (
        values,
        hashes,
        ("partial_execution_fault:" + ",".join(bad) if bad else "binary"),
    )


def signed_direction_bounds(raw, delta, scores):
    """Extremize any sign-preserving logit update shared by all questions in a window."""
    weights = (raw / 0.05).softmax(0)
    signs = delta.sign().tolist()
    negative = [i for i, x in enumerate(signs) if x < 0]
    candidates = [weights]
    # A negative action ranges from its raw weight to zero. A positive one
    # ranges from its raw weight to infinity, where it takes all learned mass.
    for k in range(1, min(len(negative), len(signs) - 1) + 1):
        for removed in combinations(negative, k):
            p = weights.clone()
            p[list(removed)] = 0
            candidates.append(p / p.sum())
    for i, x in enumerate(signs):
        if x > 0:
            vertex = torch.zeros_like(weights)
            vertex[i] = 1
            candidates.append(vertex)
    values = torch.stack(
        [contrast_bounds(0.8 * (p - weights), scores) for p in candidates]
    )
    return [float(values[:, 0].min()), float(values[:, 1].max())]


def report(base, forecast):
    out = forecast.parent
    panel = base / "late-pairs/semantic-writer/panel.json"
    split = base / "late-pairs/selection-labels/split.json"
    pf = read(forecast)
    latest = max(out.glob("through-*.json"), default=None)
    if latest is None:
        return
    full = read(latest)
    if full["forecast"] != file_hash(forecast):
        raise ValueError("Wrong full-return forecast")
    covered = {r["window"]: r for r in full["rows"] if r["partition"] == "evaluation"}
    registered = read(split)["windows"]["evaluation"]
    spec = read(panel)["windows"]
    corpus = Path(os.environ["LOCOMO_DATA"])
    key = file_hash(corpus)
    if key != read(base / "late-pairs/cohort.json")["corpus"]:
        raise ValueError("Changed registered corpus")
    histories = {
        r["sample_id"]: r
        for r in read(corpus)
        if r["sample_id"] in {w["dialogue"] for w in spec}
    }
    windows = {}
    directions = {}
    sources = {}
    for w in spec:
        name = f"{w['dialogue']}_s{w['start']}"
        if name not in registered:
            continue
        if registered[name] != w["questions"]:
            raise ValueError("Changed evaluation panel")
        f = pf["windows"][name]
        s = f["scores"]
        if name in covered:
            qf = out / "questions" / (name + ".json")
            obs = read(qf)
            if (
                obs["source"] != full["sources"][name]
                or obs["forecast"] != full["forecast"]
                or [q["index"] for q in obs["rows"]] != w["questions"]
            ):
                raise ValueError("Changed observed triplet")
            values = [q["f1"] for q in obs["rows"]]
            hashes = {"triplets": file_hash(qf)}
            status = "triplet"
        else:
            values, hashes, status = binary_scores(base, w, f, histories[w["dialogue"]])
        if "outlier" in s:
            raw = s["gate_utilities"]["raw"]
            changed = [a + b for a, b in zip(raw, s["outlier"]["credit"], strict=True)]
        else:
            raw = s["gate_utilities"]["raw"]
            changed = s["gate_utilities"]["credit"]
        with limited():

            def t(v):
                return torch.as_tensor(v, dtype=torch.float64, device="cuda")

            delta = 0.8 * ((t(changed) / 0.05).softmax(0) - (t(raw) / 0.05).softmax(0))
            if name in covered and not torch.allclose(
                delta,
                t(covered[name]["p"]["credit"]) - t(covered[name]["p"]["raw"]),
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Native mixture did not cancel")
            table = t(
                [[float("nan") if v is None else v for v in row] for row in values]
            )
            bounds = contrast_bounds(delta, table).tolist()
            controls = {}
            for key_control in ("zero_new", "previous", "unreplicated"):
                if key_control not in s.get("outlier", {}):
                    continue
                reference = t(raw) + t(s["outlier"][key_control])
                shift = 0.8 * (
                    (t(changed) / 0.05).softmax(0) - (reference / 0.05).softmax(0)
                )
                if name in covered:
                    issued = covered[name]["p"]
                    if not torch.allclose(
                        shift,
                        t(issued["credit"]) - t(issued[key_control]),
                        rtol=0,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "Expanded-credit comparison changed the issued probabilities"
                        )
                controls[key_control] = contrast_bounds(shift, table).tolist()
            if "unreplicated" in s.get("outlier", {}):
                update = t(s["outlier"]["credit"])
                envelope = signed_direction_bounds(t(raw), update, table)
                if envelope[0] > bounds[0] + 1e-10 or envelope[1] < bounds[1] - 1e-10:
                    raise ValueError("Signed class excludes the issued correction")
                directions[name] = {
                    "n": len(values),
                    "signs": update.sign().tolist(),
                    "bounds": envelope,
                }
        windows[name] = {
            "n": len(values),
            "status": status,
            "three_skill_status": full["excluded"].get(name),
            "known": [sum(row[j] is not None for row in values) for j in range(3)],
            "sum_delta_bounds": bounds,
            **({"controls": controls} if controls else {}),
            "probability_delta": delta.tolist(),
        }
        sources[name] = hashes
    n = sum(r["n"] for r in windows.values())
    if n != 161 or len(windows) != 18:
        raise ValueError("Incomplete planned evaluation")
    inputs = {
        "forecast": full["forecast"],
        "through": file_hash(latest),
        "panel": file_hash(panel),
        "split": file_hash(split),
        "sources": sources,
        "corpus": key,
    }
    folder = out / "population"
    folder.mkdir(exist_ok=True)
    register(
        folder / ("bounds-" + digest(inputs) + ".json"),
        {
            "inputs": inputs,
            "windows": windows,
            "n": n,
            "matched": sum(
                r["n"] for r in windows.values() if r["status"] == "triplet"
            ),
            "known": [sum(r["known"][j] for r in windows.values()) for j in range(3)],
            **{
                "credit_minus_" + c: [
                    sum(w["controls"][c][j] for w in windows.values()) / n
                    for j in range(2)
                ]
                for c in next(iter(windows.values())).get("controls", {})
            },
            "credit_minus_raw": [
                sum(r["sum_delta_bounds"][j] for r in windows.values()) / n
                for j in range(2)
            ],
        },
    )
    if directions:
        if set(windows) != set(directions):
            raise ValueError("Incomplete signed policy class")
        measured = folder / ("bounds-" + digest(inputs) + ".json")
        register(
            folder / ("directions-" + digest(inputs) + ".json"),
            {
                "bounds": file_hash(measured),
                "n": n,
                "windows": directions,
                "class_minus_raw": [
                    sum(r["bounds"][j] for r in directions.values()) / n
                    for j in range(2)
                ],
                "eligible_windows": sum(any(r["signs"]) for r in directions.values()),
                "eligible_questions": sum(
                    r["n"] for r in directions.values() if any(r["signs"])
                ),
            },
        )
