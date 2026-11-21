"""Score frozen pre-write Llama decisions at both roots and conditional terminal states."""

import argparse
import json
from pathlib import Path

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.audit_case_credit import verified_choice
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.retrieval_competition import budgets, summarize
from experiments.controller.writer_block_policy import trajectory_table
from src.evaluation_cache import digest

ARMS = (
    "credit",
    "semantic",
    "verified",
    "permuted",
    "rate",
    "return_only",
    "native",
    "terminal",
)


def numeric_choice(parent, joint, frozen, spec):
    found = []
    for source, key in spec["packets"]:
        d = parent if source == "parent" else joint
        p = d / "choices" / (key + ".json")
        if not p.exists():
            return None
        registration, targets = frozen[source]
        target = targets[key]
        if target["dialogue"] != spec["dialogue"]:
            raise ValueError("Incorrect packet dialogue")
        choice, usage = verified_choice(d, key, target, registration)
        found.append(
            dict(
                choice,
                usage=usage,
                stamp=p.stat().st_mtime_ns,
                fingerprint=file_hash(p),
            )
        )
    return found


def collect(base, name, w, choice, calibration):
    src = base / "source-compression"
    later = w["panel"] == "later"
    if later:
        folder = base / "late-pairs/outcomes" / name
        ref_path = src / "retrieval-competition/pairs" / (name + ".json")
        if not ref_path.exists():
            return None
        ref = read(ref_path)
        pair = read(folder / "pair.json")
        if (
            ref["inputs"]["pair"] != file_hash(folder / "pair.json")
            or pair["dialogue"] != w["dialogue"]
            or pair["start"] != w["start"]
        ):
            raise ValueError("Mismatched randomized comparison")
        paths = ["".join(map(str, b)) for b in pair["paths"]]
        pivot, bg = pair["pivot"], pair["background"]
        expected = [[r[k] for r in pair["questions"]] for k in ("temporal", "activity")]
    else:
        folder = src / "factorial" / name
        paths = ["00", "01", "10", "11"]
        ref = calibration[name]
        pivot, bg = None, None
        expected = None
    cases, costs, states, hashes = {}, {}, {}, {}
    times = []
    for i, bits in enumerate(paths):
        cell = folder / bits
        (trace,) = (cell / "traces").glob("*.jsonl")
        spans = [
            s
            for l in trace.read_bytes().splitlines()
            if (s := json.loads(l))["kind"] == "span"
        ]
        for stage, (s, a, d) in enumerate(
            zip(spans, bits, read(cell / "decisions.json"), strict=True)
        ):
            if (
                s["span_id"] != stage
                or digest(s["source"]) != choice[stage]["source"]
                or int(d["use"]) != int(a)
                or len(s["selected_skills"]) != 3
                or s["selected_skills"][-1]["source_rule"]
                != ("dated_single_turns", "dated_adjacent_turns")[int(a)]
            ):
                raise ValueError("Source or actual skill selection changed")
            node = 0 if stage == 0 else 1 + int(bits[0])
            pre = digest(
                [
                    s["source"],
                    s["refs"],
                    s["retrieved"],
                    s["selected_skills"][:-1],
                    d["state"],
                    d["base_logit"],
                ]
            )
            if states.setdefault(node, pre) != pre:
                raise ValueError("Different decision context across interventions")
        path = cell / "scores.json"
        rows = read(path)["answers"]
        if [r["qa_index"] for r in rows] != list(range(len(ref["rows"]))):
            raise ValueError("Incomplete archived reader scores")
        cases[bits] = [r["f1"] for r in rows]
        if expected is not None and cases[bits] != expected[i]:
            raise ValueError("Randomized outcome differs from its archive")
        costs[bits] = budgets(folder, bits)
        hashes[bits] = file_hash(path)
        times.append(path.stat().st_mtime_ns)
    return {
        "ref": ref,
        "nodes": sorted(states),
        "states": states,
        "cases": cases,
        "costs": costs,
        "pivot": pivot,
        "background": bg,
        "choice": choice,
        "start": w["start"],
        "strict": later and max(c["stamp"] for c in choice) < min(times),
        "inputs": {"scores": hashes, "choices": [c["fingerprint"] for c in choice]},
    }


def evaluate(data, device):
    windows = [
        dict(
            w,
            node_offsets=[w["choice"][j]["offsets"] for j in (0, 1, 1)],
            policy_details={"decision_usage": [c["usage"] for c in w["choice"]]},
        )
        for w in data
    ]
    return evaluate_offsets(
        windows,
        device,
        ARMS,
        ("credit", "semantic", "verified", "permuted"),
        ("native", "terminal", "return_only", "permuted", "semantic", "rate"),
    )


def evaluate_offsets(data, device, arms, learned, comparisons):
    """Score any frozen offsets on verified paths with matching by actual decision node."""
    if not data:
        return {"windows": 0, "questions": 0}, []
    ps = [{} for _ in data]
    for node in range(3):
        ids = [i for i, w in enumerate(data) if node in w["nodes"]]
        if not ids:
            continue
        priors = [data[i]["ref"]["probabilities"][str(node)] for i in ids]
        p = torch.tensor(
            [v["native"] for v in priors], dtype=torch.float64, device=device
        )
        z = p.logit()
        preds = {
            k: (
                z + p.new_tensor([data[i]["node_offsets"][node][k] for i in ids])
            ).sigmoid()
            for k in learned
        }
        shift = matched_offset(z, preds["credit"], torch.ones_like(z))
        preds["rate"] = (z + shift).sigmoid()
        if not torch.isclose(
            preds["credit"].mean(), preds["rate"].mean(), atol=1e-10, rtol=0
        ):
            raise ValueError("Node-specific rate control failed")
        for j, i in enumerate(ids):
            ps[i][node] = {
                **{k: float(v[j]) for k, v in preds.items()},
                **{k: priors[j][k] for k in ("native", "terminal", "return_only")},
            }
    reports = []
    for w, p in zip(data, ps, strict=True):
        r = trajectory_table(
            w["ref"],
            p,
            w["cases"],
            w["costs"],
            arms,
            ("native", "terminal", "return_only"),
            pivot=w["pivot"],
            background=w["background"],
        )
        r.update(
            inputs=w["inputs"],
            strict=w["strict"],
            start=w["start"],
            **w["policy_details"],
            state_hashes=w["states"],
        )
        reports.append(r)
    return summarize(reports, arms, "credit", comparisons), reports


def eligible(overall, fresh):
    if overall["questions"] < 0.9 * 378 or not fresh["questions"]:
        return False
    f = overall["f1"]
    return (
        all(f["credit"] >= f[k] + 0.02 for k in ("native", "terminal"))
        and all(
            f["credit"] > f[k] for k in ("return_only", "permuted", "semantic", "rate")
        )
        and overall["ht"]["credit"] > overall["ht"]["native"]
        and fresh["f1"]["credit"] > fresh["f1"]["native"]
        and sum(
            g["f1"]["credit"] > g["f1"]["native"]
            for g in overall["by_dialogue"].values()
        )
        >= 4
    )


def run(base):
    src = base / "source-compression"
    parent = src / "case-credit"
    joint = parent / "joint"
    if not (joint / "prompts.json").exists():
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select one leased GPU")
    torch.cuda.set_per_process_memory_fraction(0.06)
    locked = read(joint / "prompts.json")
    frozen = {
        k: (file_hash(d / "prompts.json"), read(d / "prompts.json")["targets"])
        for k, d in (("parent", parent), ("joint", joint))
    }
    if frozen["parent"][0] != locked["parent"]:
        raise ValueError("Parent registration changed")
    calibration = {
        w["window"]: w
        for w in read(src / "retrieval-competition/calibration.json")["windows"]
    }
    groups = {"factorial": [], "later": []}
    for name, w in locked["windows"].items():
        choice = numeric_choice(parent, joint, frozen, w)
        if choice is None:
            continue
        data = collect(base, name, w, choice, calibration)
        if data is not None:
            groups[w["panel"]].append(data)
    for panel, data in groups.items():
        if not data:
            continue
        folder = joint / panel
        folder.mkdir(exist_ok=True)
        stamp = digest([v["inputs"] for v in data])[:16]
        p = folder / f"through-{len(data):02d}-{stamp}.json"
        if p.exists():
            continue
        overall, rows = evaluate(data, "cuda")
        result = {"all": overall, "windows": rows, "registration": frozen["joint"][0]}
        if panel == "later":
            held = [w for w in data if w["start"] >= 13]
            for key, subset in (
                ("strict", [w for w in data if w["strict"]]),
                ("session13_plus", held),
                ("strict_session13_plus", [w for w in held if w["strict"]]),
            ):
                result[key], _ = evaluate(subset, "cuda")
            result["eligible_for_development"] = eligible(
                result["session13_plus"], result["strict_session13_plus"]
            )
        register(p, result)
        if len(data) == (12 if panel == "factorial" else 52):
            register(folder / "summary.json", result)
        print(panel, len(data), overall["questions"], overall["f1"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    run(p.parse_args().base)
