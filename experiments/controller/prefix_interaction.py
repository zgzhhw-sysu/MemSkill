"""Exact capacity of moving the same amount of skill use between two reached memory states."""

import argparse
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn.functional import cosine_similarity

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.terminal_capacity import optimal_rates, table

PLAN = Path(__file__).with_name("protocols") / "prefix_interaction.json"
BITS = ("00", "01", "10", "11")


def capacity(outcomes, logits, states):
    """Use the actual second state reached under each first action."""
    p = logits.sigmoid()
    first = p[0]
    reach = torch.stack((1 - first, first))
    native = p[1:]
    y = outcomes.mean(0)
    diff = y[:, 1] - y[:, 0]
    value = y[:, 0] + native * diff
    factual = reach @ value
    bounded = optimal_rates(reach, native, diff, torch.ones_like(diff), 2.0)
    free = optimal_rates(reach, native, diff, torch.ones_like(diff), None)
    root_high = (logits[0] + 2).sigmoid()
    root_low = (logits[0] - 2).sigmoid()
    root_best = torch.where(value[1] > value[0], root_high, root_low)
    if states.shape[1] % 2 or not torch.equal(*states[:, : states.shape[1] // 2]):
        raise ValueError("A different source entered the compared memory states")
    mem = states[:, states.shape[1] // 2 :]
    return {
        "questions": len(outcomes),
        "native_f1": float(factual),
        "prefix_activity": float(first),
        "suffix_activity": native.tolist(),
        "memory_cosine": float(cosine_similarity(mem[:1], mem[1:])),
        "same_memory_state": torch.equal(*mem),
        "suffix_effects": diff.tolist(),
        "interaction": float(diff[1] - diff[0]),
        "opposite_preferences": bool(diff.prod() < 0),
        "bounded_probabilities": bounded.tolist(),
        "bounded_gain": float(reach @ ((bounded - native) * diff)),
        "unrestricted_gain": float(reach @ ((free - native) * diff)),
        "root_only_gain": float((root_best - first) * (value[1] - value[0])),
    }


def measured(base, name, early, device):
    src = base / "source-compression"
    original = (
        src / "factorial" / name if early else base / "late-pairs/outcomes" / name
    )
    repaired = src / "factorial-repairs" / (name + ".json")
    completed = base / "late-pairs/donor-factorial" / name
    if early:
        expected = table(original, repaired)
        ids = [
            r["original_index"]
            for r in read(original / "measurement.json")["per_question"]
        ]
        checked = read(repaired)["inputs"]["traces"]
        source = repaired
    else:
        source = completed / "complete.json"
        ref = read(source)
        if ref["inputs"]["pair"] != file_hash(original / "pair.json"):
            raise ValueError("Changed randomized pair")
        ids = [r["index"] for r in ref["rows"]]
    values = []
    decisions = {}
    files = {}
    for bits in BITS:
        cell = original / bits
        if not (cell / "scores.json").is_file() and not early:
            cell = completed / bits
        path = cell / "scores.json"
        seen = read(cell / "decisions.json")
        if [int(d["use"]) for d in seen] != list(map(int, bits)):
            raise ValueError("Different physical intervention")
        (trace,) = (cell / "traces").glob("*.jsonl")
        hashes = {
            "scores.json": file_hash(path),
            "decisions.json": file_hash(cell / "decisions.json"),
            str(trace.relative_to(cell)): file_hash(trace),
        }
        files[bits] = hashes
        if early:
            if hashes[str(trace.relative_to(cell))] != checked[bits]:
                raise ValueError("Different factorial trajectory")
        elif any(h != ref["paths"][bits][Path(k).name] for k, h in hashes.items()):
            raise ValueError("Different complementary trajectory")
        for j, d in zip((0, 1 + int(bits[0])), seen, strict=True):
            state = {k: d[k] for k in ("base_logit", "state", "plain", "grounded")}
            if decisions.setdefault(j, state) != state:
                raise ValueError("Unequal pre-write states at the same decision")
        answers = read(path)["answers"]
        if [r["qa_index"] for r in answers] != list(range(len(ids))):
            raise ValueError("Physical questions out of order")
        values.append([r["f1"] for r in answers])
    y = torch.tensor(values, device=device, dtype=torch.float64).T.reshape(
        len(ids), 2, 2
    )
    logits = y.new_tensor([decisions[j]["base_logit"] for j in range(3)])
    states = y.new_tensor([decisions[j]["state"] for j in (1, 2)])
    result = capacity(y, logits, states)
    if not early:
        p = logits.sigmoid()
        native = (1 - p[0]) * ((1 - p[1]) * y[:, 0, 0] + p[1] * y[:, 0, 1]) + p[0] * (
            (1 - p[2]) * y[:, 1, 0] + p[2] * y[:, 1, 1]
        )
        archived = y.new_tensor([r["returns"]["native"]["exact"] for r in ref["rows"]])
        if not torch.allclose(native, archived, atol=1e-10, rtol=0):
            raise ValueError(
                "Native per-question returns from four physical cells failed to reproduce"
            )

    if early and any(
        not math.isclose(v["raw"], d, abs_tol=1e-12)
        for v, d in zip(expected, result["suffix_effects"], strict=True)
    ):
        raise ValueError("Retained effects do not reproduce original probes")
    return dict(
        result,
        window=name,
        dialogue=name.rsplit("_s", 1)[0],
        indices=ids,
        source=file_hash(source),
        files=files,
    )


def aggregate(rows):
    n = sum(r["questions"] for r in rows)
    if not n:
        return {"questions": 0, "windows": 0}
    stats = {
        k: sum(r[k] * r["questions"] for r in rows) / n
        for k in ("native_f1", "bounded_gain", "unrestricted_gain", "root_only_gain")
    }
    return dict(
        stats,
        questions=n,
        windows=len(rows),
        opposite_windows=sum(r["opposite_preferences"] for r in rows),
        opposite_question_mass=sum(
            r["questions"] for r in rows if r["opposite_preferences"]
        ),
        same_memory_windows=sum(r["same_memory_state"] for r in rows),
        mean_abs_interaction=mean(abs(r["interaction"]) for r in rows),
    )


def credited_suffixes(base, names, donors):
    src = base / "source-compression"
    blocks = src / "writer-block-repair/credit"
    block_files = read(blocks / "summary.json")["files"]
    source = src / "capped-source-credit/policy/joint-mask/fit.json"
    parent = read(source)
    if parent["evidence"]["writer_blocks"] != file_hash(blocks / "summary.json"):
        raise ValueError("Different same-skill writer controls")
    observed = parent["observations"]
    early = {}
    for name in names:
        file = blocks / (name + ".json")
        if file_hash(file) != block_files[name]:
            raise ValueError("Different signed question receipts")
        counts = [[], []]
        for g in read(file)["groups"]:
            counts[g["prefix"]] = [
                {
                    "index": q["original_index"],
                    "credit": q["credit"],
                    "status": q["status"],
                }
                for q in g["questions"]
            ]
        n = len(read(src / "factorial" / name / "measurement.json")["per_question"])
        rows = [r for r in observed if r["window"] == name and r["stage"] == 1]
        if len(rows) != 2 or any(
            not math.isclose(
                sum(q["credit"] for q in c) / n, r["credit"], abs_tol=1e-12
            )
            for c, r in zip(counts, rows, strict=True)
        ):
            raise ValueError("Per-question verified credit changed")
        early[name] = counts
    labels = base / "late-pairs/selection-labels"
    ledger = read(labels / "through-12.json")["files"]
    later = {}
    for name in donors:
        path = labels / (name + ".json")
        v = read(path)
        if file_hash(path) != ledger[name]:
            raise ValueError("Different randomized donor credit")
        results = [None, None]
        if v["pivot"] == 1:
            results[v["background"]] = [
                {
                    "index": q["original_index"],
                    "credit": q["credit"],
                    "status": q["status"],
                }
                for q in v["rows"]
                if q["credit"]
            ]
        later[name] = results
    return {
        "joint_fit": file_hash(source),
        "ledger": file_hash(labels / "through-12.json"),
        "early": early,
        "donors": later,
    }


def run(base):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use one leased GPU")
    torch.cuda.set_per_process_memory_fraction(0.06)
    src = base / "source-compression"
    out = src / "prefix-interaction"
    out.mkdir(exist_ok=True)
    early = [
        w["window"] for w in read(src / "retrieval-competition/fit.json")["training"]
    ]
    donors = read(base / "late-pairs/selection-labels/split.json")["windows"]["donor"]
    register(out / "verified-suffixes.json", credited_suffixes(base, early, donors))
    for group, names in (("early", early), ("donor", donors)):
        rows = []
        for name in names:
            if (
                group == "donor"
                and not (
                    base / "late-pairs/donor-factorial" / name / "complete.json"
                ).is_file()
            ):
                continue
            rows.append(measured(base, name, group == "early", "cuda"))
        if rows:
            report = {
                "plan": file_hash(PLAN),
                "overall": aggregate(rows),
                "windows": rows,
                "by_history": {
                    h: aggregate([r for r in rows if r["dialogue"] == h])
                    for h in sorted({r["dialogue"] for r in rows})
                },
            }
            register(out / f"{group}-{len(rows):02d}.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    run(p.parse_args().base)
