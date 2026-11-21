"""Transfer verified reader-passage contrasts to visible turns of a held training history."""

import argparse
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.evidence_transport_data import sources
from experiments.controller.late_credit_kernel import kernel_effect, score, support
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.retrieval_competition import summarize
from experiments.controller.witness_passages import registration
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest

ARMS = ("credit", "raw", "rotated", "unpaired", "group", "native", "terminal")


@contextmanager
def limited():
    torch.cuda.synchronize()
    start = time.monotonic()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        time.sleep((time.monotonic() - start) * 3)


def strength(value, reference):
    if len(reference) != 32:
        raise ValueError("Expected 32 out-of-history calibration sessions")
    p = (1 + (reference >= value).sum()) / (1.0 + len(reference))
    return (1.0 - 2 * p).clamp_min(0.0)


def contrast_scores(turns, win, lose, weights):
    """At each turn compare the actual winning and losing ranks on the same utterance."""
    good = win @ turns.T
    bad = lose @ turns.T
    match = good.clamp_min(0) * weights[:, None]
    paired = match * (good - bad).clamp_min(0)
    return torch.stack((paired.amax(), match.amax())), int(paired.argmax())


def cases(records, results):
    cases = {}
    for r in records:
        if not r["eligible"]:
            continue
        k = (r["dialogue"], r["index"])
        case = cases.setdefault(k, {"credit": r["credit"], "ranks": []})
        if not math.isclose(case["credit"], r["credit"], abs_tol=1e-12):
            raise ValueError("Repeated question has a different signed magnitude")
        role = 1.0
        if results is not None:
            v = results[r["job"]]
            role = (
                1.0
                if v["sufficient_in_loser"]
                else 0.5
                if v["necessary_in_winner"]
                else 0.0
            )
        if role:
            case["ranks"].append((r["job"], *r["keys"], role))
    return {k: v for k, v in sorted(cases.items()) if v["ranks"]}


def completed(out, rows):
    folder = out.parent / "reader-slot-credit/single-rank"
    v = read(folder / "summary.json")
    if v["requests"] != file_hash(folder / "requests.json") or v["ranks"] != len(rows):
        raise ValueError("Incomplete one-rank intervention cohort")
    result = {}
    for r in rows:
        i = r["job"]
        f = folder / f"rank-{i:02d}.json"
        item = read(f)
        if (
            item["job"] != i
            or file_hash(f) != v["files"][str(i)]
            or item["requests"] != v["requests"]
        ):
            raise ValueError("Changed single-rank responses")
        result[i] = item
    return result


def match_all(keys, bank, turns):
    packets = {}
    for label, case in keys.items():
        ranks = case["ranks"]
        win = bank[[w for _, w, _, _ in ranks]]
        lose = torch.stack(
            [
                bank[l] if l is not None else torch.zeros_like(win[0])
                for _, _, l, _ in ranks
            ]
        )
        weights = bank.new_tensor([w for _, _, _, w in ranks])
        responses = []
        hits = []
        with limited():
            for ids, matrix, _, _ in turns.values():
                sc, idx = contrast_scores(matrix.double(), win, lose, weights)
                responses.append(sc)
                hits.append([ranks[idx // len(ids)][0], ids[idx % len(ids)]])
        packets[label] = {"scores": torch.stack(responses), "hits": hits}
    return packets


def offsets(keys, packets, turns, held, index):
    donors = [k for k in keys if k[0] != held]
    if not donors:
        return {"credit": 0.0, "rotated": 0.0, "unpaired": 0.0}, []
    names = list(turns)
    signed = []
    unpaired = []
    details = []
    for k in donors:
        parents = [
            i
            for i, n in enumerate(names)
            if n.rsplit("_s", 1)[0] not in (held, k[0])
            and 5 <= int(n.rsplit("_s", 1)[1]) <= 12
        ]
        scores = packets[k]["scores"]
        q = scores[index]
        signal = [float(strength(q[col], scores[parents, col])) for col in (0, 1)]
        signed.append(signal[0])
        unpaired.append(signal[1])
        details.append(
            {
                "donor": list(k),
                "credit": keys[k]["credit"],
                "match": q.tolist(),
                "strength": signal,
                "rank_job_and_target_turn": packets[k]["hits"][index],
            }
        )
    credit = [keys[k]["credit"] for k in donors]

    def combine(values, signs):
        terms = [v * c for v, c in zip(values, signs, strict=True)]
        return 2 * (max([0.0, *terms]) + min([0.0, *terms]))

    return {
        "credit": combine(signed, credit),
        "rotated": combine(signed, credit[1:] + credit[:1]),
        "unpaired": combine(unpaired, credit),
    }, details


def forecast(base, study, corpus, stage):
    reg = registration(base, study, corpus)
    out = reg.parent
    dest = out / stage
    dest.mkdir(exist_ok=True)
    enc = out / "encoder.json"
    v = read(reg)
    existing = dest / "forecast.json"
    rank_summary = out.parent / "reader-slot-credit/single-rank/summary.json"
    if stage == "witness" and not rank_summary.is_file():
        return None
    turn_dir = out.parent / "evidence-transport"
    parents = {
        "registration": file_hash(reg),
        "encoder": file_hash(enc),
        "cohort": file_hash(base / "late-pairs/cohort.json"),
        "terminal": file_hash(base / "late-pairs/frozen.json"),
        "source_model": v["inputs"]["parent"],
        "rank_results": file_hash(rank_summary) if stage == "witness" else None,
    }
    if existing.exists():
        if read(existing)["inputs"] != parents:
            raise ValueError("Different frozen policy")
        return existing
    turns = sources(turn_dir, "cuda")
    # Sort session names to bind tie handling and matched-turn indices on every run.
    turns = dict(sorted(turns.items()))
    proto = read(study / "protocol.json")
    ref = read(enc)
    matrices = out / "passages.npz"
    if ref["registration"] != parents["registration"] or ref["vectors"] != file_hash(
        matrices
    ):
        raise ValueError("Changed passage encoder")
    with np.load(matrices, allow_pickle=False) as arr:
        if str(arr["texts"]) != digest(v["texts"]):
            raise ValueError("Wrong donor text order")
        bank = torch.as_tensor(arr["vectors"], device="cuda", dtype=torch.float64)
    data = {
        c["sample_id"]: c["conversation"]
        for c in read(corpus)
        if c["sample_id"] in proto["train"]
    }
    processor = LoCoMoProcessor(chunk_mode="full-session")
    for name, (ids, _, h, _) in turns.items():
        sid, n = name.rsplit("_s", 1)
        hist = data[sid]
        key = f"session_{int(n)}"
        (s,) = processor._extract_session_chunks(hist[key], hist[key + "_date_time"])
        if ids != [t["dia_id"] for t in hist[key]] or digest(s) != h:
            raise ValueError("Changed target session text or turn IDs")
    group = cases(v["records"], None)
    chosen = (
        group if stage == "group" else cases(v["records"], completed(out, v["records"]))
    )
    packets = match_all(chosen, bank, turns)
    broad = read(out / "group/forecast.json")["offsets"] if stage == "witness" else {}
    output = {}
    matches = {}
    for i, name in enumerate(turns):
        sid = name.rsplit("_s", 1)[0]
        with limited():
            result, details = offsets(chosen, packets, turns, sid, i)
        result["group"] = (
            result["credit"] if stage == "group" else broad[name]["credit"]
        )
        output[name] = result
        matches[name] = details
    hist = read(base / "late-pairs/cohort.json")["assignments"]
    if not all(
        f"{w['dialogue']}_s{w['start'] + j}" in output for w in hist for j in (0, 1)
    ):
        raise ValueError("Missing pre-write target offsets")
    register(
        existing,
        {
            "inputs": parents,
            "sources": {n: {"text": p[2], "vectors": p[3]} for n, p in turns.items()},
            "offsets": output,
            "matches": matches,
        },
    )
    return existing


def anchor_logits(d, model, terminal):
    bank, w, ys = model
    state = bank.new_tensor(d["state"])
    z = state.new_tensor(d["base_logit"])
    t = state.new_tensor(terminal)
    term = z + t[0] + t[1:] @ F.normalize(state, dim=0)
    raw = term + (kernel_effect(state[None], bank, w, ys)[0, 1] / 0.05).clamp(-2, 2)
    return z, term, raw


def probabilities(d, m, terminal):
    model, offsets = m
    z, term, raw = anchor_logits(d, model, terminal)
    return {
        k: float(v.sigmoid())
        for k, v in {
            "credit": raw + offsets["credit"],
            "raw": raw,
            "rotated": raw + offsets["rotated"],
            "unpaired": raw + offsets["unpaired"],
            "group": raw + offsets["group"],
            "native": z,
            "terminal": term,
        }.items()
    }


def matched_probabilities(logits, targets, weights):
    """Use one intercept per pivot, retaining the native logit ordering."""
    rate = (logits + matched_offset(logits, targets, weights)).sigmoid()
    if abs(float(weights @ (rate - targets) / weights.sum())) <= 1e-10:
        return rate
    # Combined terminal, raw and passage shifts can exceed the original +/-2 bracket.
    target = weights @ targets / weights.sum()
    if target == 0 or target == 1:
        return torch.full_like(logits, float(target))
    bound = target.logit() - logits
    low, high = float(bound.min()), float(bound.max())
    for _ in range(56):
        mid = (low + high) / 2
        if weights @ (logits + mid).sigmoid() / weights.sum() < target:
            low = mid
        else:
            high = mid
    return (logits + (low + high) / 2).sigmoid()


def report(panels, arms=ARMS):
    a = summarize(panels, arms, "credit", arms[1:])
    if not panels:
        return a
    by = defaultdict(list)
    draws = 0
    changed = 0
    for j in (0, 1):
        ps = [p for p in panels if p["pivot"] == j]
        if not ps:
            continue

        def tensor(vals):
            return torch.tensor(vals, device="cuda", dtype=torch.float64)

        logits = tensor([p["base"] for p in ps])
        wt = tensor([len(p["rows"]) for p in ps])
        targets = tensor([p["focal_probabilities"]["credit"] for p in ps])
        rate = matched_probabilities(logits, targets, wt)
        if abs(float(wt @ (rate - targets) / wt.sum())) > 1e-10:
            raise ValueError("Mismatched target use rate")
        for p, q in zip(ps, rate.tolist(), strict=True):
            by[p["dialogue"]].extend(l + q * (h - l) for l, h in p["endpoints"])
            u = int(digest(["paired-witness-action", p["window"], j])[:13], 16) / 16**13
            p["common_uniform_draw"] = {
                k: int(u < v) for k, v in p["focal_probabilities"].items()
            }
            draws += 1
            changed += (
                p["common_uniform_draw"]["credit"] != p["common_uniform_draw"]["raw"]
            )
    a["matched"] = mean(x for vs in by.values() for x in vs)
    a["matched_by_history"] = {k: mean(v) for k, v in by.items()}
    a["mean_change_from_raw"] = mean(
        abs(p["focal_probabilities"]["credit"] - p["focal_probabilities"]["raw"])
        for p in panels
    )
    a["draws_different_from_raw"] = [changed, draws]
    return a


def eligible(a, f):
    if a["windows"] != 40 or a["questions"] != 378 or not f["questions"]:
        return False
    y = a["f1"]
    ht = a["ht"]
    return (
        all(
            y["credit"] >= y[k] + 0.02 and ht["credit"] > ht[k]
            for k in ("native", "terminal")
        )
        and all(y["credit"] > y[k] for k in ("raw", "rotated", "unpaired", "group"))
        and y["credit"] > a["matched"]
        and sum(
            v["f1"]["credit"] > a["matched_by_history"][k]
            for k, v in a["by_dialogue"].items()
        )
        >= 4
        and all(f["f1"]["credit"] > f["f1"][k] for k in ("raw", "native"))
    )


def evaluate(base, path):
    root = base / "late-pairs"
    dest = path.parent
    rows = read(
        base / "source-compression/capped-source-credit/policy/joint-repaired/fit.json"
    )["observations"]
    weights = read(root / "frozen.json")["terminal"]
    archived = scored_at(root / "outcomes", path.stat().st_mtime_ns)
    register(
        dest / "preexisting.json", {"forecast": file_hash(path), "scores": archived}
    )
    before = {k.split("/")[0] for k in archived}
    folds = {
        (sid, j): support(rows, sid, j, "cuda")[0] for sid in weights for j in (0, 1)
    }
    split = read(root / "selection-labels/split.json")["windows"]
    pack = read(path)
    for label, part in (("calibration", "donor"), ("later", "evaluation")):
        panels = []
        for w in read(root / "cohort.json")["assignments"]:
            sid, n = w["dialogue"], w["start"]
            name = f"{sid}_s{n}"
            cell = root / "outcomes" / name
            if name not in split[part] or not (cell / "pair.json").exists():
                continue
            if w["questions"] != split[part][name]:
                raise ValueError("Different paired QA panel")
            pair = read(cell / "pair.json")
            target = [f"{sid}_s{n + j}" for j in (0, 1)]
            for b in pair["paths"]:
                directory = cell / "".join(map(str, b))
                (trace,) = (directory / "traces").glob("*.jsonl")
                spans = [
                    e
                    for l in trace.read_bytes().splitlines()
                    if (e := json.loads(l))["kind"] == "span"
                ]
                if [digest(r["source"]) for r in spans] != [
                    pack["sources"][k]["text"] for k in target
                ]:
                    raise ValueError("Source shown to actor differs from physical arm")
            models = {
                (sid, j): (folds[sid, j], pack["offsets"][k])
                for j, k in enumerate(target)
            }
            with limited():
                result = score(
                    root, w, models, weights, policy=probabilities, arms=ARMS
                )
            base_logit = read(
                cell / "".join(map(str, pair["paths"][0])) / "decisions.json"
            )[w["pivot"]]["base_logit"]
            result.update(
                pivot=w["pivot"],
                base=base_logit,
                endpoints=[(r["temporal"], r["activity"]) for r in pair["questions"]],
            )
            panels.append(result)
        if not panels:
            continue
        a = report(panels)
        f = report([p for p in panels if p["window"] not in before])
        directory = dest / label
        directory.mkdir(exist_ok=True)
        res = {
            "forecast": file_hash(path),
            "all": a,
            "strict": f,
            "windows": panels,
            "eligible_for_development": part == "evaluation"
            and dest.name == "witness"
            and eligible(a, f),
        }
        register(directory / f"through-{len(panels):02d}.json", res)
        if len(panels) == len(split[part]):
            register(directory / "summary.json", res)


def run(base, study, corpus):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use one guarded 251 GPU")
    torch.cuda.set_per_process_memory_fraction(0.06)
    for stage in ("group", "witness"):
        path = forecast(base, study, corpus, stage)
        if path:
            evaluate(base, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    run(a.base, a.study, a.corpus)
