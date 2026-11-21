"""Match a pre-write utterance to the causal passage and the donor evidence it represents."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.controller import file_hash, read, register
from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.evidence_transport_data import sources
from experiments.controller.witness_passage_policy import cases, completed, limited
from experiments.controller.witness_passages import registration
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "passage_evidence_advantage.json"
METRICS = ("credit", "sham", "source_only", "passage_only")
ARMS = (
    "credit",
    "raw",
    "sham",
    "source_only",
    "passage_only",
    "rotated",
    "parent",
    "native",
    "terminal",
)


def placebo_positions(positions, size):
    """Match the number of annotated turns without reusing any real annotation."""
    taken = set(positions)
    if not positions or len(taken) != len(positions) or len(positions) * 2 > size:
        return []
    result = []
    for pos in sorted(positions):
        index = next(
            i for k in range(1, size + 1) if (i := (pos + k) % size) not in taken
        )
        result.append(index)
        taken.add(index)
    return sorted(result)


def support_margin(target, donor, winner, marked):
    positive = donor[marked]
    selected = marked[int((positive @ winner).argmax())]
    other = [i for i in range(len(donor)) if i not in marked]
    competitor = (target @ donor[other].T).amax(1)
    return (target @ donor[selected] - competitor).clamp_min(0), selected


def case_scores(active, records, bank, sessions, data):
    all_turns = torch.cat([r[1] for r in sessions.values()]).double()
    lengths = [len(r[0]) for r in sessions.values()]
    groups = torch.arange(len(lengths), device=bank.device).repeat_interleave(
        torch.tensor(lengths, device=bank.device)
    )
    offsets = torch.tensor([0, *np.cumsum(lengths).tolist()], device=bank.device)
    by_case = {}
    audit = []
    for key, spec in active.items():
        sid, index = key
        annotated = set(data[sid]["qa"][index]["evidence"])
        best = bank.new_zeros((len(lengths), len(METRICS)))
        hits = [None] * len(lengths)
        for job, win, lose, weight in spec["ranks"]:
            row = records[job]
            if row["job"] != job or (row["dialogue"], row["index"]) != key:
                raise ValueError("Donor credit rank does not identify this question")
            (pivot,) = [
                j for j, (a, b) in enumerate(zip(*row["bits"], strict=True)) if a != b
            ]
            start = int(row["window"].rsplit("_s", 1)[1])
            session = f"{sid}_s{start + pivot}"
            ids, vec, _, _ = sessions[session]
            if ids != [
                t["dia_id"]
                for t in data[sid]["conversation"][f"session_{start + pivot}"]
            ]:
                raise ValueError("Changed donor turn ordering")
            positive = [i for i, t in enumerate(ids) if t in annotated]
            placebo = placebo_positions(positive, len(ids))
            receipt = {
                "job": job,
                "donor": [sid, index],
                "session": session,
                "actual_evidence": [ids[i] for i in positive],
            }
            audit.append(receipt)
            if not placebo:
                receipt["exclusion"] = (
                    "no_local_evidence" if not positive else "too_few_other_turns"
                )
                continue
            with limited():
                winner = bank[win]
                good = all_turns @ winner
                bad = (
                    all_turns @ bank[lose]
                    if lose is not None
                    else torch.zeros_like(good)
                )
                margin = good.clamp_min(0) * (good - bad).clamp_min(0)
                evidence, real = support_margin(
                    all_turns, vec.double(), winner, positive
                )
                sham, fake = support_margin(all_turns, vec.double(), winner, placebo)
                values = weight * torch.stack(
                    (margin * evidence, margin * sham, evidence, margin), dim=1
                )
                reduced = torch.zeros_like(best).scatter_reduce_(
                    0, groups[:, None].expand_as(values), values, reduce="amax"
                )
                # Recover first target turn attaining the new winning joint score.
                turn_index = torch.arange(len(values), device=bank.device)
                candidates = torch.where(
                    values[:, 0] == reduced[groups, 0], turn_index, len(values)
                )
                local = (
                    torch.full_like(offsets[:-1], len(values)).scatter_reduce_(
                        0, groups, candidates, reduce="amin"
                    )
                    - offsets[:-1]
                )
                for i in (reduced[:, 0] > best[:, 0]).nonzero().flatten().tolist():
                    hits[i] = [job, sessions[list(sessions)[i]][0][int(local[i])]]
                best = torch.maximum(best, reduced)
                receipt.update(selected=ids[real], placebo=ids[fake], weight=weight)
        if any("selected" in r for r in audit if tuple(r["donor"]) == key):
            by_case[key] = {"scores": best, "credit": spec["credit"], "hits": hits}
    return by_case, audit


def score_sessions(sessions, packets, parent):
    names = list(sessions)
    histories = [n.rsplit("_s", 1)[0] for n in names]
    numbers = [int(n.rsplit("_s", 1)[1]) for n in names]
    results = {}
    detail = defaultdict(list)
    for held in sorted(set(histories)):
        targets = [i for i, h in enumerate(histories) if h == held]
        donors = [k for k in sorted(packets) if k[0] != held]
        if not donors:
            for i in targets:
                results[names[i]] = {
                    **dict.fromkeys((*METRICS, "rotated"), 0.0),
                    "parent": parent[names[i]]["credit"],
                }
            continue
        signals = []
        credit = []
        for k in donors:
            reference = [
                i
                for i, h in enumerate(histories)
                if h not in (held, k[0]) and 5 <= numbers[i] <= 12
            ]
            if len(reference) != 32:
                raise ValueError("Reference must contain four other training histories")
            scores = packets[k]["scores"]
            val = scores[targets]
            tail = (
                1 + (scores[reference, None, :] >= val[None, :, :]).sum(0).double()
            ) / 33
            strength = (1 - 2 * tail).clamp_min(0)
            signals.append(strength)
            credit.append(packets[k]["credit"])
            for i, m, z in zip(targets, val.tolist(), strength.tolist(), strict=True):
                detail[names[i]].append(
                    {
                        "case": list(k),
                        "scores": m,
                        "strength": z,
                        "hit": packets[k]["hits"][i],
                    }
                )
        stack = torch.stack(signals, dim=0)

        def signed(v, labels):
            terms = v * v.new_tensor(labels)[:, None, None]
            return 2 * (terms.amax(0).clamp_min(0) + terms.amin(0).clamp_max(0))

        out = signed(stack, credit).tolist()
        rotated = signed(stack[:, :, :1], credit[1:] + credit[:1])[:, 0].tolist()
        for i, z, rot in zip(targets, out, rotated, strict=True):
            results[names[i]] = {
                **dict(zip(METRICS, z, strict=True)),
                "rotated": rot,
                "parent": parent[names[i]]["credit"],
            }
    return results, dict(detail)


def forecast(base, study, corpus, stage):
    origin = registration(base, study, corpus)
    src = origin.parent
    parent = src / stage / "forecast.json"
    if not parent.exists():
        return None
    out = src / "evidence-advantage" / stage
    out.mkdir(exist_ok=True, parents=True)
    path = out / "forecast.json"
    encoder = src / "encoder.json"
    matrix = src / "passages.npz"
    identity = {
        "plan": file_hash(PLAN),
        "ranks": file_hash(origin),
        "parent": file_hash(parent),
        "encoder": file_hash(encoder),
    }
    if path.exists():
        if read(path)["inputs"] != identity:
            raise ValueError("Frozen evidence actor differs from current inputs")
        return path
    prior = read(parent)
    inv = read(origin)
    sessions = dict(sorted(sources(src.parent / "evidence-transport", "cuda").items()))
    if prior["sources"] != {
        k: {"text": v[2], "vectors": v[3]} for k, v in sessions.items()
    } or read(encoder)["vectors"] != file_hash(matrix):
        raise ValueError("Cached source or passage vectors changed")
    with np.load(matrix, allow_pickle=False) as obj:
        if str(obj["texts"]) != digest(inv["texts"]):
            raise ValueError("Changed passage order")
        bank = torch.tensor(obj["vectors"], device="cuda", dtype=torch.float64)
    ids = set(read(study / "protocol.json")["train"])
    data = {r["sample_id"]: r for r in read(corpus) if r["sample_id"] in ids}
    records = inv["records"]
    active = cases(records, completed(src, records) if stage == "witness" else None)
    packets, inventory = case_scores(active, records, bank, sessions, data)
    with limited():
        adjustment, matches = score_sessions(sessions, packets, prior["offsets"])
    if set(adjustment) != set(prior["offsets"]):
        raise ValueError("Incomplete source forecast")
    register(
        path,
        {
            "inputs": identity,
            "sources": prior["sources"],
            "offsets": adjustment,
            "matches": matches,
            "inventory": inventory,
        },
    )
    return path


def exposure(base, path):
    cutoff = path.stat().st_mtime_ns
    replica = base / "source-compression/witness-passages/four-paths/cells"
    archives = {"primary": base / "late-pairs/outcomes", "replica": replica}
    before = {k: scored_at(folder, cutoff) for k, folder in archives.items()}
    register(
        path.parent / "exposure.json",
        {"forecast": file_hash(path), "cutoff_ns": cutoff, "scores": before},
    )
    return {name.split("/")[0] for files in before.values() for name in files}


def gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one guarded CUDA GPU on 251")
    free, total = torch.cuda.mem_get_info()
    budget = min(total * 0.06, free - 2048 * 2**20)
    if budget < 512 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget / total)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    gpu()
    for stage in ("group", "witness"):
        f = forecast(a.base, a.study, a.corpus, stage)
        if f:
            exposure(a.base, f)
            print(f, flush=True)
