"""Transfer paired question-level returns using only source evidence visible before a write."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from experiments.controller.evidence_transport_data import (
    factorial,
    randomized,
    source_keys,
    sources,
)
from experiments.controller.evidence_turns import PLAN
from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from src.evaluation_cache import digest

ARMS = ("credit", "raw", "blind", "sham", "equal")
TEMP = 0.05


def query_similarity(target, donor, links, stage):
    maxima = [(donor[j] @ target[j].T).amax(1) for j in range(stage + 1)]
    scores = []
    for link in links:
        visible = [
            maxima[j][indices] for j, indices in enumerate(link[: stage + 1]) if indices
        ]
        scores.append(torch.cat(visible).mean() if visible else maxima[0].new_zeros(()))
    return torch.stack(scores).double()


def coefficients(sim, groups, device):
    """Every donor history has equal mass. Within it, preserve the observed linkage mix."""
    counts = defaultdict(int)
    by_group = defaultdict(list)
    for i, (sid, link) in enumerate(groups):
        counts[sid] += 1
        by_group[sid, link].append(i)
    real = torch.empty(len(groups), device=device, dtype=torch.float64)
    blind = torch.empty_like(real)
    perm = torch.arange(len(groups), device=device)
    for (sid, _), idx in by_group.items():
        mass = len(idx) / (len(counts) * counts[sid])
        real[idx] = torch.softmax(8 * sim[idx], dim=0) * mass
        blind[idx] = mass / len(idx)
        perm[idx] = perm.new_tensor(idx[1:] + idx[:1])
    return real, blind, perm


def aggregate(sim, groups, y, device):
    weights, blind, perm = coefficients(sim, groups, device)
    return torch.stack(
        [
            (blind if k == "blind" else weights)
            @ (y[perm, i] if k == "sham" else y[:, i])
            for i, k in enumerate(ARMS)
        ]
    )


def terminal_targets(w, prefix, device):
    low, high = [
        torch.as_tensor(w["y"][f"{prefix}{b}"], device=device, dtype=torch.float64)
        for b in (0, 1)
    ]
    delta = high - low
    augmented = delta + delta.new_tensor(w["last"][prefix])
    return torch.stack([delta if k == "raw" else augmented for k in ARMS], 1)


def root_targets(w, probs, device):
    """The backup probabilities come from inner folds excluding the outer holdout and this donor."""
    values = {
        k: torch.as_tensor(v, device=device, dtype=torch.float64)[:, None]
        for k, v in w["y"].items()
    }
    p = probs.clone()
    p[:, ARMS.index("equal")] = 0.5
    use_credit = p.new_tensor([k != "raw" for k in ARMS])
    if w["pivot"] is None:
        low = values["00"] + p[0] * (values["01"] - values["00"])
        high = values["10"] + p[1] * (values["11"] - values["10"])
        c0, c1 = p.new_tensor(w["root"])
        return (
            high
            - low
            + use_credit * ((1 - p.amax(0)) * c0[:, None] + p.amin(0) * c1[:, None])
        )
    b = w["bg"]
    mass = p if b else 1 - p
    # Subtracting the same constant from both potential returns cancels in expectation over b.
    return 2 * (
        mass[1] * (values[f"1{b}"] - 0.5)
        - mass[0] * (values[f"0{b}"] - 0.5)
        + use_credit * mass.amin(0) * p.new_tensor(w["root"][b])[:, None]
    )


class Transport:
    def __init__(self, turns, donors, device):
        self.turns = turns
        self.donors = donors
        self.device = device
        self.similarities = {}
        self.inner = {}
        self.terminal_values = {
            (name, b): terminal_targets(w, b, device)
            for name, w in donors.items()
            for b in (0, 1)
            if w["pivot"] is None or (w["pivot"] == 1 and w["bg"] == b)
        }

    def similarity(self, target, name, stage):
        k = target, name, stage
        if k not in self.similarities:

            def matrices(window):
                return [self.turns[s][1] for s in source_keys(window)[: stage + 1]]

            self.similarities[k] = query_similarity(
                matrices(target), matrices(name), self.donors[name]["links"], stage
            )
        return self.similarities[k]

    def predict(self, target, excluded, stage, values):
        if target.rsplit("_s", 1)[0] not in excluded:
            raise ValueError("Target dialogue missing from exclusions")
        names = [
            name for name in sorted(values) if self.donors[name]["sid"] not in excluded
        ]
        if not names:
            raise ValueError("No independent donor trajectories remain")
        sim = torch.cat([self.similarity(target, name, stage) for name in names])
        groups = [
            (self.donors[name]["sid"], tuple(bool(ix) for ix in link))
            for name in names
            for link in self.donors[name]["links"]
        ]
        return aggregate(
            sim, groups, torch.cat([values[name] for name in names]), self.device
        )

    def terminal(self, target, excluded):
        return torch.stack(
            [
                self.predict(
                    target,
                    excluded,
                    1,
                    {n: v for (n, p), v in self.terminal_values.items() if p == b},
                )
                for b in (0, 1)
            ]
        )

    def backups(self, held):
        if held not in self.inner:
            targets = {}
            for n, w in self.donors.items():
                if w["sid"] == held or w["pivot"] == 1:
                    continue
                effect = self.terminal(n, {held, w["sid"]})
                base = effect.new_tensor(w["base"][1:])[:, None]
                probs = (base + (effect / TEMP).clamp(-2, 2)).sigmoid()
                targets[n] = root_targets(w, probs, self.device)
            self.inner[held] = targets
        return self.inner[held]

    def forecast(self, target):
        held = target.rsplit("_s", 1)[0]
        first = self.predict(target, {held}, 0, self.backups(held))
        last = self.terminal(target, {held})
        effects = torch.cat([first[None], last])
        shifts = (effects / TEMP).clamp(-2, 2)
        return {
            "effects": effects.tolist(),
            "offsets": [dict(zip(ARMS, row, strict=True)) for row in shifts.tolist()],
            "donors": sorted({v["sid"] for v in self.donors.values()} - {held}),
        }


def fit(base, study, corpus):
    src = base / "source-compression"
    dest = src / "evidence-transport"
    late = base / "late-pairs"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select one leased CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.06)
    protocol = read(study / "protocol.json")
    manifest = read(dest / "sources.json")
    if (
        file_hash(corpus) != manifest["corpus"]
        or file_hash(study / "protocol.json") != manifest["study"]
        or file_hash(PLAN) != manifest["plan"]
        or set(protocol["train"]) & set(protocol["validation"] + protocol["test"])
    ):
        raise ValueError("Different data split, source extraction or research plan")
    turns = sources(dest, "cuda")
    data = {
        d["sample_id"]: d for d in read(corpus) if d["sample_id"] in protocol["train"]
    }
    early = factorial(base, turns, data)
    extra_names = read(late / "selection-labels/split.json")["windows"]["donor"]
    extra = (
        randomized(base, turns, data)
        if all(k in turns for n in extra_names for k in source_keys(n))
        else None
    )
    specs = [(n, "factorial") for n in early] + [
        (f"{w['dialogue']}_s{w['start']}", "later")
        for w in read(late / "cohort.json")["assignments"]
    ]
    for stage, donors in (
        ("pilot", early),
        ("expanded", early | extra if extra else None),
    ):
        if not donors:
            continue
        out = dest / stage
        preds = out / "predictions"
        preds.mkdir(parents=True, exist_ok=True)
        lock = out / "donors.json"
        register(
            lock,
            {
                "source_manifest": file_hash(dest / "sources.json"),
                "algorithm": file_hash(Path(__file__)),
                "rows": digest(donors),
                "factorial": file_hash(src / "retrieval-competition/fit.json"),
                "training": {
                    n: {
                        "ids": v["ids"],
                        "inputs": v["inputs"],
                        "sources": [turns[k][3] for k in source_keys(n)],
                    }
                    for n, v in donors.items()
                },
            },
        )
        fingerprint = file_hash(lock)
        model = Transport(turns, donors, "cuda")
        for n, panel in specs:
            if stage == "expanded" and (panel == "factorial" or n in extra_names):
                continue
            path = preds / (n + ".json")
            keys = source_keys(n)
            if not all(k in turns for k in keys):
                continue
            inputs = {
                "donors": fingerprint,
                "source_files": [turns[k][3] for k in keys],
            }
            if path.exists():
                if read(path)["inputs"] != inputs:
                    raise ValueError("Already frozen predictions have different inputs")
                continue
            register(
                path,
                dict(
                    model.forecast(n),
                    inputs=inputs,
                    sources=[turns[k][2] for k in keys],
                ),
            )
        print(stage, len(list(preds.glob("*.json"))), "frozen windows", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    fit(a.base, a.study, a.corpus)
