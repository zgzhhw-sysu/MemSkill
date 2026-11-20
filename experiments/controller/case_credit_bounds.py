"""Diagnostic F1 limits under a shared per-span decision and partially recorded votes."""

from itertools import product

import torch

from experiments.controller.case_credit import KINDS, corrections, vote
from experiments.controller.factorial_rollout import read
from src.evaluation_cache import digest


def possible_offsets(target, dest):
    keys = []
    observed = {}
    for kind in KINDS:
        for text in target["prompts"][kind]:
            key = digest(
                {
                    "model": "llama3-8b",
                    "temperature": 0,
                    "max_tokens": 3,
                    "messages": [{"role": "user", "content": text}],
                }
            )
            keys.append(key)
            path = dest / "responses" / (key + ".json")
            if path.exists() and key not in observed:
                record = read(path)
                if key != record["request"]:
                    raise ValueError("Wrong source prompt response")
                observed[key] = vote(record["reply"], False)
    missing = sorted(set(keys) - observed.keys())
    offsets = set()
    for completion in product((-1, 0, 1), repeat=len(missing)):
        votes = observed | dict(zip(missing, completion, strict=True))
        by_kind = {
            kind: [votes[keys[2 * j]], -votes[keys[2 * j + 1]]]
            for j, kind in enumerate(KINDS)
        }
        offsets.add(corrections(by_kind)["credit"])
    return sorted(offsets), observed


def interval(nodes, offsets, device):
    # One source offset applies to both realized prefixes and every question.
    p = torch.tensor([n["native"] for n in nodes], dtype=torch.float64, device=device)
    reach = p.new_tensor([n["weight"] for n in nodes])
    outcomes = p.new_tensor([n["cases"] for n in nodes])
    delta = outcomes[:, 1] - outcomes[:, 0]
    probs = (p.logit() + p.new_tensor(offsets)[:, None]).sigmoid()
    values = (
        reach[None, :, None] * (outcomes[None, :, 0] + probs[:, :, None] * delta[None])
    ).sum(1)
    baseline = (reach[:, None] * (outcomes[:, 0] + p[:, None] * delta)).sum(0)
    means = values.mean(1)
    return {
        "min": float(means.min()),
        "max": float(means.max()),
        "native": float(baseline.mean()),
        "question_oracle": float(values.amax(0).mean()),
        "affected": int((delta.abs().amax(0) > 1e-12).sum()),
        "native_rows": baseline.tolist(),
    }


def bounds(w, dest, target, device):
    offsets, observed = possible_offsets(target, dest)
    result = interval(w["nodes"], offsets, device)
    if any(
        abs(a - b) > 1e-10
        for a, b in zip(result.pop("native_rows"), w["native_f1"], strict=True)
    ):
        raise ValueError("Capacity baseline differs from actual measured native return")
    return dict(
        result,
        window=w["window"],
        questions=len(w["indices"]),
        observed=len(observed),
        responses=observed,
        offsets=offsets,
        paths=w["inputs"]["paths"],
    )
