"""Compare every within-donor-history label assignment on the archived early four-path panel."""

from collections import defaultdict
from itertools import permutations, product

import torch

from experiments.controller.factorial_rollout import CELLS, read
from experiments.controller.source_compression import file_hash, register


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


def audit(trial, root, stage):
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
