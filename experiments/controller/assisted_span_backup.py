"""Cross-fitted outcome baselines for randomized training contrasts at the first write."""

from collections import defaultdict

import torch

from experiments.controller.evidence_transport import (
    ARMS,
    TEMP,
    Transport,
    root_targets,
)

CELLS = ("00", "01", "10", "11")


def strata(w):
    return [
        (int(t), tuple(bool(v) for v in link))
        for t, link in zip(w["types"], w["links"], strict=True)
    ]


def cell_table(w, device):
    return torch.tensor(
        [w["y"][k] for k in CELLS], device=device, dtype=torch.float64
    ).T.reshape(-1, 2, 2)


def outcome_baseline(anchors, excluded, donor, device):
    """Estimate four cell means from full paths in other histories; smooth sparse strata."""
    if donor["sid"] not in excluded:
        raise ValueError("Baseline would include the donor's own history")
    tables, groups = [], defaultdict(list)
    offset = 0
    for w in anchors.values():
        if w["sid"] in excluded:
            continue
        if w["pivot"] is not None:
            raise ValueError("A randomized or incomplete donor entered the baseline")
        rows = cell_table(w, device)
        tables.append(rows)
        for i, key in enumerate(strata(w)):
            groups[key].append(offset + i)
        offset += len(rows)
    if not tables:
        raise ValueError("No independent complete anchors")
    table = torch.cat(tables)
    prior = table.mean(0)
    means = {
        k: (table[ids].sum(0) + 8 * prior) / (len(ids) + 8) for k, ids in groups.items()
    }
    return torch.stack([means.get(k, prior) for k in strata(donor)])


def continuation(actor, name, held):
    w = actor.donors[name]
    effect = actor.terminal(name, {held, w["sid"]})
    logits = effect.new_tensor(w["base"][1:])[:, None]
    return (logits + (effect / TEMP).clamp(-2, 2)).sigmoid()


def adjustment(means, probs, suffix):
    """Model-assisted correction to the fixed-.5 paired estimator; zero in coin expectation."""
    p = probs.clone()
    p[:, ARMS.index("equal")] = 0.5
    m = means - 0.5
    term0 = (1 - p[1]) * m[:, 1, 0, None] - (1 - p[0]) * m[:, 0, 0, None]
    term1 = p[1] * m[:, 1, 1, None] - p[0] * m[:, 0, 1, None]
    return (term1 - term0) if suffix == 0 else (term0 - term1)


def estimates(w, probs, means, physical=None):
    paired = root_targets(w, probs, probs.device)
    if w["pivot"] is None:
        return dict.fromkeys(("paired", "difficulty", "assisted", "resolved"), paired)
    if w["pivot"] != 0:
        raise ValueError("Expected a randomized first-write intervention")
    bg = w["bg"]
    result = {
        "paired": paired,
        "difficulty": paired
        + adjustment(means.mean((1, 2), keepdim=True).expand_as(means), probs, bg),
        "assisted": paired + adjustment(means, probs, bg),
    }
    if physical is not None:
        result["resolved"] = paired + adjustment(physical, probs, bg)
    return result


class AssistedTransport:
    def __init__(self, turns, anchors, extra, device, *, physical=None):
        self.anchors = anchors
        self.physical = physical
        self.actor = Transport(turns, anchors | extra, device)
        self.device = device
        self.targets = {}

    def backups(self, held):
        if held not in self.targets:
            by_arm = defaultdict(dict)
            for n, w in self.actor.donors.items():
                if w["sid"] == held or w["pivot"] == 1:
                    continue
                p = continuation(self.actor, n, held)
                means = (
                    outcome_baseline(self.anchors, {held, w["sid"]}, w, self.device)
                    if w["pivot"] == 0
                    else None
                )
                measured = (
                    self.physical[n]
                    if self.physical is not None and w["pivot"] == 0
                    else None
                )
                for arm, v in estimates(w, p, means, measured).items():
                    by_arm[arm][n] = v
            self.targets[held] = by_arm
        return self.targets[held]

    def forecast(self, name):
        held = name.rsplit("_s", 1)[0]
        targets = self.backups(held)
        main = "resolved" if self.physical is not None else "assisted"
        first = self.actor.predict(name, {held}, 0, targets[main])
        last = self.actor.terminal(name, {held})
        effects = torch.cat([first[None], last])
        extras = {}
        for arm in ("paired", "difficulty", "assisted"):
            r = (
                first
                if arm == main
                else self.actor.predict(name, {held}, 0, targets[arm])
            )
            extras[arm] = torch.cat([r[:1], last[:, 0]])
        keys = list(ARMS) + list(extras)
        all_effects = torch.cat([effects, torch.stack(list(extras.values()), 1)], 1)
        return {
            "offsets": [
                dict(zip(keys, row, strict=True))
                for row in (all_effects / TEMP).clamp(-2, 2).tolist()
            ],
            "effects": [
                dict(zip(keys, row, strict=True)) for row in all_effects.tolist()
            ],
            "donors": sorted({w["sid"] for w in self.actor.donors.values()} - {held}),
        }
