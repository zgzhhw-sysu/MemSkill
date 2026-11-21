"""Discover sparse span interventions and confirm them on separate donor histories."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.passage_advantage import evaluate
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.passage_evidence_prefix import evaluate as evaluate_prefix
from experiments.controller.selective_span_bounds import measure
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "selective_span_policy.json"
MASS = 0.65
MIX = ("credit", "base_rule", "raw_rule", "rotated", "unscreened")
ARMS = (*MIX, "parent", "raw", "terminal", "native")


def load_donors(base, study, corpus):
    src = base / "source-compression"
    root = src / "matched-span-rank"
    parent = root / "root-augmented"
    fitted = parent / "models.json"
    forecast = read(parent / "forecast.json")
    models = read(fitted)
    proto = read(study / "protocol.json")
    if (
        models["inputs"] != forecast["inputs"]
        or proto["input_sha256"] != file_hash(corpus)
        or set(proto["train"]) & set(proto["validation"] + proto["test"])
    ):
        raise ValueError("Different parent training run")
    hashes = {
        "models": file_hash(fitted),
        "protocol": file_hash(study / "protocol.json"),
        "corpus": file_hash(corpus),
    }
    values = {}
    for d, k in (
        (root, "features"),
        (root / "earlier-sources", "earlier_features"),
        (root / "earlier-root-sources", "root_features"),
    ):
        f = d / "features.json"
        if file_hash(f) != models["inputs"][k]:
            raise ValueError("Changed donor geometry")
        features = read(f)["features"]
        if any(values[n] != features[n] for n in values.keys() & features):
            raise ValueError("Inconsistent source encoding")
        values.update(features)
    units = []
    for row in models["units"]:
        h, n = row["node"].rsplit("_s", 1)
        j, b = row["stage"], row["background"]
        start = int(n) - j
        key = f"{h}_s{start}"
        if h != row["history"] or h not in proto["train"]:
            raise ValueError("Unexpected donor history")
        if "raw" in row:
            f = parent / ("early" if start <= 7 else "calibration") / (key + ".json")
            v = read(f)
            if (
                v["dialogue"] != h
                or len(v["rows"]) != row["questions"]
                or v["inputs"]["forecast"] != file_hash(parent / "forecast.json")
            ):
                raise ValueError("Changed registered training pair")
            p = v["probabilities"][str(1 + b if j else 0)]["native"]
        else:
            f = study / "train" / key / "rows.json"
            reg = src / "legacy-terminal-symmetry/registration.json"
            if file_hash(f) != read(reg)["windows"][key]["rows"]:
                raise ValueError("Changed earlier decision")
            # The archived conditional probability under the checkpoint is sigmoid(base_logit).
            p = None
        hashes[
            str(
                f.relative_to(parent)
                if f.is_relative_to(parent)
                else f.relative_to(study)
            )
        ] = file_hash(f)
        units.append(
            dict(
                row, native=p, old_logit=read(f)[j]["base_logit"] if p is None else None
            )
        )
    if (
        len(units) != 84
        or sum("raw" in u for u in units) != 60
        or len(set(proto["train"])) != 6
    ):
        raise ValueError("Incomplete registered donors")
    return values, units, hashes


def weights(rows, ids, key, tensor):
    counts = Counter(
        (r["history"], r["node"]) for r in rows if r["history"] in ids and key in r
    )
    histories = Counter(h for h, _ in counts)
    return tensor(
        [
            1
            / (
                len(histories)
                * histories[r["history"]]
                * counts[r["history"], r["node"]]
            )
            if r["history"] in ids and key in r
            else 0.0
            for r in rows
        ]
    )


def regions(x, discovery, nodes, stage):
    # Thresholds belong only to unique discovery sources; backgrounds may repeat a source.
    first = {}
    for i, (node, used) in enumerate(zip(nodes, discovery, strict=True)):
        if used:
            first.setdefault(node, i)
    cuts = (
        torch.quantile(
            x[list(first.values()), :12], x.new_tensor([0.25, 0.5, 0.75]), dim=0
        )
        .T.cpu()
        .tolist()
    )
    specs = [(-1, 0.0, 0), (-1, 0.0, 1)]
    for k, values in enumerate(cuts):
        for cut in sorted(set(values)):
            specs.extend([(k, cut, -1), (k, cut, 1)])
    if stage:
        specs.extend([(12, 0.5, -1), (12, 0.5, 1)])
    return specs


def matches(x, specs):
    features = torch.tensor([s[0] for s in specs], device=x.device, dtype=torch.long)
    cuts = x.new_tensor([s[1] for s in specs])
    sides = x.new_tensor([s[2] for s in specs])
    selected = x[:, features.clamp_min(0)].T
    return torch.where(
        features[:, None] < 0,
        (sides == 1)[:, None],
        torch.where(
            sides[:, None] < 0, selected <= cuts[:, None], selected > cuts[:, None]
        ),
    )


def prob(native, action):
    return (1 - MASS * action.abs()) * native + MASS * (action > 0).to(native.dtype)


def best_rule(mask, current, native, target, w, groups):
    choices = native.new_tensor([0.0, -1.0, 1.0])
    actions = torch.where(
        mask[:, None, :], choices[None, :, None], current[None, None, :]
    )
    change = prob(native, actions) - prob(native, current)
    gains = ((w * target) * change).sum(-1)
    movement = (w * change.abs()).sum(-1)
    represented = (mask.to(w.dtype) @ ((w > 0)[:, None] * groups).to(w.dtype) > 0).sum(
        -1
    )
    valid = represented >= 2
    valid[0] = True
    gains = gains.masked_fill(~valid[:, None], -torch.inf)
    # Empty region comes first and wins any zero-gain tie.
    near = gains >= gains.max() - 1e-12
    smallest = movement.masked_fill(~near, torch.inf).min()
    flat = ((near & (movement <= smallest + 1e-12)).flatten().nonzero()[0]).item()
    i, j = divmod(flat, len(choices))
    return i, int(choices[j]), actions[i, j], float(gains[i, j])


def fit_split(rows, x, native, raw, credit, rotated, discovery, confirm, stage):
    specs = regions(
        x, [r["history"] in discovery for r in rows], [r["node"] for r in rows], stage
    )
    masks = matches(x, specs)
    histories = sorted(discovery | confirm)
    groups = x.new_tensor([[r["history"] == h for h in histories] for r in rows])
    zeros = torch.zeros_like(native)
    wr = weights(rows, discovery, "raw", x.new_tensor)
    wc = weights(rows, discovery, "credit", x.new_tensor)
    vr = weights(rows, confirm, "raw", x.new_tensor)
    vc = weights(rows, confirm, "credit", x.new_tensor)
    idx, sign, base, discovery_gain = best_rule(masks, zeros, native, raw, wr, groups)
    controls = {}
    for key, signal, weight, held_w in (
        ("credit", credit, wc, vc),
        ("rotated", rotated, wc, vc),
        ("raw_rule", raw, wr, vr),
    ):
        i, s, offer, g = best_rule(masks, base, native, signal, weight, groups)
        dp = prob(native, offer) - prob(native, base)
        physical = float((vr * raw * dp).sum())
        held = float((held_w * signal * dp).sum())
        controls[key] = {
            "region": specs[i],
            "action": s,
            "discovery_gain": g,
            "confirmation_raw": physical,
            "confirmation_signal": held,
            "accepted": g > 1e-12 and min(physical, held) > 1e-12,
        }
    return {
        "discovery": sorted(discovery),
        "confirmation": sorted(confirm),
        "base": {"region": specs[idx], "action": sign, "gain": discovery_gain},
        "offers": controls,
    }


def allocations(models, targets):
    """Average individual probability mixtures; opposite votes retain their intervention mass."""

    def apply(rule, previous):
        mask = matches(targets, [rule["region"]])[0]
        return torch.where(mask, rule["action"], previous)

    result = {k: [] for k in MIX}
    for m in models:
        original = apply(m["base"], torch.zeros_like(targets[:, 0]))
        result["base_rule"].append(original)
        for k, r in m["offers"].items():
            offer = apply(r, original)
            result[k].append(offer if r["accepted"] else original)
            if k == "credit":
                result["unscreened"].append(offer)
    output = [{} for _ in targets]
    for arm, decisions in result.items():
        votes = torch.stack(decisions)
        toward_temporal = (votes < 0).to(targets.dtype).mean(0)
        toward_activity = (votes > 0).to(targets.dtype).mean(0)
        masses = MASS * torch.stack((toward_temporal, toward_activity), dim=1)
        for row, pair in zip(output, masses.tolist(), strict=True):
            row[arm] = pair
    return output


def train(rows, values, nodes, held, stage):
    rows = [r for r in rows if r["history"] != held and r["stage"] == stage]
    histories = sorted({r["history"] for r in rows})
    if len(histories) != 5:
        raise ValueError("Expected five donor histories")
    tensor = lambda a: torch.tensor(a, device="cuda", dtype=torch.float64)
    x = tensor([[*values[r["node"]], r["background"] if stage else 0] for r in rows])
    old = tensor([r["old_logit"] or 0 for r in rows]).sigmoid()
    native = torch.where(
        x.new_tensor([r["native"] is None for r in rows]).bool(),
        old,
        tensor([r["native"] or 0 for r in rows]),
    )
    raw = tensor([r.get("raw", 0.0) for r in rows])
    credit = tensor([r["credit"] for r in rows])
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[r["history"], r["background"]].append(i)
    rotation = list(range(len(rows)))
    for order in strata.values():
        order.sort(key=lambda i: rows[i]["node"])
        for i, j in zip(order, order[1:] + order[:1], strict=True):
            rotation[i] = j
    rotated = credit[rotation]
    models = []
    for split in combinations(histories, 3):
        with limited():
            models.append(
                fit_split(
                    rows,
                    x,
                    native,
                    raw,
                    credit,
                    rotated,
                    set(split),
                    set(histories) - set(split),
                    stage,
                )
            )
    keys = [(n, b) for n in nodes for b in (range(2) if stage else (0,))]
    with limited():
        targets = tensor([[*values[n], b] for n, b in keys])
        result = allocations(models, targets)
    return models, dict(zip(keys, result, strict=True))


def freeze(base, study, corpus):
    root = base / "source-compression/matched-span-rank"
    trial = root / "selective-rules"
    prefix = trial / "long-prefix/forecast.json"
    local = trial / "forecast.json"
    prefix.parent.mkdir(exist_ok=True, parents=True)
    reference = root / "consensus/original"
    a, b = (read(reference / f) for f in ("forecast.json", "long-prefix/forecast.json"))
    values, units, hashes = load_donors(base, study, corpus)
    identity = {
        "plan": file_hash(PLAN),
        "donors": hashes,
        "local_parent": file_hash(reference / "forecast.json"),
        "prefix_parent": file_hash(reference / "long-prefix/forecast.json"),
        "manifest": b["inputs"]["manifest"],
    }
    register(trial / "registration.json", identity)
    inputs = {
        "manifest": b["inputs"]["manifest"],
        "registration": file_hash(trial / "registration.json"),
    }
    if local.exists() and prefix.exists():
        if read(local)["inputs"] != inputs or read(prefix)["inputs"] != {
            **inputs,
            "local": file_hash(local),
        }:
            raise ValueError("Frozen rule forecast changed")
        return local, prefix
    nodes = sorted(set(a["sources"]) | set(b["sources"]))
    mixtures = {n: [None] * 3 for n in nodes}
    models = {}
    for h in sorted({r["history"] for r in units}):
        targets = [n for n in nodes if n.rsplit("_s", 1)[0] == h]
        models[h] = {}
        for stage in (0, 1):
            m, pred = train(units, values, targets, h, stage)
            models[h][str(stage)] = m
            for (n, prior), v in pred.items():
                mixtures[n][1 + prior if stage else 0] = v
    if any(v is None for row in mixtures.values() for v in row):
        raise ValueError("Missing conditional action forecast")
    register(trial / "models.json", {"inputs": inputs, "folds": models})
    by_stage = {
        j: {n: {"parent": r["credit"]} for n, r in d.items()}
        for j, d in a["by_stage"].items()
    }
    register(
        local,
        {
            "inputs": inputs,
            "sources": a["sources"],
            "by_stage": by_stage,
            "mixtures": {n: mixtures[n] for n in a["sources"]},
            "arms": ARMS,
        },
    )
    register(
        prefix,
        {
            "inputs": {**inputs, "local": file_hash(local)},
            "sources": b["sources"],
            "offsets": {n: {"parent": v["credit"]} for n, v in b["offsets"].items()},
            "mixtures": {n: mixtures[n] for n in b["sources"]},
            "arms": ARMS,
        },
    )
    return local, prefix


def run(base, study, corpus):
    local, prefix = freeze(base, study, corpus)
    evaluate_prefix(base, prefix)
    evaluate(
        base,
        study,
        corpus,
        local,
        final_stage=local.parent.name,
        gate_controls=("base_rule", "raw_rule", "rotated", "raw", "parent"),
        required_prefix=prefix.parent / "summary.json",
    )

    measure(base)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    gpu()
    run(a.base, a.study, a.corpus)
