"""Learn one exceptional-skill update per uniquely verified question at a deployed span."""

import argparse
from itertools import permutations
from pathlib import Path

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import weights
from experiments.controller.three_action_bounds import report as population_bounds
from experiments.controller.three_skill_margin import margin_fit, score
from experiments.controller.three_skill_shared_credit import EDGES, SKILLS, registration
from experiments.controller.three_skill_temporal_gate import allocate as gate_allocate
from experiments.controller.three_skill_temporal_gate import simplex, utilities
from experiments.controller.witness_passage_policy import limited
from experiments.controller.writer_format_policy import (
    qualified_weights,
    rotation,
    scores_at,
)

EARLY_MODES = ("outlier-initial", "outlier-final")
MODES = ("outlier-total", "outlier-components", *EARLY_MODES)


def original(job):
    pairs = [p for p, c in zip(EDGES, job["edges"], strict=True) if c]
    if len(pairs) != 2:
        return None
    (i,) = set(pairs[0]).intersection(pairs[1])
    other = [v for j, v in enumerate(job["scores"]) if j != i]
    if abs(other[0] - other[1]) > 1e-10:
        return None
    raw = job["scores"][i] - sum(other) / 2
    if not raw:
        return None
    for j, edge in enumerate(EDGES):
        c = job["edges"][j]
        if c and abs(c - (job["scores"][edge[1]] - job["scores"][edge[0]])) > 1e-10:
            raise ValueError("Different exceptional-action endpoint")
    return i, raw


def donors(base, mode):
    path = registration(base)
    reg = read(path)
    folder = path.parent
    full = mode == "outlier-components"
    physical = read(folder / "outcomes-complete.json") if full else {}
    if full and set(physical) != set(reg["jobs"]):
        raise ValueError("Incomplete six-cell census")
    parent = base / "late-pairs/semantic-writer/three-writer"
    data = read(parent / "skill-graph/complete/models.json")["donors"]
    rows = []
    cases = []
    for w in data:
        name = w["window"]
        n = len(w["points"])
        credit = [[0.0, 0.0] for _ in SKILLS]
        for iq, job in reg["jobs"].get(name, {}).items():
            outcome = original(job)
            if outcome is None:
                continue
            j, raw = outcome
            if full:
                f = folder / "outcomes" / (name + ".json")
                r = read(f)
                if physical[name] != file_hash(f) or r["inputs"] != {
                    "registration": file_hash(path),
                    "parent": file_hash(folder / "prepared" / f.name),
                }:
                    raise ValueError("Changed six-cell intervention")
                result = r["rows"].get(iq, {})
                if result.get("status") != "unique":
                    cases.append(
                        {
                            "window": name,
                            "index": iq,
                            "status": result.get("status", r["status"]),
                        }
                    )
                    continue
                shares = result["credit"]
                if result["outlier"] != SKILLS[j] or abs(sum(shares) - raw) > 1e-10:
                    raise ValueError(
                        "Changed exceptional skill or double-counted credit"
                    )
            else:
                shares = [raw / 2, raw / 2]
            for k in range(2):
                credit[j][k] += shares[k] / n
            cases.append(
                {
                    "window": name,
                    "index": iq,
                    "skill": SKILLS[j],
                    "raw": raw,
                    "components": shares if full else None,
                }
            )
        rows.append(
            {
                "node": name,
                "history": w["history"],
                "features": w["features"],
                "points": n,
                "credit": credit,
                "raw": 0.0,
            }
        )
    return rows, cases


def early_donors(early, marker):
    folder = marker.parent
    batch = read(marker)
    reg = folder / "registration.json"
    roster = read(reg)
    if (
        file_hash(reg) != batch["registration"]
        or file_hash(early / "forecast.json") != roster["inputs"]["forecast"]
    ):
        raise ValueError("Different early action cohort")
    names = roster["windows"] if marker.stem == "final" else roster["initial"]
    if set(batch["outcomes"]) != set(names):
        raise ValueError("Incomplete early wave")
    ws = read(early / "forecast.json")["windows"]
    rows, cases = [], []
    for name, checksum in batch["outcomes"].items():
        path = folder / "outcomes" / (name + ".json")
        packet = read(path)
        if checksum != file_hash(path) or packet["inputs"] != {
            "registration": file_hash(reg),
            "parent": file_hash(folder / "prepared" / path.name),
        }:
            raise ValueError("Changed early six-cell outcome")
        if packet["status"] != "matched":
            continue
        w = ws[name]
        n = len(w["questions"])
        credit = [[0.0, 0.0] for _ in SKILLS]
        for idx, c in packet["candidates"].items():
            r = packet["rows"].get(idx, {})
            status = r.get("status", c["status"])
            case = {"window": name, "index": idx, "status": status}
            if status == "unique":
                skill = SKILLS.index(r["outlier"])
                shares = r["credit"]
                raw = r["mean"]["raw"]
                if (
                    int(idx) not in w["questions"]
                    or c["status"] != "certified"
                    or c["outlier"] != r["outlier"]
                    or abs(sum(shares) - raw) > 1e-10
                ):
                    raise ValueError("Changed physical unique-question return")
                for k in range(2):
                    credit[skill][k] += shares[k] / n
                case.update(skill=r["outlier"], raw=raw, components=shares)
            cases.append(case)
        rows.append(
            {
                "node": name,
                "history": w["history"],
                "features": w["features"],
                "points": n,
                "credit": credit,
                "raw": 0.0,
            }
        )
    return rows, cases


def expanded_corrections(model, old, features, total):
    upd = corrections(model, features, "cuda", total)
    if old is not None:
        upd["previous"] = corrections(old, features, "cuda", False)["credit"]
        upd["zero_new"] = corrections(model["zero_new"], features, "cuda", False)[
            "credit"
        ]
    return upd


def fit_correction(rows, held, device):
    donors = [r for r in rows if r["history"] != held]
    hs = {r["history"] for r in donors}

    def t(x):
        return torch.as_tensor(x, device=device, dtype=torch.float64)

    rw = weights(donors, hs, "raw", t)
    raw = t([r["features"] for r in donors])
    g = raw[:, 1]
    center = rw @ g
    sd = (rw @ (g - center).square()).sqrt().clamp_min(0.02)
    z = torch.zeros_like(g)
    one = z + 1
    w0 = torch.stack((one, raw[:, 0] / 2, z, z), 1)
    w1 = torch.stack((z, z, one, ((g - center) / sd).clamp(-3, 3)), 1)
    whole = w0 + w1
    each = torch.cat((w0, w1))
    targets = t([r["credit"] for r in donors])
    perm = rotation(donors)
    coef = {}
    for scheme in (
        "components",
        "total",
        "swapped",
        "rotated",
        "equal",
        "total_rotated",
    ):
        fitted = []
        for j in range(len(SKILLS)):
            c = targets[:, j]
            if scheme in ("rotated", "total_rotated"):
                c = c[perm]
            if scheme == "swapped":
                c = c.flip((1,))
            summed = c.sum(1)
            if scheme == "equal":
                c = summed[:, None].expand(-1, 2) / 2
            v = qualified_weights(donors, hs, c, t)
            joint = scheme in ("total", "total_rotated")
            _, b = margin_fit(
                whole,
                z,
                rw,
                summed if joint else c.T.reshape(-1),
                v if joint else v.repeat(2),
                constraint_x=whole if joint else each,
                penalty=t([0.04, 0.25, 0.04, 0.25]),
            )
            fitted.append(b.tolist())
        coef[scheme] = fitted
    return {
        "coef": coef,
        "center": float(center),
        "sd": float(sd),
        "windows": [r["node"] for r in donors],
        "histories": sorted(hs),
    }


def corrections(model, features, device, total):
    z = max(-3.0, min(3.0, (features[1] - model["center"]) / model["sd"]))
    x = torch.tensor([1.0, features[0] / 2, 1.0, z], device=device, dtype=torch.float64)
    result = {k: (x.new_tensor(v) @ x).tolist() for k, v in model["coef"].items()}
    return {
        "raw": [0.0] * 3,
        "credit": result["total" if total else "components"],
        "rotated": result["total_rotated" if total else "rotated"],
        "undivided": result["total"],
        "swapped": result["swapped"],
        "equal": result["equal"],
    }


def corrected_utilities(raw, delta):
    return {k: [a + b for a, b in zip(raw, v, strict=True)] for k, v in delta.items()}


def allocate(native, packet):
    old = gate_allocate(native, packet)
    return {
        **simplex(
            native,
            corrected_utilities(packet["gate_utilities"]["raw"], packet["outlier"]),
        ),
        "native": native,
        "uniform": old["uniform"],
        "old_gate": old["credit"],
        "fixed_t_credit": old["fixed_t_credit"],
        "joint_graph": old["joint_graph"],
    }


def freeze(base, mode):
    reg = registration(base)
    p = base / "late-pairs/semantic-writer/three-writer"
    early = base / "source-compression/earliest-three-skills"
    folder = p / mode
    folder.mkdir(exist_ok=True)
    eo = early / mode
    eo.mkdir(exist_ok=True)
    gate = p / "temporal-gate/forecast.json"
    gate_early = early / "temporal-gate/forecast.json"
    gate_models = p / "temporal-gate/models.json"
    routed = p / "writer-format-verified/models.json"
    inputs = {
        "plan": file_hash(reg),
        "gate": file_hash(gate),
        "early": file_hash(gate_early),
        "models": file_hash(gate_models),
        "route": file_hash(routed),
    }
    total = mode == "outlier-total"
    if not total:
        inputs["six"] = file_hash(reg.parent / "outcomes-complete.json")
    expanded = mode in EARLY_MODES
    previous = None
    if expanded:
        marker = early / "shared-weak-credit" / (mode.split("-")[1] + ".json")
        prior = p / "outlier-components/models.json"
        inputs.update(wave=file_hash(marker), previous=file_hash(prior))
        previous = read(prior)["models"]
    f = folder / "forecast.json"
    ef = eo / "forecast.json"
    if f.exists() and ef.exists():
        if (
            read(f)["inputs"] != inputs
            or read(f)["models"] != file_hash(folder / "models.json")
            or read(ef)["parent"] != file_hash(f)
        ):
            raise ValueError("Changed unique-question fit")
        return f
    rows, cases = donors(base, "outlier-components" if expanded else mode)
    if expanded:
        more, added = early_donors(early, marker)
        zeroed = rows + [dict(r, credit=[[0.0, 0.0] for _ in SKILLS]) for r in more]
        rows += more
        cases += added
    parent = read(gate)
    ep = read(gate_early)
    if ep["parent"] != inputs["gate"] or parent["models"] != inputs["models"]:
        raise ValueError("Different gate model")
    gm = read(gate_models)["models"]
    rm = read(routed)["models"]
    models = {}
    later = {}
    earlier = {}
    for h in sorted(gm):
        with limited():
            m = fit_correction(rows, h, "cuda")
            old = previous[h] if expanded else None
            if expanded:
                comparison = fit_correction(zeroed, h, "cuda")
                has_new_credit = any(
                    any(any(shares) for shares in r["credit"])
                    for r in more
                    if r["history"] != h
                )
                if not has_new_credit and m != comparison:
                    raise ValueError("A held dialogue changed its own credit model")
                m["zero_new"] = comparison
            models[h] = m
            for k, w in parent["windows"].items():
                if k.rsplit("_s", 1)[0] != h:
                    continue
                s = w["scores"]
                upd = expanded_corrections(m, old, s["source_features"], total)
                later[k] = {**w, "scores": {**s, "outlier": upd}}
            for k, w in ep["windows"].items():
                if w["history"] != h:
                    continue
                x = w["features"]
                ps = w["probabilities"]
                n = torch.tensor(ps["native"], dtype=torch.float64, device="cuda")
                raw = utilities(gm[h], scores_at(rm[h], x, "cuda"), x, "cuda")["raw"]
                upd = expanded_corrections(m, old, x, total)
                prob = simplex(n, corrected_utilities(raw, upd))
                if not torch.allclose(
                    prob["raw"], n.new_tensor(ps["gate_raw"]), rtol=0, atol=1e-12
                ):
                    raise ValueError("Unique-question raw benchmark changed")
                extra = {"outlier_" + k: v.tolist() for k, v in prob.items()}
                earlier[k] = {**w, "probabilities": {**ps, **extra}}
    if len(later) != 25 or len(earlier) != 12:
        raise ValueError("Incomplete outlier panel")
    register(
        folder / "models.json",
        {"inputs": inputs, "models": models, "donors": rows, "cases": cases},
    )
    register(
        f,
        {
            "inputs": inputs,
            "models": file_hash(folder / "models.json"),
            "windows": later,
        },
    )
    exposure = eo / "exposure.json"
    if not exposure.exists():
        register(
            exposure,
            {
                k: {f.stem: file_hash(f) for f in (early / k).glob("conv-*.json")}
                for k in ("writers", "readers", "pairs", "judged")
            },
        )
    register(
        ef,
        {
            "inputs": ep["inputs"],
            "parent": file_hash(f),
            "exposure": file_hash(exposure),
            "windows": earlier,
        },
    )
    return f


def isolated_changes(raw, delta, native):
    """Expected policies for every subset of the three additive logit updates."""
    ids = torch.arange(8, device=raw.device)
    bits = (ids[:, None] >> torch.arange(3, device=raw.device)) & 1
    logs = ((raw + bits * delta) / 0.05).log_softmax(1)
    error = []
    for j in range(3):
        peers = [k for k in range(3) if j != k]
        odds = logs[:, peers[0]] - logs[:, peers[1]]
        error.append(float((odds[1 << j] - odds[0]).abs()))
    return 0.2 * native + 0.8 * logs.exp(), error


def skill_report(path):
    out = path.parent
    spec = read(path)
    for f in sorted(out.glob("through-*.json")):
        dest = out / ("skill-effects-" + f.stem.split("-")[-1] + ".json")
        if dest.is_file():
            if read(dest)["through"] != file_hash(f):
                raise ValueError("Changed outlier evaluation")
            continue
        result = read(f)
        windows = {}
        for w in result["rows"]:
            name = w["window"]
            qfile = out / "questions" / (name + ".json")
            qs = read(qfile)
            if (
                qs["forecast"] != file_hash(path)
                or qs["source"] != result["sources"][name]
            ):
                raise ValueError("Different per-question three-action outcomes")
            pkt = spec["windows"][name]["scores"]
            with limited():

                def t(v):
                    return torch.as_tensor(v, dtype=torch.float64, device="cuda")

                ps, errors = isolated_changes(
                    t(pkt["gate_utilities"]["raw"]),
                    t(pkt["outlier"]["credit"]),
                    t(w["p"]["native"]),
                )
                if max(errors) > 1e-10 or not torch.allclose(
                    ps[[0, 7]], t([w["p"]["raw"], w["p"]["credit"]]), rtol=0, atol=1e-12
                ):
                    raise ValueError(
                        "Isolated credit affected other log odds or changed the issued strategy"
                    )
                values = t([q["f1"] for q in qs["rows"]]) @ ps.T
                parts = values.new_zeros((len(qs["rows"]), 3))
                for perm in permutations(range(3)):
                    mask = 0
                    for j in perm:
                        nxt = mask | (1 << j)
                        parts[:, j] += (values[:, nxt] - values[:, mask]) / 6
                        mask = nxt
                actual = t(
                    [q["expected"]["credit"] - q["expected"]["raw"] for q in qs["rows"]]
                )
                if not torch.allclose(parts.sum(1), actual, rtol=0, atol=1e-12):
                    raise ValueError(
                        "Per-question utility changes do not explain measured F1"
                    )
                windows[name] = {
                    "partition": w["partition"],
                    "n": w["n"],
                    "max_peer_log_odds_error": max(errors),
                    "mean": parts.mean(0).tolist(),
                    "by_question": {
                        q["index"]: v
                        for q, v in zip(qs["rows"], parts.tolist(), strict=True)
                    },
                }
        selected = [w for w in windows.values() if w["partition"] == "evaluation"]
        n = sum(w["n"] for w in selected)
        register(
            dest,
            {
                "through": file_hash(f),
                "skills": SKILLS,
                "windows": windows,
                "evaluation_questions": n,
                "evaluation": [
                    sum(w["n"] * w["mean"][j] for w in selected) / n for j in range(3)
                ]
                if n
                else [],
            },
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("mode", choices=MODES)
    a = cli.parse_args()
    gpu()
    f = freeze(a.base, a.mode)
    score(
        a.base,
        f,
        allocate,
        extra_controls=(
            ("undivided", "swapped", "equal") if a.mode != "outlier-total" else ()
        )
        + (("zero_new", "previous") if a.mode in EARLY_MODES else ())
        + ("old_gate", "fixed_t_credit", "joint_graph"),
        replay_choices=True,
    )
    skill_report(f)
    population_bounds(a.base, f)
