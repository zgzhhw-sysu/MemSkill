"""Learn temporal mass and the activity/relation split from separately verified edges."""

import argparse
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import weights
from experiments.controller.skill_graph_policy import allocate as graph_allocate
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_action_bounds import report as population_bounds
from experiments.controller.three_skill_margin import margin_fit, score
from experiments.controller.witness_passage_policy import limited
from experiments.controller.writer_format_policy import allocate as fixed_allocate
from experiments.controller.writer_format_policy import (
    fit,
    qualified_weights,
    rotation,
    scores_at,
)

PLAN = Path(__file__).with_name("protocols") / "three_skill_temporal_gate.json"
NAME = "temporal-gate"
KEYS = ("credit", "raw", "gate_only", "route_only", "rotated")
ROUTE = {
    "credit": "credit",
    "gate_only": "raw",
    "route_only": "credit",
    "raw": "raw",
    "rotated": "rotated",
}


def registration(base):
    p = base / "late-pairs/semantic-writer/three-writer"
    early = base / "source-compression/earliest-three-skills"
    out = p / NAME
    out.mkdir(exist_ok=True)
    f = out / "registration.json"
    inputs = {
        "plan": file_hash(PLAN),
        "verified": file_hash(p / "writer-format-verified/forecast.json"),
        "route_models": file_hash(p / "writer-format-verified/models.json"),
        "early": file_hash(early / "writer-format-verified/forecast.json"),
        "graph": file_hash(p / "skill-graph/complete/forecast.json"),
        "triplets": file_hash(p / "skill-graph/complete/models.json"),
    }
    if not f.exists():
        register(
            f,
            {
                "inputs": inputs,
                "later_at_registration": {
                    x.stem: file_hash(x) for x in p.glob("conv-*.json")
                },
                "early_at_registration": {
                    k: {x.stem: file_hash(x) for x in (early / k).glob("conv-*.json")}
                    for k in ("writers", "readers", "pairs", "judged")
                },
            },
        )
    elif read(f)["inputs"] != inputs:
        raise ValueError("Changed temporal gate inputs")
    return f


def gate_targets(points):
    value = 0.0
    edges = [0.0, 0.0]
    qualified = []
    for q in points:
        t, a, r = q["scores"]
        c = q["edges"]
        for j, delta in enumerate((a - t, r - t, r - a)):
            if c[j] and abs(c[j] - delta) > 1e-10:
                raise ValueError("Temporal label contradicts its physical return")
        active = sum(x != 0 for x in c[:2])
        if active:
            qualified.append({"id": q["index"], "edges": c, "temporal_edges": active})
            for j in range(2):
                edges[j] += c[j] / active
        value += (a + r) / 2 - t
    return value / len(points), [z / len(points) for z in edges], qualified


def train(donors, held, inner_models, device):
    rows = [r for r in donors if r["history"] != held]

    def t(v):
        return torch.as_tensor(v, device=device, dtype=torch.float64)

    ids = {r["history"] for r in rows}
    w = weights(rows, ids, "raw", t)
    x = t([r["features"] for r in rows])
    center = w @ x[:, 1]
    sd = (w @ (x[:, 1] - center).square()).sqrt().clamp_min(0.02)
    design = torch.stack(
        (torch.ones_like(w), x[:, 0] / 2, ((x[:, 1] - center) / sd).clamp(-3, 3)), dim=1
    )
    offsets = []
    for r in rows:
        h = r["history"]
        model = inner_models[h]
        if {held, h}.intersection(model["histories"]):
            raise ValueError("Inner route model saw its target or the held dialogue")
        offsets.append(scores_at(model, r["features"], device))
    y = t([r["gate_raw"] for r in rows])
    c = t([r["temporal_edges"] for r in rows])
    perm = rotation(rows)
    coef = {}
    for k in ("credit", "gate_only", "rotated"):
        targets = c[perm] if k == "rotated" else c
        d = t([o[ROUTE[k]] for o in offsets]) / 2
        v = qualified_weights(rows, ids, targets, t)
        raw, b = margin_fit(
            design,
            y,
            w,
            targets.T.reshape(-1),
            v.repeat(2),
            constraint_x=design.repeat(2, 1),
            penalty=t([0.02, 0.25, 0.25]),
            constraint_offset=torch.cat((-d, d)),
        )
        coef[k] = b.tolist()
    coef.update(raw=raw.tolist(), route_only=raw.tolist())
    return {
        "histories": sorted(ids),
        "windows": [r["node"] for r in rows],
        "center": float(center),
        "sd": float(sd),
        "coef": coef,
        "inner_routes": {
            h: {"histories": m["histories"], "windows": m["windows"]}
            for h, m in inner_models.items()
        },
        "offsets": {
            r["node"]: {k: o[k] for k in ("raw", "credit", "rotated")}
            for r, o in zip(rows, offsets, strict=True)
        },
    }


def utilities(model, route, features, device):
    z = max(-3.0, min(3.0, (features[1] - model["center"]) / model["sd"]))
    x = torch.tensor([1.0, features[0] / 2, z], device=device, dtype=torch.float64)
    out = {}
    for k in KEYS:
        b = float(x @ x.new_tensor(model["coef"][k]))
        d = route[ROUTE[k]] / 2
        out[k] = [0.0, b - d, b + d]
    return out


def simplex(native, utilities):
    return {
        k: 0.20 * native + 0.80 * (native.new_tensor(u) / 0.05).softmax(0)
        for k, u in utilities.items()
    }


def allocate(native, packet):
    fixed = fixed_allocate(native, packet)
    graph = graph_allocate(native, packet["gate_graph"])
    return {
        **simplex(native, packet["gate_utilities"]),
        "native": native,
        "uniform": 0.20 * native + 0.80 / 3,
        "fixed_t_credit": fixed["credit"],
        "fixed_t_raw": fixed["raw"],
        "joint_graph": graph["credit"],
        "two_edge": graph["parent_credit"],
    }


def freeze(base):
    reg = registration(base)
    out = reg.parent
    p = out.parent
    early = base / "source-compression/earliest-three-skills"
    eo = early / NAME
    eo.mkdir(exist_ok=True)
    f = out / "forecast.json"
    ef = eo / "forecast.json"
    if f.exists() and ef.exists():
        if read(f)["registration"] != file_hash(reg) or read(ef)["parent"] != file_hash(
            f
        ):
            raise ValueError("Changed gate forecast")
        return f
    fixed = read(p / "writer-format-verified/forecast.json")
    models = read(p / "writer-format-verified/models.json")
    if fixed["models"] != file_hash(p / "writer-format-verified/models.json"):
        raise ValueError("Unbound route model")
    graph = read(p / "skill-graph/complete/forecast.json")
    physical = {
        r["window"]: r for r in read(p / "skill-graph/complete/models.json")["donors"]
    }
    donors = []
    for r in models["donors"]:
        w = physical[r["node"]]
        if w["features"] != r["features"] or w["history"] != r["history"]:
            raise ValueError("Changed decision geometry")
        value, edges, qs = gate_targets(w["points"])
        donors.append(
            {**r, "gate_raw": value, "temporal_edges": edges, "qualified_temporal": qs}
        )
    early_fixed = read(early / "writer-format-verified/forecast.json")
    if early_fixed["parent"] != file_hash(p / "writer-format-verified/forecast.json"):
        raise ValueError("Wrong frozen early route")
    ms = {}
    forecasts = {}
    forecasts_early = {}
    for h, full in models["models"].items():
        with limited():
            remaining = [r for r in donors if r["history"] != h]
            inner = {
                i: fit(remaining, i, "cuda")
                for i in sorted({r["history"] for r in remaining})
            }
            m = train(donors, h, inner, "cuda")
            ms[h] = m
            if h in full["histories"]:
                raise ValueError("Outer route includes evaluation dialogue")
            for n, w in fixed["windows"].items():
                if n.rsplit("_s", 1)[0] != h:
                    continue
                other = graph["windows"][n]
                if (
                    w["source"] != other["source"]
                    or w["background"] != other["background"]
                ):
                    raise ValueError("Different graph decision")
                s = w["scores"]
                fs = scores_at(full, s["source_features"], "cuda")
                if fs != s["factors"]:
                    raise ValueError("Changed outer route forecast")
                forecasts[n] = {
                    **w,
                    "scores": {
                        **s,
                        "gate_utilities": utilities(
                            m, fs, s["source_features"], "cuda"
                        ),
                        "gate_graph": other["scores"],
                    },
                }
            for n, w in early_fixed["windows"].items():
                if w["history"] != h:
                    continue
                prob = w["probabilities"]
                native = torch.tensor(
                    prob["native"], dtype=torch.float64, device="cuda"
                )
                u = utilities(
                    m, scores_at(full, w["features"], "cuda"), w["features"], "cuda"
                )
                ps = {k: v.tolist() for k, v in simplex(native, u).items()}
                ps.update(
                    fixed_t_credit=prob["factor_credit"],
                    fixed_t_raw=prob["factor_raw"],
                    joint_graph=prob["graph_credit"],
                    two_edge=prob["direct_parent_credit"],
                    uniform=(0.2 * native + 0.8 / 3).tolist(),
                )
                forecasts_early[n] = {
                    **w,
                    "probabilities": {
                        **prob,
                        **{"gate_" + k: v for k, v in ps.items()},
                    },
                }
    if len(forecasts) != 25 or len(forecasts_early) != 12:
        raise ValueError("Incomplete forecast panels")
    register(
        out / "models.json",
        {"registration": file_hash(reg), "donors": donors, "models": ms},
    )
    register(
        f,
        {
            "registration": file_hash(reg),
            "models": file_hash(out / "models.json"),
            "windows": forecasts,
        },
    )
    x = eo / "exposure.json"
    if not x.exists():
        register(
            x,
            {
                k: {a.stem: file_hash(a) for a in (early / k).glob("conv-*.json")}
                for k in ("writers", "readers", "pairs", "judged")
            },
        )
    register(
        ef,
        {
            "inputs": early_fixed["inputs"],
            "parent": file_hash(f),
            "exposure": file_hash(x),
            "windows": forecasts_early,
        },
    )
    return f


def attribution(p0, p1, means):
    def combine(t, q):
        return t * means[0] + (1 - t) * ((1 - q) * means[1] + q * means[2])

    t0, t1 = p0[0], p1[0]
    q0, q1 = p0[2] / (1 - t0), p1[2] / (1 - t1)
    a, b, c, d = combine(t0, q0), combine(t1, q0), combine(t0, q1), combine(t1, q1)
    return [(b - a + d - c) / 2, (c - a + d - b) / 2]


def budget_report(path):
    out = path.parent
    for f in sorted(out.glob("through-*.json")):
        dest = out / ("budgets-" + f.stem.split("-")[-1] + ".json")
        if dest.exists():
            if read(dest)["source"] != file_hash(f):
                raise ValueError("Changed physical budget anchor")
            continue
        result = read(f)
        rows = {}
        for w in result["rows"]:
            probs = w["p"]
            y = w["means"]
            g, q = attribution(probs["raw"], probs["credit"], y)
            if abs(g + q - (w["f1"]["credit"] - w["f1"]["raw"])) > 1e-10:
                raise ValueError(
                    "Gate and routing changes did not add to observed contrast"
                )
            ceilings = {}
            for k in ("fixed_t_raw", "raw"):
                t = probs[k][0]
                ceilings[k] = t * y[0] + (1 - t) * max(y[1:])
            ceilings.update(
                unrestricted=max(y), native_mix=0.2 * w["f1"]["native"] + 0.8 * max(y)
            )
            rows[w["window"]] = {
                "partition": w["partition"],
                "questions": w["n"],
                "gate": g,
                "route": q,
                "ceilings": ceilings,
            }
        sample = [v for v in rows.values() if v["partition"] == "evaluation"]
        n = sum(v["questions"] for v in sample)
        totals = (
            {
                k: sum(v["questions"] * v[k] for v in sample) / n
                for k in ("gate", "route")
            }
            if n
            else {}
        )
        if n:
            totals["ceilings"] = {
                k: sum(v["questions"] * v["ceilings"][k] for v in sample) / n
                for k in sample[0]["ceilings"]
            }
        register(
            dest,
            {
                "source": file_hash(f),
                "windows": rows,
                "questions": n,
                "evaluation": totals,
            },
        )


def run(base, *, register_only=False):
    if register_only:
        return registration(base)
    gpu()
    f = freeze(base)
    score(
        base,
        f,
        allocate,
        extra_controls=(
            "fixed_t_credit",
            "gate_only",
            "route_only",
            "joint_graph",
            "two_edge",
        ),
        replay_choices=True,
    )
    budget_report(f)
    population_bounds(base, f)
    return f


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("--register-only", action="store_true")
    a = cli.parse_args()
    run(a.base, register_only=a.register_only)
