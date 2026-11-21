"""Freeze three-skill policies at the twelve original pre-write training states."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.direct_skill_allocation import (
    direct_distributions,
    features,
)
from experiments.controller.resume_status import check_equal_source_choice
from experiments.controller.semantic_writer import (
    REL,
    probabilities,
    sha,
    source_offsets,
)
from experiments.controller.skill_graph_policy import allocate as graph_allocate
from experiments.controller.skill_graph_policy import predict
from experiments.controller.three_writer_credit import ACTIVITY, three_probabilities
from experiments.controller.witness_passage_policy import limited
from src.episode_trace import skill_spec
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "earliest_three_skills.json"


def frozen_policies(native, geometry, offset, bg, graph, margin, direct):
    def t(v):
        return native.new_tensor(v)

    m = {k: t(graph[k]) for k in ("center", "sd", "shift", "anchor")}
    m["coef"] = {k: t(v) for k, v in graph["coef"].items()}
    gp = graph_allocate(native, {"graph": predict(m, [offset, geometry[5]])})
    act = margin["activity"]
    g = t(geometry)
    ax = torch.cat(
        (
            t([1.0]),
            ((g - t(act["center"])) / t(act["sd"])).clamp(-3, 3) / len(g) ** 0.5,
            t([2 * bg - 1.0]),
        )
    )
    rx = t([1.0, offset / 2])
    two = {
        k: [float(ax @ t(v)), float(rx @ t(margin["relation"]["coef"][k]))]
        for k, v in act["coef"].items()
    }
    dx = features(g[5], t(offset), t(bg), t(direct["center"]), t(direct["sd"]), False)
    dp = direct_distributions(
        native,
        {
            "two_edge": two,
            "direct": {k: float(dx @ t(v)) for k, v in direct["coef"].items()},
        },
    )
    return {
        "native": native,
        **{"graph_" + k: v for k, v in gp.items() if k != "native"},
        **{"direct_" + k: v for k, v in dp.items() if k != "native"},
    }


def freeze(trainer, study, parent, specs):
    b = parent.parents[5]
    src = b / "source-compression"
    root = src / "earliest-three-skills"
    root.mkdir(exist_ok=True)
    earlier = src / "earlier-terminal-returns"
    trial = b / "late-pairs/semantic-writer/three-writer"
    graph = trial / "skill-graph/complete"
    direct = trial / "direct-allocation/time"
    margin = trial / "margin"
    if parent != graph / "forecast.json":
        raise ValueError("Different graph checkpoint")
    modelpaths = {
        k: d / "models.json"
        for k, d in (("graph", graph), ("direct", direct), ("margin", margin))
    }
    geo = src / "matched-span-rank/earlier-sources/features.json"
    sf = src / "semantic-writer/source-forecast.json"
    rel_spec = src / "semantic-writer/conv-26_s5/choice.json"
    old = src / "legacy-terminal-symmetry/registration.json"
    oldreg = read(old)
    reg = earlier / "registration.json"
    cohort = read(reg)
    proto = read(study / "protocol.json")
    inputs = {
        k: file_hash(p)
        for k, p in {
            "plan": PLAN,
            "parent": parent,
            "geometry": geo,
            "affinity": sf,
            "original": old,
            "cohort": reg,
            "eligible": earlier / "prepared-complete.json",
            "skill": rel_spec,
            "protocol": study / "protocol.json",
            **modelpaths,
        }.items()
    }
    f = root / "forecast.json"
    if f.exists():
        if read(f)["inputs"] != inputs:
            raise ValueError("Changed preregistered early forecast")
        return
    if (
        oldreg["protocol"] != inputs["protocol"]
        or cohort["archive"] != inputs["original"]
        or proto["skills"] != specs
        or set(proto["train"]) & set(proto["validation"] + proto["test"])
        or len(cohort["windows"]) != 12
    ):
        raise ValueError("Different earlier training cohort")
    if any(root.glob("writers/conv-*.json")):
        raise ValueError("Relation writer preceded forecast")
    model = {k: read(path)["folds"] for k, path in modelpaths.items()}
    for k, d in (("graph", graph), ("direct", direct), ("margin", margin)):
        if read(d / "forecast.json")["models"] != inputs[k]:
            raise ValueError("Changed frozen model")
    idx = {op["name"]: i for i, op in enumerate(specs)}
    relation = skill_spec(
        SimpleNamespace(**dict(specs[idx[REL]], source_rule="dated_single_turns"))
    )
    if relation != read(rel_spec)["selected"][-1]:
        raise ValueError("Different third writer")
    affinities = read(sf)["rows"]
    vec = read(geo)["features"]
    with limited():
        a = torch.tensor(
            [r["affinity"] for r in affinities if r["session"] <= 4],
            device="cuda",
            dtype=torch.float64,
        )
        if len(a) != 24:
            raise ValueError("Changed source scaling")
        scaled = source_offsets(a, a)
        offsets = {
            (r["dialogue"], r["session"]): (r["source"], v)
            for r, v in zip(
                [r for r in affinities if r["session"] <= 4],
                scaled.tolist(),
                strict=True,
            )
        }
    records = {}
    for name, ids in cohort["windows"].items():
        h, n = name.rsplit("_s", 1)
        n = int(n) + 1
        ep = study / "train" / name
        (tr,) = (ep / "traces").glob("*.jsonl")
        rf = ep / "rows.json"
        if (
            h not in proto["train"]
            or file_hash(tr) != oldreg["windows"][name]["trace"]
            or file_hash(rf) != oldreg["windows"][name]["rows"]
        ):
            raise ValueError("Changed early trajectory")
        spans = [
            e
            for line in tr.read_bytes().splitlines()
            if (e := json.loads(line))["kind"] == "span"
        ]
        first, event = spans
        r0, row = read(rf)
        bg = int(r0["use"])
        check_equal_source_choice(r0, specs, first["selected_skills"])
        check_equal_source_choice(row, specs, event["selected_skills"])
        v, off = offsets[h, n]
        node = f"{h}_s{n}"
        if sha(event["source"]) != v or REL in [
            q["name"] for q in event["selected_skills"][:-1]
        ]:
            raise ValueError("Invalid pre-action source or slots")
        if any(
            h in ds
            for ds in (
                model["graph"][h]["histories"],
                model["direct"][h]["histories"],
                model["margin"][h]["activity"]["histories"],
                model["margin"][h]["relation"]["histories"],
            )
        ):
            raise ValueError("Target history used in fit")
        with limited():
            p = [
                probabilities(trainer, row["state"], idx, 0, alternative=k)
                for k in (ACTIVITY, REL)
            ]
            if abs(row["base_logit"] - p[0]["base_logit"]) > 1e-5:
                raise ValueError("Native checkpoint failed to reproduce activity logit")
            nv = three_probabilities(
                *torch.tensor(
                    [x["native"] for x in p], device="cuda", dtype=torch.float64
                )
            )
            policies = frozen_policies(
                nv,
                vec[node],
                off,
                bg,
                model["graph"][h],
                model["margin"][h],
                model["direct"][h],
            )
        live = restore_text([first])
        versions = {c["memory_id"]: c["version"] for c in first["changes"]}
        if [(live[mid], versions[mid]) for mid, _ in event["refs"]] != list(
            zip(event["retrieved"], [ver for _, ver in event["refs"]], strict=True)
        ):
            raise ValueError("Different pre-write memory versions")
        records[name] = {
            "history": h,
            "background": bg,
            "native_activity": row["use"],
            "source": v,
            "features": [off, vec[node][5]],
            "trace": file_hash(tr),
            "rows": file_hash(rf),
            "selected": [*event["selected_skills"][:-1], relation],
            "native_logits": [x["base_logit"] for x in p],
            "probabilities": {k: v.tolist() for k, v in policies.items()},
            "questions": ids,
        }
    register(
        f,
        {
            "inputs": inputs,
            "earlier_outcomes_at_freeze": {
                p.stem: file_hash(p) for p in (earlier / "outcomes").glob("conv-*.json")
            },
            "windows": records,
        },
    )
