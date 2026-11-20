"""Audit held-question rank shifts after the deployable numerical forecasts are frozen."""

from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "query_displacement_audit.json"


def score(rows):
    if not rows:
        return {"questions": 0}
    moving = [r for r in rows if r["actual"] != 0]
    return {
        "questions": len(rows),
        "covered": sum(r["histories"] > 0 for r in rows),
        "non_tied": len(moving),
        "mae": {
            k: mean(abs(r[k] - r["actual"]) for r in rows)
            for k in ("translated", "untranslated", "source_only", "zero")
        },
        "sign_matches": {
            k: sum(r[k] * r["actual"] > 0 for r in moving)
            for k in ("translated", "untranslated", "source_only", "zero")
        },
    }


def perform(reg, local, prefix, metric="rank"):
    from experiments.controller.query_displacement_policy import (
        compile_fold,
        donor_advantages,
        load_banks,
        projected_queries,
    )

    out = reg.parent
    dest = local.parent / "held-query-audit.json"
    identity = {
        "local": file_hash(local),
        "prefix": file_hash(prefix),
        "plan": file_hash(PLAN),
        "registration": file_hash(reg),
    }
    if dest.exists():
        if read(dest)["inputs"] != identity:
            raise ValueError("Different audit freeze")
        return
    packet = read(reg)
    queries, banks = load_banks(out.parents[1], reg, "cuda")
    rows = packet["rows"]
    donors = donor_advantages(rows, queries, banks, metric)
    histories = sorted({r["history"] for r in rows})
    # The audit represents each unique donor question in each focal session once.
    distinct = {}
    for i, row in enumerate(rows):
        if i in donors:
            key = (row["node"], row["key"])
            if key in distinct and donors[distinct[key]] != donors[i]:
                raise ValueError("Source-rule proxy differs by writer background")
            distinct[key] = i
    selected = [dict(rows[i], credit=0.0, raw=0.0) for i in distinct.values()]
    blank = {i: 0.0 for i in range(len(selected))}
    folds = {
        (h, j): compile_fold(selected, h, j, histories, queries, banks, blank)
        for h in histories
        for j in (0, 1)
    }
    by_node = defaultdict(list)
    for (node, _), i in distinct.items():
        by_node[node].append(i)
    audit = []
    for node, indices in by_node.items():
        with limited():
            r0 = rows[indices[0]]
            f = folds[r0["history"], r0["stage"]]
            mapped, similarity, values = projected_queries(f, banks[node], metric)
            for i in indices:
                row = rows[i]
                annotated = mapped.new_tensor(row["linked"])
                overlap = (
                    (mapped[:, :, None] == annotated).any(dim=2) & f["mask"]
                ).sum(dim=1) / f["mask"].sum(dim=1)
                outvotes = defaultdict(list)
                for h in f["histories"]:
                    take = (overlap > 0) & similarity.new_tensor(
                        [o == h for o in f["owners"]], dtype=torch.bool
                    )
                    if take.any():
                        weights = (
                            similarity[take] / 0.08 + overlap[take].log()
                        ).softmax(0)
                        for arm, x in values.items():
                            outvotes[arm].append(float(weights @ x[take]))
                forecasts = {
                    k: mean(outvotes[k]) if outvotes[k] else 0.0 for k in values
                }
                audit.append(
                    {
                        "node": node,
                        "question": row["key"],
                        "history": row["history"],
                        "actual": donors[i],
                        "zero": 0.0,
                        "translated": forecasts["credit"],
                        "source_only": forecasts["source_only"],
                        "untranslated": forecasts["untranslated"],
                        "histories": len(outvotes["credit"]),
                    }
                )
    register(
        dest,
        {
            "inputs": identity,
            "all": score(audit),
            "by_history": {
                h: score([r for r in audit if r["history"] == h]) for h in histories
            },
            "rows": audit,
        },
    )
