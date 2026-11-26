"""Fit two action-value contrasts with soft verified-span preference constraints."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import load_donors, weights
from experiments.controller.turn_key_credit import paired_interval
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "three_skill_margin.json"
MODELS = ("raw", "credit", "rotated")


def margin_fit(x, y, w, c, v, *, constraint_x=None, penalty=None, constraint_offset=None):
    """Convex squared-hinge regularization; the raw ridge solution is the zero-credit limit."""
    if constraint_x is None:
        constraint_x = x
    if penalty is None:
        penalty = x.new_full((x.shape[1],), 0.25)
        penalty[0] = 0.02
    gram = x.T @ (w[:, None] * x) + torch.diag(penalty)
    rhs = x.T @ (w * y)
    raw = torch.linalg.solve(gram, rhs)
    if not v.any():
        return raw, raw

    offset = 0.0 if constraint_offset is None else constraint_offset

    def objective(b):
        residual = (c.abs() - c.sign() * (constraint_x @ b + offset)).clamp_min(0)
        return (
            (w * ((x @ b) - y).square()).sum()
            + (penalty * b.square()).sum()
            + (v * residual.square()).sum()
        ) * 0.5

    beta = raw.clone()
    for _ in range(40):
        gap = constraint_x @ beta + offset - c
        active = v * (c.sign() * gap < 0)
        gradient = gram @ beta - rhs + constraint_x.T @ (active * gap)
        if gradient.abs().max() < 1e-10:
            break
        step = torch.linalg.solve(
            gram + constraint_x.T @ (active[:, None] * constraint_x), gradient
        )
        old = objective(beta)
        for _ in range(32):
            proposal = beta - step
            if objective(proposal) <= old - 1e-4 * (gradient @ step) + 1e-14:
                break
            step = step / 2
        else:
            raise ValueError("Margin solver did not descend")
        beta = proposal
    else:
        raise ValueError("Margin solver failed to converge")
    return raw, beta


def relation_donors(src):
    semantic = src / "semantic-writer"
    audit = src / "relation-skill-repairs/0"
    hashes = read(audit / "outcomes-complete.json")
    source = read(semantic / "source-forecast.json")["rows"]
    source = {(v["dialogue"], v["session"]): v for v in source if "offset" in v}
    inputs = {}
    rows = []
    for k, h in sorted(hashes.items()):
        f = audit / "outcomes" / (k + ".json")
        pf = semantic / k / "pair.json"
        o = read(f)
        p = read(pf)
        prep = audit / "prepared" / f.name
        if (
            file_hash(f) != h
            or o["inputs"]["parent"] != file_hash(prep)
            or read(prep)["inputs"]["parent"] != file_hash(pf)
            or o["matched"] != p["matched"]
            or o["questions"] != p["questions"]
        ):
            raise ValueError("Unbound relation donor")
        inputs[k] = {"outcome": h, "pair": file_hash(pf)}
        if not p["matched"]:
            continue
        h, n = k.rsplit("_s", 1)
        j = int(n) + 1
        if source[h, j] != p["source_forecast"]:
            raise ValueError("Different relation source")
        if any(
            r["credit"] != (r["raw"] if r["status"] == "confirmed" else 0.0)
            for r in o["jobs"]
        ):
            raise ValueError("Invalid relation credit")
        raw = mean(r["relation"]["f1"] - r["temporal"]["f1"] for r in p["rows"])
        rows.append(
            {
                "history": h,
                "node": f"{h}_s{j}",
                "background": 0,
                "offset": source[h, j]["offset"],
                "raw": raw,
                "credit": sum(r["credit"] for r in o["jobs"]) / p["questions"],
            }
        )
    if len(hashes) != 12 or len(rows) != 10:
        raise ValueError("Incomplete original relation screen")
    return rows, inputs


def fit_contrasts(rows, x, y, w, ids):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["history"], r["background"]].append(i)
    credit = x.new_tensor([r["credit"] for r in rows])
    rotation = list(range(len(rows)))
    for ix in groups.values():
        ix.sort(key=lambda i: rows[i]["node"])
        for i, k in zip(ix, ix[1:] + ix[:1], strict=True):
            rotation[i] = k
    coef = {}
    counts = {}
    for name, c in (("credit", credit), ("rotated", credit[rotation])):
        has = [
            dict(r, qualified=True) if value else r
            for r, value in zip(rows, c.tolist(), strict=True)
        ]
        v = weights(has, ids, "qualified", x.new_tensor)
        raw, b = margin_fit(x, y, w, c, v)
        coef.update(raw=raw)
        coef[name] = b
        counts[name] = sum("qualified" in r for r in has)
    return coef, counts


def fit_edge(rows, values, held, edge):
    rows = [r for r in rows if r["history"] != held]
    ids = {r["history"] for r in rows}
    if len(ids) != 5:
        raise ValueError("Expected five independent training histories")

    def tensor(v):
        return torch.as_tensor(v, dtype=torch.float64, device="cuda")

    w = weights(rows, ids, "raw", tensor)
    y = tensor([r.get("raw", 0.0) for r in rows])
    if edge == "activity":
        source = tensor([values[r["node"]] for r in rows])
        center = w @ source
        sd = (w @ (source - center).square()).sqrt().clamp_min(0.02)
        x = torch.cat(
            [
                torch.ones_like(w[:, None]),
                ((source - center) / sd).clamp(-3, 3) / source.shape[1] ** 0.5,
                tensor([[2 * r["background"] - 1] for r in rows]),
            ],
            1,
        )
    else:
        center = sd = None
        x = tensor([[1.0, r["offset"] / 2] for r in rows])
    coef, counts = fit_contrasts(rows, x, y, w, ids)
    return {
        "center": center,
        "sd": sd,
        "coef": coef,
        "support": counts,
        "histories": sorted(ids),
    }


def freeze(base, study, corpus):
    late = base / "late-pairs"
    semantic = late / "semantic-writer"
    parent = semantic / "three-writer"
    out = parent / "margin"
    out.mkdir(exist_ok=True)
    values, all_units, activity_inputs = load_donors(base, study, corpus)
    activity = [r for r in all_units if r["stage"] == 1]
    relation, relation_inputs = relation_donors(base / "source-compression")
    panel = read(semantic / "panel.json")["windows"]
    forecast = read(semantic / "forecast.json")["targets"]
    affinity = {(r["dialogue"], r["session"]): r for r in forecast}
    if (
        len(activity) != 43
        or len(panel) != 25
        or sum(len(w["questions"]) for w in panel) != 209
    ):
        raise ValueError("Changed terminal scope")
    split = late / "selection-labels/split.json"
    inputs = {
        "plan": file_hash(PLAN),
        "activity": activity_inputs,
        "relation": relation_inputs,
        "panel": file_hash(semantic / "panel.json"),
        "affinity": file_hash(semantic / "forecast.json"),
        "split": file_hash(split),
        "parent": file_hash(parent / "registration.json"),
    }
    path = out / "forecast.json"
    if path.exists():
        if read(path)["inputs"] != inputs:
            raise ValueError("Changed three-action training inputs")
        return path
    forecasts = {}
    folds = {}
    for h in sorted({w["dialogue"] for w in panel}):
        with limited():
            a = fit_edge(activity, values, h, "activity")
            r = fit_edge(relation, values, h, "relation")

            def pack(m):
                return {
                    "coef": {k: v.tolist() for k, v in m["coef"].items()},
                    "center": m["center"].tolist() if m["center"] is not None else None,
                    "sd": m["sd"].tolist() if m["sd"] is not None else None,
                    "support": m["support"],
                    "histories": m["histories"],
                }

            folds[h] = {"activity": pack(a), "relation": pack(r)}
            for w in panel:
                if w["dialogue"] != h:
                    continue
                n = w["start"] + 1
                name = f"{h}_s{w['start']}"
                node = f"{h}_s{n}"
                x = a["center"].new_tensor(values[node])
                ax = torch.cat(
                    [
                        x.new_ones(1),
                        ((x - a["center"]) / a["sd"]).clamp(-3, 3) / len(x) ** 0.5,
                        x.new_tensor([2 * w["background"] - 1]),
                    ]
                )
                rx = x.new_tensor([1.0, affinity[h, n]["offset"] / 2])
                forecasts[name] = {
                    "source": affinity[h, n]["source"],
                    "background": w["background"],
                    "scores": {
                        k: [float(ax @ a["coef"][k]), float(rx @ r["coef"][k])]
                        for k in MODELS
                    },
                }
    register(out / "models.json", {"inputs": inputs, "folds": folds})
    register(
        path,
        {
            "inputs": inputs,
            "models": file_hash(out / "models.json"),
            "windows": forecasts,
        },
    )
    return path


def distributions(native, scores):
    def mix(a, r):
        effects = native.new_tensor([0.0, a, r])
        return 0.2 * native + 0.8 * torch.softmax(effects / 0.05, dim=0)

    result = {k: mix(*scores[k]) for k in MODELS}
    result.update(
        activity_only=mix(scores["credit"][0], scores["raw"][1]),
        relation_only=mix(scores["raw"][0], scores["credit"][1]),
        native=native,
        uniform=mix(0, 0),
    )
    if not all(
        abs(float(p.sum()) - 1) < 1e-10 and bool((p >= 0).all())
        for p in result.values()
    ):
        raise ValueError("Wrong three-action simplex")
    return result


def aggregate(rows, *, direct=False):
    if not rows:
        return {"windows": 0, "questions": 0}
    n = sum(x["n"] for x in rows)

    def avg(fn):
        return sum(w["n"] * fn(w) for w in rows) / n

    names = rows[0]["f1"]
    rates = {
        k: [avg(lambda w, k=k, i=i: w["p"][k][i]) for i in range(3)] for k in names
    }
    fixed = [avg(lambda w, i=i: w["means"][i]) for i in range(3)]
    result = {
        "windows": len(rows),
        "questions": n,
        "f1": {k: avg(lambda w, k=k: w["f1"][k]) for k in names},
        "fixed": fixed,
        "rates": rates,
        "matched_rate": sum(p * v for p, v in zip(rates["credit"], fixed, strict=True)),
        "window_oracle": avg(lambda w: max(w["means"])),
    }
    if direct:
        for arm in ("credit", "soft"):
            if arm not in rates:
                continue
            relation_rate = rates[arm][2] / (1 - rates[arm][0])
            key = (
                "conditional_matched_rate"
                if arm == "credit"
                else "parent_conditional_matched_rate"
            )
            result[key] = avg(
                lambda w, arm=arm, rate=relation_rate: (
                    w["p"][arm][0] * w["means"][0]
                    + (1 - w["p"][arm][0])
                    * ((1 - rate) * w["means"][1] + rate * w["means"][2])
                )
            )
    return result


def score(
    base,
    path,
    allocate=distributions,
    *,
    direct=False,
    eligible=True,
    extra_controls=(),
    replay_choices=False,
):
    out = path.parent
    late = base / "late-pairs"
    semantic = late / "semantic-writer"
    parent = semantic / "three-writer"
    packet = read(path)
    specs = read(semantic / "panel.json")["windows"]
    split = read(late / "selection-labels/split.json")["windows"]
    preexisting = {}
    for w in specs:
        k = f"{w['dialogue']}_s{w['start']}"
        files = [
            semantic / k / "write.json",
            semantic / k / "pair.json",
            *(late / "outcomes" / k).glob("*/scores.json"),
        ]
        preexisting[k] = {
            str(f.relative_to(late)): file_hash(f)
            for f in files
            if f.is_file() and f.stat().st_mtime_ns <= path.stat().st_mtime_ns
        }
    register(out / "exposure.json", {"forecast": file_hash(path), "paths": preexisting})
    rows = []
    question_reports = []
    question_dir = out / "questions"
    question_dir.mkdir(exist_ok=True)
    hashes = {}
    excluded = {}
    for w in specs:
        name = f"{w['dialogue']}_s{w['start']}"
        f = parent / (name + ".json")
        if not f.exists():
            continue
        d = read(f)
        hashes[name] = file_hash(f)
        pf = semantic / name / "pair.json"
        if (
            d["semantic"] != file_hash(pf)
            or d["name"] != name
            or d["questions"] != len(w["questions"])
        ):
            raise ValueError("Changed physical terminal pair")
        if not d["matched"]:
            excluded[name] = d["reason"]
            continue
        j = packet["windows"][name]
        q = read(pf)
        if (
            q["source_forecast"]["source"] != j["source"]
            or j["background"] != w["background"]
            or [x["index"] for x in d["rows"]] != w["questions"]
        ):
            raise ValueError("Different pre-action source or questions")
        key = "donor" if name in split["donor"] else "evaluation"
        if split[key][name] != w["questions"]:
            raise ValueError("Invalid panel partition")
        with limited():
            p = allocate(
                torch.tensor(
                    d["probabilities"]["native"], device="cuda", dtype=torch.float64
                ),
                j["scores"],
            )
            scores = next(iter(p.values())).new_tensor([q["f1"] for q in d["rows"]])
            if direct:
                actual = scores @ (p["credit"] - p["raw"])
                predicted = (p["credit"][2] - p["raw"][2]) * (
                    scores[:, 2] - scores[:, 1]
                )
                if abs(
                    float(p["credit"][0] - p["raw"][0])
                ) > 1e-12 or not torch.allclose(actual, predicted, atol=1e-12, rtol=0):
                    raise ValueError(
                        "Direct credit changed the temporal probability budget"
                    )
            means = scores.mean(0)
            projected = (scores @ torch.stack(list(p.values()), dim=1)).tolist()
            questions = [
                {**question, "expected": dict(zip(p, expected, strict=True))}
                for question, expected in zip(d["rows"], projected, strict=True)
            ]
            register(
                question_dir / f.name,
                {
                    "forecast": file_hash(path),
                    "source": hashes[name],
                    "rows": questions,
                },
            )
            if key == "evaluation":
                question_reports.extend(questions)
            rows.append(
                {
                    "window": name,
                    "history": w["dialogue"],
                    "partition": key,
                    "strict": not preexisting[name],
                    "n": len(w["questions"]),
                    "means": means.tolist(),
                    "p": {k: v.tolist() for k, v in p.items()},
                    "f1": {k: float(means @ v) for k, v in p.items()},
                }
            )
    panels = {
        k: [r for r in rows if r["partition"] == k] for k in ("donor", "evaluation")
    }
    eval_rows = panels["evaluation"]
    by_history = {
        h: aggregate([r for r in eval_rows if r["history"] == h], direct=direct)
        for h in sorted({w["dialogue"] for w in specs})
    }
    totals = {k: aggregate(v, direct=direct) for k, v in panels.items()}
    strict = aggregate([r for r in eval_rows if r["strict"]], direct=direct)
    if hashes:
        result = {
            "forecast": file_hash(path),
            "sources": hashes,
            "excluded": excluded,
            "rows": rows,
            "all": aggregate(rows, direct=direct),
            "by_partition": totals,
            "strict_evaluation": strict,
            "evaluation_by_history": by_history,
            "registered_questions": 209,
            "registered_windows": 25,
        }
        register(out / f"through-{len(hashes):02d}.json", result)
        if replay_choices:
            from experiments.controller.skill_choice_replay import record

            record(base, path, result)
        active = [v for v in by_history.values() if v["questions"]]
        intervals = {}
        if len(active) >= 2:
            baselines = ["raw", "rotated", "native", "uniform"]
            baselines.extend(k for k in ("zero_new", "previous", "unreplicated") if k in extra_controls)
            for arm in baselines:
                intervals[arm] = paired_interval(
                    [v["questions"] for v in active],
                    [v["f1"]["credit"] - v["f1"][arm] for v in active],
                )
        categories = {}
        for category in sorted({q["category"] for q in question_reports}):
            group = [q for q in question_reports if q["category"] == category]
            categories[str(category)] = {
                "questions": len(group),
                "f1": {
                    a: mean(q["expected"][a] for q in group)
                    for a in group[0]["expected"]
                },
            }
        register(
            question_dir / f"evaluation-{len(hashes):02d}.json",
            {
                "through": file_hash(out / f"through-{len(hashes):02d}.json"),
                "evaluation_histories": len(active),
                "paired_dialogue_intervals": intervals,
                "categories": categories,
            },
        )
        if len(hashes) == 25:
            ev = totals["evaluation"]
            den = sum(
                len(w["questions"])
                for w in specs
                if f"{w['dialogue']}_s{w['start']}" in split["evaluation"]
            )
            ok = False
            if eligible and ev["questions"] >= 0.9 * den and strict["questions"]:
                y = ev["f1"]
                ok = (
                    y["credit"] >= y["native"] + 0.02
                    and y["credit"]
                    > max(
                        y["raw"],
                        y["rotated"],
                        y["uniform"],
                        *(y[k] for k in extra_controls),
                        *ev["fixed"],
                        ev["matched_rate"],
                    )
                    and strict["f1"]["credit"]
                    > max(strict["f1"]["raw"], strict["f1"]["native"])
                    and sum(
                        d["questions"] > 0 and d["f1"]["credit"] > d["f1"]["raw"]
                        for d in by_history.values()
                    )
                    >= 4
                )
            if ok and direct:
                compared = [
                    v
                    for k, v in ev["f1"].items()
                    if k in ("parent_raw", "parent_credit", "time_only")
                ]
                ok = ev["f1"]["credit"] > max(
                    ev["conditional_matched_rate"],
                    ev.get("parent_conditional_matched_rate", 0.0),
                    *compared,
                )
            register(
                out / "summary.json",
                dict(result, coverage=ev["questions"] / den, training_gate=ok),
            )


def run(base, study, corpus):
    score(base, freeze(base, study, corpus))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    gpu()
    run(a.base, a.study, a.corpus)
