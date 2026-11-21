"""Score frozen original-state policies on measured branches, including partial identification."""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.controller.earliest_three_skills import (
    alternate_slots,
    verify_response,
)
from experiments.controller.factorial_rollout import read
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.skill_choice_replay import decisions, uniform
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited
from src.evaluation_cache import digest

ARMS = ("temporal", "activity", "relation")
FAMILIES = ("direct", "graph")


def contrast_bounds(delta, scores):
    """Sum credit-minus-raw F1 over questions, allowing each missing score in [0,1]."""
    if scores.ndim != 2 or scores.shape[1] != len(delta):
        raise ValueError("Wrong contrast dimensions")
    unknown = scores.isnan()
    if ((~unknown) & ((scores < 0) | (scores > 1))).any():
        raise ValueError("Invalid F1")
    observed = (scores.nan_to_num() * delta).sum()
    return torch.stack(
        [
            observed + (unknown * v).sum()
            for v in (delta.clamp_max(0), delta.clamp_min(0))
        ]
    )


def files_at(root):
    return {p.name: file_hash(p) for p in sorted(root.glob("conv-*.json"))}


def available(out, earlier):
    return {
        "forecast": file_hash(out / "forecast.json"),
        "binary": files_at(earlier / "outcomes"),
        **{k: files_at(out / k) for k in ("readers", "pairs", "judged")},
    }


def arm_data(out, earlier, study, name, w, inputs):
    file = name + ".json"
    count = len(w["questions"])
    scores = [[None] * 3 for _ in range(count)]
    levels = {}
    binary_path = earlier / "outcomes" / file
    prep_path = earlier / "prepared" / file
    prep = read(prep_path)
    for stage, directory in (
        ("binary", earlier / "outcomes"),
        ("readers", out / "readers"),
        ("pairs", out / "pairs"),
        ("judged", out / "judged"),
    ):
        if file in inputs[stage]:
            path = directory / file
            if file_hash(path) != inputs[stage][file]:
                raise ValueError("Changed measured stage")
            levels[stage] = read(path)
    bins = levels.get("binary", {})
    if bins and bins["inputs"]["parent"] != file_hash(prep_path):
        raise ValueError("Unbound two-skill outcomes")
    if bins.get("status") == "matched":
        for i, (idx, row, spec) in enumerate(
            zip(w["questions"], bins["rows"], prep["questions"], strict=True)
        ):
            if idx != row["original_index"] or idx != spec["qa"]["original_index"]:
                raise ValueError("Wrong binary question")
            for key, arm in row["arms"].items():
                a = int((key == "native") == w["native_activity"])
                verify_response(
                    earlier / "responses", arm["receipt"], spec[key]["hash"]
                )
                scores[i][a] = arm["f1"]
            if abs(scores[i][1] - scores[i][0] - row["delta"]) > 1e-10:
                raise ValueError("Wrong binary orientation")
    relation = levels.get("readers", {})
    if relation.get("status") == "answered":
        packet = out / "prepared" / file
        if (
            relation["inputs"]["parent"] != file_hash(packet)
            or relation["inputs"]["forecast"] != inputs["forecast"]
        ):
            raise ValueError("Unbound relation readers")
        for i, (idx, r, spec) in enumerate(
            zip(
                w["questions"], relation["rows"], read(packet)["questions"], strict=True
            )
        ):
            if idx != r["index"] or idx != spec["index"]:
                raise ValueError("Wrong relation question")
            verify_response(out / "responses", r["receipt"], spec["hash"])
            scores[i][2] = r["f1"]
    pair = levels.get("pairs", {})
    costs = None
    ordered_skills = {}
    groups = {}
    if pair.get("status") == "matched":
        if (
            pair["inputs"]["early_outcomes"] != file_hash(binary_path)
            or pair["inputs"]["parent"] != inputs["readers"][file]
            or pair["inputs"]["forecast"] != inputs["forecast"]
            or pair["f1"] != scores
        ):
            raise ValueError("Unbound triplet")
        (trace,) = (study / "train" / name / "traces").glob("*.jsonl")
        if file_hash(trace) != w["trace"]:
            raise ValueError("Wrong native writer")
        spans = [
            v
            for l in trace.read_bytes().splitlines()
            if (v := json.loads(l))["kind"] == "span"
        ]
        receipts = spans[-1]["executor_responses"]
        native = [
            sum(r[key] for r in receipts) for key in ("input_tokens", "output_tokens")
        ]
        alternate = read(earlier / "writers" / file)
        if prep["inputs"]["parent"] != file_hash(earlier / "writers" / file):
            raise ValueError("Changed alternative writer")
        alt = alternate["response"]["tokens"]
        writer = [alt, native] if w["native_activity"] else [native, alt]
        rel = read(out / "writers" / file)
        verify_response(out / "responses", rel["response"], rel["prompt_hash"])
        writer.append(rel["response"]["tokens"])
        factual, other = (
            ("activity", "temporal")
            if w["native_activity"]
            else ("temporal", "activity")
        )
        decisions_file = trace.parent.parent / "rows.json"
        if file_hash(decisions_file) != w["rows"]:
            raise ValueError("Different three-slot controller action")
        selected = alternate_slots(
            read(decisions_file)[1],
            read(study / "protocol.json")["skills"],
            spans[-1],
            alternate,
        )
        for arm, skills in (
            (factual, spans[-1]["selected_skills"]),
            (other, selected),
            ("relation", w["selected"]),
        ):
            ordered_skills[arm] = [
                {"name": s["name"], "version": s["version"]} for s in skills
            ]
        reader = [
            [sum(r[a]["receipt"]["tokens"][i] for r in pair["rows"]) for i in (0, 1)]
            for a in ARMS
        ]
        costs = [w_cost + r_cost for w_cost, r_cost in zip(writer, reader, strict=True)]
        for i, q in enumerate(prep["questions"]):
            groups[i] = (
                "all",
                w["history"],
                "focal" if q["linked"] else "earlier_only",
                "category_" + str(q["qa"]["category"]),
            )
    return scores, levels, costs, groups, ordered_skills


def population(out, study, corpus):
    f = out / "population.json"
    root = out / "forecast.json"
    identity = {
        "forecast": file_hash(root),
        "protocol": file_hash(study / "protocol.json"),
        "corpus": file_hash(corpus),
    }
    if f.exists():
        if read(f)["inputs"] != identity:
            raise ValueError("Changed early question population")
        return f
    pred = read(root)
    proto = read(study / "protocol.json")
    if (
        pred["inputs"]["protocol"] != identity["protocol"]
        or proto["input_sha256"] != identity["corpus"]
    ):
        raise ValueError("Changed question archive")
    corpus_rows = {
        c["sample_id"]: c["qa"]
        for c in read(corpus)
        if c["sample_id"] in proto["train"]
    }
    counts = Counter()
    windows = {}
    unique = set()
    for name, w in pred["windows"].items():
        (tr,) = (study / "train" / name / "traces").glob("*.jsonl")
        if file_hash(tr) != w["trace"]:
            raise ValueError("Changed focal span")
        _, event = [
            v
            for line in tr.read_bytes().splitlines()
            if (v := json.loads(line))["kind"] == "span"
        ]
        turns = set(event["source_turns"])
        h = w["history"]
        rows = {}
        for idx in w["questions"]:
            q = corpus_rows[h][idx]
            linked = bool(turns.intersection(q["evidence"]))
            rows[str(idx)] = {"category": q["category"], "focal": linked}
            counts.update(
                (
                    "all",
                    h,
                    "focal" if linked else "earlier_only",
                    "category_" + str(q["category"]),
                )
            )
            unique.add((h, idx))
        windows[name] = rows
    register(
        f,
        {
            "inputs": identity,
            "planned": dict(counts),
            "unique_questions": len(unique),
            "windows": windows,
        },
    )
    return f


def paired_histories(strata, families):
    observed = [v for k, v in strata.items() if k.startswith("conv-")]
    if len(observed) < 2:
        return {}
    with limited():
        device = "cuda"
        ids = torch.arange(len(observed), device=device)
        draws = torch.cartesian_prod(*([ids] * len(observed)))
        intervals = {}
        for family in families:
            raw, credit = family + "_raw", family + "_credit"
            x = torch.tensor(
                [
                    [r[raw][0], r[credit][1] - r[raw][1], r[credit][2] - r[raw][2]]
                    for r in observed
                ],
                dtype=torch.float64,
                device=device,
            )
            sums = x[draws].sum(1)
            lo, hi = torch.quantile(
                sums[:, 1:] / sums[:, :1], x.new_tensor([0.025, 0.975]), dim=0
            ).tolist()
            intervals[family] = {
                "expected": [lo[0], hi[0]],
                "drawn": [lo[1], hi[1]],
                "histories": len(observed),
            }
    return intervals


def evaluate(
    base, study, corpus, *, state=False, factor=None, gate=False, outlier=None
):
    src = base / "source-compression"
    earlier = src / "earlier-terminal-returns"
    out = src / "earliest-three-skills"
    family = None
    policy = out
    if state:
        family, policy = "state", out / "checkpoint-odds"
    if factor:
        family, policy = "factor", out / ("writer-format-" + factor)
    if gate:
        family, policy = "gate", out / "temporal-gate"
    if outlier:
        family, policy = "outlier", out / outlier
    forecast = policy / "forecast.json"
    results = forecast.parent / "evaluation"
    results.mkdir(exist_ok=True)
    inputs = available(out, earlier)
    census = population(out, study, corpus)
    inputs["population"] = file_hash(census)
    if family:
        inputs[family + "_policy"] = file_hash(forecast)
    path = results / ("snapshot-" + digest(inputs) + ".json")
    final = (
        (out / "judged-complete.json").is_file()
        and (earlier / "outcomes-complete.json").is_file()
        and len(inputs["judged"]) == len(inputs["binary"]) == 12
    )
    if path.exists():
        if final:
            register(
                results / "summary.json",
                {"snapshot": path.name, "hash": file_hash(path)},
            )
        return path
    pred = read(forecast)
    if family and pred["inputs"] != read(out / "forecast.json")["inputs"]:
        raise ValueError("Different original-state cohort")
    windows = {}
    strata = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    families = (family,) if family else FAMILIES
    totals = {family: [0.0, 0.0] for family in families}
    matched = binary = relation = denominator = 0
    for name, w in pred["windows"].items():
        scores, stages, costs, groups, ordered_skills = arm_data(
            out, earlier, study, name, w, inputs
        )
        n = len(scores)
        denominator += n
        binary += n if stages.get("binary", {}).get("status") == "matched" else 0
        relation += n if stages.get("readers", {}).get("status") == "answered" else 0
        matched += n if costs is not None else 0
        draw = uniform(pred["inputs"]["cohort"], name)
        keys = list(w["probabilities"])
        reports = {}
        with limited():

            def matrix(v):
                return torch.tensor(v, dtype=torch.float64, device="cuda")

            table = matrix(
                [[float("nan") if x is None else x for x in r] for r in scores]
            )
            prob = matrix([w["probabilities"][k] for k in keys])
            chosen, _ = decisions(prob, 0, draw)
            for family, bounds in totals.items():
                raw = keys.index(family + "_raw")
                j = keys.index(family + "_credit")
                delta = prob[j] - prob[raw]
                lo, hi = contrast_bounds(delta, table).tolist()
                _, change = decisions(prob, raw, draw)
                reports[family] = {
                    "sum_delta_f1_bounds": [lo, hi],
                    "switch_probability": float(change[j]),
                    "probability_delta": delta.tolist(),
                    "draw_changed": bool(chosen[raw] != chosen[j]),
                }
                bounds[0] += lo
                bounds[1] += hi
            choices = {k: ARMS[a] for k, a in zip(keys, chosen.tolist(), strict=True)}
            usage = {}
            if costs is not None:
                expected = (table @ prob.T).tolist()
                sampled = table[:, chosen].tolist()
                ecost = (prob @ matrix(costs)).tolist()
                for j, k in enumerate(keys):
                    usage[k] = {"expected": ecost[j], "drawn": costs[chosen[j].item()]}
                    for i, names in groups.items():
                        for group in names:
                            acc = strata[group][k]
                            acc[0] += 1
                            acc[1] += expected[i][j]
                            acc[2] += sampled[i][j]
        windows[name] = {
            "questions": n,
            "draw": draw,
            "selected": choices,
            "stages": {k: v["status"] for k, v in stages.items()},
            "cases": stages.get("judged", {}).get("cases", []),
            "contrasts": reports,
            "token_costs": usage,
            **({"ordered_skills": ordered_skills} if ordered_skills else {}),
        }
    register(
        path,
        {
            "inputs": inputs,
            "windows": windows,
            "questions": denominator,
            "matched_questions": matched,
            "binary_questions": binary,
            "relation_questions": relation,
            "population_delta_bounds": {
                k: [v / denominator for v in pair] for k, pair in totals.items()
            },
            "population_by_stratum": read(census)["planned"],
            "paired_dialogue_intervals": paired_histories(strata, totals),
            "matched_f1": {
                g: {
                    k: {"n": n, "expected": e / n, "drawn": s / n}
                    for k, (n, e, s) in rows.items()
                }
                for g, rows in strata.items()
            },
            "cost_order": [
                "terminal_writer_input",
                "terminal_writer_output",
                "all_reader_input",
                "all_reader_output",
            ],
        },
    )
    if final:
        register(
            results / "summary.json", {"snapshot": path.name, "hash": file_hash(path)}
        )
    print(path.name, binary, relation, matched, flush=True)
    return path


def switched_evidence(base, snapshot, family):
    """Join actual common-draw switches to the matching physical question and repair."""
    out = base / "source-compression/earliest-three-skills"
    target = snapshot.parent / (
        "switches-" + snapshot.stem.removeprefix("snapshot-") + ".json"
    )
    if target.exists():
        if read(target)["snapshot"] != file_hash(snapshot):
            raise ValueError("Different decision snapshot")
        return
    d = read(snapshot)
    pop = read(out / "population.json")["windows"]
    windows = {}
    for name, w in d["windows"].items():
        if not w["contrasts"][family]["draw_changed"]:
            continue
        raw, chosen = (w["selected"][family + "_" + k] for k in ("raw", "credit"))
        entry = {"from": raw, "to": chosen, "questions": w["questions"], "rows": []}
        windows[name] = entry
        if not w["token_costs"]:
            entry["status"] = "pending_or_unmatched"
            continue
        pn = name + ".json"
        pair = out / "pairs" / pn
        if d["inputs"]["pairs"][pn] != file_hash(pair):
            raise ValueError("Changed physical switch outcomes")
        judged = {}
        if pn in d["inputs"]["judged"]:
            jf = out / "judged" / pn
            repairs = out / "repairs" / pn
            obs = read(jf)
            if (
                file_hash(jf) != d["inputs"]["judged"][pn]
                or obs["inputs"]["parent"] != file_hash(repairs)
                or read(repairs)["inputs"]["parent"] != file_hash(pair)
            ):
                raise ValueError("Wrong weaker-skill adjudication for observed switch")
            judged = {(c["index"], c["baseline"]): c for c in obs["cases"]}
        baseline = raw if chosen == "relation" else chosen
        for q in read(pair)["rows"]:
            idx = q["original_index"]
            before = q[raw]["f1"]
            after = q[chosen]["f1"]
            delta = after - before
            c = judged.get((idx, baseline)) if "relation" in (raw, chosen) else None
            status = "no_audited_pair" if c is None else c["status"]
            extra = {}
            if c is not None:
                orientation = 1 if chosen == "relation" else -1
                if abs(orientation * c["raw"] - delta) > 1e-10:
                    raise ValueError(
                        "Switched branch orientation contradicts the measured repair"
                    )
                extra = {
                    "signed_choice_credit": orientation * c["credit"],
                    "veto": c.get("veto", []),
                }
            entry["rows"].append(
                {
                    "index": idx,
                    "category": q["category"],
                    "focal": pop[name][str(idx)]["focal"],
                    "from_f1": before,
                    "to_f1": after,
                    "delta": delta,
                    "adjudication": status,
                    **extra,
                }
            )
        entry["status"] = "matched"
        entry["sum_delta"] = sum(r["delta"] for r in entry["rows"])
    n = d["matched_questions"]
    if n:
        actual = sum(w.get("sum_delta", 0.0) for w in windows.values()) / n
        scores = d["matched_f1"]["all"]
        expected = (
            scores[family + "_credit"]["drawn"] - scores[family + "_raw"]["drawn"]
        )
        if abs(actual - expected) > 1e-10:
            raise ValueError("Switches did not reproduce drawn F1")
    register(
        target,
        {"snapshot": file_hash(snapshot), "matched_questions": n, "windows": windows},
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("study", type=Path)
    model = cli.add_mutually_exclusive_group()
    model.add_argument("--state", action="store_true")
    model.add_argument("--gate", action="store_true")
    model.add_argument(
        "--outlier",
        choices=(
            "outlier-total",
            "outlier-components",
            "outlier-initial",
            "outlier-final",
            "outlier-replicated-initial",
            "outlier-replicated-final",
        ),
    )
    model.add_argument(
        "--factor",
        choices=(
            "sensitivity",
            "verified",
            "dense",
            "history",
            "transfer",
            "proxy",
            "sample1",
            "sample2",
        ),
    )
    a = cli.parse_args()
    gpu()
    snapshot = evaluate(
        a.base,
        a.study,
        Path(os.environ["LOCOMO_DATA"]),
        state=a.state,
        factor=a.factor,
        gate=a.gate,
        outlier=a.outlier,
    )

    if a.gate or a.outlier:
        switched_evidence(a.base, snapshot, "outlier" if a.outlier else "gate")
