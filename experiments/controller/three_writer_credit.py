"""Combine two frozen pairwise action models at an observed three-skill decision."""

import argparse
import json
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from experiments.controller import file_hash, read, register
from experiments.controller.semantic_writer import REL, TEMP, sha
from experiments.controller.third_skill_capacity import run as measure_capacity
from src.credit_assignment import _digest
from src.evaluation_cache import digest
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "three_writer_credit.json"
ACTIVITY = "capture_activity_details"
ARMS = {
    "native": ("native", "native"),
    "credit_tail": ("credit", "tail"),
    "return_tail": ("return_only", "tail"),
    "terminal_tail": ("terminal", "tail"),
    "native_tail": ("native", "tail"),
    "credit_native": ("credit", "native"),
    "credit_whole": ("credit", "whole"),
}


def three_probabilities(activity, relation):
    if not ((activity > 0) & (activity < 1) & (relation > 0) & (relation < 1)).all():
        raise ValueError("Pairwise forecasts must have finite odds")
    return torch.stack(
        [torch.zeros_like(activity), activity.logit(), relation.logit()], dim=-1
    ).softmax(dim=-1)


def registration(base):
    native = base / "late-pairs/outcomes"
    semantic = base / "late-pairs/semantic-writer"
    out = semantic / "three-writer"
    out.mkdir(exist_ok=True)
    path = out / "registration.json"
    identity = {
        "plan": file_hash(PLAN),
        "panel": file_hash(semantic / "panel.json"),
        "forecast": file_hash(semantic / "forecast.json"),
        "fit": file_hash(base / "source-compression/retrieval-competition/fit.json"),
    }
    if not path.exists():
        register(
            path,
            dict(
                identity,
                preexisting_relation={
                    str(p.relative_to(semantic)): file_hash(p)
                    for kind in ("write.json", "pair.json")
                    for p in semantic.glob("conv-*/" + kind)
                },
                preexisting_two_skill={
                    str(p.relative_to(native)): file_hash(p)
                    for p in native.glob("conv-*/*/scores.json")
                },
            ),
        )
    record = read(path)
    if (
        any(record[k] != v for k, v in identity.items())
        or any(
            file_hash(semantic / k) != v
            for k, v in record["preexisting_relation"].items()
        )
        or any(
            file_hash(native / k) != v
            for k, v in record["preexisting_two_skill"].items()
        )
    ):
        raise ValueError("Protocol, forecast or previously published outcome changed")
    return record


def question_links(base, corpus, panel):
    """Stratify realized effects without exposing evidence to either policy."""
    registered = read(base / "late-pairs/cohort.json")
    if file_hash(corpus) != registered["corpus"]:
        raise ValueError("Different corpus for evidence attribution")
    histories = {h["sample_id"]: h for h in read(corpus)}
    result = {}
    for w in panel:
        sid, start = w["dialogue"], w["start"]
        history = histories[sid]
        turn_ids = [
            {t["dia_id"] for t in history["conversation"][f"session_{n}"]}
            for n in (start, start + 1)
        ]
        links = []
        for i in w["questions"]:
            evidence = set(history["qa"][i]["evidence"])
            if not evidence or evidence - turn_ids[0] - turn_ids[1]:
                raise ValueError("Evidence outside the registered two-session window")
            if evidence <= turn_ids[0]:
                links.append("first_only")
            elif evidence <= turn_ids[1]:
                links.append("last_only")
            else:
                links.append("both")
        result[f"{sid}_s{start}"] = links
    return result


def endpoints(base, w, registered, links):
    name = f"{w['dialogue']}_s{w['start']}"
    outcomes = base / "late-pairs/outcomes" / name
    semantic = base / "late-pairs/semantic-writer" / name
    older = base / "source-compression/retrieval-competition/pairs" / (name + ".json")
    pair = read(semantic / "pair.json")
    if (
        pair["forecast"] != registered["forecast"]
        or pair["prefix_action"] != w["background"]
    ):
        raise ValueError("Misaligned semantic decision")
    result = {
        "name": name,
        "dialogue": w["dialogue"],
        "questions": len(w["questions"]),
        "matched": pair["matched"],
        "semantic": file_hash(semantic / "pair.json"),
    }
    if not pair["matched"]:
        return dict(result, reason=pair["reason"])
    competition = read(older)
    if (
        competition["inputs"]
        != {"pair": file_hash(outcomes / "pair.json"), "fit": registered["fit"]}
        or competition["window"] != name
        or competition["pivot"] != 1
    ):
        raise ValueError("Misaligned activity model or archived pair")
    spans, scores = [], []
    for suffix, skill in enumerate((TEMP, ACTIVITY)):
        bits = f"{w['background']}{suffix}"
        folder = outcomes / bits
        (trace,) = (folder / "traces").glob("*.jsonl")
        if file_hash(folder / "scores.json") != competition["paths"][bits]:
            raise ValueError("Changed activity or temporal outcomes")
        if suffix == 0 and file_hash(trace) != pair["source_trace"]:
            raise ValueError("Different temporal control for the relation branch")
        events = [
            r
            for line in trace.read_bytes().splitlines()
            if (r := json.loads(line))["kind"] == "span"
        ]
        if len(events) != 2 or events[1]["selected_skills"][-1]["name"] != skill:
            raise ValueError("Unexpected intervention or trace length")
        spans.append(events)
        scores.append(read(folder / "scores.json")["answers"])
    (prefix, event), (other_prefix, other) = spans
    refs = list(restore_text([prefix]).items())
    choice = pair["choice"]
    forecasts = pair["source_forecast"]
    state = digest([event["source"], [text for _, text in refs]])
    if (
        list(restore_text([other_prefix]).items()) != refs
        or choice["fixed_prefix"] != _digest(refs)
        or any(
            event[k] != other[k]
            for k in ("source", "retrieved", "refs", "source_turns")
        )
        or prefix["selected_skills"] != other_prefix["selected_skills"]
        or choice["selected"][:-1] != event["selected_skills"][:-1]
        or other["selected_skills"][:-1] != event["selected_skills"][:-1]
        or choice["selected"][-1]["name"] != REL
        or sha(event["source"]) != forecasts["source"]
        or competition["state_hashes"][str(w["background"] + 1)] != state
        or read(semantic / "choice.json") != choice
        or read(semantic / "write.json")["choice"]
        != file_hash(semantic / "choice.json")
    ):
        raise ValueError(
            "Third-slot interventions do not share their pre-write context"
        )
    relation = {
        "native": choice["native"],
        "whole": forecasts["whole"],
        "tail": forecasts["tail"],
    }
    focus = competition["focal_probabilities"]
    inputs = [(focus[a], relation[b]) for a, b in ARMS.values()]
    x = torch.tensor(inputs, device="cuda", dtype=torch.float64)
    ps = dict(zip(ARMS, three_probabilities(*x.T).tolist(), strict=True))
    rows = []
    for i, (lo, act, r, index, category) in enumerate(
        zip(*scores, pair["rows"], w["questions"], w["types"], strict=True)
    ):
        old = r["temporal"]
        if (
            lo["qa_index"] != i
            or act["qa_index"] != i
            or int(r["category"]) != category
            or r["original_index"] != index
            or old["f1"] != lo["f1"]
            or old["answer"] != lo["prediction"]
            or old["indices"] != lo["retrieved_indices"]
            or old["tokens"] != lo["tokens"]
        ):
            raise ValueError(
                "Three endpoints do not share their question or temporal control"
            )
        rows.append(
            {
                "index": index,
                "category": category,
                "linkage": links[i],
                "f1": [lo["f1"], act["f1"], r["relation"]["f1"]],
            }
        )
    return dict(
        result, probabilities=ps, rows=rows, activity=file_hash(older), state=state
    )


def paired_differences(matched, groups):
    """Question-weighted dialogue bootstrap, re-matching skill visits in every draw."""
    names = sorted({w["dialogue"] for w in matched})
    groups = [groups[name] for name in names]
    counts = np.array([g["questions"] for g in groups])
    visits = np.array([sum(w["dialogue"] == name for w in matched) for name in names])
    probability_sums = np.array(
        [
            np.sum(
                [
                    w["probabilities"]["credit_tail"]
                    for w in matched
                    if w["dialogue"] == name
                ],
                axis=0,
            )
            for name in names
        ]
    )
    skill_sums = (
        np.array(
            [[g["f1"][k] for k in ("temporal", "activity", "relation")] for g in groups]
        )
        * counts[:, None]
    )
    draws = np.array(list(product(range(len(names)), repeat=len(names))))
    question_counts = counts[draws].sum(axis=1)
    boot_rates = (
        probability_sums[draws].sum(axis=1) / visits[draws].sum(axis=1)[:, None]
    )
    candidate = counts * [g["f1"]["credit_tail"] for g in groups]
    results = {}
    for arm in (
        "native",
        "terminal_tail",
        "return_tail",
        "rate",
        "temporal",
        "activity",
        "relation",
    ):
        if arm not in groups[0]["f1"]:
            continue
        gap = np.array([g["f1"]["credit_tail"] - g["f1"][arm] for g in groups])
        if arm == "rate":
            reference = (boot_rates * skill_sums[draws].sum(axis=1)).sum(axis=1)
            sampled = (candidate[draws].sum(axis=1) - reference) / question_counts
        else:
            sampled = (counts * gap)[draws].sum(axis=1) / question_counts
        results[arm] = {
            "difference": float(counts @ gap / counts.sum()),
            "dialogues_positive": int((gap > 1e-12).sum()),
            "interval": np.quantile(sampled, [0.025, 0.975]).tolist()
            if len(names) >= 2
            else None,
        }
    return results


def aggregate(reports):
    matched = [r for r in reports if r["matched"]]
    result = {
        "windows": len(reports),
        "matched_windows": len(matched),
        "unmatched_questions": sum(r["questions"] for r in reports if not r["matched"]),
    }
    if not matched:
        return result
    rate = [
        mean(r["probabilities"]["credit_tail"][k] for r in matched) for k in range(3)
    ]
    strata = {}
    for w in matched:
        probs = dict(
            w["probabilities"],
            rate=rate,
            temporal=(1, 0, 0),
            activity=(0, 1, 0),
            relation=(0, 0, 1),
        )
        for row in w["rows"]:
            values = {
                k: sum(p * f for p, f in zip(weights, row["f1"], strict=True))
                for k, weights in probs.items()
            }
            for key in (
                "all",
                w["dialogue"],
                "category_" + str(row["category"]),
                "evidence_" + row["linkage"],
            ):
                group = strata.setdefault(key, {k: [] for k in values})
                for arm, v in values.items():
                    group[arm].append(v)
    groups = {
        k: {
            "questions": len(g["native"]),
            "f1": {arm: mean(scores) for arm, scores in g.items()},
        }
        for k, g in strata.items()
    }
    return dict(
        result,
        comparisons=paired_differences(matched, groups),
        matched_rate=rate,
        mean_total_variation=mean(
            sum(
                abs(a - b)
                for a, b in zip(
                    w["probabilities"]["native"],
                    w["probabilities"]["credit_tail"],
                    strict=True,
                )
            )
            / 2
            for w in matched
        ),
        groups=groups,
    )


def run(base, corpus):
    semantic = base / "late-pairs/semantic-writer"
    out = semantic / "three-writer"
    registration_data = registration(base)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "A single leased GPU is required to compose the action policies"
        )
    torch.cuda.set_per_process_memory_fraction(0.05)
    panel = read(semantic / "panel.json")["windows"]
    links = question_links(base, corpus, panel)
    register(
        out / "linkage.json", {"panel": registration_data["panel"], "links": links}
    )
    prior = {Path(k).parts[0] for k in registration_data["preexisting_relation"]}
    rows = []
    for w in panel:
        name = f"{w['dialogue']}_s{w['start']}"
        p = semantic / name / "pair.json"
        if not p.exists() or (
            read(p)["matched"]
            and not (
                base
                / "source-compression/retrieval-competition/pairs"
                / (name + ".json")
            ).exists()
        ):
            break
        item = endpoints(base, w, registration_data, links[name])
        register(out / (name + ".json"), item)
        rows.append(item)
    if rows:
        result = {
            "registration": file_hash(out / "registration.json"),
            "registered_windows": len(panel),
            "registered_questions": sum(len(w["questions"]) for w in panel),
            "all": aggregate(rows),
            "strict": aggregate([r for r in rows if r["name"] not in prior]),
        }
        register(out / f"through-{len(rows):02d}.json", result)
        if len(rows) == len(panel):
            register(out / "summary.json", result)
        if (base / "late-pairs/selection-labels/split.json").is_file():
            measure_capacity(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    run(args.base, args.corpus)
