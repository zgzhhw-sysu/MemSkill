"""Transfer physically separated reader effects from source and memory features."""

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.credit_rank_allocation import audit_long_prefix
from experiments.controller.factorial_rollout import read
from experiments.controller.late_credit_kernel import kernel_effect, score, support
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.reader_slot_credit import (
    EARLY,
    PLAN,
    components,
    reader_key,
)
from experiments.controller.retrieval_competition import summarize
from experiments.controller.source_compression import file_hash, register
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import read_response

ARMS = ("credit", "raw", "unsplit", "rotated", "native", "terminal")


def crossed_scores(out, model, row, q, evaluator, files=None):
    scores = {row[k]["prompt_sha256"]: row[k]["f1"] for k in ("low", "high")}
    receipts = {}
    for h, p in row["novel"].items():
        key = reader_key(model, p)
        f = out / "hybrids" / (key + ".json")
        if not f.exists() and files is None:
            return None, {}
        v = read(f)
        digest = file_hash(f)
        raw = read_response(out / "responses", key)
        if not raw:
            archive = (
                out.parent / "factorial"
                if os.environ["MEMSKILL_BACKEND_ID"] == EARLY
                else out.parents[1] / "late-pairs/outcomes"
            )
            raw = read_response(archive / "responses", key)
        if (
            not raw
            or raw["sha256"] != v["response_sha256"]
            or not raw["payload"]["prompt_sha256"] == h == v["prompt_sha256"]
            or raw["payload"]["response"] != v["answer"]
            or raw["payload"]["tokens"] != v["tokens"]
            or v["key"] != key
            or (files is not None and files[key] != digest)
        ):
            raise ValueError("Crossed answer cannot be traced to the physical response")
        scores[h] = evaluator.compute_f1(v["answer"], evaluator.get_ground_truth(q), q)
        receipts[key] = digest
    return scores, receipts


def verified_cases(out, corpus):
    plan = read(out / "cohort.json")
    original = read(out / "registration.json")
    if file_hash(corpus) != original["corpus"]:
        raise ValueError("Unexpected evaluation labels")
    subjects = {v["dialogue"] for v in plan["contrasts"]}
    data = {c["sample_id"]: c["qa"] for c in read(corpus) if c["sample_id"] in subjects}
    ev = LoCoMoEvaluator(SimpleNamespace())
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    cases = []
    files = {}
    for c in plan["contrasts"]:
        os.environ["MEMSKILL_BACKEND_ID"] = c["backend"]
        for r in c["rows"]:
            if not r["credit"]:
                continue
            scores, digests = crossed_scores(
                out, plan["model"], r, data[c["dialogue"]][r["index"]], ev
            )
            if scores is None:
                return
            files.update(digests)
            lo, hi = [r[k]["f1"] for k in ("low", "high")]
            hs = [scores[h] for h in r["hybrids"]]
            d, bonus = components(lo, hi, hs, r["credit"])
            cases.append(
                {
                    "window": c["window"],
                    "bits": c["bits"],
                    "index": r["index"],
                    "low": lo,
                    "high": hi,
                    "hybrids": hs,
                    "credit": r["credit"],
                    "direct": d,
                    "other": hi - lo - d,
                    "direct_credit": bonus,
                }
            )
    register(
        out / "verified-cases.json",
        {"cohort": file_hash(out / "cohort.json"), "measured": files, "rows": cases},
    )


def freeze(out, parent, stage, corpus):
    dest = out / stage
    dest.mkdir(exist_ok=True)
    path = dest / "fit.json"
    reg = read(out / "registration.json")
    identity = {
        "cohort": file_hash(out / "cohort.json"),
        "parent": file_hash(parent),
        "plan": file_hash(PLAN),
        "hybrids": file_hash(out / "readers-complete.json")
        if stage == "complete"
        else None,
    }
    if (
        reg["fit"] != identity["parent"]
        or reg["plan"] != identity["plan"]
        or reg["corpus"] != file_hash(corpus)
    ):
        raise ValueError("Training inputs changed")
    if path.exists():
        if read(path)["inputs"] != identity:
            raise ValueError("Changed reader-slot fit")
        return path
    plan = read(out / "cohort.json")
    original = read(parent)["observations"]
    ev = LoCoMoEvaluator(SimpleNamespace())
    data = {
        c["sample_id"]: c["qa"]
        for c in read(corpus)
        if c["sample_id"] in {r["dialogue"] for r in original}
    }
    files = {}
    if stage == "complete":
        complete = read(out / "readers-complete.json")
        if complete["cohort"] != identity["cohort"]:
            raise ValueError("Mismatched hybrid reader cohort")
        files = complete["files"]
    targets = []
    audit = []
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    for i, (obs, c) in enumerate(zip(original, plan["contrasts"], strict=True)):
        if (
            c["observation"] != i
            or c["window"] != obs["window"]
            or c["stage"] != obs["stage"]
            or not math.isclose(
                mean(r["credit"] for r in c["rows"]), obs["credit"], abs_tol=1e-12
            )
        ):
            raise ValueError("Reader evidence does not align with the trained span")
        os.environ["MEMSKILL_BACKEND_ID"] = c["backend"]
        ds = []
        cs = []
        for r in c["rows"]:
            scores = {a["prompt_sha256"]: a["f1"] for a in (r["low"], r["high"])}
            if stage == "complete":
                q = data[c["dialogue"]][r["index"]]
                scores, _ = crossed_scores(out, plan["model"], r, q, ev, files)
            resolved = all(h in scores for h in r["hybrids"])
            d = bonus = None
            lo, hi = r["low"]["f1"], r["high"]["f1"]
            if resolved:
                d, bonus = components(
                    lo, hi, [scores[h] for h in r["hybrids"]], r["credit"]
                )
            # Unresolved cases transfer their full target via the ordinary state kernel.
            ds.append(d if resolved and not c["masked"] else 0.0)
            cs.append(bonus if resolved and not c["masked"] else 0.0)
            audit.append(
                {
                    "contrast": i,
                    "index": r["index"],
                    "group": r["group"],
                    "observed": hi - lo,
                    "direct": d,
                    "direct_credit": bonus,
                }
            )
        targets.append({"direct": mean(ds), "direct_credit": mean(cs)})
    register(path, {"inputs": identity, "targets": targets, "audit": audit})
    return path


def rotations(rows):
    indices = defaultdict(list)
    for i, r in enumerate(rows):
        indices[r["dialogue"], r["stage"], r["weight"]].append(i)
    rotated = [None] * len(rows)
    for ids in indices.values():
        for i, j in zip(ids, ids[1:] + ids[:1], strict=True):
            rotated[i] = rows[j]["direct"] + rows[j]["direct_credit"]
    return [dict(r, rotated=v) for r, v in zip(rows, rotated, strict=True)]


def direct_range(row):
    low, high = (row[k]["f1"] for k in ("low", "high"))
    known = {row[k]["prompt_sha256"]: row[k]["f1"] for k in ("low", "high")}
    a, b = row["hybrids"]
    if a == b:
        extremes = [[0.0, 0.0]] * 2
    else:
        extremes = [
            [known.get(a, 0.0), known.get(b, 1.0)],
            [known.get(a, 1.0), known.get(b, 0.0)],
        ]
    bounds = [
        sum(components(low, high, crossed, row["credit"])) for crossed in extremes
    ]
    if bounds[0] > bounds[1] or row["credit"] * (high - low) < 0:
        raise ValueError("Direct-label monotonicity is violated")
    return bounds


def target_ranges(plan):
    ranges = []
    for c in plan["contrasts"]:
        bounds = [direct_range(r) for r in c["rows"]]
        ranges.append(
            [0.0, 0.0] if c["masked"] else [mean(v) for v in zip(*bounds, strict=True)]
        )
    return ranges


def possible_probability(decision, model):
    states, weights, ys, _, bounds = model
    state = states.new_tensor(decision["state"])

    def coefficients(s, bank):
        k = (F.normalize(bank, dim=1) @ F.normalize(s, dim=0)).clamp(0, 1).pow(
            8
        ) * weights
        return k / (1 + k.sum())

    full = coefficients(state, states)
    n = state.shape[0] // 2
    change = coefficients(state[:n], states[:, :n]) - full
    ends = change[:, None] * bounds
    total = full @ ys[:, 0]
    effect = total + torch.stack(
        (ends.min(dim=1).values.sum(), ends.max(dim=1).values.sum())
    )
    return (decision["base_logit"] + (effect / 0.05).clamp(-2, 2)).sigmoid().tolist()


def model_for(rows, sid, stage, device):
    donor = [r for r in rows if r["dialogue"] != sid and r["stage"] == stage]
    (states, weights, ys), _ = support(donor, None, stage, device)
    if states.shape[1] != 2048:
        raise ValueError(
            "Expected contiguous 1024-dimensional source and memory features"
        )
    parts = states.new_tensor(
        [[r["direct"] + r["direct_credit"], r["direct"], r["rotated"]] for r in donor]
    )
    return states, weights, ys, parts, states.new_tensor([r["bounds"] for r in donor])


def effects(state, model):
    states, w, ys, parts, _ = model
    totals = kernel_effect(state[None], states, w, ys)[0]
    via_full = kernel_effect(state[None], states, w, parts)[0]
    source = state.shape[-1] // 2
    via_source = kernel_effect(state[None, :source], states[:, :source], w, parts)[0]
    shift = via_source - via_full
    return torch.stack(
        (totals[0] + shift[0], totals[1] + shift[1], totals[0], totals[0] + shift[2])
    )


def probabilities(decision, model, terminal):
    state = model[0].new_tensor(decision["state"])
    base = state.new_tensor(decision["base_logit"])
    t = state.new_tensor(terminal)
    shifts = torch.cat(
        (
            (effects(state, model) / 0.05).clamp(-2, 2),
            torch.stack((base * 0, t[0] + t[1:] @ F.normalize(state, dim=0))),
        )
    )
    return dict(zip(ARMS, (base + shifts).sigmoid().tolist(), strict=True))


def summary(windows):
    stats = summarize(windows, ARMS, "credit", ARMS[1:])
    if not windows:
        return stats
    matched = []
    for pivot in (0, 1):
        group = [w for w in windows if w["pivot"] == pivot]
        if not group:
            continue

        def vec(seq):
            return torch.tensor(seq, dtype=torch.float64, device="cuda")

        base = vec([w["base_logit"] for w in group])
        candidate = vec([w["focal_probabilities"]["credit"] for w in group])
        weights = vec([len(w["rows"]) for w in group])
        rates = (base + matched_offset(base, candidate, weights)).sigmoid()
        if abs(float(weights @ (rates - candidate) / weights.sum())) > 1e-10:
            raise ValueError("Rate control did not match the candidate")
        for w, p in zip(group, rates.tolist(), strict=True):
            matched.extend(
                (w["dialogue"], lo + p * (hi - lo)) for lo, hi in w["endpoints"]
            )
    bounds = [0.0, 0.0]
    for window in windows:
        ps = window["possible_completed_probability"]
        low = mean(v[0] for v in window["endpoints"])
        delta = mean(hi - lo for lo, hi in window["endpoints"])
        lo, hi = sorted(low + p * delta for p in ps)
        bounds[0] += lo * len(window["rows"])
        bounds[1] += hi * len(window["rows"])
    stats["possible_completed_f1"] = [v / stats["questions"] for v in bounds]
    stats["matched"] = mean(v for _, v in matched)
    stats["matched_by_history"] = {
        h: mean(v for s, v in matched if s == h) for h in stats["by_dialogue"]
    }
    stats["movement_from_unsplit"] = mean(
        abs(w["focal_probabilities"]["credit"] - w["focal_probabilities"]["unsplit"])
        for w in windows
    )
    stats["movement_from_raw"] = mean(
        abs(w["focal_probabilities"]["credit"] - w["focal_probabilities"]["raw"])
        for w in windows
    )
    return stats


def passes(all_rows, strict):
    if (
        all_rows["windows"] != 40
        or all_rows["questions"] != 378
        or not strict["questions"]
    ):
        return False
    f = all_rows["f1"]
    ht = all_rows["ht"]
    return (
        all(
            f["credit"] >= f[k] + 0.02 and ht["credit"] > ht[k]
            for k in ("native", "terminal")
        )
        and all(f["credit"] > f[k] for k in ("raw", "unsplit", "rotated"))
        and f["credit"] > all_rows["matched"]
        and sum(
            v["f1"]["credit"] > all_rows["matched_by_history"][h]
            for h, v in all_rows["by_dialogue"].items()
        )
        >= 4
        and all(strict["f1"]["credit"] > strict["f1"][k] for k in ("native", "raw"))
    )


def prefix_record(decision, model, pivot):
    return {
        "pivot": pivot,
        "base": decision["base_logit"],
        "possible_completed_probability": possible_probability(decision, model),
    }


def prefix_summary(panels, device):
    enriched = []
    for p in panels:
        enriched.append(
            dict(
                p,
                pivot=p["focal"]["pivot"],
                base_logit=p["focal"]["base"],
                possible_completed_probability=p["focal"][
                    "possible_completed_probability"
                ],
                endpoints=[(r["low"], r["low"] + r["delta"]) for r in p["observed"]],
            )
        )
    return summary(enriched)


def run(base, corpus, freeze_only, long_prefix=False):
    out = base / "source-compression/reader-slot-credit"
    parent = out.parent / "capped-source-credit/policy/joint-mask/fit.json"
    late = base / "late-pairs"
    splits = read(late / "selection-labels/split.json")["windows"]
    cohort = read(late / "cohort.json")["assignments"]
    terminal = read(late / "frozen.json")["terminal"]
    if not freeze_only:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Policy evaluation requires one leased GPU")
        torch.cuda.set_per_process_memory_fraction(0.06)
    verified_cases(out, corpus)
    for stage in ("aliases", "complete"):
        if stage == "complete" and not (out / "readers-complete.json").is_file():
            continue
        path = freeze(out, parent, stage, corpus)
        old = scored_at(late / "outcomes", path.stat().st_mtime_ns)
        register(
            path.parent / "preexisting.json", {"fit": file_hash(path), "scores": old}
        )
        if freeze_only:
            continue
        supplemental = PLAN.with_name("reader_slot_bounds.json")
        limits = target_ranges(read(out / "cohort.json"))
        register(
            out / "direct-bounds.json",
            {
                "plan": file_hash(supplemental),
                "cohort": file_hash(out / "cohort.json"),
                "training_ranges": limits,
            },
        )
        rows = rotations(
            [
                dict(r, **t, bounds=lb)
                for r, t, lb in zip(
                    read(parent)["observations"],
                    read(path)["targets"],
                    limits,
                    strict=True,
                )
            ]
        )
        models = {
            (sid, j): model_for(rows, sid, j, "cuda")
            for sid in sorted({r["dialogue"] for r in rows})
            for j in (0, 1)
        }
        if long_prefix:
            audit_long_prefix(
                late,
                path.parent,
                models,
                terminal,
                splits,
                "cuda",
                policy=probabilities,
                record=prefix_record,
                summarize_panel_fn=prefix_summary,
                protocol=PLAN.with_name("reader_slot_long_prefix.json"),
            )
            continue
        scored = {k.split("/")[0] for k in old}
        for label, key in (("calibration", "donor"), ("later", "evaluation")):
            panels = []
            for w in cohort:
                name = f"{w['dialogue']}_s{w['start']}"
                folder = late / "outcomes" / name
                if name not in splits[key] or not (folder / "pair.json").is_file():
                    continue
                if splits[key][name] != w["questions"]:
                    raise ValueError("Different held-history test panel")
                v = score(late, w, models, terminal, policy=probabilities, arms=ARMS)
                pair = read(folder / "pair.json")
                bits = "".join(map(str, pair["paths"][0]))
                v.update(
                    pivot=w["pivot"],
                    base_logit=read(folder / bits / "decisions.json")[w["pivot"]][
                        "base_logit"
                    ],
                    endpoints=[
                        [r["temporal"], r["activity"]] for r in pair["questions"]
                    ],
                )
                decision = read(folder / bits / "decisions.json")[w["pivot"]]
                ps = possible_probability(decision, models[w["dialogue"], w["pivot"]])
                if (
                    stage == "complete"
                    and not ps[0] - 1e-10
                    <= v["focal_probabilities"]["credit"]
                    <= ps[1] + 1e-10
                ):
                    raise ValueError("Completed policy contradicts its physical bounds")
                v["possible_completed_probability"] = ps
                panels.append(v)
            if not panels:
                continue
            all_stats = summary(panels)
            strict = summary([v for v in panels if v["window"] not in scored])
            p = path.parent / label
            p.mkdir(exist_ok=True)
            report = {
                "fit": file_hash(path),
                "bounds": file_hash(out / "direct-bounds.json"),
                "all": all_stats,
                "strict": strict,
                "windows": panels,
                "eligible_for_development": stage == "complete"
                and label == "later"
                and passes(all_stats, strict),
            }
            register(p / f"through-{len(panels):02d}.json", report)
            if len(panels) == len(splits[key]):
                register(p / "summary.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--long-prefix", action="store_true")
    a = p.parse_args()
    run(a.base, a.corpus, a.freeze, a.long_prefix)
