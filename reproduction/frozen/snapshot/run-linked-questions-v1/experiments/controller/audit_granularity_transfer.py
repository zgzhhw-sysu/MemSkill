"""Compare last-span grouping effects on the same training questions across two memory banks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller.granularity_placement import load_panel
from experiments.controller.writer_granularity import source_variants
from src.eval import get_evaluator
from src.evaluation_cache import digest, read_response


def direction(gain):
    return 1 if gain > 1e-9 else -1 if gain < -1e-9 else 0


def summarize(rows):
    directions = Counter()
    for r in rows:
        source, writer = direction(r["source_delta"]), direction(r["writer_delta"])
        if source and writer:
            outcome = "agree" if source == writer else "opposite"
        else:
            outcome = "source_only" if source else "writer_only" if writer else "both_tied"
        directions[outcome] += 1
    return {"questions": len(rows), "direction_counts": dict(directions),
            "source_delta": mean(r["source_delta"] for r in rows),
            "writer_delta": mean(r["writer_delta"] for r in rows)}


def placement_ceiling(groups):
    """One grouping per final span, with QA-weighted return over the selected training windows."""
    choices = {k: float(direction(mean(r["source_delta"] for r in rows)) > 0) for k, rows in groups.items()}
    rates = defaultdict(list)
    for k, rows in groups.items():
        rates[rows[0]["dialogue"]].append(choices[k])
    matched = {sid: mean(values) for sid, values in rates.items()}
    totals = Counter()
    for k, rows in groups.items():
        oracle = float(mean(r["writer_delta"] for r in rows) > 0)
        for r in rows:
            for arm, p in (("fixed_pair", 1), ("fixed_single", 0), ("best_per_window", oracle),
                           ("source_preferred", choices[k]), ("source_count_within_dialogue", matched[r["dialogue"]])):
                totals[arm] += r["writer_f1"]["single"] + p * r["writer_delta"]
    n = sum(len(rows) for rows in groups.values())
    return {"f1": {arm: val / n for arm, val in totals.items()}, "source_preferred_pair_windows": int(sum(choices.values())),
            "windows_with_exchangeable_choices": sum(len(values) for values in rates.values() if len(set(values)) > 1)}


def audit(study, data, source_dir, writer_dir):
    panel = load_panel(study, data, source_dir, include_questions=True)
    manifest = json.loads((writer_dir / "manifest.json").read_bytes())
    if set(manifest["windows"]) != {f'{w["dialogue"]}_s{w["start"]}' for w in panel}:
        raise ValueError("The two interventions cover different training windows")
    summary = json.loads((writer_dir / "summary.json").read_bytes())
    jobs_file = writer_dir / "jobs.json"
    jobs = None
    if jobs_file.exists():
        plan = json.loads(jobs_file.read_bytes())
        jobs = {(j["episode"], j["original_index"]): j for j in plan["jobs"]}
        if digest(plan) != manifest["plan"] or len(jobs) != len(plan["jobs"]) or plan["windows"] != manifest["windows"]:
            raise ValueError("Saved retrieval plan differs from the scored writer-bank inputs")
    cache = writer_dir / "responses"
    receipts = set()
    for path in cache.glob("*/*.json"):
        payload = read_response(cache, path.stem)["payload"]
        receipts.add((payload["prompt_sha256"], payload["response"], tuple(payload["tokens"])))
    ev = get_evaluator("locomo", SimpleNamespace())
    groups = defaultdict(list)
    expected_files = set()
    for w in panel:
        episode = f'{w["dialogue"]}_s{w["start"]}'
        trace, = (study / "train" / episode / "traces").glob("*.jsonl")
        raw = trace.read_bytes()
        entry = manifest["windows"][episode]
        if hashlib.sha256(raw).hexdigest() != entry["trace"]:
            raise ValueError(f"Different writer trace for {episode}")
        spans = [s for s in map(json.loads, raw.splitlines()) if s["kind"] == "span"]
        eligible = source_variants(spans) is not None
        if eligible != entry["source_inserted"]:
            raise ValueError(f"Source eligibility differs at {episode}")
        if not eligible:
            continue
        for i, item in w["questions"].items():
            path = writer_dir / f"{episode}-q{i}.json"
            expected_files.add(path)
            result = json.loads(path.read_bytes())
            if result["episode"] != episode or result["original_index"] != i or set(result["scores"]) != {"single", "pair"}:
                raise ValueError("Mismatched training question or grouping")
            q = item["question"]
            job = None if jobs is None else jobs[episode, i]
            if job is not None and (job["question"]["question"] != q["question"] or
                                    result["pair_retrieval_reproduced"] != job["pair_retrieval_reproduced"]):
                raise ValueError(f"Writer-bank question or retrieval differs from the saved plan at {path}")
            scores = {}
            for arm, r in result["scores"].items():
                if job is not None and hashlib.sha256(job["prompts"][arm].encode()).hexdigest() != r["prompt_sha256"]:
                    raise ValueError(f"Writer-bank prompt differs from the saved plan at {path}")
                if (r["prompt_sha256"], r["answer"], tuple(r["tokens"])) not in receipts:
                    raise ValueError(f"Missing measured writer-bank answer in {path}")
                scores[arm] = float(ev.compute_f1(r["answer"], ev.get_ground_truth(q), q))
                if not math.isclose(scores[arm], r["f1"], abs_tol=1e-12):
                    raise ValueError(f"Writer-bank F1 disagrees with the reference in {path}")
            cells = item["f1"]
            first_single = cells["01"] - cells["00"]
            first_pair = cells["11"] - cells["10"]
            groups[episode].append({"original_index": i, "dialogue": w["dialogue"],
                                    "source_delta": (first_single + first_pair) / 2,
                                    "source_given_first_single": first_single, "source_given_first_pair": first_pair,
                                    "writer_delta": scores["pair"] - scores["single"], "writer_f1": scores,
                                    "original_retrieval_matches": result["pair_retrieval_reproduced"]})
    rows = [r for group in groups.values() for r in group]
    matched = [r for r in rows if r["original_retrieval_matches"]]
    if (not rows or set(writer_dir.glob("*-q*.json")) != expected_files or len(rows) != manifest["questions"]
            or len(rows) != summary["questions"] or len(matched) != summary["original_retrieval_matches"]
            or any(not math.isclose(mean(r["writer_f1"][a] for r in rows), summary["mean_f1"][a], abs_tol=1e-12)
                   for a in ("pair", "single"))):
        raise ValueError("Joined training observations differ from the completed writer-bank panel")
    return {"training_dialogues": len({r["dialogue"] for r in rows}), "eligible_windows": len(groups),
            "writer_backend": manifest["backend"], "all": summarize(rows),
            "placement": placement_ceiling(groups),
            "original_retrieval_matches": summarize(matched) if matched else None,
            "by_window": {k: summarize(v) for k, v in groups.items()}, "rows": dict(groups)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "locomo", "source_dir", "writer_dir"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.study, args.locomo, args.source_dir, args.writer_dir), indent=2))
