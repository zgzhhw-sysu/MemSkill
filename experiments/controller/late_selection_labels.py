"""Collect counterfactual cause labels for the registered later randomized interventions."""

import argparse
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.factorial_repairs import checked_cell, final_probe
from experiments.controller.factorial_rollout import read
from experiments.controller.late_pair_policy import assignments
from experiments.controller.late_pair_trial import paths
from experiments.controller.paired_grounding import window
from experiments.controller.root_write_repairs import check_root, seed_controls
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import recovered
from src.evaluation_cache import history_embeddings
from src.rag_utils import get_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "late_selection_labels.json"


def registration(late):
    out = late / "selection-labels"
    out.mkdir(exist_ok=True)
    cohort = read(late / "cohort.json")
    identity = {
        "plan": file_hash(PLAN),
        "cohort": file_hash(late / "cohort.json"),
        "outcomes": file_hash(late / "outcomes/manifest.json"),
    }
    path = out / "registration.json"
    if not path.exists():
        archive = late / "outcomes"
        existing = {
            str(p.relative_to(archive)): file_hash(p)
            for p in archive.glob("conv-*/*/scores.json")
            if p.parent.name in ("00", "01", "10", "11")
        }
        register(path, dict(identity, preexisting=existing))
    saved = read(path)
    if any(saved[k] != v for k, v in identity.items()) or any(
        file_hash(late / "outcomes" / k) != v for k, v in saved["preexisting"].items()
    ):
        raise ValueError("Registration or previously published paths changed")
    splits = {}
    for label, donor in (("donor", True), ("evaluation", False)):
        rows = [w for w in cohort["assignments"] if (w["start"] <= 11) == donor]
        splits[label] = {f"{w['dialogue']}_s{w['start']}": w["questions"] for w in rows}
    if [sum(map(len, g.values())) for g in splits.values()] != [88, 378] or [
        len(g) for g in splits.values()
    ] != [12, 40]:
        raise ValueError("Unexpected chronological separation of the randomized cohort")
    register(out / "split.json", {"registration": file_hash(path), "windows": splits})
    return out


def labels(trainer, convo, spec, cells, raw, cache, episode, *, receipts_only=False):
    pivot, bg = spec["pivot"], spec["background"]
    chosen = paths(pivot, bg)
    low, high = [cells[k] for k in chosen]
    a, b = [c["spans"][pivot] for c in (low, high)]
    same = (
        a["selected_skills"][:-1] == b["selected_skills"][:-1]
        and all(a[k] == b[k] for k in ("retrieved", "refs", "source", "source_turns"))
        and (
            pivot == 0
            or restore_text(low["spans"][:1]) == restore_text(high["spans"][:1])
        )
    )
    rows = []
    for i, (q, record) in enumerate(zip(convo["qa"], raw["questions"], strict=True)):
        before, after = [c["scores"][i] for c in (low, high)]
        if (
            q["original_index"] != record["original_index"]
            or int(q["category"]) != record["type"]
            or (before["f1"], after["f1"]) != (record["temporal"], record["activity"])
        ):
            raise ValueError("Factual archive differs from paired question or F1")
        truth = trainer.evaluator.get_ground_truth(q)
        sign = int(recovered(before, after, truth)) - int(
            recovered(after, before, truth)
        )
        linked = bool(set(a["source_turns"]) & set(q["evidence"]))
        reason = (
            "different_fixed_context"
            if not same
            else (
                "unlinked"
                if not linked
                else ("probe_pending" if sign else "no_strict_recovery")
            )
        )
        rows.append(
            {
                "qa_index": i,
                "original_index": q["original_index"],
                "category": q["category"],
                "background": bg,
                "recovery_sign": sign,
                "linked": linked,
                "delta": after["f1"] - before["f1"],
                "status": reason,
                "credit": 0.0,
            }
        )
    candidates = [r for r in rows if r["status"] == "probe_pending"]
    if candidates and receipts_only:
        return None
    if candidates:
        for bits in chosen:
            seed_controls(
                episode / "".join(map(str, bits)), episode.parent / "responses", cache
            )
        queries = get_embeddings(
            trainer.args.retriever, [q["question"] for q in convo["qa"]], "query"
        )
        banks = {}
        for r in candidates:
            i = r["qa_index"]
            probe = (
                check_root(trainer, convo, r, cells, queries[i])
                if pivot == 0
                else final_probe(trainer, convo, r, cells, banks, queries[i])
            )
            if probe["signed_effect"] and (
                probe["status"] != "survived_probes"
                or probe["signed_effect"] != r["delta"]
            ):
                raise ValueError("Invalid local credit")
            r.update(probe=probe, status=probe["status"], credit=probe["signed_effect"])
    return rows


def run(trainer, study, locked, train, specs, limit, *, receipts_only=False):
    late = locked.parent
    out = registration(late)
    archive = late / "outcomes"
    manifest = read(archive / "manifest.json")
    cohort = read(late / "cohort.json")
    data = {c["sample_id"]: c for c in train}
    if (
        (not receipts_only and manifest["backend"] != os.environ["MEMSKILL_BACKEND_ID"])
        or manifest["forecast"] != file_hash(locked)
        or manifest["protocol"] != file_hash(study / "protocol.json")
        or cohort["corpus"] != file_hash(Path(trainer.args.data_file))
        or cohort["assignments"] != assignments(data)
        or manifest["skill_versions"] != [s["version"] for s in specs]
    ):
        raise ValueError(
            "Expected the original randomized trajectories, encoder and reader"
        )
    cache = out / "responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "usage.jsonl")
    reports = []
    for w in cohort["assignments"][:limit]:
        name = f"{w['dialogue']}_s{w['start']}"
        folder = archive / name
        orig = folder / "pair.json"
        if not orig.is_file():
            break
        report = out / (name + ".json")
        pair = read(orig)
        chosen = paths(w["pivot"], w["background"])
        inputs = {
            "pair": file_hash(orig),
            "registration": file_hash(out / "registration.json"),
            "scores": {
                "".join(map(str, k)): file_hash(
                    folder / "".join(map(str, k)) / "scores.json"
                )
                for k in chosen
            },
        }
        if not report.exists():
            convo = window(data[w["dialogue"]], w["start"], 2)
            if (
                pair["paths"] != [list(k) for k in chosen]
                or [q["original_index"] for q in convo["qa"]] != w["questions"]
                or any(
                    pair[k] != w[k]
                    for k in ("dialogue", "start", "pivot", "background")
                )
            ):
                raise ValueError("Wrong randomized assignment")
            with history_embeddings():
                cells = {
                    k: checked_cell(
                        trainer,
                        convo,
                        folder / "".join(map(str, k)),
                        k,
                        specs,
                        archive / "responses",
                    )
                    for k in chosen
                }
                rows = labels(
                    trainer,
                    convo,
                    w,
                    cells,
                    pair,
                    cache,
                    folder,
                    receipts_only=receipts_only,
                )
            if rows is None:
                break
            register(
                report,
                {
                    "inputs": inputs,
                    "window": name,
                    "dialogue": w["dialogue"],
                    "pivot": w["pivot"],
                    "background": w["background"],
                    "rows": rows,
                },
            )
        doc = read(report)
        if doc["inputs"] != inputs:
            raise ValueError("Measured donor or trajectory changed")
        reports.append(doc)
    if reports:
        rows = [r for w in reports for r in w["rows"]]
        stats = {
            "windows": len(reports),
            "questions": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "positive": sum(r["credit"] > 0 for r in rows),
            "negative": sum(r["credit"] < 0 for r in rows),
            "signed_dialogues": sorted(
                {w["dialogue"] for w in reports if any(r["credit"] for r in w["rows"])}
            ),
            "files": {
                w["window"]: file_hash(out / (w["window"] + ".json")) for w in reports
            },
        }
        register(out / f"through-{len(reports):02d}.json", stats)
        if len(reports) == 52:
            register(out / "summary.json", stats)


if __name__ == "__main__":
    # Verifying saved answers with zero qualifying swaps requires only the data processor and the scorer.
    from src.data_processing.datasets import load_dataset, split_data
    from src.data_processing.locomo import LoCoMoProcessor
    from src.eval.locomo import LoCoMoEvaluator

    parser = argparse.ArgumentParser(
        description="Verify completed, zero-credit windows until a model probe is required"
    )
    parser.add_argument("late", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("corpus", type=Path)
    inputs = parser.parse_args()
    options = SimpleNamespace(data_file=inputs.corpus)
    reader = SimpleNamespace(
        args=options,
        evaluator=LoCoMoEvaluator(options),
        data_processor=LoCoMoProcessor("full-session"),
    )
    train, _, _ = split_data(load_dataset(str(inputs.corpus), "locomo"), "locomo")
    run(
        reader,
        inputs.study,
        inputs.late / "frozen.json",
        train,
        read(inputs.study / "protocol.json")["skills"],
        None,
        receipts_only=True,
    )
