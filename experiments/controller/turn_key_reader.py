"""Measure real reader returns for one terminal activity write with two-turn retrieval keys."""

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.controller.factorial_repairs import checked_cell, replay_writes
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.semantic_writer import measured_readers, sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.turn_key_credit import (
    PLAN,
    keyed_scores,
    paired_interval,
    screen_summary,
)
from experiments.controller.writer_granularity import measured_answer
from src.credit_assignment import _digest
from src.eval import get_evaluator
from src.evaluation_cache import history_embeddings
from src.rag_utils import get_embeddings
from src.replay import dated_group_parts, restore_text


def live_keys(spans):
    ev = spans[-1]
    groups = defaultdict(set)
    for parent, parts in dated_group_parts(ev["source"], 2):
        groups[parent].add(tuple(text for _, text in parts))
    live = restore_text(spans)
    index = {mid: i for i, mid in enumerate(live)}
    acts = [
        r for r in ev["parsed_actions"] if r["action_type"] == "INSERT" and r["success"]
    ]
    changes = [c for c in ev["changes"] if c["action"] == "INSERT"]
    out = {}
    for action, c in zip(acts, changes, strict=True):
        text = action["memory_content"]
        if text != c["after"] or live.get(c["memory_id"]) != text:
            raise ValueError("Live item does not match the recorded focal insertion")
        if action["reasoning"] == "dated_adjacent_turns":
            choices = groups[text]
            if not choices:
                raise ValueError("Writer indexed a missing source group")
            if len(choices) == 1:
                (parts,) = choices
                if len(parts) == 2:
                    out[index[c["memory_id"]]] = parts
    return out


def reorder(scores, factual_ranking, k):
    # Preserve the archived native order for exact score ties.
    order = torch.as_tensor(factual_ranking, device=scores.device)
    return order[scores[order].argsort(descending=True, stable=True)[:k]].tolist()


def build(trainer, convo, folder, archive, specs, forecast, mass):
    arms = {}
    for bit, name in enumerate(("temporal", "activity")):
        path = folder / f"0{bit}"
        cell = checked_cell(
            trainer, convo, path, (0, bit), specs, archive / "responses"
        )
        arms[name] = (
            cell,
            replay_writes(trainer, cell["spans"]),
            measured_readers(path, archive / "responses"),
        )
    temporal, activity = [arms[k][0]["spans"] for k in ("temporal", "activity")]
    if (
        temporal[:-1] != activity[:-1]
        or any(
            temporal[-1][k] != activity[-1][k] for k in ("source", "refs", "retrieved")
        )
        or temporal[-1]["selected_skills"][:-1] != activity[-1]["selected_skills"][:-1]
        or forecast["source_sha256"] != sha(activity[-1]["source"])
    ):
        raise ValueError("Source, first two slots or fixed temporal prefix changed")
    keyed = live_keys(activity)
    bank = arms["activity"][1]
    contents = bank.get_all_contents()
    questions = convo["qa"]
    vectors = get_embeddings(
        trainer.args.retriever, [q["question"] for q in questions], "query"
    )
    query = F.normalize(torch.as_tensor(vectors, device="cuda").float(), dim=1)
    old = (
        query
        @ torch.as_tensor(bank._get_embeddings_matrix(False), device="cuda").float().T
    )
    if keyed:
        texts = [t for parts in keyed.values() for t in parts]
        raw = get_embeddings(trainer.args.retriever, texts, "context", batch_size=2)
        children = F.normalize(torch.as_tensor(raw, device="cuda").float(), dim=1)
        key_scores = (query @ children.T).reshape(len(query), len(keyed), 2)
    else:
        key_scores = old.new_empty((len(query), 0, 2))
    new, rot = keyed_scores(old, key_scores, list(keyed))

    def conditional(arm):
        return mass["01"][arm] / (mass["00"][arm] + mass["01"][arm])

    native = conditional("native")
    p = 1 / (1 + math.exp(-(math.log(native / (1 - native)) + forecast["offset"])))
    jobs = []
    for i, (q, vec) in enumerate(zip(questions, vectors, strict=True)):
        jobs.append(
            {
                "original_index": q["original_index"],
                "category": q["category"],
                "qa": q,
                "known": {},
                "prompts": {},
                "reads": {},
            }
        )
        job = jobs[-1]
        for name, (cell, measured, receipts) in arms.items():
            ctx, indices = measured.retrieve(vec, top_k=trainer.config.mem_top_k_eval)
            prompt = trainer.evaluator.build_prompt(q["question"], ctx, q)
            digest = sha(prompt)
            saved = receipts.get(digest)
            observed = cell["scores"][i]
            if (
                list(map(int, indices)) != observed["indices"]
                or not saved
                or saved["response"].strip() != observed["answer"]
                or saved["tokens"] != observed["tokens"]
            ):
                job["exclusion"] = f"unreproduced_{name}_reader"
                break
            job["known"][digest] = {
                "answer": observed["answer"],
                "f1": observed["f1"],
                "tokens": observed["tokens"],
            }
            job["reads"][name] = {
                "indices": observed["indices"],
                "prompt": digest,
                "words": sum(len(t.split()) for t in ctx),
            }
        if "exclusion" in job:
            continue
        _, orig = bank.retrieve(vec, top_k=len(contents))
        k = len(job["reads"]["activity"]["indices"])
        if reorder(old[i], orig, k) != job["reads"]["activity"]["indices"]:
            job["exclusion"] = "unreproduced_full_vector_order"
            continue
        for name, scores in (("keyed", new), ("rotated", rot)):
            indices = reorder(scores[i], orig, k)
            ctx = [contents[j] for j in indices]
            prompt = trainer.evaluator.build_prompt(q["question"], ctx, q)
            digest = sha(prompt)
            job["reads"][name] = {
                "indices": indices,
                "prompt": digest,
                "words": sum(len(t.split()) for t in ctx),
            }
            if digest not in job["known"]:
                job["prompts"][digest] = prompt
    return {
        "source": forecast["source_sha256"],
        "trace": {k: v[0]["trace"] for k, v in arms.items()},
        "native": native,
        "terminal": conditional("terminal"),
        "offset": forecast["offset"],
        "p_keyed_activity": p,
        "extra_keys": 2 * len(keyed),
        "activity_bank_hash": _digest(contents),
        "models": trainer.args.model,
        "jobs": jobs,
    }


def prepare(trainer, study, forecast_path, train, specs):
    archive = forecast_path.parent / "factorial"
    root = forecast_path.parent / "turn-keys"
    forecast = read(root / "forecast.json")
    screen = read(root / "manifest.json")
    panel = read(archive / "manifest.json")
    screen_summary(root)
    if (
        screen["plan"] != file_hash(PLAN)
        or screen["protocol"] != file_hash(study / "protocol.json")
        or panel["protocol"] != screen["protocol"]
        or panel["forecast"] != file_hash(forecast_path)
        or forecast["manifest"] != file_hash(root / "manifest.json")
        or screen["backend"] != os.environ["MEMSKILL_BACKEND_ID"]
    ):
        raise ValueError("Different frozen panel, retriever or writer")
    data = {c["sample_id"]: c for c in train}
    offsets = {(r["dialogue"], r["session"]): r for r in forecast["forecasts"]}
    dest = root / "reader"
    dest.mkdir(exist_ok=True)
    for sid, start in panel["windows"]:
        name = f"{sid}_s{start}"
        before = archive / name / "measurement.json"
        if not before.is_file():
            continue
        if sid in forecast["fits"][sid]["donors"]:
            raise ValueError("Target dialogue entered its own fit")
        path = dest / name
        path.mkdir(exist_ok=True)
        inputs = {
            "measurement": file_hash(before),
            "forecast": file_hash(root / "forecast.json"),
            "backend": screen["backend"],
        }
        receipt = path / "jobs.json"
        if receipt.exists():
            if read(receipt)["inputs"] != inputs:
                raise ValueError("Frozen reader inputs changed")
            continue
        with history_embeddings():
            table = build(
                trainer,
                window(data[sid], start, 2),
                before.parent,
                archive,
                specs,
                offsets[sid, start + 1],
                read(before)["path_mass"],
            )
        register(receipt, dict(table, inputs=inputs, dialogue=sid, start=start))
        print(
            name,
            len(table["jobs"]),
            sum(len(j["prompts"]) for j in table["jobs"]),
            "new prompt entries",
            flush=True,
        )


def score_window(root, folder):
    path = folder / "jobs.json"
    jobs = read(path)
    identity = {"jobs": file_hash(path), "backend": os.environ["MEMSKILL_BACKEND_ID"]}
    if identity["backend"] != jobs["inputs"]["backend"]:
        raise ValueError("Reader backend changed")
    result = folder / "pair.json"
    if result.exists():
        v = read(result)
        if v["inputs"] != identity:
            raise ValueError("Different reader inputs")
        return v
    evaluator = get_evaluator("locomo", SimpleNamespace())
    rows = []
    os.environ["MEMSKILL_USAGE_LOG"] = str(folder / "usage.jsonl")
    for job in jobs["jobs"]:
        q = job["qa"]
        row = {"original_index": job["original_index"], "category": job["category"]}
        rows.append(row)
        if "exclusion" in job:
            row["exclusion"] = job["exclusion"]
            continue
        responses = dict(job["known"])
        for digest, prompt in job["prompts"].items():
            if sha(prompt) != digest:
                raise ValueError("Reader prompt changed")
            responses[digest] = measured_answer(
                prompt, jobs["models"], q, evaluator, root / "reader-responses"
            )
        scores = {}
        for arm, r in job["reads"].items():
            v = responses[r["prompt"]]
            actual = evaluator.compute_f1(
                v["answer"].strip(), evaluator.get_ground_truth(q), q
            )
            if min(v["tokens"]) <= 0 or not math.isclose(
                actual, v["f1"], abs_tol=1e-12
            ):
                raise ValueError("Reader answer or F1 failed verification")
            scores[arm] = dict(v, **r)
        row["arms"] = scores
    table = {
        k: jobs[k]
        for k in (
            "dialogue",
            "start",
            "native",
            "terminal",
            "p_keyed_activity",
            "extra_keys",
        )
    }
    register(result, dict(table, inputs=identity, rows=rows))
    return read(result)


def summarize(group):
    total = sum(len(g["rows"]) for g in group)
    active = [g for g in group if any("arms" in r for r in g["rows"])]
    if not active:
        return {
            "windows": len(group),
            "intended_questions": total,
            "matched_questions": 0,
        }
    rate = mean(g["p_keyed_activity"] for g in active)
    values = defaultdict(list)
    details = defaultdict(lambda: defaultdict(list))
    costs = defaultdict(list)

    def collect(by, scores):
        for k, v in scores.items():
            by[k].append(v)

    for g in active:
        for r in g["rows"]:
            if "arms" not in r:
                continue
            scores = {k: v["f1"] for k, v in r["arms"].items()}
            for arm, measurement in r["arms"].items():
                costs[arm].append(measurement["words"])
            low = scores["temporal"]
            for name, p, arm in (
                ("native", g["native"], "activity"),
                ("terminal", g["terminal"], "activity"),
                ("old_index_policy", g["p_keyed_activity"], "activity"),
                ("new_native", g["native"], "keyed"),
                ("policy", g["p_keyed_activity"], "keyed"),
                ("matched_rate", rate, "keyed"),
                ("sham_policy", g["p_keyed_activity"], "rotated"),
            ):
                scores[name] = low + p * (scores[arm] - low)
            collect(values, scores)
            collect(details[g["dialogue"]], scores)
            collect(details["type-" + str(r["category"])], scores)

    def average(v):
        return {"n": len(v["policy"]), "f1": {k: mean(x) for k, x in v.items()}}

    result = {
        "windows": len(group),
        "intended_questions": total,
        "matched_questions": len(values["policy"]),
        "p_mean": rate,
        "mean_abs_p_change": mean(
            abs(g["native"] - g["p_keyed_activity"]) for g in active
        ),
        **average(values),
        "mean_read_words": {k: mean(v) for k, v in costs.items()},
        "per_group": {k: average(v) for k, v in details.items()},
    }

    by_dialogue = [average(v) for k, v in details.items() if not k.startswith("type-")]
    if len(by_dialogue) >= 2:
        result["paired_dialogue_95pct"] = {
            "policy-" + k: paired_interval(
                [g["n"] for g in by_dialogue],
                [g["f1"]["policy"] - g["f1"][k] for g in by_dialogue],
            )
            for k in (
                "native",
                "new_native",
                "terminal",
                "matched_rate",
                "keyed",
                "sham_policy",
            )
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--limit", type=int, default=12)
    args = p.parse_args()
    src = args.root.parent / "factorial/manifest.json"
    group = []
    for sid, start in read(src)["windows"][: args.limit]:
        folder = args.root / "reader" / f"{sid}_s{start}"
        if not (folder / "jobs.json").exists():
            break
        group.append(score_window(args.root, folder))
    if group:
        stats = summarize(group)
        register(args.root / f"reader-through-{len(group)}.json", stats)
        if len(group) == len(read(src)["windows"]):
            register(args.root / "reader-summary.json", stats)
        print(stats.get("f1", {}), flush=True)
