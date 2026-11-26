"""Score every question for both measured final actions where forward-fitted training policies disagree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller.audit_trajectory_return import audit
from experiments.controller.paired_grounding import window
from experiments.controller.probe_omitted import atomic_json
from experiments.controller.writer_granularity import measured_answer
from src.credit_assignment import _digest
from src.eval import get_evaluator
from src.executor import ExecutionResult, Executor
from src.memory_bank import MemoryBank
from src.rag_utils import get_embeddings, set_retriever_flash_attn
from src.replay import restore_text


def final_switches(episode):
    uses = episode["uses"]
    probabilities = episode["probabilities"]
    contrasts = {}
    for target, control in (("credit", "credit_rate"), ("anchor", "anchor_rate"), ("novelty", "anchor")):
        a, b = probabilities[target], probabilities[control]
        if (a[-1] >= .5) != (b[-1] >= .5):
            contrasts[target] = {"control": control, "probabilities": [a[-1], b[-1]],
                "both_follow_recorded_prefix": all((x >= .5) == u and (y >= .5) == u
                                                  for x, y, u in zip(a[:-1], b[:-1], uses[:-1], strict=True))}
    return contrasts


def summarize_window(result):
    rows = result["questions"]
    return {"questions": len(rows), "original_reads_and_answers_reproduced": sum(
                row["archived_factual_match"] for row in rows),
            "f1": result["f1"], "comparisons": {
                name: dict(pair, **result["effects"][name]) for name, pair in result["inputs"]["comparisons"].items()}}


def measure(study, locomo, out, *, progress=False):
    panel = audit(study, locomo, progress=progress)
    protocol = json.loads((study / "protocol.json").read_text())
    data = {c["sample_id"]: c for c in json.loads(locomo.read_text()) if c["sample_id"] in protocol["train"]}
    ev = get_evaluator("locomo", SimpleNamespace())
    retriever = protocol["retriever"]
    executor = Executor(SimpleNamespace(retriever=retriever))
    set_retriever_flash_attn(False)
    out.mkdir(exist_ok=True, parents=True)
    eligible, missing = [], []
    for e in panel["episodes"]:
        pairs = final_switches(e)
        if not pairs:
            continue
        sid, start = e["dialogue"], e["start"]
        directory = study / "train" / f"{sid}_s{start}"
        trace, = (directory / "traces").glob("*.jsonl")
        raw = trace.read_bytes()
        inputs = {"corpus_sha256": protocol["input_sha256"], "trace_sha256": hashlib.sha256(raw).hexdigest(), "backend": os.environ["MEMSKILL_BACKEND_ID"],
                  "model": protocol["model"], "retriever": retriever, "retrieval_k": protocol["retrieval_k"][1],
                  "comparisons": pairs}
        dest = out / (directory.name + ".json")
        eligible.append(directory.name)
        if dest.exists():
            if json.loads(dest.read_text())["inputs"] != inputs:
                raise ValueError(f"Different frozen inputs at {dest}")
            continue
        events = list(map(json.loads, raw.splitlines()))
        spans = [v for v in events if v["kind"] == "span"]
        last = [v["diagnosis"] for v in events if v["kind"] == "span_credit"][-1]
        branches = [last.get(k, {}) for k in ("control", "swap")]
        if last.get("execution_fault") or any(b.get("contract_fault") or not b.get("bank_hash") for b in branches):
            missing.append(directory.name)
            continue
        if (last["span_id"] != len(spans)-1 or any(b["suffix"] or b["failed_span"] is not None for b in branches)
                or branches[0]["actions"] != spans[-1]["parsed_actions"]):
            raise ValueError("Recorded interventions do not start at the final factual state")
        prior = restore_text(spans[:-1])
        factual = restore_text(spans)
        if _digest(list(factual.values())) != branches[0]["bank_hash"]:
            raise ValueError("Final observed memory bank failed its hash")
        positions = {mid: i for i, mid in enumerate(prior)}
        indices = [positions[mid] for mid, _ in spans[-1]["refs"]]
        texts = list(prior.values())
        vecs = get_embeddings(retriever, texts, "context") if texts else []
        banks = {}
        for i, branch in enumerate(branches):
            bank = MemoryBank(retriever, top_k=protocol["retrieval_k"][1])
            for text, vec in zip(texts, vecs, strict=True):
                bank.add_memory(text, vec)
            actions = [ExecutionResult(**a) for a in branch["actions"]]
            if (not executor.apply_to_memory_bank(actions, bank, indices)
                    or _digest(bank.get_all_contents()) != branch["bank_hash"]):
                raise ValueError("Reconstructed branch differs from the measured writer output")
            banks[e["uses"][-1] if i == 0 else not e["uses"][-1]] = bank
        convo = window(data[sid], start, 2)
        qa = convo["qa"]
        original = {v["query_id"]: v for v in events if v["kind"] == "qa"}
        vectors = get_embeddings(retriever, [q["question"] for q in qa], "query")
        rows = []
        for i, (q, v) in enumerate(zip(qa, vectors, strict=True)):
            hist = original[f"{sid}_{i}"]
            archived_context = [factual[mid] for mid, _ in hist["refs"]]
            scores = {}
            for choice, bank in banks.items():
                ctx, ids = bank.retrieve(v)
                prompt = ev.build_prompt(q["question"], ctx, q)
                score = measured_answer(prompt, protocol["model"], q, ev, out / "responses")
                if min(score["tokens"]) <= 0:
                    raise ValueError(f"Reader did not complete {sid}, question {q['original_index']}")
                score.update(indices=ids, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                scores["grounded" if choice else "plain"] = score
                if choice == e["uses"][-1]:
                    matched = ctx == archived_context and score["answer"].strip() == hist["prediction"].strip()
            rows.append({"original_index": q["original_index"], "archived_factual_match": matched, "scores": scores})
        f1 = {arm: mean(r["scores"][arm]["f1"] for r in rows) for arm in ("grounded", "plain")}
        results = {}
        for target, pair in pairs.items():
            a, b = pair["probabilities"]
            effect = f1["grounded"] - f1["plain"]
            results[target] = {"greedy_f1_difference": (int(a >= .5) - int(b >= .5)) * effect,
                               "stochastic_f1_difference_at_final_state": (a - b) * effect}
        atomic_json(dest, {"inputs": inputs, "episode": directory.name, "grounded_factual": e["uses"][-1], "questions": rows, "f1": f1, "effects": results})
        print(directory.name, len(rows), f1, results, flush=True)
    complete = {name: summarize_window(json.loads((out / (name + ".json")).read_text()))
                for name in eligible if name not in missing}
    atomic_json(out / ("progress.json" if panel["pending_windows"] else "summary.json"),
                {"eligible": eligible, "measured": complete, "missing_paired_writes": missing,
                 "pending_windows": panel["pending_windows"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "locomo", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    measure(args.study, args.locomo, args.output, progress=args.progress)
