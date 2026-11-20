"""Compare paired and individual source turns in otherwise identical saved training memory banks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller.paired_grounding import window
from experiments.controller.probe_omitted import atomic_json
from experiments.controller.resume_status import load_episode
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest
from src.eval import get_evaluator
from src.llm_utils import get_llm_response_via_api
from src.memory_bank import MemoryBank
from src.rag_utils import get_embeddings, set_retriever_flash_attn
from src.replay import attributed_turn_pairs, restore_text


def source_variants(spans):
    last = spans[-1]
    pairs = [a["memory_content"] for a in last["parsed_actions"]
             if a["success"] and a["reasoning"] == "attributed_turn_pairs"]
    if not pairs:
        return None
    live = restore_text(spans)
    tail = last["changes"][-len(pairs):]
    if ([c["after"] for c in tail] != pairs or any(c["action"] != "INSERT" for c in tail)
            or list(live)[-len(pairs):] != [c["memory_id"] for c in tail]):
        raise ValueError("Source writes do not match the final inserted memory IDs")
    header, *lines = last["source"].splitlines()
    date = header.removeprefix("DATE: ") + " | "
    singles = [turn for line in lines for turn in attributed_turn_pairs(header + "\n" + line)]
    split = {}
    for i in range(0, len(singles), 2):
        turns = singles[i:i + 2]
        joined = date + " ".join(turn.removeprefix(date) for turn in turns)
        split.setdefault(joined, turns)
    if any(pair not in split for pair in pairs):
        raise ValueError("Splitting the source changed the stored speaker, date or utterance")
    fixed = list(live.values())[:-len(pairs)]
    return live, fixed, {"pair": pairs, "single": [s for pair in pairs for s in split[pair]]}


def missing_multiline_turns(processor, turns, date, recorded_source):
    source, = processor._extract_session_chunks(turns, date)
    if source != recorded_source:
        raise ValueError("Structured dialogue differs from the recorded writer input")
    missing = {}
    for turn in turns:
        text, = processor._extract_session_chunks([turn], date)
        if attributed_turn_pairs(text):
            continue
        normalized = dict(turn)
        for key in ("text", "blip_caption"):
            if key in turn:
                normalized[key] = " ".join(turn[key].splitlines())
        clean, = processor._extract_session_chunks([normalized], date)
        items = attributed_turn_pairs(clean)
        if clean != text and len(items) == 1:
            missing[turn["dia_id"]] = items[0]
    return missing


def build_jobs(study, data_path, *, repair_multiline=False, skill_budget=None):
    protocol = json.loads((study / "protocol.json").read_text())
    raw = data_path.read_bytes()
    train = set(protocol["train"])
    if hashlib.sha256(raw).hexdigest() != protocol["input_sha256"] or train & set(protocol["validation"] + protocol["test"]):
        raise ValueError("The saved training split and input corpus must match")
    data = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in train}
    ev = get_evaluator("locomo", SimpleNamespace())
    processor = LoCoMoProcessor(chunk_mode="full-session") if repair_multiline or skill_budget is not None else None
    if skill_budget is not None:
        from experiments.controller.skill_source_budget import SKILLS, retained_turns
        descriptions = {s["name"]: s["description"] for s in protocol["skills"] if s["name"] in SKILLS.values()}
        picks = {(r["dialogue"], r["session"]): r for r in skill_budget["rows"]}
        if (repair_multiline or skill_budget["checkpoint"] != protocol["checkpoint_sha256"]
                or skill_budget["turn_budget_per_skill"] != protocol["retrieval_k"][1]
                or skill_budget["skill_descriptions"] != descriptions or len(picks) != len(skill_budget["rows"])
                or {sid for sid, _ in picks} != train):
            raise ValueError("Skill turn selections differ from the training study")
    jobs, windows = [], {}
    set_retriever_flash_attn(False)
    for start, end in protocol["train_sessions"]:
        for sid in protocol["train"]:
            episode = study / "train" / f"{sid}_s{start}"
            convo = window(data[sid], start, end - start + 1)
            ids = [q["original_index"] for q in convo["qa"]]
            load_episode(episode, sid, end-start+1, ids, deployment_writer=protocol.get("deployment_writer", False))
            trace, = (episode / "traces").glob("*.jsonl")
            raw_trace = trace.read_bytes()
            events = list(map(json.loads, raw_trace.splitlines()))
            spans = [e for e in events if e["kind"] == "span"]
            variants = source_variants(spans)
            windows[episode.name] = {"trace": hashlib.sha256(raw_trace).hexdigest(), "source_inserted": variants is not None}
            if variants is None and skill_budget is None:
                continue
            if variants is None:
                live = restore_text(spans)
                fixed, tails = list(live.values()), {}
            else:
                live, fixed, tails = variants
            if skill_budget is not None:
                session = f"session_{end}"
                turns = convo["conversation"][session]
                date = convo["conversation"][session + "_date_time"]
                recorded, = processor._extract_session_chunks(turns, date)
                if recorded != spans[-1]["source"]:
                    raise ValueError("Frozen turn choices and the original writer saw different sessions")
                tails = retained_turns(turns, date, picks[sid, end]["selected"], processor, protocol["retrieval_k"][1])
                # Include the factual bank to verify retrieval, without generating an extra reader answer.
                tails["original"] = list(live.values())[len(fixed):]
            snapshot, = [e for e in events if e["kind"] == "snapshot"]
            versions = {c["memory_id"]: c["version"] for e in events if e["kind"] == "span" for c in e["changes"]}
            bank_hash = hashlib.sha256(json.dumps(list(live.items()), ensure_ascii=False).encode()).hexdigest()
            if snapshot["text_sha256"] != bank_hash or snapshot["memory_refs"] != [[i, versions[i]] for i in live]:
                raise ValueError("Restored bank differs from the final recorded snapshot")
            measured = {int(e["query_id"].rsplit("_", 1)[1]): e for e in events if e["kind"] == "qa" and e["success"]}
            questions = ev.filter_qa_list(convo["qa"])
            if measured.keys() != {i for i, _ in questions}:
                raise ValueError("Missing original questions")
            missing = {}
            if repair_multiline:
                session = f"session_{end}"
                missing = missing_multiline_turns(processor, convo["conversation"][session],
                    convo["conversation"][session + "_date_time"], spans[-1]["source"])
                windows[episode.name]["omitted_turns"] = list(missing)
                if not any(set(q["evidence"]).intersection(missing) for _, q in questions):
                    continue
                present = set(live.values())
                extras = list(dict.fromkeys(t for t in missing.values() if t not in present))
                tails = {"pair": tails["pair"], "repair": tails["pair"] + extras}
            retriever = protocol["retriever"]
            vectors = get_embeddings(retriever, [q["question"] for _, q in questions], "query")
            # Encode each distinct memory once so both banks use the same vector for each shared entry.
            texts = list(dict.fromkeys(fixed + [t for additions in tails.values() for t in additions]))
            embeddings = dict(zip(texts, get_embeddings(retriever, texts, "context", batch_size=8), strict=True))
            banks = {}
            for name, extra in tails.items():
                bank = MemoryBank(retriever, top_k=protocol["retrieval_k"][1])
                for t in fixed + extra:
                    bank.add_memory(t, embeddings[t])
                banks[name] = bank
            for (i, q), vector in zip(questions, vectors, strict=True):
                if repair_multiline and not set(q["evidence"]).intersection(missing):
                    continue
                if any(versions[mid] != version for mid, version in measured[i]["refs"]):
                    raise ValueError("Reader used a different memory version")
                if measured[i]["question"] != q["question"]:
                    raise ValueError("The recorded question differs from the corpus")
                contexts = {name: bank.retrieve(vector)[0] for name, bank in banks.items()}
                archived = [live[mid] for mid, _ in measured[i]["refs"]]
                if skill_budget is None:
                    agreement = {"pair_retrieval_reproduced": archived == contexts["pair"]}
                else:
                    agreement = {"original_retrieval_reproduced": archived == contexts.pop("original")}
                job = {"episode": episode.name, "original_index": ids[i], "question": q, **agreement,
                       "prompts": {a: ev.build_prompt(q["question"], ctx, q) for a, ctx in contexts.items()}}
                if repair_multiline:
                    job["omitted_turns_retrieved"] = {a: [tid for tid, text in missing.items() if text in ctx]
                                                     for a, ctx in contexts.items()}
                jobs.append(job)
    return {"windows": windows, "jobs": jobs, "model": protocol["model"]}


def measured_answer(prompt, model, q, ev, cache):
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    answer, n_in, n_out = get_llm_response_via_api(
        prompt, LLM_MODEL=model, MAX_TOKENS=32, TAU=0.0, base_url=os.environ["MEMSKILL_API_BASE"],
        api_key=os.environ.get("MEMSKILL_API_KEY", "local-only"))
    return {"answer": answer, "tokens": [n_in, n_out], "f1": ev.compute_f1(answer, ev.get_ground_truth(q), q)}


def run(study, data_path, output, *, repair_multiline=False, skill_budget=None):
    plan = build_jobs(study, data_path, repair_multiline=repair_multiline, skill_budget=skill_budget)
    score_jobs(plan, output, repair_multiline=repair_multiline)


def score_jobs(plan, output, *, repair_multiline=False):
    """Score the saved reader prompts after the GPU used for retrieval has been released."""
    output.mkdir(parents=True, exist_ok=True)
    if not plan["jobs"]:
        raise ValueError("The saved training panel contains no eligible source writes and questions")
    backend = os.environ["MEMSKILL_BACKEND_ID"]
    signature = {"plan": digest(plan), "backend": backend, "windows": plan["windows"], "questions": len(plan["jobs"])}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Previously measured panel has different inputs")
    atomic_json(manifest, signature)
    ev = get_evaluator("locomo", SimpleNamespace())
    rows = []
    for job in plan["jobs"]:
        q = job["question"]
        scores = {}
        for name, prompt in job["prompts"].items():
            scores[name] = measured_answer(prompt, plan["model"], q, ev, output / "responses")
            scores[name]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            if repair_multiline:
                # An independent greedy repeat has its own measured response cache.
                scores[name]["repeat"] = measured_answer(prompt, plan["model"], q, ev, output / "repeat-responses")
        row = {k: v for k, v in job.items() if k not in ("question", "prompts")}
        row["scores"] = scores
        rows.append(row)
        atomic_json(output / f'{job["episode"]}-q{job["original_index"]}.json', row)
    ref_key = "original_retrieval_reproduced" if "original_retrieval_reproduced" in rows[0] else "pair_retrieval_reproduced"
    summary = {"questions": len(rows), "original_retrieval_matches": sum(r[ref_key] for r in rows),
               "mean_f1": {a: mean(r["scores"][a]["f1"] for r in rows) for a in rows[0]["scores"]}}
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "locomo", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--repair-multiline", action="store_true",
                        help="Independently repeat both answers to every training question linked to an omitted utterance")
    parser.add_argument("--skill-budget", type=Path, help="Measured equal-budget source turn selections from skill_source_budget")
    args = parser.parse_args()
    selections = json.loads(args.skill_budget.read_bytes()) if args.skill_budget else None
    run(args.study, args.locomo, args.output, repair_multiline=args.repair_multiline, skill_budget=selections)
