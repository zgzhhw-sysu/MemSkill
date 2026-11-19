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
        split[joined] = turns
    if any(pair not in split for pair in pairs):
        raise ValueError("Splitting the source changed the stored speaker, date or utterance")
    fixed = list(live.values())[:-len(pairs)]
    return live, fixed, {"pair": pairs, "single": [s for pair in pairs for s in split[pair]]}


def build_jobs(study, data_path):
    protocol = json.loads((study / "protocol.json").read_text())
    raw = data_path.read_bytes()
    train = set(protocol["train"])
    if hashlib.sha256(raw).hexdigest() != protocol["input_sha256"] or train & set(protocol["validation"] + protocol["test"]):
        raise ValueError("The saved training split and input corpus must match")
    data = {c["sample_id"]: c for c in json.loads(raw) if c["sample_id"] in train}
    ev = get_evaluator("locomo", SimpleNamespace())
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
            variants = source_variants([e for e in events if e["kind"] == "span"])
            windows[episode.name] = {"trace": hashlib.sha256(raw_trace).hexdigest(), "source_inserted": variants is not None}
            if variants is None:
                continue
            live, fixed, tails = variants
            snapshot, = [e for e in events if e["kind"] == "snapshot"]
            versions = {c["memory_id"]: c["version"] for e in events if e["kind"] == "span" for c in e["changes"]}
            bank_hash = hashlib.sha256(json.dumps(list(live.items()), ensure_ascii=False).encode()).hexdigest()
            if snapshot["text_sha256"] != bank_hash or snapshot["memory_refs"] != [[i, versions[i]] for i in live]:
                raise ValueError("Restored bank differs from the final recorded snapshot")
            measured = {int(e["query_id"].rsplit("_", 1)[1]): e for e in events if e["kind"] == "qa" and e["success"]}
            questions = ev.filter_qa_list(convo["qa"])
            if measured.keys() != {i for i, _ in questions}:
                raise ValueError("Missing original questions")
            retriever = protocol["retriever"]
            vectors = get_embeddings(retriever, [q["question"] for _, q in questions], "query")
            # Encode each distinct memory once so both banks use the same vector for each shared entry.
            texts = list(dict.fromkeys(fixed + tails["pair"] + tails["single"]))
            embeddings = dict(zip(texts, get_embeddings(retriever, texts, "context", batch_size=8), strict=True))
            banks = {}
            for name, extra in tails.items():
                bank = MemoryBank(retriever, top_k=protocol["retrieval_k"][1])
                for t in fixed + extra:
                    bank.add_memory(t, embeddings[t])
                banks[name] = bank
            for (i, q), vector in zip(questions, vectors, strict=True):
                if any(versions[mid] != version for mid, version in measured[i]["refs"]):
                    raise ValueError("Reader used a different memory version")
                if measured[i]["question"] != q["question"]:
                    raise ValueError("The recorded question differs from the corpus")
                contexts = {name: bank.retrieve(vector)[0] for name, bank in banks.items()}
                archived = [live[mid] for mid, _ in measured[i]["refs"]]
                jobs.append({"episode": episode.name, "original_index": ids[i], "question": q,
                             "pair_retrieval_reproduced": archived == contexts["pair"],
                             "prompts": {a: ev.build_prompt(q["question"], ctx, q) for a, ctx in contexts.items()}})
    return {"windows": windows, "jobs": jobs, "model": protocol["model"]}


def run(study, data_path, output):
    output.mkdir(parents=True, exist_ok=True)
    plan = build_jobs(study, data_path)
    if not plan["jobs"]:
        raise ValueError("The saved training panel contains no eligible source writes and questions")
    backend = os.environ["MEMSKILL_BACKEND_ID"]
    signature = {"plan": digest(plan), "backend": backend, "windows": plan["windows"], "questions": len(plan["jobs"])}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Previously measured panel has different inputs")
    atomic_json(manifest, signature)
    ev = get_evaluator("locomo", SimpleNamespace())
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(output / "responses")
    rows = []
    for job in plan["jobs"]:
        q = job["question"]
        scores = {}
        for name, prompt in job["prompts"].items():
            answer, n_in, n_out = get_llm_response_via_api(prompt, LLM_MODEL=plan["model"], MAX_TOKENS=32, TAU=0.0,
                base_url=os.environ["MEMSKILL_API_BASE"], api_key=os.environ.get("MEMSKILL_API_KEY", "local-only"))
            scores[name] = {"answer": answer, "tokens": [n_in, n_out], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "f1": ev.compute_f1(answer, ev.get_ground_truth(q), q)}
        row = {k: job[k] for k in ("episode", "original_index", "pair_retrieval_reproduced")}
        row["scores"] = scores
        rows.append(row)
        atomic_json(output / f'{job["episode"]}-q{job["original_index"]}.json', row)
    atomic_json(output / "summary.json", {"questions": len(rows), "original_retrieval_matches": sum(r["pair_retrieval_reproduced"] for r in rows),
                "mean_f1": {a: mean(r["scores"][a]["f1"] for r in rows) for a in ("pair", "single")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "locomo", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.study, args.locomo, args.output)
