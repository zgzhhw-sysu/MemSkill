"""Build memories and answer questions for the text datasets."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from prompts.prompt_pool import LLM_JUDGE_GENERAL_PROMPT
from src.episode_trace import EpisodeTrace
from src.eval_utils import llm_judge
from src.inference.memory import (
    _build_memory_bank_from_sessions,
    _load_memory_bank,
    _memory_cache_path,
    _resolve_sample_id,
    _save_memory_bank,
)
from src.llm_utils import get_llm_response
from src.memory_bank import MemoryBank
from src.rag_utils import get_embeddings
from src.trainer import BaseTrainer


def evaluate_text_dataset_queries(trainer: BaseTrainer,
                                  test_data,
                                  memory_banks: Dict[str, Any],
                                  args) -> Dict[str, Any]:
    eval_args = trainer.evaluator.prepare_eval_args()
    top_k_eval = getattr(trainer.config, "mem_top_k_eval", trainer.config.mem_top_k)
    is_hotpotqa = args.dataset == "hotpotqa"
    extractor = getattr(trainer.evaluator, "_extract_answer", None) if is_hotpotqa else None

    task_args = []
    meta_by_qid = {}
    next_qid = 0

    for conv_idx, conversation in enumerate(test_data):
        if args.dataset == "longmemeval":
            question_id = str(conversation.get("question_id", ""))
            if question_id.endswith("_abs") and not getattr(args, "include_abstentions", False):
                print(f"[Eval] Skipping _abs sample question_id={question_id}.")
                continue
        sample_id = _resolve_sample_id(trainer, conversation, conv_idx)
        memory_bank = memory_banks.get(sample_id)
        if memory_bank is None:
            if getattr(args, "eval_f1_only", False):
                raise ValueError(f"Missing evaluation memory bank for {sample_id}")
            print(f"[Eval] Missing memory bank for sample_id={sample_id}, skipping.")
            continue

        qa_list = trainer.data_processor.get_qa_list(conversation)
        valid_qa = trainer.evaluator.filter_qa_list(qa_list)
        if not valid_qa:
            print(f"[Eval] No valid QA items for sample_id={sample_id}, skipping.")
            continue

        questions = [qa['question'] for _, qa in valid_qa]
        q_embeddings = get_embeddings(
            eval_args.retriever,
            questions,
            'query'
        )

        for idx, (qa_idx, qa) in enumerate(valid_qa):
            question = qa['question']
            q_embedding = q_embeddings[idx]
            retrieved_mems, retrieved_indices = memory_bank.retrieve(
                q_embedding, top_k=top_k_eval, use_state_encoder=False
            )
            # print(retrieved_mems)
            # print(top_k_eval, len(retrieved_mems))
            prompt = trainer.evaluator.build_prompt(question, retrieved_mems, qa)
            task_args.append((next_qid, prompt, eval_args))
            meta_by_qid[next_qid] = {
                "qa": qa,
                "qa_idx": qa_idx,
                "sample_id": sample_id,
                "retrieved_memories": retrieved_mems,
                "retrieved_indices": list(retrieved_indices)
            }
            next_qid += 1

    if not task_args:
        print("No evaluation queries found.")
        return {}

    ret = get_llm_response(args=eval_args, task_args=task_args)
    if getattr(args, "eval_f1_only", False):
        actual_ids = [qid for qid, _, _, _ in ret]
        if (len(actual_ids) != len(meta_by_qid) or set(actual_ids) != set(meta_by_qid)
                or any(not ok for _, _, _, ok in ret)):
            raise RuntimeError("Missing, duplicate or failed reader responses")

    predictions = {}
    metrics = {
        "f1": [],
        "llm_judge": []
    }
    category_metrics = {}

    for qid, response, tokens, success in ret:
        meta = meta_by_qid.get(qid, {})
        qa = meta.get("qa", {})
        ground_truth = trainer.evaluator.get_ground_truth(qa)
        prediction = response.strip() if success and response is not None else ""
        if callable(extractor):
            prediction = extractor(prediction)
        # print(prediction)
        predictions[qid] = prediction
        f1 = trainer.evaluator.compute_f1(prediction, ground_truth, qa)
        bank = memory_banks[meta["sample_id"]]
        if bank.trace is not None:
            bank.trace.write("qa", {"query_id": meta["qa_idx"], "question": qa["question"],
                                    "prediction": prediction, "success": success, "f1": f1, "tokens": tokens,
                                    "refs": [bank.trace.ref(i) for i in meta["retrieved_indices"]]})

        metrics["f1"].append(f1)

        category = qa.get("category")
        if category is not None:
            bucket = category_metrics.setdefault(category, {"f1": [], "llm_judge": []})
            bucket["f1"].append(f1)

    if getattr(args, "eval_f1_only", False):
        rows = [
            {"sample_id": meta_by_qid[qid]["sample_id"], "qa_index": meta_by_qid[qid]["qa_idx"],
             "retrieved_indices": [int(i) for i in meta_by_qid[qid]["retrieved_indices"]],
             "prediction": predictions[qid], "f1": score, "tokens": tokens}
            for (qid, _, tokens, _), score in zip(ret, metrics["f1"])
        ]
        report = {"count": len(rows), "f1": float(np.mean(metrics["f1"])), "answers": rows}
        with open(args.out_file, "w") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report

    # LLM judge
    judge_task_args = []
    for qid, meta in meta_by_qid.items():
        qa = meta.get("qa", {})
        ground_truth = trainer.evaluator.get_ground_truth(qa)
        if isinstance(ground_truth, list):
            ground_truth_str = ", ".join(str(ans) for ans in ground_truth)
        else:
            ground_truth_str = str(ground_truth)
        prediction = predictions.get(qid, "")
        judge_task_args.append((
            qid,
            LLM_JUDGE_GENERAL_PROMPT.format(
                question=qa.get("question", ""),
                ground_truth=ground_truth_str,
                model_answer=prediction
            ),
            eval_args
        ))

    llm_judge_scores = {}
    if judge_task_args:
        judge_scores = llm_judge(task_args=judge_task_args, args=eval_args)
        for idx, (qid, _, _) in enumerate(judge_task_args):
            llm_judge_scores[qid] = judge_scores[idx]

    for qid, score in llm_judge_scores.items():
        metrics["llm_judge"].append(float(score))
        meta = meta_by_qid.get(qid, {})
        qa = meta.get("qa", {})
        category = qa.get("category")
        if category is not None and category in category_metrics:
            category_metrics[category]["llm_judge"].append(float(score))

    def _avg(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    print("\n" + "=" * 80)
    print(f"{args.dataset} Evaluation (query-wise averages)")
    print("=" * 80)
    print(f"Total queries: {len(metrics['f1'])}")
    print(f"F1: {_avg(metrics['f1']):.4f}")
    print(f"LLM Judge: {_avg(metrics['llm_judge']):.4f}")

    if category_metrics:
        print("\nBy category:")
        for category in sorted(category_metrics.keys()):
            data = category_metrics[category]
            print(
                f"Category {category}: "
                f"F1={_avg(data['f1']):.4f}, "
                f"LLM Judge={_avg(data['llm_judge']):.4f}"
            )

    return {}


def infer_text_dataset_memories(trainer: BaseTrainer, test_data, args):
    print("\n" + "="*80)
    print("Constructing Memory Banks for Test Set")
    print("="*80)

    trainer.controller.eval()

    memory_dir = os.path.join(os.path.dirname(args.out_file), 'memories')
    os.makedirs(memory_dir, exist_ok=True)

    memory_banks: Dict[str, MemoryBank] = {}
    overwrite = bool(getattr(args, "overwrite", False))
    inference_workers = int(
        getattr(args, "inference_workers", getattr(trainer.config, "inference_workers", 1)) or 1
    )
    inference_workers = max(1, inference_workers)
    inference_session_workers = int(
        getattr(args, "inference_session_workers", getattr(trainer.config, "inference_session_workers", 1)) or 1
    )
    inference_session_workers = max(1, inference_session_workers)
    loaded_count = 0
    computed_count = 0
    pending_jobs = []

    def _compute_single_sample_memory(conversation, sample_id: str, memory_path: str, use_local_executor: bool):
        exec_impl = trainer.executor.__class__(args) if use_local_executor else trainer.executor
        sessions, episode_length, precompute = trainer._prepare_sessions(conversation)
        if precompute and isinstance(sessions, list):
            session_embeddings = trainer.state_encoder._encode_texts(sessions)
            if hasattr(session_embeddings, "ndim") and session_embeddings.ndim == 1:
                session_embeddings = session_embeddings.reshape(1, -1)
        else:
            session_embeddings = None

        total = episode_length if episode_length is not None and episode_length > 0 else None
        memory_bank = _build_memory_bank_from_sessions(
            trainer,
            sessions,
            session_embeddings=session_embeddings,
            total=total,
            executor=exec_impl,
            show_progress=True,
            session_parallel_workers=inference_session_workers,
            trace=EpisodeTrace(args.trace_dir, sample_id, conversation.get("conversation"))
            if getattr(args, "trace_dir", None) else None
        )
        return sample_id, memory_path, memory_bank

    for conv_idx, conversation in enumerate(tqdm(test_data, desc="Processing")):
        if args.dataset == "longmemeval":
            question_id = str(conversation.get("question_id", ""))
            if question_id.endswith("_abs"):
                print(f"[Memory] Skipping _abs sample question_id={question_id}.")
                continue
        sample_id = _resolve_sample_id(trainer, conversation, conv_idx)
        memory_path = _memory_cache_path(memory_dir, args, sample_id)
        if not overwrite and not getattr(args, "trace_dir", None):
            cached_bank = _load_memory_bank(memory_path)
            if cached_bank is not None:
                memory_banks[sample_id] = cached_bank
                loaded_count += 1
                continue
        pending_jobs.append((conversation, sample_id, memory_path))

    if inference_workers == 1:
        for conversation, sample_id, memory_path in pending_jobs:
            sid, out_path, memory_bank = _compute_single_sample_memory(
                conversation=conversation,
                sample_id=sample_id,
                memory_path=memory_path,
                use_local_executor=False
            )
            memory_banks[sid] = memory_bank
            _save_memory_bank(out_path, memory_bank)
            computed_count += 1
    else:
        with ThreadPoolExecutor(max_workers=inference_workers) as pool:
            futures = [
                pool.submit(
                    _compute_single_sample_memory,
                    conversation,
                    sample_id,
                    memory_path,
                    True
                )
                for conversation, sample_id, memory_path in pending_jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing memories"):
                sid, out_path, memory_bank = future.result()
                memory_banks[sid] = memory_bank
                _save_memory_bank(out_path, memory_bank)
                computed_count += 1

    print(
        f"\nLoaded {loaded_count} cached memory banks; "
        f"computed {computed_count} new memory banks."
    )
    print(f"Memory cache directory: {memory_dir}")
    print("Running evaluation on test queries...")
    evaluate_text_dataset_queries(trainer, test_data, memory_banks, args)
    return memory_dir
