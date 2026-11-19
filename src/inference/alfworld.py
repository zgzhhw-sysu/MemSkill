"""Build and evaluate ALFWorld memories from offline task trajectories."""
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.data_processing.alfworld import chunk_trajectories_by_tokens
from src.inference.memory import (
    _build_memory_bank_from_sessions,
    _load_memory_bank,
    _memory_cache_path,
    _save_memory_bank,
)
from src.rag_utils import get_embeddings
from src.trainer import BaseTrainer


def _collect_alfworld_trajectories(train_data) -> list:
    trajectories = []
    if isinstance(train_data, dict):
        for _, games in train_data.items():
            if not isinstance(games, dict):
                continue
            for _, entry in games.items():
                if isinstance(entry, dict):
                    traj = entry.get("trajectory") or ""
                    if isinstance(traj, str) and traj.strip():
                        trajectories.append(traj.strip())
    elif isinstance(train_data, list):
        for entry in train_data:
            if isinstance(entry, dict):
                traj = entry.get("trajectory") or ""
                if isinstance(traj, str) and traj.strip():
                    trajectories.append(traj.strip())
    return trajectories


def _collect_alfworld_eval_entries(test_data) -> List[Dict[str, Any]]:
    entries = []
    if isinstance(test_data, dict):
        for task_type, games in test_data.items():
            if not isinstance(games, dict):
                continue
            for gamefile, entry in games.items():
                if isinstance(entry, dict):
                    entries.append({
                        "task_type": task_type,
                        "gamefile": gamefile,
                        "entry": entry
                    })
    elif isinstance(test_data, list):
        for entry in test_data:
            if isinstance(entry, dict):
                entries.append({
                    "task_type": entry.get("task_type", "unknown"),
                    "gamefile": entry.get("gamefile", ""),
                    "entry": entry
                })
    return entries


def _extract_alfworld_objective(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Your task is to:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    objective = match.group(1).strip()
    if not objective:
        return ""
    objective = objective.splitlines()[0].strip()
    return objective.rstrip(".")


def infer_alfworld_memories(trainer: BaseTrainer, train_data, test_data, args):
    from src.alfworld_env_runner import run_alfworld_episode
    print("\n" + "="*80)
    print("Constructing Memory Bank from ALFWorld Train Trajectories")
    print("="*80)

    trainer.controller.eval()

    memory_dir = os.path.join(os.path.dirname(args.out_file), 'memories')
    os.makedirs(memory_dir, exist_ok=True)
    inference_session_workers = int(
        getattr(args, "inference_session_workers", getattr(trainer.config, "inference_session_workers", 1)) or 1
    )
    inference_session_workers = max(1, inference_session_workers)

    overwrite = bool(getattr(args, "overwrite", False))
    memory_path = _memory_cache_path(
        memory_dir,
        args,
        "alfworld_train",
        session_mode="alfworld",
        chunk_size=getattr(trainer.config, "alfworld_pair_chunk_size", None)
        or getattr(trainer.config, "chunk_size", None)
    )
    cached_bank = None
    if not overwrite:
        cached_bank = _load_memory_bank(memory_path)
    if cached_bank is not None:
        memory_bank = cached_bank
        print(f"Loaded cached ALFWorld memory bank from {memory_path}")
    else:
        trajectories = _collect_alfworld_trajectories(train_data)
        chunk_size = getattr(trainer.config, "alfworld_pair_chunk_size", None)
        if chunk_size is None:
            chunk_size = getattr(trainer.config, "chunk_size", None)
        chunks = chunk_trajectories_by_tokens(trajectories, chunk_size)

        if not chunks:
            print("No ALFWorld trajectories found; skipping memory extraction.")
            return memory_dir

        session_embeddings = trainer.state_encoder._encode_texts(chunks)
        if hasattr(session_embeddings, "ndim") and session_embeddings.ndim == 1:
            session_embeddings = session_embeddings.reshape(1, -1)

        memory_bank = _build_memory_bank_from_sessions(
            trainer,
            chunks,
            session_embeddings=session_embeddings,
            total=len(chunks),
            executor=trainer.executor,
            session_parallel_workers=inference_session_workers
        )
        _save_memory_bank(memory_path, memory_bank)
        print(f"Saved ALFWorld memory bank to {memory_path}")

    if not test_data:
        print("No ALFWorld eval data provided; skipping eval.")
        return memory_dir

    print("\n" + "="*80)
    print("Evaluating ALFWorld Test Environments")
    print("="*80)

    eval_entries = _collect_alfworld_eval_entries(test_data)
    if not eval_entries:
        print("No ALFWorld eval entries found; skipping eval.")
        return memory_dir

    query_source = str(getattr(trainer.config, "alfworld_eval_query_source", "first_observation")).lower()
    if query_source not in ("objective", "first_observation"):
        query_source = "first_observation"

    queries = []
    objectives = []
    for item in eval_entries:
        entry = item["entry"]
        first_obs = entry.get("first_observation") or ""
        if not isinstance(first_obs, str):
            first_obs = str(first_obs)
        objective = entry.get("objective") or ""
        if not isinstance(objective, str):
            objective = ""
        if not objective and first_obs:
            objective = _extract_alfworld_objective(first_obs)
        objectives.append(objective)
        if query_source == "objective":
            query_text = objective or first_obs
        else:
            query_text = first_obs or objective
        queries.append(query_text or "")

    retriever = getattr(trainer.args, "retriever", None) or "contriever"
    top_k_eval = getattr(trainer.config, "mem_top_k_eval", trainer.config.mem_top_k)

    query_embeddings = []
    if any(q.strip() for q in queries):
        safe_queries = [q if q.strip() else " " for q in queries]
        query_embeddings = get_embeddings(retriever, safe_queries, "query")

    tasks = []
    for idx, item in enumerate(eval_entries):
        q_text = queries[idx]
        retrieved_memories = []
        retrieved_indices = []
        if q_text.strip() and len(memory_bank.memories) > 0 and len(query_embeddings) > 0:
            emb = query_embeddings[idx]
            if hasattr(emb, "ndim") and emb.ndim == 2:
                emb = emb[0]
            retrieved_memories, retrieved_indices = memory_bank.retrieve(
                emb, top_k=top_k_eval, use_state_encoder=False
            )
        tasks.append({
            "task_type": item["task_type"],
            "gamefile": item["gamefile"],
            "objective": objectives[idx],
            "query": q_text,
            "retrieved_memories": list(retrieved_memories),
            "retrieved_indices": list(retrieved_indices)
        })

    llm_args = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": args.api_key,
        "temperature": getattr(trainer.config, "alfworld_action_temperature", 0.0),
        "top_p": getattr(trainer.config, "alfworld_action_top_p", 1.0),
        "max_tokens": getattr(trainer.config, "alfworld_action_max_tokens", 32),
        "seed": args.seed
    }
    include_inventory = bool(getattr(trainer.config, "alfworld_include_inventory", True))
    use_icl_prompt = bool(getattr(trainer.config, "alfworld_use_icl", False))
    max_steps = int(getattr(trainer.config, "alfworld_pair_max_steps", 50))
    workers = int(getattr(trainer.config, "alfworld_pair_b_workers", 0) or 0)
    if workers <= 0:
        workers = max(1, getattr(trainer.config, "batch_size", 1))

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                run_alfworld_episode,
                task["gamefile"],
                task["objective"],
                task["retrieved_memories"],
                max_steps,
                llm_args,
                include_inventory,
                query_source,
                task_type=task.get("task_type"),
                use_icl_prompt=use_icl_prompt
            ): task for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="ALFWorld eval"):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = {
                    "success": False,
                    "episode_length": max_steps,
                    "error": str(exc)
                }
            merged = dict(task)
            merged.update(outcome)
            results.append(merged)

    if not results:
        print("No ALFWorld eval results produced.")
        return memory_dir

    success_count = sum(1 for r in results if r.get("success"))
    avg_success = success_count / max(len(results), 1)
    step_counts = []
    for r in results:
        steps = r.get("episode_length")
        if not isinstance(steps, int):
            steps = max_steps
        step_counts.append(steps)
    avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

    print("\n" + "=" * 80)
    print("ALFWorld Evaluation Metrics")
    print("=" * 80)
    print(f"Total envs: {len(results)}")
    print(f"Success rate: {avg_success:.4f}")
    print(f"Avg interaction steps: {avg_steps:.2f}")
    return memory_dir
