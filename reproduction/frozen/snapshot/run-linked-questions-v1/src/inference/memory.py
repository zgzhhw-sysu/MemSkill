"""Construct, merge and cache memory banks for inference."""
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.executor import ExecutionResult, deployment_operations
from src.memory_bank import MemoryBank
from src.trainer import BaseTrainer


def _extract_memory_actions_step(trainer: BaseTrainer,
                                 memory_bank,
                                 session_text: str,
                                 session_embedding: np.ndarray = None,
                                 executor=None):
    if session_embedding is None:
        session_embedding = trainer.state_encoder._encode_texts(session_text)
        if session_embedding.ndim == 2:
            session_embedding = session_embedding[0]

    retrieved_memories, retrieved_indices, retrieved_memory_embeddings = memory_bank.retrieve(
        session_embedding, use_state_encoder=True, return_embeddings=True
    )

    state_embedding = trainer.state_encoder.encode(
        session_text,
        retrieved_memories,
        session_embedding=session_embedding,
        memory_embeddings=retrieved_memory_embeddings
    )

    candidate_ops = trainer.operation_bank.get_candidate_operations()
    op_embeddings = np.vstack([op.embedding for op in candidate_ops])

    state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(trainer.device)
    op_tensor = torch.tensor(op_embeddings, dtype=torch.float32).to(trainer.device)

    record_policy = memory_bank.trace is not None and os.environ.get("MEMSKILL_TRACE_NATIVE_POLICY") == "1"
    policy_options = {"policy_trace": {}} if record_policy else {}
    action_idx, log_prob, _ = trainer.controller(state_tensor, op_tensor, deterministic=True, **policy_options)
    if isinstance(action_idx, list):
        selected_ops = [candidate_ops[idx] for idx in action_idx]
    else:
        selected_ops = [candidate_ops[action_idx]]

    executor_ops = deployment_operations(selected_ops)
    # print([i.instruction_template for i in executor_ops])
    exec_impl = executor if executor is not None else trainer.executor
    audit = {} if memory_bank.trace is None else {"audit": []}
    exec_result = exec_impl.execute_operation(
        operation=executor_ops,
        session_text=session_text,
        retrieved_memories=retrieved_memories, **audit
    )

    if memory_bank.trace is not None:
        memory_bank.trace.begin(memory_bank.timestep, session_text, executor_ops, retrieved_memories,
                                retrieved_indices, selected=selected_ops)
        if record_policy:
            memory_bank.trace.current["policy"] = {
                **policy_options["policy_trace"], "candidates": [op.name for op in candidate_ops],
                "ordered_log_probability": log_prob,
                "state_sha256": hashlib.sha256(np.asarray(state_embedding, dtype=np.float32).tobytes()).hexdigest(),
            }
        memory_bank.trace.current["executor_responses"] = audit["audit"]
    operation_names = []
    seen = set()
    for op in selected_ops:
        name = getattr(op, "name", None)
        if name and name not in seen:
            seen.add(name)
            operation_names.append(name)
    return {
        "results": exec_result,
        "retrieved_indices": list(retrieved_indices),
        "operation_names": operation_names
    }


def _apply_extracted_memory_actions(trainer: BaseTrainer,
                                    memory_bank,
                                    extracted: Dict[str, Any],
                                    executor=None):
    exec_impl = executor if executor is not None else trainer.executor
    applied = exec_impl.apply_to_memory_bank(
        results=extracted.get("results", []),
        memory_bank=memory_bank,
        retrieved_indices=extracted.get("retrieved_indices", []),
        operation_name=extracted.get("operation_names", [])
    )
    if memory_bank.trace is not None:
        memory_bank.trace.finish(extracted["results"], applied)


def _normalize_insert_content(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _clone_snapshot_memory_bank(snapshot_dict: Dict[str, Any], state_encoder=None) -> MemoryBank:
    snapshot_bank = MemoryBank.from_dict(snapshot_dict)
    if state_encoder is not None:
        snapshot_bank.set_state_encoder(state_encoder)
    return snapshot_bank


def _merge_batch_extracted_actions(extracted_batch: List[Dict[str, Any]],
                                   batch_memory_size: int) -> Dict[str, Any]:
    ordered = sorted(extracted_batch, key=lambda x: x["session_idx"])

    update_by_actual: Dict[int, ExecutionResult] = {}
    delete_by_actual: Dict[int, ExecutionResult] = {}
    insert_seen = set()
    merged_inserts: List[ExecutionResult] = []
    merged_operation_names = []
    op_seen = set()

    for entry in ordered:
        extracted = entry.get("extracted", {})

        for op_name in extracted.get("operation_names", []) or []:
            name = str(op_name).strip()
            if name and name not in op_seen:
                op_seen.add(name)
                merged_operation_names.append(name)

        retrieved_indices = extracted.get("retrieved_indices", []) or []
        results = extracted.get("results", []) or []
        for result in results:
            if not getattr(result, "success", False):
                continue

            action_type = str(getattr(result, "action_type", "")).upper()
            reasoning = str(getattr(result, "reasoning", "") or "")

            if action_type == "INSERT":
                content = str(getattr(result, "memory_content", "") or "").strip()
                if not content:
                    continue
                norm_key = _normalize_insert_content(content)
                if not norm_key or norm_key in insert_seen:
                    continue
                insert_seen.add(norm_key)
                merged_inserts.append(ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=content,
                    reasoning=reasoning
                ))
                continue

            if action_type not in ("UPDATE", "DELETE"):
                continue

            try:
                rel_idx = int(getattr(result, "memory_index", -1))
            except Exception:
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    "invalid MEMORY_INDEX value."
                )
                continue
            if rel_idx < 0 or rel_idx >= len(retrieved_indices):
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    f"MEMORY_INDEX {rel_idx} out of range for retrieved_indices size {len(retrieved_indices)}."
                )
                continue

            try:
                actual_idx = int(retrieved_indices[rel_idx])
            except Exception:
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    f"retrieved_indices[{rel_idx}] is not a valid integer index."
                )
                continue
            if actual_idx < 0 or actual_idx >= batch_memory_size:
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    f"actual memory index {actual_idx} out of snapshot range [0, {batch_memory_size})."
                )
                continue

            if action_type == "DELETE":
                delete_by_actual[actual_idx] = ExecutionResult(
                    action_type="DELETE",
                    success=True,
                    memory_index=actual_idx,
                    reasoning=reasoning
                )
                if actual_idx in update_by_actual:
                    del update_by_actual[actual_idx]
            else:
                if actual_idx in delete_by_actual:
                    continue
                content = str(getattr(result, "memory_content", "") or "").strip()
                if not content:
                    continue
                update_by_actual[actual_idx] = ExecutionResult(
                    action_type="UPDATE",
                    success=True,
                    memory_index=actual_idx,
                    memory_content=content,
                    reasoning=reasoning
                )

    merged_results: List[ExecutionResult] = []
    for actual_idx in sorted(update_by_actual.keys()):
        merged_results.append(update_by_actual[actual_idx])
    for actual_idx in sorted(delete_by_actual.keys()):
        merged_results.append(delete_by_actual[actual_idx])
    merged_results.extend(merged_inserts)

    return {
        "results": merged_results,
        "retrieved_indices": list(range(batch_memory_size)),
        "operation_names": merged_operation_names
    }


def _apply_memory_extraction_step(trainer: BaseTrainer,
                                  memory_bank,
                                  session_text: str,
                                  session_embedding: np.ndarray = None,
                                  executor=None):
    extracted = _extract_memory_actions_step(
        trainer,
        memory_bank,
        session_text=session_text,
        session_embedding=session_embedding,
        executor=executor
    )
    _apply_extracted_memory_actions(
        trainer,
        memory_bank,
        extracted=extracted,
        executor=executor
    )
    memory_bank.step()


def _resolve_sample_id(trainer: BaseTrainer, conversation: Dict[str, Any], conv_idx: int) -> str:
    sample_id = None
    if hasattr(trainer, "data_processor") and trainer.data_processor is not None:
        try:
            sample_id = trainer.data_processor.get_sample_id(conversation)
        except Exception:
            sample_id = None
    if sample_id is None:
        sample_id = conversation.get('sample_id', conversation.get('index', conv_idx))
    return str(sample_id)


def _build_memory_bank_from_sessions(trainer: BaseTrainer,
                                     sessions,
                                     session_embeddings: np.ndarray = None,
                                     total: int = None,
                                     executor=None,
                                     show_progress: bool = True,
                                     session_parallel_workers: int = 1, trace=None):
    memory_bank = trainer._initialize_memory_bank()

    memory_bank.trace = trace
    workers = max(1, int(session_parallel_workers or 1))
    if trace is not None and workers != 1:
        raise ValueError("Use --inference-session-workers 1 for ordered span traces")
    if workers == 1:
        iterator = enumerate(sessions)
        if show_progress:
            iterator = tqdm(iterator, total=total, desc="Sessions")
        for session_idx, session_text in iterator:
            session_embedding = None
            if session_embeddings is not None:
                session_embedding = session_embeddings[session_idx]
            _apply_memory_extraction_step(
                trainer,
                memory_bank,
                session_text=session_text,
                session_embedding=session_embedding,
                executor=executor
            )
        if trace is not None:
            trace.verify(memory_bank)
        return memory_bank

    session_texts = sessions if isinstance(sessions, list) else list(sessions)
    session_records = []
    for session_idx, session_text in enumerate(session_texts):
        session_embedding = None
        if session_embeddings is not None:
            session_embedding = session_embeddings[session_idx]
        session_records.append((session_idx, session_text, session_embedding))

    total_sessions = total if total is not None else len(session_records)
    progress = tqdm(total=total_sessions, desc="Sessions") if show_progress else None

    try:
        for start in range(0, len(session_records), workers):
            batch = session_records[start:start + workers]
            if not batch:
                continue

            snapshot_dict = memory_bank.to_dict()
            snapshot_size = len(memory_bank.memories)
            local_executor_cls = executor.__class__ if executor is not None else trainer.executor.__class__

            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
                future_to_idx = {
                    pool.submit(
                        _extract_memory_actions_step,
                        trainer,
                        _clone_snapshot_memory_bank(snapshot_dict, memory_bank.state_encoder),
                        session_text,
                        session_embedding,
                        local_executor_cls(trainer.args)
                    ): session_idx
                    for session_idx, session_text, session_embedding in batch
                }

                extracted_batch = []
                for future in as_completed(future_to_idx):
                    session_idx = future_to_idx[future]
                    extracted = future.result()
                    extracted_batch.append({
                        "session_idx": session_idx,
                        "extracted": extracted
                    })

            merged = _merge_batch_extracted_actions(
                extracted_batch=extracted_batch,
                batch_memory_size=snapshot_size
            )
            _apply_extracted_memory_actions(
                trainer,
                memory_bank,
                extracted=merged,
                executor=executor
            )

            for _ in range(len(batch)):
                memory_bank.step()

            if progress is not None:
                progress.update(len(batch))
    finally:
        if progress is not None:
            progress.close()

    return memory_bank


def _save_memory_bank(memory_path: str, memory_bank: MemoryBank):
    import pickle
    payload = {"memory_bank": memory_bank.to_dict()}
    with open(memory_path, 'wb') as f:
        pickle.dump(payload, f)


def _load_memory_bank(memory_path: str) -> Optional[MemoryBank]:
    import pickle
    if not os.path.exists(memory_path):
        return None
    with open(memory_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'memory_bank' in data:
        bank_dict = data.get('memory_bank')
        if isinstance(bank_dict, dict) and 'memories' in bank_dict:
            return MemoryBank.from_dict(bank_dict)
    if isinstance(data, dict) and 'memories' in data:
        return MemoryBank.from_dict(data)
    if isinstance(data, dict) and len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, dict) and 'memories' in only_value:
            return MemoryBank.from_dict(only_value)
    return None


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    return sanitized.strip('_') or 'unknown'


def _memory_cache_path(
    memory_dir: str,
    args,
    sample_id: str,
    session_mode: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    action_top_k: Optional[int] = None,
    retriever: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    dataset = getattr(args, 'dataset', 'data')
    if session_mode is None:
        session_mode = getattr(args, 'session_mode', 'turn-pair')
    if chunk_size is None:
        chunk_size = getattr(args, 'chunk_size', None)
    if chunk_overlap is None:
        chunk_overlap = getattr(args, 'chunk_overlap', None)
    if action_top_k is None:
        action_top_k = getattr(args, 'action_top_k', None)
    if retriever is None:
        retriever = getattr(args, 'retriever', None)
    if model is None:
        model = getattr(args, 'model', None)
    # print(model)
    name_parts = [
        "memory",
        _sanitize_filename(str(dataset)),
        f"sample_{_sanitize_filename(str(sample_id))}",
        f"mode_{_sanitize_filename(str(session_mode))}",
        f"chunk_{_sanitize_filename(str(chunk_size))}",
        f"overlap_{_sanitize_filename(str(chunk_overlap))}",
        f"topk_{_sanitize_filename(str(action_top_k))}",
        f"retriever_{_sanitize_filename(str(retriever))}",
        f"model_{_sanitize_filename(str(model).lower())}",
    ]
    if getattr(args, 'skip_load_operation_bank', False):
        name_parts.append('skipopbank')
    cache_suffix = str(getattr(args, 'memory_cache_suffix', '') or '').strip()
    if cache_suffix:
        name_parts.append(_sanitize_filename(cache_suffix))
    filename = "_".join(name_parts) + ".pkl"
    return os.path.join(memory_dir, filename)
