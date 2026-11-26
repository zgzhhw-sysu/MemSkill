"""Run MemSkill and keep the historical Python import paths available."""
from src.cli import main, set_seed
from src.data_processing.datasets import load_dataset, split_data
from src.inference.memory import (
    _apply_extracted_memory_actions,
    _apply_memory_extraction_step,
    _build_memory_bank_from_sessions,
    _clone_snapshot_memory_bank,
    _extract_memory_actions_step,
    _load_memory_bank,
    _memory_cache_path,
    _merge_batch_extracted_actions,
    _normalize_insert_content,
    _resolve_sample_id,
    _sanitize_filename,
    _save_memory_bank,
)
from src.inference.text import evaluate_text_dataset_queries, infer_text_dataset_memories
from src.inference.alfworld import (
    _collect_alfworld_eval_entries,
    _collect_alfworld_trajectories,
    _extract_alfworld_objective,
    infer_alfworld_memories,
)


if __name__ == "__main__":
    main()
