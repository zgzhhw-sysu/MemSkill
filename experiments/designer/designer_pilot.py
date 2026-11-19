"""Matched 8B pilot on real LoCoMo prefixes, using the native controller and Designer."""
import copy
import json
from pathlib import Path
import re

from main import (load_dataset, split_data, set_seed, evaluate_text_dataset_queries,
                  _build_memory_bank_from_sessions)
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.controller import PPOBuffer
from src.designer import DesignerCase
from src.episode_trace import EpisodeTrace
from src.operation_bank import OperationBank
from src.trainer import get_trainer


def prefix(conv, length=2):
    sessions = {key: value for key, value in conv["conversation"].items()
                if (m := re.fullmatch(r"session_(\d+)(?:_date_time)?", key)) and int(m[1]) <= length}
    source = {turn["dia_id"] for turns in sessions.values() if isinstance(turns, list) for turn in turns}
    qa = [dict(q, original_index=i) for i, q in enumerate(conv["qa"])
          if q.get("evidence") and set(q["evidence"]) <= source and int(q["category"]) != 5]
    return dict(conv, conversation=sessions, qa=qa)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def run():
    args = get_agentic_memory_args()
    out = Path(args.out_file).parent
    out.mkdir(exist_ok=True, parents=True)
    train, val, _ = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    fit = [prefix(c) for c in train[:2]]
    held = [prefix(c) for c in val]
    save(out / "protocol.json", {
        "training_samples": {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in fit},
        "validation_samples": {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in held},
        "prefix_sessions": 2, "seed": args.seed, "model": args.model,
        "comparison": "unchanged checkpoint, original Designer, paired-replay Designer",
        "primary_metric": "unweighted native question F1; count positive and negative paired changes",
        "case_selection": "first 4 replayed failures; identical cases for both Designers",
        "extra_cost": "diagnostic probes in replay arm only; Designer edit budget held constant"})
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    trainer.designer.case_collector.clear(reset_pool=True)
    base = copy.deepcopy(trainer.operation_bank.to_dict())
    combined = PPOBuffer()
    for c in fit:
        bank = trainer._initialize_memory_bank()
        sessions, length, precompute = trainer._prepare_sessions(c)
        local = PPOBuffer()
        trainer.train_episode(c, bank, sessions, local, episode_length=length, precompute_embeddings=precompute)
        combined.merge(local)
    # Verify PPO, checkpoint save and reload, then restore the same controller for both arms.
    update = trainer.update_controller(combined)
    trainer.save_checkpoint("pilot")
    pilot_checkpoint = next(Path(args.save_dir).glob("*pilot*.pt"))
    trainer.load_checkpoint(str(pilot_checkpoint))
    cases = [x for x in trainer.designer.case_collector.get_all_cases() if x.replay][:4]
    if not cases:
        raise RuntimeError("No real failures with replay records")
    save(out / "training_feedback.json", [
        {k: v for k, v in c.to_dict().items() if k != "memory_bank_snapshot"} for c in cases])
    save(out / "ppo.json", {k: float(v) for k, v in update.items()})
    trainer.load_checkpoint(args.load_checkpoint)
    trainer.controller.eval()
    results = {}
    for arm in ("checkpoint", "original_designer", "replay_designer"):
        trainer.operation_bank = OperationBank.from_dict(copy.deepcopy(base), encoder=trainer.op_encoder)
        if arm != "checkpoint":
            selected = [DesignerCase.from_dict(c.to_dict()) for c in cases]
            if arm == "original_designer":
                for case in selected:
                    case.replay = None
            prompt = trainer.designer.build_analysis_prompt(selected, trainer.operation_bank)
            proposed = trainer.designer.run_evolution(trainer.operation_bank, {
                "analysis_prompt": prompt, "analysis_cases": selected})
            applied = trainer.designer.apply_evolution(trainer.operation_bank, proposed)
            save(out / f"{arm}_edit.json", {"prompt": prompt, "applied": applied, "proposal": proposed})
        banks = {}
        for c in held:
            sessions, _, _ = trainer._prepare_sessions(c)
            trace = EpisodeTrace(out / "validation_traces" / arm, c["sample_id"])
            banks[c["sample_id"]] = _build_memory_bank_from_sessions(trainer, sessions, trace=trace)
        args.out_file = str(out / f"{arm}_scores.json")
        result = evaluate_text_dataset_queries(trainer, held, banks, args)
        results[arm] = result["f1"]
    save(out / "comparison.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    run()
