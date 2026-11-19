"""Matched on-policy update: same native transitions, frozen skills, three reward treatments."""
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from experiments.designer.designer_pilot import prefix, save
from main import load_dataset, set_seed, split_data
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.controller import PPOBuffer
from src.episode_trace import skill_spec
from src.revision_validation import evaluate_bank
from src.trainer import get_trainer


def log_margins(trainer, batch, alternatives):
    """Compare joint log-probability of two ordered K-tuples on the sampled states."""
    def joint(i, action):
        state = torch.as_tensor(batch.states[i], device=trainer.device).float()[None]
        ops = torch.as_tensor(batch.op_embs[i], device=trainer.device).float()[None]
        mask = torch.as_tensor(batch.new_op_masks[i], device=trainer.device)[None]
        action = torch.tensor(action, device=trainer.device).reshape(1, -1)
        with torch.no_grad():
            logp, *_ = trainer.controller.evaluate_actions(state, ops, action, new_op_masks=mask)
        return logp.item()
    return [joint(i, batch.actions[i]) - joint(i, alt) if alt is not None else None
            for i, alt in enumerate(alternatives)]


def run():
    args = get_agentic_memory_args()
    if not args.span_credit_weight or args.enable_designer:
        raise ValueError("Matched study requires positive span credit and a fixed skill bank")
    out = Path(args.out_file).parent
    out.mkdir(exist_ok=False, parents=True)
    train, val, _ = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    fit = [prefix(train[i], 2) for i in (0, 1, 4, 5)]
    held = [prefix(c, 3) for c in val]
    def dataset_ids(group):
        return {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in group}
    spec = {"seed": args.seed, "train": dataset_ids(fit), "validation": dataset_ids(held),
            "checkpoint": hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest(),
            "dataset": hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest(),
            "config": {k: v for k, v in vars(args).items() if k != "api_key"},
            "arms": ["B0_upstream", "B1_terminal", "C1_credit"],
            "comparison": "one paired native PPO update from identical sampled episodes and Adam state"}
    save(out / "protocol.json", spec)
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    set_seed(args.seed)
    bank = trainer.operation_bank
    versions = [skill_spec(op) for op in bank.get_candidate_operations()]
    names = [s["name"] for s in versions]
    spec["skills"] = versions
    save(out / "protocol.json", spec)
    batch = PPOBuffer()
    upstream, terminal, auxiliary, alternatives, samples = [], [], [], [], []
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "train_usage.jsonl")
    for c in fit:
        local = PPOBuffer()
        sessions, length, precompute = trainer._prepare_sessions(c)
        memory = trainer._initialize_memory_bank()
        log = trainer.train_episode(c, memory, sessions, local, episode_length=length, precompute_embeddings=precompute)
        records = [json.loads(s) for s in memory.trace.path.read_text().splitlines()]
        diagnostics = [r["diagnosis"] for r in records if r["kind"] == "span_credit"]
        for i, (d, row) in enumerate(zip(diagnostics, log["reward_components"], strict=True)):
            upstream.append(row["upstream_reward"])
            terminal.append(log["total_reward"] if i == len(diagnostics)-1 else 0.0)
            auxiliary.append(row["credit"])
            alternatives.append([names.index(s["name"]) for s in d["alternative"]] if d.get("alternative") else None)
            samples.append({"sample": c["sample_id"], "span_id": i, "terminal_f1": log["total_reward"],
                            "labels": dict(Counter(q["cause"] for q in d["questions"])),
                            "sha": memory.trace.path.name, **row})
        batch.merge(local)
        save(out / "rollout.json", samples)
        print("collected", c["sample_id"], "span credit", [r["credit"] for r in log["reward_components"]], flush=True)

    np.savez_compressed(out / "on_policy.npz", states=np.stack(batch.states), candidates=np.stack(batch.op_embs),
                        actions=batch.actions, old_log_probs=batch.log_probs, old_values=batch.values,
                        new_op_masks=batch.new_op_masks, dones=batch.dones, upstream=upstream,
                        credit=batch.rewards, terminal=terminal)
    initial = copy.deepcopy(trainer.controller.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    initial_steps = trainer.total_steps
    margins_before = log_margins(trainer, batch, alternatives)
    result = {"arms": {}, "margins_before": margins_before}
    # Reward vectors differ only in these treatments. No counterfactual transition enters the buffer.
    treatments = {"B0_upstream": upstream, "B1_terminal": terminal, "C1_credit": list(batch.rewards)}
    for name, rewards in treatments.items():
        set_seed(args.seed + 1000)
        trainer.controller.load_state_dict(initial)
        trainer.optimizer.load_state_dict(copy.deepcopy(optimizer))
        trainer.total_steps = initial_steps
        batch.rewards = list(rewards)
        raw_returns, advantages = batch.compute_returns_and_advantages(trainer.controller.gamma, trainer.controller.gae_lambda)
        update = trainer.update_controller(batch)
        margins_after = log_margins(trainer, batch, alternatives)
        os.environ["MEMSKILL_USAGE_LOG"] = str(out / (name + "_validation_usage.jsonl"))
        scores = evaluate_bank(trainer, bank, held, out / name)
        assert versions == [skill_spec(op) for op in trainer.operation_bank.get_candidate_operations()]
        torch.save(trainer.controller.state_dict(), out / name / "controller.pt")
        summaries = {c["sample_id"]: mean(r["f1"] for r in scores["answers"] if r["sample_id"] == c["sample_id"])
                     for c in held}
        result["arms"][name] = {"validation_f1": scores["f1"], "count": scores["count"], "by_conversation": summaries,
                                "update": update, "returns": raw_returns.tolist(), "advantages": advantages.tolist(),
                                "margins_after": margins_after}
        save(out / "results.json", result)
        print(name, scores["count"], scores["f1"], flush=True)


if __name__ == "__main__":
    run()
