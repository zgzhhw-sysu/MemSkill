"""Evaluate the native trainer's frozen pre/post revision with one trained controller on the full test split."""
import copy
import hashlib
import json
import os
from pathlib import Path

from designer_pilot import save
from main import load_dataset, set_seed, split_data
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.episode_trace import skill_spec
from src.operation_bank import OperationBank
from src.revision_validation import compare_revisions, evaluate_bank
from src.trainer import get_trainer


def run():
    args = get_agentic_memory_args()
    out = Path(args.out_file).parent
    out.mkdir(parents=True, exist_ok=False)
    _, _, test = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    set_seed(args.seed)
    proposal_file = os.environ.get("MEMSKILL_ACCEPTED_PROPOSAL")
    args.skip_load_snapshot_manager = bool(proposal_file)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    if proposal_file:
        path = Path(proposal_file)
        pending = json.loads(path.read_text())["proposal"]
        arm = path.name.removesuffix("_proposal.json")
        decision = json.loads((path.parent / "calibration" / arm / "decision.json").read_text())
        if not decision["accept"] or not all(decision["exposure"].values()):
            raise ValueError("External proposal must have passed calibration with actual exposure")
        pending["validation"] = decision
        prior = trainer.operation_bank
        current = OperationBank.from_dict(copy.deepcopy(prior.to_dict()), encoder=trainer.op_encoder)
        if not trainer.designer.apply_evolution(current, copy.deepcopy(pending)):
            raise ValueError("Proposal no longer applies to this checkpoint")
    else:
        # Before and after banks in the native snapshot share the same trained PPO controller.
        manager = trainer.snapshot_manager
        if len(manager.snapshots) != 1:
            raise ValueError("This paired evaluation expects one completed native evolution")
        pending = manager.pending_evolution_result
        if not (pending and pending.get("validation", {}).get("accept")):
            raise ValueError("Checkpoint contains no accepted calibrated proposal")
        prior = OperationBank.from_dict(copy.deepcopy(manager.snapshots[0].operation_bank_dict), encoder=trainer.op_encoder)
        current = trainer.operation_bank
    sample_id = os.environ.get("MEMSKILL_TEST_SAMPLE_ID")
    if sample_id:
        test = [c for c in test if c["sample_id"] == sample_id]
        if len(test) != 1:
            raise ValueError("Unknown test conversation")
    ids = {c["sample_id"]: [i for i, q in enumerate(c["qa"]) if int(q["category"]) != 5] for c in test}
    spec = {"checkpoint_sha256": hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest(),
            "test_questions": ids, "comparison": "frozen pre-/post-Designer banks, identical checkpoint controller",
            "before_skills": sorted(prior.operations), "after_skills": sorted(current.operations),
            "model": args.model, "api_base": args.api_base, "seed": args.seed, "session_mode": args.session_mode,
            "action_top_k": args.action_top_k, "max_new_tokens": args.max_new_tokens, "proposal": pending}
    spec["changed_versions"] = {op.name: skill_spec(op)["version"] for op in current.get_all_operations()
                                if op.name not in prior.operations or skill_spec(op) != skill_spec(prior.operations[op.name])}
    save(out / "protocol.json", spec)
    reports = {}
    for label, bank in (("before", prior), ("after", current)):
        os.environ["MEMSKILL_USAGE_LOG"] = str(out / (label + "_usage.jsonl"))
        scores = evaluate_bank(trainer, bank, test, out / label)
        expected = {(sid, i) for sid, indices in ids.items() for i in indices}
        if {(r["sample_id"], r["qa_index"]) for r in scores["answers"]} != expected:
            raise ValueError("Incomplete test coverage")
        reports[label] = scores
    paired = compare_revisions(reports["before"], reports["after"])
    save(out / "comparison.json", {"before_f1": reports["before"]["f1"], "after_f1": reports["after"]["f1"],
                                   "count": reports["before"]["count"], "paired": paired})


if __name__ == "__main__":
    run()
