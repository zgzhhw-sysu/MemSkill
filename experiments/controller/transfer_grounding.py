"""Zero-shot LongMemEval-S transfer of a frozen LoCoMo span-credit gate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import torch

from experiments.controller.paired_grounding import score
from experiments.controller.paired_policy import PairedPolicy
from main import load_dataset, set_seed
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.episode_trace import skill_spec
from src.trainer import get_trainer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--study-arm", choices=("native", "terminal", "credit", "always"), required=True)
    parser.add_argument("--study-reverse", action="store_true")
    options, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    args = get_agentic_memory_args()
    if args.dataset != "longmemeval" or args.session_mode != "full-session" or not args.eval_f1_only or args.enable_designer:
        raise ValueError("Transfer requires complete LongMemEval histories, frozen skills and F1 evaluation")
    out = options.study_dir.resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    cohort_file = Path(__file__).with_name("protocols") / "paired_grounding_cohort.json"
    cohort = json.loads(cohort_file.read_text())["external"]
    if hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest() != cohort["file_sha256"]:
        raise ValueError("LongMemEval-S dataset differs from the locked cohort")
    if hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest() != protocol["checkpoint_sha256"]:
        raise ValueError("Wrong base Controller")
    split = Path("data/longmemeval_s_splits.json")
    if hashlib.sha256(split.read_bytes()).hexdigest() != cohort["split_sha256"]:
        raise ValueError("Test split differs from the locked cohort")
    data = load_dataset(args.data_file, "longmemeval")
    held = [data[i] for i in json.loads(split.read_text())["test"]]
    assert [c["question_id"] for c in held] == cohort["ids"]
    assert sum(len(c["haystack_sessions"]) for c in held) == cohort["history_sessions"]
    assert [args.mem_top_k, args.mem_top_k_eval] == protocol["retrieval_k"]
    assert args.model == protocol["model"] and args.action_top_k == 3 and args.temperature == 0
    args.include_abstentions = True
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    target = "capture_activity_details"
    trainer.operation_bank.get_operation(target).source_rule = "attributed_turn_pairs"
    skills = [skill_spec(op) for op in trainer.operation_bank.get_candidate_operations()]
    if skills != protocol["skills"]:
        raise ValueError("Training and transfer skill libraries differ")
    trainer.controller = PairedPolicy(trainer.controller, [s["name"] for s in skills].index(target))
    arm = options.study_arm
    weights = torch.tensor(json.loads((out / (arm + ".json")).read_text())) if arm in ("credit", "terminal") else None
    dest = out / "longmemeval" / arm
    dest.mkdir(exist_ok=True, parents=True)
    for c in reversed(held) if options.study_reverse else held:
        sid = str(c["question_id"])
        if Path(sid).name != sid:
            raise ValueError("Invalid dataset ID")
        with (dest / (sid + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            score(trainer, c, dest / sid, arm, weights)


if __name__ == "__main__":
    main()
