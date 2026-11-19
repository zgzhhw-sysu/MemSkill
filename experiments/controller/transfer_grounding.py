"""Zero-shot LongMemEval-S transfer of a frozen LoCoMo span-credit gate."""
import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import sys

from experiments.controller.paired_grounding import load_arm_weights, score
from experiments.controller.paired_policy import PairedPolicy
from src.cli import set_seed
from src.data_processing.datasets import load_dataset
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.episode_trace import skill_spec
from src.evaluation_cache import history_embeddings
from src.trainer import get_trainer


def external_histories(data, splits, formal, validation=None):
    if validation is None:
        rows = [data[i] for i in splits["test"]]
        if [r["question_id"] for r in rows] != formal["ids"]:
            raise ValueError("Test histories differ from the locked order")
        spec = formal
    else:
        by_id = {data[i]["question_id"]: data[i] for i in splits["val"]}
        rows = [by_id[sid] for sid in validation["ids"]]
        families = {r["question_id"].removesuffix("_abs") for r in rows}
        excluded = {sid.removesuffix("_abs") for sid in formal["ids"]}
        ordinary = [r for r in rows if not r["question_id"].endswith("_abs")]
        if (len(families) != len(rows) or families & excluded or len(rows) - len(ordinary) != validation["abstentions"]
                or Counter(r["question_type"] for r in ordinary) != validation["answerable_types"]):
            raise ValueError("External validation strata or question families differ from the manifest")
        spec = validation
    if sum(len(r["haystack_sessions"]) for r in rows) != spec["history_sessions"]:
        raise ValueError("Incomplete external histories")
    return rows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--study-arm", choices=("native", "terminal", "credit", "always", "rate", "count", "all"), required=True)
    parser.add_argument("--study-external-validation", action="store_true")
    parser.add_argument("--study-reverse", action="store_true")
    parser.add_argument("--study-writer-tokenizer", type=Path, help="Local writer model directory for context budgeting")
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
    dev = json.loads(cohort_file.with_name("paired_external_validation.json").read_text()) if options.study_external_validation else None
    held = external_histories(data, json.loads(split.read_text()), cohort, dev)
    assert [args.mem_top_k, args.mem_top_k_eval] == protocol["retrieval_k"]
    assert args.model == protocol["model"] and args.action_top_k == 3 and args.temperature == 0
    args.include_abstentions = True
    set_seed(args.seed)
    config = AgenticMemoryConfig().update_from_args(args)
    config.share_serial_encoders = args.inference_workers == args.inference_session_workers == 1
    trainer = get_trainer(args, config)
    trainer.load_checkpoint(args.load_checkpoint)
    if options.study_writer_tokenizer:
        from src.writer_context import WriterContext
        trainer.executor.writer_context = WriterContext(options.study_writer_tokenizer)
    target = "capture_activity_details"
    trainer.operation_bank.get_operation(target).source_rule = "attributed_turn_pairs"
    skills = [skill_spec(op) for op in trainer.operation_bank.get_candidate_operations()]
    if skills != protocol["skills"]:
        raise ValueError("Training and transfer skill libraries differ")
    trainer.controller = PairedPolicy(trainer.controller, [s["name"] for s in skills].index(target))
    phase = "longmemeval-validation" if dev else "longmemeval"
    arms = ("credit", "count", "terminal", "rate", "always", "native") if options.study_arm == "all" else (options.study_arm,)
    weights = {arm: load_arm_weights(out, arm) for arm in arms}
    for c in reversed(held) if options.study_reverse else held:
        sid = str(c["question_id"])
        if Path(sid).name != sid:
            raise ValueError("Invalid dataset ID")
        with (history_embeddings() if len(arms) > 1 else nullcontext()) as stats:
            for arm in arms:
                dest = out / phase / arm
                dest.mkdir(exist_ok=True, parents=True)
                with (dest / (sid + ".lock")).open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    score(trainer, c, dest / sid, arm, weights[arm])
            if stats is not None:
                print(sid, "encoder batches", stats(), flush=True)


if __name__ == "__main__":
    main()
