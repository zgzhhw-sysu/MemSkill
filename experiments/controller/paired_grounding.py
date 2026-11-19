"""Train and evaluate a paired, fixed-K Controller from reproducible span interventions."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

from experiments.controller.paired_policy import PairedPolicy, fit_policy
from experiments.designer.designer_pilot import prefix, save
from main import load_dataset, set_seed, split_data
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.controller import PPOBuffer
from src.episode_trace import skill_spec
from src.revision_validation import evaluate_bank
from src.trainer import get_trainer


def window(conversation, start, length):
    subset = prefix(conversation, start + length - 1)
    keys = [f"session_{i}" for i in range(start, start + length)]
    content = {k: subset["conversation"][k] for name in keys for k in (name, name + "_date_time")}
    turns = {r["dia_id"] for k in keys for r in content[k]}
    return dict(subset, conversation=content, qa=[q for q in subset["qa"] if set(q["evidence"]) <= turns])


def collect(trainer, convo, seed, dest):
    if dest.exists():
        return json.loads((dest / "rows.json").read_text())
    policy = trainer.controller
    policy.mode = "explore"
    policy.decisions.clear()
    set_seed(seed)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        work = Path(tmp) / "episode"
        work.mkdir()
        trainer.args.trace_dir = str(work / "traces")
        os.environ["MEMSKILL_USAGE_LOG"] = str(work / "usage.jsonl")
        buffer = PPOBuffer()
        sessions, length, precompute = trainer._prepare_sessions(convo)
        memory = trainer._initialize_memory_bank()
        log = trainer.train_episode(convo, memory, sessions, buffer, episode_length=length,
                                    precompute_embeddings=precompute)
        trace = [json.loads(line) for line in memory.trace.path.read_text().splitlines()]
        diagnostics = [e["diagnosis"] for e in trace if e["kind"] == "span_credit"]
        rows = []
        for i, (decision, diagnosis) in enumerate(zip(policy.decisions, diagnostics, strict=True)):
            chosen = decision["grounded"] if decision["use"] else decision["plain"]
            assert buffer.actions[i] == chosen and abs(buffer.log_probs[i] - math.log(0.5)) < 1e-6
            other = decision["plain"] if decision["use"] else decision["grounded"]
            if diagnosis.get("alternative"):
                specs = [skill_spec(op) for op in trainer.operation_bank.get_candidate_operations()]
                assert [specs[j] for j in other] == diagnosis["alternative"]
            delta = diagnosis["mean_controller_delta"]
            assert all(q["controller_delta"] == 0 or q["cause"] == "selection" for q in diagnosis["questions"])
            rows.append(dict(decision, episode=dest.name, span=i, delta=delta, terminal=log["total_reward"],
                             old_log_probability=buffer.log_probs[i], causes=dict(Counter(q["cause"] for q in diagnosis["questions"]))))
        save(work / "rows.json", rows)
        save(work / "questions.json", [q["original_index"] for q in convo["qa"]])
        work.rename(dest)
    return rows


def reuse_identical_choices(dest, arm, weights, signature):
    """Identical choices at each observed state imply the same deterministic rollout by induction."""
    for reference in sorted(dest.parent.parent.glob("*/" + dest.name)):
        if reference == dest or not (reference / "inputs.json").is_file():
            continue
        source = json.loads((reference / "inputs.json").read_text())
        if any(source.get(k) != v for k,v in signature.items() if k not in {"arm","weights"}):
            continue
        decisions = json.loads((reference / "decisions.json").read_text())
        if not decisions:
            continue
        revised = []
        for row in decisions:
            logit = row["base_logit"]
            if weights is not None:
                state = torch.tensor(row["state"], dtype=weights.dtype)
                vector = torch.cat((state.new_ones(1), torch.nn.functional.normalize(state, dim=0)))
                logit += float(vector @ weights)
            use = arm == "always" or logit >= 0
            if use != row["use"] or (arm != "always" and abs(logit) < 1e-5):
                break
            revised.append(dict(row, prob=1.0 if arm == "always" else float(torch.sigmoid(torch.tensor(logit)))))
        if len(revised) != len(decisions):
            continue
        with TemporaryDirectory(dir=dest.parent) as tmp:
            directory = Path(tmp) / "eval"
            directory.mkdir()
            for name in ("scores.json", "question_ids.json"):
                os.link(reference / name, directory / name)
            shutil.copytree(reference / "traces", directory / "traces", copy_function=os.link)
            save(directory / "inputs.json", signature)
            save(directory / "decisions.json", revised)
            save(directory / "reuse.json", {"reference":str(reference.relative_to(dest.parent.parent)),
                "checked_decisions":len(revised), "score_sha256":hashlib.sha256((reference/"scores.json").read_bytes()).hexdigest()})
            (directory / "usage.jsonl").touch()
            directory.rename(dest)
        print(arm, dest.name, "verified identical to", reference.parent.name, flush=True)
        return True
    return False


def score(trainer, conversation, dest, arm, weights):
    signature = {
        "arm": arm,
        "weights": None if weights is None else hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
        "conversation": hashlib.sha256(json.dumps(conversation, sort_keys=True).encode()).hexdigest(),
        "skills": [skill_spec(op)["version"] for op in trainer.operation_bank.get_candidate_operations()],
        "reader": trainer.args.model, "writer_tokens": trainer.args.max_new_tokens,
        "retriever": trainer.args.retriever, "backend": os.environ.get("MEMSKILL_BACKEND_ID", trainer.args.api_base),
    }
    if dest.exists():
        if json.loads((dest / "inputs.json").read_text()) != signature:
            raise ValueError(f"Existing evaluation uses different inputs at {dest}")
        return
    if reuse_identical_choices(dest, arm, weights, signature):
        return
    policy = trainer.controller
    policy.weights = weights
    policy.mode = arm
    policy.decisions.clear()
    with TemporaryDirectory(dir=dest.parent) as tmp:
        directory = Path(tmp) / "eval"
        os.environ["MEMSKILL_USAGE_LOG"] = str(Path(tmp) / "usage.jsonl")
        report = evaluate_bank(trainer, trainer.operation_bank, [conversation], directory)
        answers = report["answers"]
        qa_list = trainer.data_processor.get_qa_list(conversation)
        expected = dict(trainer.evaluator.filter_qa_list(qa_list))
        if len(answers) != len(expected) or {r["qa_index"] for r in answers} != expected.keys():
            raise RuntimeError("Missing or duplicate scored questions")
        for row in answers:
            qa = expected[row["qa_index"]]
            actual = trainer.evaluator.compute_f1(row["prediction"], trainer.evaluator.get_ground_truth(qa), qa)
            if min(row["tokens"]) <= 0 or not math.isclose(actual, row["f1"]):
                raise RuntimeError("Invalid prediction or score")
        save(directory / "inputs.json", signature)
        save(directory / "decisions.json", policy.decisions)
        save(directory / "question_ids.json", [q.get("original_index", i) for i,q in enumerate(qa_list)])
        (Path(tmp) / "usage.jsonl").rename(directory / "usage.jsonl")
        directory.rename(dest)
        print(arm, report["answers"][0]["sample_id"], len(answers), report["f1"], flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--study-phase", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--study-arm", choices=["native", "terminal", "credit", "always", "all"], default="all")
    parser.add_argument("--study-reverse", action="store_true", help="Collect independent episodes in reverse order")
    options, remainder = parser.parse_known_args()
    sys.argv[1:] = remainder
    args = get_agentic_memory_args()
    if (args.dataset != "locomo" or not args.api or args.enable_designer or not args.eval_f1_only
            or args.action_top_k != 3 or args.span_credit_rule_version != 2 or args.temperature != 0):
        raise ValueError("This protocol requires LoCoMo, a frozen K=3 policy, greedy 8B responses and evidence rule 2")
    out = options.study_dir.resolve()
    out.mkdir(exist_ok=True, parents=True)
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    bank = trainer.operation_bank
    target = "capture_activity_details"
    bank.get_operation(target).source_rule = "attributed_turn_pairs"
    specs = [skill_spec(op) for op in bank.get_candidate_operations()]
    trainer.controller = PairedPolicy(trainer.controller, [s["name"] for s in specs].index(target))
    train, val, test = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    protocol = {
        "input_sha256": hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest(),
        "seed": args.seed, "skills": specs, "model": args.model, "writer_tokens": args.max_new_tokens,
        "retriever": args.retriever, "retrieval_k": [args.mem_top_k, args.mem_top_k_eval],
        "train": [c["sample_id"] for c in train], "validation": [c["sample_id"] for c in val],
        "test": [c["sample_id"] for c in test], "train_sessions": [[1, 2], [3, 4]],
        "sampling": "fair coin for one fixed-K substitution; same coin on branch-specific suffix state",
        "training_signals": "leave-one-episode-out centered terminal F1 in both arms; credit adds signed rule-2 single-slot F1 deltas",
        "validation_sessions": 6, "test_sessions": "all", "decision_threshold": "greedy, p>=0.5",
        "target": "at least +0.02 micro mean per-question F1 vs terminal and native, positive on each independent test dialogue",
        "weight": 1, "ridge": 0.01, "rule": 2, "span_budget": args.span_credit_spans, "question_budget": args.span_credit_questions,
    }
    path = out / "protocol.json"
    with (out / "protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError("Study input differs from the saved protocol")
        elif options.study_phase == "train":
            with TemporaryDirectory(dir=out) as tmp:
                temporary = Path(tmp) / path.name
                save(temporary, protocol)
                temporary.replace(path)
        else:
            raise ValueError("Run training first")
    if options.study_phase == "test":
        locked = json.loads((Path(__file__).with_name("protocols") / "paired_grounding_cohort.json").read_text())["primary"]
        if protocol["input_sha256"] != locked["file_sha256"] or protocol["test"] != locked["ids"]:
            raise ValueError("Test inputs differ from the locked LoCoMo cohort")
    if options.study_phase == "train":
        root = out / "train"
        root.mkdir(exist_ok=True)
        episodes = [(start, i, c) for start in (1, 3) for i, c in enumerate(train)]
        for start, i, c in reversed(episodes) if options.study_reverse else episodes:
            dest = root / f"{c['sample_id']}_s{start}"
            with (root / (dest.name + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                batch = collect(trainer, window(c, start, 2), args.seed + i*100 + start, dest)
            print(dest.name, [(r["use"], r["delta"]) for r in batch], flush=True)
        rows = [row for start, i, c in episodes
                for row in json.loads((root / f"{c['sample_id']}_s{start}" / "rows.json").read_text())]
        with (out / "fit.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for arm, weight in (("terminal", 0), ("credit", 1)):
                weights = fit_policy(rows, weight)
                with TemporaryDirectory(dir=out) as tmp:
                    temp = Path(tmp) / (arm + ".json")
                    save(temp, weights.tolist())
                    temp.replace(out / temp.name)
        print("Fitted", len(rows), "training spans;", sum(r["delta"] != 0 for r in rows), "nonzero credits", flush=True)
        return
    group = [prefix(c, 6) for c in val] if options.study_phase == "validation" else test
    root = out / options.study_phase
    for arm in (["native", "terminal", "credit", "always"] if options.study_arm == "all" else [options.study_arm]):
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        weights = torch.tensor(json.loads((out / (arm + ".json")).read_text())) if arm in ("credit", "terminal") else None
        for conversation in group:
            score(trainer, conversation, directory / conversation["sample_id"], arm, weights)


if __name__ == "__main__":
    main()
