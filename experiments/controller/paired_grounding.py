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

from experiments.controller.paired_policy import PairedPolicy, count_schedule, fit_policy, fit_anchored_credit, match_training_probability
from experiments.controller.resume_status import load_episode
from experiments.designer.designer_pilot import prefix, save
from src.cli import set_seed
from src.data_processing.datasets import load_dataset, split_data
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.controller import PPOBuffer
from src.episode_trace import skill_spec
from src.revision_validation import evaluate_bank
from src.trainer import get_trainer


def configure_policy(trainer, equal_sources):
    bank = trainer.operation_bank
    target = "capture_activity_details"
    alternative = "capture_temporal_context"
    bank.get_operation(target).source_rule = "dated_adjacent_turns" if equal_sources else "attributed_turn_pairs"
    if equal_sources:
        bank.get_operation(alternative).source_rule = "dated_single_turns"
    specs = [skill_spec(op) for op in bank.get_candidate_operations()]
    names = [spec["name"] for spec in specs]
    trainer.controller = PairedPolicy(trainer.controller, names.index(target),
                                      names.index(alternative) if equal_sources else None)
    return specs


def window(conversation, start, length):
    subset = prefix(conversation, start + length - 1)
    keys = [f"session_{i}" for i in range(start, start + length)]
    content = {k: subset["conversation"][k] for name in keys for k in (name, name + "_date_time")}
    turns = {r["dia_id"] for k in keys for r in content[k]}
    return dict(subset, conversation=content, qa=[q for q in subset["qa"] if set(q["evidence"]) <= turns])


def verify_linked_questions(events, questions):
    """Require every eligible factual answer and every question linked to either span."""
    from experiments.controller.probe_omitted import check_responses
    answers = [e for e in events if e["kind"] == "qa"]
    answered = sorted(int(e["query_id"].rsplit("_", 1)[1]) for e in answers if e["success"])
    if len(answers) != len(questions) or answered != list(range(len(questions))):
        raise ValueError("Missing or repeated factual training answer")
    spans = [e for e in events if e["kind"] == "span"]
    records = [e["diagnosis"] for e in events if e["kind"] == "span_credit"]
    for span, record in zip(spans, records, strict=True):
        turns = set(span["source_turns"])
        expected = [(i, q["original_index"]) for i, q in enumerate(questions) if turns.intersection(q["evidence"])]
        observed = [(q["qa_index"], q["original_index"]) for q in record["questions"]]
        if span["span_id"] != record["span_id"] or record["evidence_rule_version"] != 2 or observed != expected:
            raise ValueError("Incomplete or mismatched rule-2 question panel")
        check_responses(record)


def collect(trainer, convo, seed, dest):
    all_linked = getattr(trainer.args, "study_all_linked_questions", False)
    if dest.exists():
        if all_linked:
            trace, = (dest / "traces").glob("*.jsonl")
            verify_linked_questions([json.loads(line) for line in trace.read_bytes().splitlines()], convo["qa"])
        ids = [q["original_index"] for q in convo["qa"]]
        spans = sum(k.startswith("session_") and not k.endswith("_date_time") for k in convo["conversation"])
        return load_episode(dest, convo["sample_id"], spans, ids,
                            deployment_writer=getattr(trainer.args, "span_credit_deployment_writer", False))
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
        if all_linked:
            verify_linked_questions(trace, convo["qa"])
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
        if os.environ.get("MEMSKILL_BACKEND_ID"):
            save(work / "backend.json", {"inference": os.environ["MEMSKILL_BACKEND_ID"]})
        work.rename(dest)
    return rows


def reuse_identical_choices(dest, arm, weights, signature):
    """Identical choices at each observed state imply the same deterministic rollout by induction."""
    policy_fields = {"arm", "weights", "count_control"}
    shared = {k: v for k, v in signature.items() if k not in policy_fields}
    for reference in sorted(dest.parent.parent.glob("*/" + dest.name)):
        if reference == dest or not (reference / "inputs.json").is_file():
            continue
        source = json.loads((reference / "inputs.json").read_text())
        if {k: v for k, v in source.items() if k not in policy_fields} != shared:
            continue
        decisions = json.loads((reference / "decisions.json").read_text())
        schedule = signature["count_control"]["schedule"] if arm == "count" else None
        if not decisions or (schedule is not None and len(schedule) != len(decisions)):
            continue
        revised = []
        for index, row in enumerate(decisions):
            logit = row["base_logit"]
            if weights is not None:
                state = torch.tensor(row["state"], dtype=weights.dtype)
                vector = torch.cat((state.new_ones(1), torch.nn.functional.normalize(state, dim=0)))
                logit += float(vector @ weights)
            use = logit >= 0
            probability = float(torch.sigmoid(torch.tensor(logit)))
            if arm in ("always", "plain"):
                use = arm == "always"
                probability = float(use)
            if schedule is not None:
                use = schedule[index]
                probability = float(use)
            if use != row["use"] or (arm not in ("always", "plain", "count") and abs(logit) < 1e-5):
                break
            revised.append(dict(row, prob=probability))
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
    writer_context = getattr(getattr(trainer, "executor", None), "writer_context", None)
    if writer_context is not None:
        signature["writer_context"] = writer_context.signature
    schedule = None
    if arm == "count":
        study = dest.parents[2]
        credit = dest.parent.parent / "credit" / dest.name
        credit.parent.mkdir(exist_ok=True)
        with (credit.parent / (credit.name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            score(trainer, conversation, credit, "credit", load_arm_weights(study, "credit"))
        source = (credit / "decisions.json").read_bytes()
        seed = json.loads((study / "protocol.json").read_text())["seed"]
        schedule = count_schedule(seed, dest.name, [r["use"] for r in json.loads(source)])
        signature["count_control"] = {"schedule": schedule, "credit_decisions_sha256": hashlib.sha256(source).hexdigest()}
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
    policy.schedule = schedule
    with TemporaryDirectory(dir=dest.parent) as tmp:
        directory = Path(tmp) / "eval"
        os.environ["MEMSKILL_USAGE_LOG"] = str(Path(tmp) / "usage.jsonl")
        report = evaluate_bank(trainer, trainer.operation_bank, [conversation], directory)
        if schedule is not None and [d["use"] for d in policy.decisions] != schedule:
            raise ValueError("Count control did not execute its assigned span schedule")
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


def load_arm_weights(study, arm):
    if arm in ("native", "always", "plain", "count"):
        return None
    path = study / (arm + ".json")
    if arm == "rate" and not path.exists():
        # Older training runs saved this training-only control at their first validation.
        protocol = json.loads((study / "protocol.json").read_text())
        rows = [row for start, end in protocol["train_sessions"] for sid in protocol["train"]
                for row in load_episode(study / "train" / f"{sid}_s{start}", sid, end - start + 1,
                                        deployment_writer=protocol.get("deployment_writer", False))]
        weights = match_training_probability(rows, load_arm_weights(study, "terminal"), load_arm_weights(study, "credit"))
        with TemporaryDirectory(dir=study) as tmp:
            temporary = Path(tmp) / path.name
            save(temporary, weights.tolist())
            temporary.replace(path)
        return weights
    return torch.tensor(json.loads(path.read_text()), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--study-phase", required=True, choices=["train", "validation", "test", "probe-omitted", "probe-weaker", "replay-validation-choices", "training-factorial", "training-late-pairs"])
    parser.add_argument("--study-forecast", type=Path, help="Frozen training-only probabilities for four-path evaluation")
    parser.add_argument("--study-pair-limit", type=int, help="Run the first N registered later training pairs")
    parser.add_argument("--study-reference", type=Path, help="Completed development study whose observed credit choices to replay")
    parser.add_argument("--study-probe-through", type=int, choices=(4, 6, 8),
                        help="Last session of the completed training panel for the weaker-branch audit")
    parser.add_argument("--study-weaker-omitted", action="store_true",
                        help="Audit positive sampled-action credit in completed omitted-final question receipts")
    parser.add_argument("--study-probe-completed", action="store_true",
                        help="Probe saved training windows on their original backend while other windows are collecting")
    parser.add_argument("--study-share-probe-encoder", action="store_true",
                        help="Reuse the same frozen CUDA Qwen for serial omitted-question state, skill and retrieval calls")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--study-arm", choices=["native", "terminal", "credit", "always", "plain", "rate", "count", "all"], default="all")
    parser.add_argument("--study-reverse", action="store_true", help="Collect independent episodes in reverse order")
    parser.add_argument("--study-train-sessions", type=int, choices=(4, 8),
                        help="Use the first 4 or 8 sessions of each training dialogue, in disjoint pairs")
    parser.add_argument("--study-equal-sources", action="store_true",
                        help="Compare the activity and temporal skills with paired and single source turns in slot three")
    parser.add_argument("--study-deployment-writer", action="store_true",
                        help="Use evaluation-time writer prompts in training and counterfactual branches")
    parser.add_argument("--study-all-linked-questions", action="store_true",
                        help="Probe every evidence-linked training question in each two-session window")
    parser.add_argument("--study-center-credit", action="store_true",
                        help="Center and scale training states for the abstention-anchored correction")
    parser.add_argument("--study-anchor-abstentions", action="store_true",
                        help="Penalize probability changes at training states with no verified local credit")
    options, remainder = parser.parse_known_args()
    if options.study_probe_completed and options.study_phase != "probe-omitted":
        parser.error("--study-probe-completed requires --study-phase probe-omitted")
    if options.study_probe_completed and options.study_probe_through is not None:
        parser.error("--study-probe-completed checks the whole training protocol")
    if options.study_share_probe_encoder and options.study_phase != "probe-omitted":
        parser.error("--study-share-probe-encoder requires --study-phase probe-omitted")
    paired_trial = options.study_phase in ("training-factorial", "training-late-pairs")
    if options.study_pair_limit is not None and (options.study_phase != "training-late-pairs" or options.study_pair_limit < 1):
        parser.error("--study-pair-limit requires a positive limit and training-late-pairs")
    if paired_trial != bool(options.study_forecast):
        parser.error("--study-forecast is required exclusively for paired trajectory experiments")
    sys.argv[1:] = remainder
    args = get_agentic_memory_args()
    if (args.dataset != "locomo" or not args.api or args.enable_designer or not args.eval_f1_only
            or args.action_top_k != 3 or args.span_credit_rule_version != 2 or args.temperature != 0):
        raise ValueError("This protocol requires LoCoMo, a frozen K=3 policy, greedy 8B responses and evidence rule 2")
    out = options.study_dir.resolve()
    out.mkdir(exist_ok=True, parents=True)
    path = out / "protocol.json"
    saved_protocol = json.loads(path.read_text()) if path.exists() else {}
    args.span_credit_deployment_writer = options.study_deployment_writer or saved_protocol.get("deployment_writer", False)
    train_sessions = options.study_train_sessions
    if train_sessions is None:
        train_sessions = saved_protocol["train_sessions"][-1][-1] if saved_protocol else 4
    train_windows = [[start, start + 1] for start in range(1, train_sessions, 2)]
    train, val, test = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    args.study_all_linked_questions = options.study_all_linked_questions or saved_protocol.get("all_linked_questions", False)
    if args.study_all_linked_questions:
        if args.span_credit_spans < 2:
            parser.error("An exhaustive two-session window requires probing both spans")
        args.span_credit_questions = max(1, max(len(window(c, start, end - start + 1)["qa"])
                                               for c in train for start, end in train_windows))
    set_seed(args.seed)
    config = AgenticMemoryConfig().update_from_args(args)
    config.share_serial_encoders = args.inference_workers == args.inference_session_workers == 1
    if paired_trial:
        if args.device != "cuda" or not config.share_serial_encoders or torch.cuda.device_count() != 1:
            parser.error("training-factorial requires a single leased GPU and serial encoders")
        free, total = torch.cuda.mem_get_info()
        budget = min(int(total * .76), free - 2048 * 2**20)
        if budget < 2304 * 2**20:
            if options.study_phase == "training-late-pairs":
                print(f"Waiting for encoder headroom: CUDA reports {free//2**20} MiB free; need 4352 MiB", flush=True)
                raise SystemExit(75)
            raise RuntimeError("Insufficient free memory for the bounded factorial encoder")
        torch.cuda.set_per_process_memory_fraction(budget/total)
        print(f"Training factorial CUDA allocator limit {budget//2**20} MiB", flush=True)
    trainer = get_trainer(args, config)
    trainer.load_checkpoint(args.load_checkpoint)
    equal_sources = options.study_equal_sources or saved_protocol.get("equal_sources", False)
    specs = configure_policy(trainer, equal_sources)
    protocol = {
        "input_sha256": hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest(),
        "seed": args.seed, "skills": specs, "model": args.model, "writer_tokens": args.max_new_tokens,
        "retriever": args.retriever, "retrieval_k": [args.mem_top_k, args.mem_top_k_eval],
        "train": [c["sample_id"] for c in train], "validation": [c["sample_id"] for c in val],
        "test": [c["sample_id"] for c in test], "train_sessions": train_windows,
        "sampling": "fair coin for one fixed-K substitution; same coin on branch-specific suffix state",
        "training_signals": "leave-one-episode-out centered terminal F1 in both arms; credit adds signed rule-2 single-slot F1 deltas",
        "validation_sessions": 6, "test_sessions": "all", "decision_threshold": "greedy, p>=0.5",
        "target": "at least +0.02 micro mean per-question F1 vs terminal and native, positive on each independent test dialogue",
        "weight": 1, "ridge": 0.01, "rule": 2, "span_budget": args.span_credit_spans, "question_budget": args.span_credit_questions,
    }
    if args.study_all_linked_questions:
        protocol["all_linked_questions"] = True
    if options.study_center_credit or saved_protocol.get("center_credit"):
        if not (options.study_anchor_abstentions or saved_protocol.get("anchor_abstentions")):
            parser.error("--study-center-credit requires --study-anchor-abstentions")
        protocol["center_credit"] = True
    if equal_sources:
        protocol["equal_sources"] = True
    if args.span_credit_deployment_writer:
        protocol["deployment_writer"] = True
    if options.study_anchor_abstentions or saved_protocol.get("anchor_abstentions"):
        protocol["anchor_abstentions"] = True
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
    if paired_trial:
        from src.rag_utils import reuse_qwen_retriever
        with reuse_qwen_retriever(trainer.state_encoder._base_encoder, args.retriever):
            if options.study_phase == "training-late-pairs":
                from experiments.controller.late_pair_trial import run
                run(trainer, out, options.study_forecast, train, specs, options.study_pair_limit)
            else:
                from experiments.controller.factorial_rollout import run
                run(trainer, out, options.study_forecast, train, specs)
        return
    if options.study_phase == "replay-validation-choices":
        from experiments.controller.replay_validation_choices import replay
        if options.study_reference is None:
            parser.error("replay-validation-choices requires --study-reference")
        with (out / "validation-schedule.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            replay(trainer, out, options.study_reference, [prefix(c, protocol["validation_sessions"]) for c in val],
                   load_arm_weights(out, "credit"))
        return
    if options.study_phase == "test":
        locked = json.loads((Path(__file__).with_name("protocols") / "paired_grounding_cohort.json").read_text())["primary"]
        if protocol["input_sha256"] != locked["file_sha256"] or protocol["test"] != locked["ids"]:
            raise ValueError("Test inputs differ from the locked LoCoMo cohort")
    if options.study_phase in ("probe-omitted", "probe-weaker"):
        samples = {c["sample_id"]: c for c in train}
        through = options.study_probe_through if options.study_phase == "probe-weaker" else None
        through = train_windows[-1][1] if through is None else through
        if through not in {end for _, end in train_windows}:
            raise ValueError("Requested panel exceeds the training protocol")
        windows = ((window(samples[sid], start, end - start + 1), out / "train" / f"{sid}_s{start}")
                   for start, end in train_windows if end <= through for sid in protocol["train"])
        if options.study_phase == "probe-weaker":
            from experiments.controller.probe_weaker import probe_weaker_windows
            probe_weaker_windows(trainer, out, windows, through, omitted=options.study_weaker_omitted)
        else:
            from experiments.controller.probe_omitted import probe_training_windows
            from contextlib import nullcontext
            from src.rag_utils import reuse_qwen_retriever
            if options.study_share_probe_encoder and not config.share_serial_encoders:
                parser.error("--study-share-probe-encoder requires one serial inference worker")
            sharing = (reuse_qwen_retriever(trainer.state_encoder._base_encoder, args.retriever)
                       if options.study_share_probe_encoder else nullcontext())
            with sharing:
                probe_training_windows(trainer, out, windows, completed_only=options.study_probe_completed)
        return
    if options.study_phase == "train":
        root = out / "train"
        root.mkdir(exist_ok=True)
        episodes = [(start, i, c) for start, _ in train_windows for i, c in enumerate(train)]
        for start, i, c in reversed(episodes) if options.study_reverse else episodes:
            dest = root / f"{c['sample_id']}_s{start}"
            with (root / (dest.name + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                batch = collect(trainer, window(c, start, 2), args.seed + i*100 + start, dest)
            print(dest.name, [(r["use"], r["delta"]) for r in batch], flush=True)
        rows = [row for start, _, c in episodes
                for row in load_episode(root / f"{c['sample_id']}_s{start}", c["sample_id"], 2,
                                        deployment_writer=args.span_credit_deployment_writer)]
        with (out / "fit.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with torch.device(args.device):
                terminal = fit_policy(rows, 0)
                if protocol.get("anchor_abstentions"):
                    credit = fit_anchored_credit(rows, terminal, center_states=protocol.get("center_credit", False))
                else:
                    credit = fit_policy(rows, 1)
                rate = match_training_probability(rows, terminal, credit)
            print(f"Fitted terminal, credit and rate policies on {credit.device}", flush=True)
            for arm, weights in (("terminal", terminal), ("credit", credit), ("rate", rate)):
                with TemporaryDirectory(dir=out) as tmp:
                    temp = Path(tmp) / (arm + ".json")
                    save(temp, weights.tolist())
                    temp.replace(out / temp.name)
        print("Fitted", len(rows), "training spans;", sum(r["delta"] != 0 for r in rows), "nonzero credits", flush=True)
        return
    group = [prefix(c, 6) for c in val] if options.study_phase == "validation" else test
    root = out / options.study_phase
    arms = ["native", "terminal", "credit", "always", "rate"]
    if equal_sources:
        arms.append("plain")
    for arm in arms if options.study_arm == "all" else [options.study_arm]:
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        weights = load_arm_weights(out, arm)
        for conversation in group:
            dest = directory / conversation["sample_id"]
            with (directory / (dest.name + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                score(trainer, conversation, dest, arm, weights)


if __name__ == "__main__":
    main()
