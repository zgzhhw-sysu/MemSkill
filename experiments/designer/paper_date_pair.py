"""Evaluate the locked paper cohorts and collect original LoCoMo development trajectories."""
import copy
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from experiments.controller import file_hash, measured_cost, register
from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort, question_scores, read_locked_json, score_answers, validation_decision
from src.cli import set_seed
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.episode_trace import inspect_trace, skill_spec
from src.evaluation_cache import publish_response, read_response
from src.inference.text import infer_text_dataset_memories
from src.rag_utils import reuse_qwen_retriever
from src.revision_validation import compare_revisions
from src.trainer import get_trainer


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
CHECKPOINT_SHA = "fe784612b0d4028f431379e7cae618d743761175532f0f8feb14959ed37d0226"
SKILL = "capture_temporal_context"
VERSIONS = {"native": "d2be7d28f4fee9e180c628d2bf99ddf8dc663611cab3f40e755193e0a2b12779",
            "dated": "b7ac4330629e8940405853e49e5713202ba4806976a3ba9b295d19b8b63f4080"}


def save(path, obj):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


@contextmanager
def collect_policy_inputs(trainer, directory):
    """Save the actual inputs from each native decision and verify them against its trace."""
    import numpy as np
    import torch

    ops = trainer.operation_bank.get_candidate_operations()
    names = [op.name for op in ops]
    vectors = np.asarray([op.embedding for op in ops], dtype=np.float32)
    states = []

    def capture(module, inputs):
        state, candidates = inputs
        if not state.is_cuda or not candidates.is_cuda:
            raise ValueError("Development policy inputs must come from the GPU Controller")
        if not np.array_equal(candidates.detach().cpu().numpy(), vectors):
            raise ValueError("Candidate vectors changed within a frozen history")
        states.append(state.detach().cpu().numpy().copy())

    with trainer.controller.register_forward_pre_hook(capture):
        yield
    trace, = (directory / "traces").glob("*.jsonl")
    with trace.open() as stream:
        spans = [(r["span_id"], r["policy"], [op["name"] for op in r["selected_skills"]])
                 for line in stream if (r := json.loads(line))["kind"] == "span"]
    if not states or len(states) != len(spans):
        raise ValueError("Controller forwards and completed spans differ in count")
    path = directory / "controller-inputs.npz"
    np.savez_compressed(path, states=np.stack(states), candidate_vectors=vectors,
                        candidate_names=names, span_ids=[sid for sid, _, _ in spans])
    with np.load(path, allow_pickle=False) as saved, torch.no_grad():
        matrix = torch.from_numpy(saved["candidate_vectors"]).to(trainer.device)
        for state, (_, observed, selected) in zip(saved["states"], spans, strict=True):
            if names != observed["candidates"] or hashlib.sha256(state.tobytes()).hexdigest() != observed["state_sha256"]:
                raise ValueError("Archived input differs from the recorded selection state")
            policy = {}
            chosen, log_prob, _ = trainer.controller(torch.from_numpy(state).to(trainer.device), matrix,
                                                      deterministic=True, policy_trace=policy)
            if ([names[i] for i in chosen] != selected or log_prob != observed["ordered_log_probability"]
                    or policy["logits"] != observed["logits"]
                    or policy["first_draw_probabilities"] != observed["first_draw_probabilities"]):
                raise ValueError("Archived inputs did not reproduce the native policy")


def run_traced_history(trainer, sample, args, directory):
    local = copy.copy(args)
    local.out_file = str(directory / "scores.json")
    local.trace_dir = str(directory / "traces")
    os.environ["MEMSKILL_USAGE_LOG"] = str(directory / "usage.jsonl")
    infer_text_dataset_memories(trainer, [sample], local)


def history_spans(directory):
    trace, = (directory / "traces").glob("*.jsonl")
    with trace.open() as stream:
        return [r for line in stream if (r := json.loads(line))["kind"] == "span"]


def seed_writer_prefix(original, spans, start, source_cache, cache):
    """Initialize an arm with only the historical writes strictly before the intervention."""
    with (original / "usage.jsonl").open() as stream:
        writes = [r for line in stream if (r := json.loads(line))["kind"] == "executor"]
    if len(writes) != len(spans) or source_cache.resolve() == cache.resolve():
        raise ValueError("Independent repeat requires the original writer ledger and a separate cache")
    receipts = {}
    for event, usage in zip(spans[:start], writes[:start], strict=True):
        response, = event["executor_responses"]
        key = usage["cache_key"]
        receipt = read_response(source_cache, key)
        expected = {"key": key, "prompt_sha256": response["prompt_sha256"], "response": response["response"],
                    "tokens": [response["input_tokens"], response["output_tokens"]]}
        if (receipt is None or receipt["sha256"] != usage["response_sha256"]
                or receipt["payload"] != expected or expected["tokens"] != [usage["input"], usage["output"]]):
            raise ValueError("Prefix writer trace differs from the native response receipt")
        receipts[key] = receipt
    manifest = {"spans": start, "responses": {k: v["sha256"] for k, v in receipts.items()}}
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache.parent) as tmp:
            staged = Path(tmp) / "seeded"
            staged.mkdir()
            for key, receipt in receipts.items():
                row = receipt["payload"]
                publish_response(staged, key, row["prompt_sha256"], row["response"], *row["tokens"])
            save(staged / "prefix.json", manifest)
            staged.rename(cache)
    if (json.loads((cache / "prefix.json").read_bytes()) != manifest
            or any(read_response(cache, k) != v for k, v in receipts.items())):
        raise ValueError("Independent arm cache does not match its recorded prefix")


def measure_training_swap(trainer, sample, args, root, original, questions, start, repeat=0):
    """Measure one training action with the native end-to-end writer, retrieval and reader."""
    from src.replay import swap_boundary_logits, verify_native_swap

    spans = history_spans(original)
    sid = trainer.data_processor.get_sample_id(sample)
    if not 0 <= start < len(spans):
        raise ValueError("Requested span is outside the native history")
    event = spans[start]
    turns = set(event["source_turns"])
    linked = sorted(i for (_, i), qa in questions.items() if turns.intersection(qa.get("evidence") or []))
    panel = json.loads(PROTOCOL.read_text())["training_boundary_swap"]["histories"].get(sid)
    registered = panel is not None and panel["span"] == start
    if registered and panel != {"span": start, "total_spans": len(spans), "questions": len(questions),
                                "linked_question_indices": linked, "source_sha256": hashlib.sha256(event["source"].encode()).hexdigest()}:
        raise ValueError("Measured training span differs from its preregistered source and question IDs")
    order = ("swap", "native") if repeat % 2 else ("native", "swap")
    source_cache = Path(os.environ["MEMSKILL_GREEDY_EVAL_CACHE"]) if repeat else None
    arms = order if repeat else ("swap",)
    for arm in arms:
        directory = root / arm / sid
        directory.parent.mkdir(exist_ok=True)
        if repeat:
            cache = root.parent / "greedy-cache" / arm
            seed_writer_prefix(original, spans, start, source_cache, cache)
        if directory.exists():
            continue
        with TemporaryDirectory(dir=directory.parent) as tmp:
            staged = Path(tmp) / sid
            staged.mkdir()
            try:
                if repeat:
                    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
                with swap_boundary_logits(trainer.controller, spans, start) if arm == "swap" else nullcontext():
                    run_traced_history(trainer, sample, args, staged)
            finally:
                if repeat:
                    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(source_cache)
            stats = inspect_trace(staged / "traces", sid)
            observed = history_spans(staged)
            measured_cost(staged, observed, len(questions))
            question_scores("locomo", questions, [staged / "scores.json"])
            if arm == "swap":
                verify_native_swap(spans, observed, start)
            else:
                if (len(observed) != len(spans) or observed[:start] != spans[:start]
                        or any(observed[start][k] != event[k] for k in
                               ("source", "sample", "span_id", "source_turns", "retrieved", "refs", "policy", "selected_skills", "skills"))):
                    raise ValueError("Repeated native arm differs from the original before the target write")
                save(staged / "audit.json", stats)
            staged.rename(directory)

    directory = root / "swap" / sid
    control_dir = root / "native" / sid if repeat else original
    native = history_spans(control_dir) if repeat else spans
    changed = history_spans(directory)
    record = verify_native_swap(native, changed, start)
    control = question_scores("locomo", questions, [control_dir / "scores.json"])
    after = question_scores("locomo", questions, [directory / "scores.json"])
    deltas = {key[1]: after[key] - control[key] for key in questions}
    cost = {"native": measured_cost(control_dir, native, len(control)),
            "swap": measured_cost(directory, changed, len(after))}
    record.update(sample_id=sid, span_id=start, questions=len(deltas), cost=cost,
                  registered_span=registered, linked_question_indices=linked,
                  native_f1=100 * mean(control.values()), swap_f1=100 * mean(after.values()),
                  difference_points=100 * mean(deltas.values()), f1_deltas={str(i): d for i, d in deltas.items()},
                  **inspect_trace(directory / "traces", sid))
    trace, = (directory / "traces").glob("*.jsonl")
    record["artifacts"] = {str(p.relative_to(directory)): file_hash(p)
                           for p in (trace, directory / "usage.jsonl", directory / "scores.json")}
    if repeat:
        for arm in order:
            arm_cache = root.parent / "greedy-cache" / arm
            seeded = json.loads((arm_cache / "prefix.json").read_bytes())["responses"]
            with (root / arm / sid / "usage.jsonl").open() as stream:
                usage = [json.loads(line) for line in stream]
            if any(row["kind"] != "executor" or not row["cache_hit"] or seeded.get(row["cache_key"]) != row["response_sha256"]
                   for row in usage[:start]):
                raise ValueError("Independent arm did not replay its historical prefix")
            suffix = usage[start:]
            for row in suffix:
                receipt = read_response(arm_cache, row["cache_key"])
                if receipt is None or receipt["sha256"] != row["response_sha256"]:
                    raise ValueError("Independent arm is missing a measured post-fork response")
            cost[arm]["postfork"] = {"requests": len(suffix), "generated_this_attempt": sum(not r["cache_hit"] for r in suffix),
                                    "distinct_new_requests": len({r["cache_key"] for r in suffix} - seeded.keys()),
                                    "requests_matching_prefix": sum(r["cache_key"] in seeded for r in suffix)}
        record.update(repeat=repeat, execution_order=list(order), native_scores_sha256=file_hash(control_dir / "scores.json"),
                      native_usage_sha256=file_hash(control_dir / "usage.jsonl"))

    measurement = directory / "measurement.json"
    if measurement.exists():
        if json.loads(measurement.read_text()) != record:
            raise ValueError("Published training measurement differs from its original and swap artifacts")
    else:
        save(measurement, record)


def summarize_if_complete(dataset, output, questions, types, active_arms, published, split="test", native_root=None):
    histories = dict.fromkeys(sid for sid, _ in questions)
    reports = {arm: [output.parent / arm / sid / "scores.json" for sid in histories] for arm in active_arms}
    if native_root is not None:
        reports["native"] = [native_root / "native" / sid / "scores.json" for sid in histories]
    if any(not path.exists() for paths in reports.values() for path in paths):
        return
    scores_by_arm = {arm: question_scores(dataset, questions, paths) for arm, paths in reports.items()}
    by_type = {}
    for arm, scores in scores_by_arm.items():
        grouped = {}
        for (sid, index), score in scores.items():
            group = types[sid] if types else str(questions[sid, index]["category"])
            grouped.setdefault(group, []).append(score)
        for group, values in grouped.items():
            by_type.setdefault(group, {"questions": len(values)})[arm] = 100 * mean(values)
    native = 100 * mean(scores_by_arm["native"].values())
    summary = {"questions": len(questions), "native_f1": native}
    if split == "test":
        summary.update(paper_f1=published, difference_from_paper_points=native - published)
    else:
        summary["split"] = split
    summary["f1_by_type"] = by_type
    if "dated" in scores_by_arm:
        arms = {arm: {"answers": [{"sample_id": sid, "qa_index": i, "f1": v} for (sid, i), v in scores.items()]}
                for arm, scores in scores_by_arm.items()}
        dated = 100 * mean(scores_by_arm["dated"].values())
        summary.update(dated_f1=dated, gain_f1_points=dated - native,
                       paired=compare_revisions(arms["native"], arms["dated"]))
    if "span" in scores_by_arm:
        for arm in active_arms:
            summary[arm + "_f1"] = 100 * mean(scores_by_arm[arm].values())
        if split == "test":
            summary["difference_from_paper_by_arm"] = {
                arm: summary[arm + "_f1"] - published for arm in scores_by_arm
            }
        summary["by_history"] = {sid: {arm: 100 * mean(v for k, v in vals.items() if k[0] == sid)
                                       for arm, vals in scores_by_arm.items()} for sid in histories}
        summary["paired"] = {}
        from experiments.controller.turn_key_credit import paired_interval
        keys = ("spans", "eligible_spans", "probability_changes", "selection_changes", "membership_changes", "incoming_selections")
        summary["exposure"] = {}
        for arm in active_arms:
            exposure = {}
            for sid in histories:
                audit = json.loads((output.parent / arm / sid / "audit.json").read_bytes())
                exposure[sid] = {k: audit[k] for k in keys}
            summary["exposure"][arm] = exposure
        for other in ("native", "terminal", "rate"):
            deltas = [scores_by_arm["span"][k] - scores_by_arm[other][k] for k in questions]
            sizes = [sum(k[0] == sid for k in questions) for sid in histories]
            differences = [row["span"] - row[other] for row in summary["by_history"].values()]
            summary["paired"][other] = {"history_bootstrap_95_points": paired_interval(sizes, differences), "difference_points": 100 * mean(deltas), "higher": sum(d > 0 for d in deltas),
                                         "lower": sum(d < 0 for d in deltas), "equal": sum(d == 0 for d in deltas)}
    save(output, summary)


def run(args, root, rank, workers):
    paper = json.loads(PROTOCOL.read_text())
    spec = paper["datasets"][args.dataset]
    split = os.environ.get("MEMSKILL_PAPER_SPLIT", "test")
    credit_base = os.environ.get("MEMSKILL_STATE_CREDIT_BASE")
    if credit_base:
        if (split not in ("validation", "test") or args.dataset != "locomo" or workers != 1
                or os.environ.get("MEMSKILL_SWAP_HISTORY")):
            raise ValueError("Frozen state-credit comparison requires the LoCoMo validation or test split")
        validation = Path(credit_base + "-locomo-credit-validation") / "locomo"
        if split == "test":
            decision = validation_decision(validation, paper)
            if not decision["selected"]:
                raise ValueError("The frozen span policy did not qualify on validation")
            if root.resolve() == validation.resolve():
                raise ValueError("Test output must be distinct from validation")
    samples, questions = load_split_cohort(args.dataset, Path(args.data_file), spec, split)
    if split != "test" and (os.environ.get("MEMSKILL_PAPER_NATIVE_ONLY") != "1"
                            or os.environ.get("MEMSKILL_TRACE_NATIVE_POLICY") != "1" or args.device != "cuda"):
        raise ValueError("Development collection requires native GPU inference and policy traces")
    if file_hash(Path(args.load_checkpoint)) != CHECKPOINT_SHA:
        raise ValueError("Unexpected conversation Controller checkpoint")
    model_info = json.loads(Path(os.environ["MEMSKILL_MODEL_MANIFEST"]).read_text())
    if (model_info["model"], model_info["revision"]) != (MODEL, REVISION):
        raise ValueError("Unexpected model snapshot")
    with urlopen(args.api_base.rstrip("/") + "/models", timeout=10) as response:
        if [m["id"] for m in json.load(response)["data"]] != [MODEL]:
            raise ValueError("Unexpected reader/writer service")
    settings = paper["inference_defaults"].copy()
    launcher = os.environ.get("MEMSKILL_LONGMEMEVAL_LAUNCHER") == "1"
    if launcher:
        reference = paper["longmemeval_launcher"]
        if (args.dataset != "longmemeval" or split != "test" or args.encode_batch_size != reference["encode_batch_size"]
                or os.environ.get("MEMSKILL_PAPER_NATIVE_ONLY") != "1"):
            raise ValueError("LongMemEval launcher settings require the complete native batch-64 test cohort")
        settings["action_top_k"] = reference["action_top_k"]
    for name, value in settings.items():
        setattr(args, name, value)
    args.model = MODEL
    args.eval_only = args.eval_f1_only = args.skip_load_snapshot_manager = True
    args.inference_workers = args.inference_session_workers = 1
    args.temperature, args.seed, args.max_new_tokens = 0.0, 42, 2048
    config = AgenticMemoryConfig().update_from_args(args)
    config.share_serial_encoders = True
    memory_search = os.environ.get("MEMSKILL_MEMORY_SEARCH", "numpy")
    if memory_search not in {"numpy", "cuda", "cuda-skylakex"}:
        raise ValueError("Unknown memory search device")
    if memory_search != "numpy":
        import torch
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA memory search requires the encoder GPU lease")
        torch.set_float32_matmul_precision("highest")
        if memory_search == "cuda-skylakex":
            memory_search = "cuda-skylakex-fp32-v1"
        else:
            memory_search = "cuda-native-ties-v1"
    plan = {"dataset": args.dataset,
            "model": model_info, "backend": os.environ["MEMSKILL_BACKEND_ID"],
            "checkpoint": CHECKPOINT_SHA, "skills": VERSIONS, "settings": settings,
            "seed": args.seed, "temperature": args.temperature, "max_writer_tokens": args.max_new_tokens,
            "history_workers": workers, "memory_search": memory_search,
            "encode_batch_size": args.encode_batch_size, "flash_attention": config.use_flash_attn}
    if launcher:
        plan["upstream_launcher"] = reference["script"]
    if split == "test":
        plan["test_keys"] = spec["scored_key_sha256"]
    else:
        plan.update(split=split, question_keys=spec["development_splits"][split]["scored_key_sha256"],
                    policy_inputs="verified-fp32-v1")
    if os.environ.get("MEMSKILL_TRACE_NATIVE_POLICY") == "1":
        plan["native_policy_trace"] = "first-draw-logits-v1"
    prefill_only = os.environ.get("MEMSKILL_QWEN_PREFILL_ONLY") == "1"
    if prefill_only:
        import torch
        if args.device != "cuda":
            raise ValueError("The Qwen prefill comparison requires a GPU")
        free, total = torch.cuda.mem_get_info()
        # Longer LongMemEval batches can exceed the LoCoMo encoder allocation.
        cap_mib = 5500 if launcher else 3328
        budget = min(cap_mib * 2**20, free - 2**30)
        if budget < 2250 * 2**20:
            raise RuntimeError("Insufficient free GPU memory for the batch-64 encoder")
        torch.cuda.set_per_process_memory_fraction(budget / total)
        plan["encoder_kv_cache"] = False
    swap_history = os.environ.get("MEMSKILL_SWAP_HISTORY")
    if swap_history:
        if (split != "train" or args.dataset != "locomo" or workers != 1
                or swap_history not in {sid for sid, _ in questions}):
            raise ValueError("A single-span swap requires one native LoCoMo training history")
        start = int(os.environ["MEMSKILL_SWAP_SPAN"])
        repeat = int(os.environ.get("MEMSKILL_SWAP_REPEAT", "0"))
        if repeat < 0:
            raise ValueError("Repeat index must be nonnegative")
        native_root = Path(os.environ["MEMSKILL_NATIVE_TRAIN_ROOT"]).resolve()
        if root == native_root or json.loads((native_root / "protocol.json").read_text()) != plan:
            raise ValueError("Training measurement and native reference must have identical inference settings and distinct outputs")
        original = native_root / "native" / swap_history
        trace, = (original / "traces").glob("*.jsonl")
        plan["training_swap"] = {"history": swap_history, "span": start, "intervention": "exchange-kth-and-next-logits",
                                 "trace_sha256": file_hash(trace), "scores_sha256": file_hash(original / "scores.json")}
        if repeat:
            plan["training_swap"].update(repeat=repeat, arms="independent-postfork-v1",
                                         usage_sha256=file_hash(original / "usage.jsonl"))
    native_reference = None
    if credit_base:
        from experiments.controller.controller_credit_study import freeze_paper_credit, learned_skill_boundary
        native_run = "-locomo-batch64" if split == "test" else "-locomo-validation"
        native_reference = Path(credit_base + native_run) / "locomo"
        if root.resolve() == native_reference.resolve():
            raise ValueError("Use distinct directories for the policy comparison and native baseline")
        if json.loads((native_reference / "protocol.json").read_bytes()) != plan or not (native_reference / "baseline.json").is_file():
            raise ValueError("Comparison needs the complete native run with identical settings")
        if split == "test":
            model = read_locked_json(validation / "frozen.json", decision["frozen_sha256"])
            register(root / "frozen.json", model)
            if file_hash(root / "frozen.json") != decision["frozen_sha256"]:
                raise ValueError("The test run requires the validated frozen weights")
            register(root / "validation-decision.json", decision)
            plan["validation_decision_sha256"] = file_hash(root / "validation-decision.json")
        else:
            model = freeze_paper_credit(Path(credit_base + "-locomo-train") / "locomo", Path(credit_base + "-locomo-train-swap"),
                                       root / "frozen.json", args.device, args.data_file)
        if model["checkpoint"] != CHECKPOINT_SHA:
            raise ValueError("Training and evaluation Controller checkpoints differ")
        plan["state_credit_sha256"] = file_hash(root / "frozen.json")
    protocol = root / "protocol.json"
    with (root / "protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if protocol.exists():
            existing = json.loads(protocol.read_text())
            existing.setdefault("memory_search", "numpy")
            if existing != plan:
                raise ValueError("Existing runs use a different protocol")
        else:
            save(protocol, plan)
    histories = list(dict.fromkeys(sid for sid, _ in questions))
    assigned = set(histories[rank::workers])
    if not assigned:
        return
    set_seed(args.seed)
    trainer = get_trainer(args, config)
    if prefill_only:
        base = trainer.state_encoder._base_encoder
        if base is not trainer.op_encoder._base_encoder or base is not trainer.memory_bank_encoder._base_encoder:
            raise ValueError("Expected one shared Qwen encoder for the serial comparison")
        base.model[0].auto_model.config.use_cache = False
    trainer.load_checkpoint(args.load_checkpoint)
    skill = trainer.operation_bank.operations[SKILL]
    types = {row["question_id"]: row["question_type"] for row in samples} if args.dataset == "longmemeval" else {}
    active_arms = ("native",) if os.environ.get("MEMSKILL_PAPER_NATIVE_ONLY") == "1" else tuple(VERSIONS)
    if credit_base:
        active_arms = (*model["treatments"], "rate")
    with reuse_qwen_retriever(trainer.state_encoder._base_encoder, args.retriever):
        if swap_history:
            sample, = [s for s in samples if trainer.data_processor.get_sample_id(s) == swap_history]
            subset = {k: q for k, q in questions.items() if k[0] == swap_history}
            return measure_training_swap(trainer, sample, args, root, original, subset, start, repeat)
        for index, sample in enumerate(samples):
            sid = trainer.data_processor.get_sample_id(sample)
            if sid not in assigned:
                continue
            for arm in active_arms:
                # The edit changes only the executor hook. Keep the checkpoint's skill embeddings in place.
                skill.source_rule = None if arm != "dated" else "dated_relative_turns"
                if skill_spec(skill)["version"] != VERSIONS["dated" if arm == "dated" else "native"]:
                    raise ValueError(f"Unexpected {arm} skill version")
                directory = root / arm / sid
                directory.parent.mkdir(exist_ok=True)
                if not directory.exists():
                    with TemporaryDirectory(dir=directory.parent) as temp:
                        staged = Path(temp) / sid
                        staged.mkdir()
                        subset = {k: q for k, q in questions.items() if k[0] == sid}
                        context = collect_policy_inputs(trainer, staged) if split != "test" else nullcontext()
                        if credit_base:
                            context = learned_skill_boundary(trainer, model, arm)
                        with context as decisions:
                            run_traced_history(trainer, sample, args, staged)
                        stats = inspect_trace(staged / "traces", sid)
                        if credit_base:
                            spans = history_spans(staged)
                            for decision, event in zip(decisions, spans, strict=True):
                                if decision["changed_logits"] != event["policy"]["logits"]:
                                    raise ValueError("Deployed logits differ from the actual Controller forward")
                                selected = [s["name"] for s in event["selected_skills"]]
                                decision["selected_top_k"] = selected
                                decision["first_draw_changed"] = decision["native_probabilities"] != event["policy"]["first_draw_probabilities"]
                            stats["eligible_spans"] = sum(d["eligible"] for d in decisions)
                            stats["incoming_selections"] = sum(model["spec"]["incoming"] in d["selected_top_k"] for d in decisions)
                            stats["probability_changes"] = sum(d["first_draw_changed"] for d in decisions)
                            stats["selection_changes"] = sum(d["selected_top_k"] != d["native_top_k"] for d in decisions)
                            stats["membership_changes"] = sum(set(d["selected_top_k"]) != set(d["native_top_k"]) for d in decisions)
                            save(staged / "policy-comparison.json", decisions)
                            stats["artifacts"] = {"policy-comparison.json": file_hash(staged / "policy-comparison.json")}
                            stats["cost"] = measured_cost(staged, spans, len(subset))
                        elif split != "test":
                            stats["policy_inputs_sha256"] = file_hash(staged / "controller-inputs.npz")
                        stats["f1_percent"] = score_answers(args.dataset, subset, [staged / "scores.json"])
                        save(staged / "audit.json", stats)
                        staged.rename(directory)
                else:
                    inspect_trace(directory / "traces", sid)
                if credit_base or split != "test":
                    audit = json.loads((directory / "audit.json").read_bytes())
                    if credit_base:
                        if file_hash(directory / "policy-comparison.json") != audit["artifacts"]["policy-comparison.json"]:
                            raise ValueError(f"Archived policy comparison changed for {sid}")
                    elif file_hash(directory / "controller-inputs.npz") != audit["policy_inputs_sha256"]:
                        raise ValueError(f"Archived policy inputs changed for {sid}")
                print(f"Complete {args.dataset} {index+1}/{len(samples)} {sid} {arm}", flush=True)
    published = paper["reported_f1"]["Llama-3.1-8B-Instruct"][args.dataset]
    summarize_if_complete(args.dataset, Path(args.out_file), questions, types, active_arms, published, split, native_reference)


if __name__ == "__main__":
    args = get_agentic_memory_args()
    root = Path(args.out_file).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("MEMSKILL_HISTORY_RANK", "0"))
    workers = int(os.environ.get("MEMSKILL_HISTORY_WORKERS", "1"))
    if not 0 <= rank < workers:
        raise ValueError("Invalid history shard")
    with (root / f"run-{rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, root, rank, workers)
