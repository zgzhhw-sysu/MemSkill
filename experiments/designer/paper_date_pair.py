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

from experiments.controller import file_hash
from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort, question_scores, score_answers
from src.cli import set_seed
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.episode_trace import skill_spec
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


def inspect_trace(trace_dir, sample_id):
    paths = list(trace_dir.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"Expected one fresh history trace for {sample_id}")
    spans = applied = 0
    with paths[0].open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["kind"] != "span":
                continue
            spans += 1
            actions = event["parsed_actions"]
            if any("API call failed" in a["reasoning"] for a in actions):
                raise RuntimeError(f"Writer request failed in {sample_id}")
            dates = {a["memory_content"] for a in actions if a["success"] and a["reasoning"] == "dated_relative_turns"}
            applied += any(c["action"] == "INSERT" and c["after"] in dates for c in event["changes"])
    if not spans:
        raise ValueError(f"No memory construction trace for {sample_id}")
    return {"spans": spans, "spans_with_date_inserts": applied}


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


def summarize_if_complete(dataset, output, questions, types, active_arms, published, split="test"):
    histories = dict.fromkeys(sid for sid, _ in questions)
    reports = {arm: [output.parent / arm / sid / "scores.json" for sid in histories] for arm in active_arms}
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
    save(output, summary)


def run(args, root, rank, workers):
    paper = json.loads(PROTOCOL.read_text())
    spec = paper["datasets"][args.dataset]
    split = os.environ.get("MEMSKILL_PAPER_SPLIT", "test")
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
    for name, value in paper["inference_defaults"].items():
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
            "checkpoint": CHECKPOINT_SHA, "skills": VERSIONS, "settings": paper["inference_defaults"],
            "seed": args.seed, "temperature": args.temperature, "max_writer_tokens": args.max_new_tokens,
            "history_workers": workers, "memory_search": memory_search,
            "encode_batch_size": args.encode_batch_size, "flash_attention": config.use_flash_attn}
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
        # Leave 1 GiB free on the shared 251 card, and cap this encoder at 3.25 GiB.
        budget = min(3328 * 2**20, free - 2**30)
        if budget < 2250 * 2**20:
            raise RuntimeError("Insufficient free GPU memory for the batch-64 encoder")
        torch.cuda.set_per_process_memory_fraction(budget / total)
        plan["encoder_kv_cache"] = False
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
    with reuse_qwen_retriever(trainer.state_encoder._base_encoder, args.retriever):
        for index, sample in enumerate(samples):
            sid = trainer.data_processor.get_sample_id(sample)
            if sid not in assigned:
                continue
            for arm in active_arms:
                # The edit changes only the executor hook. Keep the checkpoint's skill embeddings in place.
                skill.source_rule = None if arm == "native" else "dated_relative_turns"
                if skill_spec(skill)["version"] != VERSIONS[arm]:
                    raise ValueError(f"Unexpected {arm} skill version")
                directory = root / arm / sid
                directory.parent.mkdir(exist_ok=True)
                if not directory.exists():
                    with TemporaryDirectory(dir=directory.parent) as temp:
                        staged = Path(temp) / sid
                        staged.mkdir()
                        local = copy.copy(args)
                        local.out_file = str(staged / "scores.json")
                        local.trace_dir = str(staged / "traces")
                        os.environ["MEMSKILL_USAGE_LOG"] = str(staged / "usage.jsonl")
                        with collect_policy_inputs(trainer, staged) if split != "test" else nullcontext():
                            infer_text_dataset_memories(trainer, [sample], local)
                        stats = inspect_trace(Path(local.trace_dir), sid)
                        if split != "test":
                            stats["policy_inputs_sha256"] = file_hash(staged / "controller-inputs.npz")
                        subset = {k: q for k, q in questions.items() if k[0] == sid}
                        stats["f1_percent"] = score_answers(args.dataset, subset, [Path(local.out_file)])
                        save(staged / "audit.json", stats)
                        staged.rename(directory)
                elif split != "test":
                    audit = json.loads((directory / "audit.json").read_bytes())
                    if file_hash(directory / "controller-inputs.npz") != audit["policy_inputs_sha256"]:
                        raise ValueError(f"Archived policy inputs changed for {sid}")
                print(f"Complete {args.dataset} {index+1}/{len(samples)} {sid} {arm}", flush=True)
    published = paper["reported_f1"]["Llama-3.1-8B-Instruct"][args.dataset]
    summarize_if_complete(args.dataset, Path(args.out_file), questions, types, active_arms, published, split)


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
