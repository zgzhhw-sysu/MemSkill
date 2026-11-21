"""Run the frozen date skill and its original checkpoint on the locked paper test cohort."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from experiments.controller.paper_f1 import PROTOCOL, load_cohort, question_scores, score_answers
from src.cli import set_seed
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.data_processing.datasets import load_dataset, split_data
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


def summarize_if_complete(dataset, output, questions, types, active_arms, published):
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
    summary = {"questions": len(questions), "native_f1": native, "paper_f1": published,
               "difference_from_paper_points": native - published, "f1_by_type": by_type}
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
    questions = load_cohort(args.dataset, Path(args.data_file), spec)
    if hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest() != CHECKPOINT_SHA:
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
    if memory_search not in {"numpy", "cuda"}:
        raise ValueError("Unknown memory search device")
    if memory_search == "cuda":
        import torch
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA memory search requires the encoder GPU lease")
        torch.set_float32_matmul_precision("highest")
        memory_search = "cuda-native-ties-v1"
    plan = {"dataset": args.dataset, "test_keys": spec["scored_key_sha256"],
            "model": model_info, "backend": os.environ["MEMSKILL_BACKEND_ID"],
            "checkpoint": CHECKPOINT_SHA, "skills": VERSIONS, "settings": paper["inference_defaults"],
            "seed": args.seed, "temperature": args.temperature, "max_writer_tokens": args.max_new_tokens,
            "history_workers": workers, "memory_search": memory_search,
            "encode_batch_size": args.encode_batch_size, "flash_attention": config.use_flash_attn}
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
    trainer.load_checkpoint(args.load_checkpoint)
    skill = trainer.operation_bank.operations[SKILL]
    _, _, test = split_data(load_dataset(args.data_file, args.dataset), args.dataset)
    types = {row["question_id"]: row["question_type"] for row in test} if args.dataset == "longmemeval" else {}
    active_arms = ("native",) if os.environ.get("MEMSKILL_PAPER_NATIVE_ONLY") == "1" else tuple(VERSIONS)
    with reuse_qwen_retriever(trainer.state_encoder._base_encoder, args.retriever):
        for index, sample in enumerate(test):
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
                        infer_text_dataset_memories(trainer, [sample], local)
                        stats = inspect_trace(Path(local.trace_dir), sid)
                        subset = {k: q for k, q in questions.items() if k[0] == sid}
                        stats["f1_percent"] = score_answers(args.dataset, subset, [Path(local.out_file)])
                        save(staged / "audit.json", stats)
                        staged.rename(directory)
                print(f"Complete {args.dataset} {index+1}/{len(test)} {sid} {arm}", flush=True)
    published = paper["reported_f1"]["Llama-3.1-8B-Instruct"][args.dataset]
    summarize_if_complete(args.dataset, Path(args.out_file), questions, types, active_arms, published)


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
