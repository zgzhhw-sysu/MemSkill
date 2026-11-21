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

from experiments.controller.paper_f1 import PROTOCOL, load_cohort, score_answers
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
    tmp = path.with_suffix(".tmp")
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


def run(args, root):
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
    plan = {"dataset": args.dataset, "test_keys": spec["scored_key_sha256"],
            "model": model_info, "backend": os.environ["MEMSKILL_BACKEND_ID"],
            "checkpoint": CHECKPOINT_SHA, "skills": VERSIONS, "settings": paper["inference_defaults"],
            "seed": args.seed, "temperature": args.temperature, "max_writer_tokens": args.max_new_tokens,
            "encode_batch_size": args.encode_batch_size, "flash_attention": config.use_flash_attn}
    protocol = root / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != plan:
        raise ValueError("Existing runs use a different protocol")
    save(protocol, plan)
    set_seed(args.seed)
    trainer = get_trainer(args, config)
    trainer.load_checkpoint(args.load_checkpoint)
    skill = trainer.operation_bank.operations[SKILL]
    _, _, test = split_data(load_dataset(args.data_file, args.dataset), args.dataset)
    histories = {sid for sid, _ in questions}
    types = {row["question_id"]: row["question_type"] for row in test} if args.dataset == "longmemeval" else {}
    reports = {arm: [] for arm in VERSIONS}
    with reuse_qwen_retriever(trainer.state_encoder._base_encoder, args.retriever):
        for index, sample in enumerate(test):
            sid = trainer.data_processor.get_sample_id(sample)
            if sid not in histories:
                continue
            for arm, version in VERSIONS.items():
                # The edit changes only the executor hook. Keep the checkpoint's skill embeddings in place.
                skill.source_rule = None if arm == "native" else "dated_relative_turns"
                if skill_spec(skill)["version"] != version:
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
                reports[arm].append(directory / "scores.json")
                print(f"Complete {args.dataset} {index+1}/{len(test)} {sid} {arm}", flush=True)
    arms = {arm: {"f1_percent": score_answers(args.dataset, questions, paths),
                  "answers": [row for path in paths for row in json.loads(path.read_text())["answers"]]}
            for arm, paths in reports.items()}
    comparison = compare_revisions(arms["native"], arms["dated"])
    by_type = {}
    for arm, report in arms.items():
        grouped = {}
        for row in report["answers"]:
            sid = row["sample_id"]
            group = types[sid] if types else str(questions[sid, row["qa_index"]]["category"])
            grouped.setdefault(group, []).append(row["f1"])
        for group, scores in grouped.items():
            by_type.setdefault(group, {"questions": len(scores)})[arm] = 100 * mean(scores)
    gain = arms["dated"]["f1_percent"] - arms["native"]["f1_percent"]
    if abs(gain - 100 * comparison["delta"]) > 1e-8:
        raise ValueError("Stored scores differ from reference-based paired F1")
    save(Path(args.out_file), {"questions": len(questions), "native_f1": arms["native"]["f1_percent"],
                               "dated_f1": arms["dated"]["f1_percent"],
                               "gain_f1_points": gain, "paired": comparison, "f1_by_type": by_type})


if __name__ == "__main__":
    args = get_agentic_memory_args()
    root = Path(args.out_file).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, root)
