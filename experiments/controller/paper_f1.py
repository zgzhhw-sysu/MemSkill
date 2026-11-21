"""Verify the MemSkill paper test cohorts and score native --eval-f1-only outputs."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocols") / "memskill_paper_f1.json"


def read_locked_json(path, expected_hash):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return json.loads(data)


def load_split_cohort(dataset, path, spec, split="test"):
    cohort = spec
    if split != "test":
        if dataset != "locomo" or split not in spec["development_splits"]:
            raise ValueError(f"Unsupported {dataset} split: {split}")
        cohort = spec["development_splits"][split]
    data = read_locked_json(path, spec["data_sha256"])
    questions = {}
    if dataset == "locomo":
        indices = spec["test_indices"] if split == "test" else cohort["indices"]
        samples = [data[i] for i in indices]
        for dialogue in samples:
            for i, qa in enumerate(dialogue["qa"]):
                if qa["category"] != spec["excluded_category"]:
                    questions[dialogue["sample_id"], i] = qa
    else:
        splits = read_locked_json(ROOT / spec["split_file"], spec["split_sha256"])
        samples = [data[i] for i in splits["test"]]
        for row in samples:
            sample_id = row["question_id"]
            if not sample_id.endswith(spec["excluded_id_suffix"]):
                questions[sample_id, 0] = {"question": row["question"], "answer": row["answer"]}
    keys = json.dumps(list(questions), separators=(",", ":")).encode()
    digest = hashlib.sha256(keys).hexdigest()
    if len(questions) != cohort["scored_questions"] or digest != cohort["scored_key_sha256"]:
        raise ValueError(f"{dataset}/{split}: question IDs differ from the locked cohort")
    return samples, questions


def load_cohort(dataset, path, spec):
    return load_split_cohort(dataset, path, spec)[1]


def question_scores(dataset, questions, answer_files):
    from src.eval import get_evaluator

    evaluator = get_evaluator(dataset, args=None)
    scores = {}
    for path in answer_files:
        rows = json.loads(path.read_text())["answers"]
        for row in rows:
            key = (row["sample_id"], row["qa_index"])
            if type(key[1]) is not int or key not in questions:
                raise ValueError(f"Unexpected question ID {key!r} in {path}")
            if key in scores:
                raise ValueError(f"Duplicate question ID {key!r} in {path}")
            prediction = row["prediction"]
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction for {key!r} in {path}")
            qa = questions[key]
            gold = evaluator.get_ground_truth(qa)
            scores[key] = float(evaluator.compute_f1(prediction, gold, qa))
    if len(scores) != len(questions):
        missing = questions.keys() - scores.keys()
        raise ValueError(f"Missing {len(missing)} question(s), including {min(missing)!r}")
    return scores


def score_answers(dataset, questions, answer_files):
    return 100 * mean(question_scores(dataset, questions, answer_files).values())


def run_baseline(dataset, data, output_dir, engine_args, defaults):
    fixed = {"dataset": dataset, "data_file": data.resolve(),
             "out_file": output_dir / "result.json", "save_dir": output_dir / "checkpoints",
             "log_dir": output_dir / "logs", **defaults}
    switches = ["--eval-only", "--eval-f1-only", "--skip-load-snapshot-manager"]
    for name, value in fixed.items():
        switches.extend(["--" + name.replace("_", "-"), str(value)])
    output_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "main.py", *engine_args, *switches], cwd=ROOT, check=True)
    return fixed["out_file"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("locomo", "longmemeval"))
    parser.add_argument("data", type=Path, help="Original, full dataset JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--answers", nargs="+", type=Path,
                      help="One native report, or disjoint reports from the same run")
    mode.add_argument("--run-baseline", nargs=argparse.REMAINDER, metavar="ARG",
                      help="New output directory, then main.py checkpoint/model/API options")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    spec = protocol["datasets"][args.dataset]
    try:
        questions = load_cohort(args.dataset, args.data, spec)
        report = {"dataset": args.dataset, "questions": len(questions),
                  "scored_key_sha256": spec["scored_key_sha256"]}
        answers = args.answers
        if args.run_baseline is not None:
            if not args.run_baseline or args.run_baseline[0].startswith("--"):
                parser.error("--run-baseline requires a new output directory")
            output = Path(args.run_baseline[0]).resolve()
            answers = [run_baseline(args.dataset, args.data, output, args.run_baseline[1:],
                                    protocol["inference_defaults"])]
        if answers:
            report["f1_percent"] = score_answers(args.dataset, questions, answers)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"MemSkill evaluation failed with exit code {exc.returncode}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
