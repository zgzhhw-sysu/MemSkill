"""Verify the MemSkill paper test cohorts and score native --eval-f1-only outputs."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


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


def credit_selection(model, paper):
    if "repair_rule" in model:
        rule = paper["repair_credit"]
        if model["repair_rule"] != rule or model["treatments"] != rule["treatments"]:
            raise ValueError("Frozen repair-aware policy differs from the registered training rule")
        return rule
    return paper["state_credit_test"]


def verified_policy_changes(directory, summary, candidate="span"):
    """Recount Controller exposure from the actual deployed policy and ordered skill traces."""
    changes = {"probability_changes": 0, "membership_changes": 0, "joint_changes": 0}
    for sid, reported in summary["exposure"][candidate].items():
        history = directory / candidate / sid
        audit = json.loads((history / "audit.json").read_bytes())
        decisions = read_locked_json(history / "policy-comparison.json", audit["artifacts"]["policy-comparison.json"])
        trace, = (history / "traces").glob("*.jsonl")
        counts = dict.fromkeys(changes, 0)
        with trace.open() as stream:
            events = (event for line in stream if (event := json.loads(line))["kind"] == "span")
            for expected_span, (decision, event) in enumerate(zip(decisions, events, strict=True)):
                chosen = [skill["name"] for skill in event["selected_skills"]]
                different = decision["native_probabilities"] != event["policy"]["first_draw_probabilities"]
                if (event["sample"] != sid or event["span_id"] != expected_span or decision["changed_logits"] != event["policy"]["logits"]
                        or chosen != decision["selected_top_k"] or different != decision["first_draw_changed"]):
                    raise ValueError(f"Recorded Controller decision differs from the forward trace for {sid}")
                membership = set(chosen) != set(decision["native_top_k"])
                counts["probability_changes"] += different
                counts["membership_changes"] += membership
                counts["joint_changes"] += different and membership
        if (len(decisions) != reported["spans"] or len(decisions) != audit["spans"]
                or any(counts[k] != reported[k] or counts[k] != audit[k] for k in ("probability_changes", "membership_changes"))):
            raise ValueError(f"Reported Controller exposure differs from the trace counts for {sid}")
        for key, value in counts.items():
            changes[key] += value
    return changes


def validation_decision(directory, paper):
    """Decide from independent validation receipts before opening any held-out answers."""
    cohort = paper["datasets"]["locomo"]["development_splits"]["validation"]
    source = directory / "upstream-score.json"
    receipt_bytes = source.read_bytes()
    receipt = json.loads(receipt_bytes)
    protocol = read_locked_json(directory / "protocol.json", receipt["protocol_sha256"])
    summary = read_locked_json(directory / "comparison.json", receipt["comparison_sha256"])
    frozen_sha = hashlib.sha256((directory / "frozen.json").read_bytes()).hexdigest()
    selection = credit_selection(read_locked_json(directory / "frozen.json", frozen_sha), paper)
    if receipt["frozen_sha256"] != frozen_sha:
        raise ValueError("Validation frozen weights differ from the author-score receipt")
    if (receipt["dataset"] != "locomo" or receipt["split"] != "validation"
            or receipt["scorer_commit"] != paper["upstream_revision"]
            or protocol["split"] != "validation" or protocol["question_keys"] != cohort["scored_key_sha256"]
            or protocol["state_credit_sha256"] != frozen_sha or summary["split"] != "validation"
            or receipt["questions"] != cohort["scored_questions"] or summary["questions"] != receipt["questions"]):
        raise ValueError("Selection requires the complete locked LoCoMo validation cohort")
    arm = selection["candidate"]
    scores = receipt["f1_percent"]
    differences = {other: summary["paired"][other]["difference_points"] for other in selection["comparators"]}
    if any(not 0 <= scores[name] <= 100 for name in (arm, *differences)):
        raise ValueError("Validation F1 must be a finite percentage")
    for other, delta in differences.items():
        if not math.isclose(delta, scores[arm] - scores[other], rel_tol=0, abs_tol=1e-10):
            raise ValueError("Validation contrast differs from the independently rescored answers")
    exposure = summary["exposure"][arm]
    changes = {key: sum(h[key] for h in exposure.values()) for key in ("probability_changes", "membership_changes")}
    verified = receipt["candidate_changes"]
    if (any(verified[k] != n for k, n in changes.items()) or type(verified["joint_changes"]) is not int
            or not 0 <= verified["joint_changes"] <= min(changes.values())):
        raise ValueError("Validation exposure differs from the verified Controller decisions")
    return {"selected": all(d > 1e-10 for d in differences.values()) and verified["joint_changes"] > 0,
            "candidate": arm, "validation_questions": receipt["questions"], "difference_points": differences, **verified,
            "frozen_sha256": frozen_sha, "validation_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "rule": selection}


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


def verify_credit_scores(data, directory, native, paper, validation=None):
    """Recompute all four arms using the locked author's evaluator in an isolated CLI process."""
    settings_file = directory / "protocol.json"
    summary_file = directory / "comparison.json"
    settings = json.loads(settings_file.read_bytes())
    split = settings.get("split", "test")
    frozen = settings.pop("state_credit_sha256")
    read_locked_json(directory / "frozen.json", frozen)
    if split == "test":
        if validation is None:
            raise ValueError("A test receipt requires the original validation directory")
        expected = validation_decision(validation, paper)
        actual = read_locked_json(directory / "validation-decision.json", settings.pop("validation_decision_sha256"))
        if not expected["selected"] or actual != expected or frozen != expected["frozen_sha256"]:
            raise ValueError("Held-out policy differs from the policy selected on validation")
    if split not in ("validation", "test") or settings != json.loads((native / "protocol.json").read_bytes()):
        raise ValueError("The policy and original must use identical inference settings")
    _, questions = load_split_cohort("locomo", data, paper["datasets"]["locomo"], split)
    selection = credit_selection(read_locked_json(directory / "frozen.json", frozen), paper)
    candidate, controls = selection["candidate"], selection["comparators"]
    summary = json.loads(summary_file.read_bytes())
    histories = sorted({sid for sid, _ in questions})
    if (summary["questions"] != len(questions) or summary.get("split", "test") != split
            or set(summary["exposure"][candidate]) != set(histories)):
        raise ValueError("Incomplete Controller comparison")
    scores, percentages, files = {}, {}, {}
    revision = paper["upstream_revision"]
    archive = subprocess.check_output(["git", "-C", str(ROOT), "archive", revision, "src",
                                       "eval_utils.py", "llm_utils.py", "rag_utils.py", "prompts"])
    with TemporaryDirectory() as upstream:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(upstream, filter="data")
        sys.path.insert(0, upstream)
        try:
            from src.eval import get_evaluator
            import eval_utils
            evaluator = get_evaluator("locomo", None)
            for module in (eval_utils, sys.modules[type(evaluator).__module__]):
                if not Path(module.__file__).resolve().is_relative_to(upstream):
                    raise ValueError("Run the author scorer in a fresh Python process")
            for arm in (*controls, candidate):
                source = native if arm == "native" else directory
                paths = [source / arm / sid / "scores.json" for sid in histories]
                values = question_scores("locomo", questions, paths)
                scores[arm] = values
                files[arm] = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                percentages[arm] = 100 * mean(values.values())
        finally:
            sys.path.remove(upstream)

    def check(measured, reported):
        if not math.isclose(measured, reported, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Author score {measured} disagrees with reported score {reported}")

    for arm, values in scores.items():
        check(percentages[arm], summary[arm + "_f1"])
        for sid in histories:
            check(100 * mean(v for k, v in values.items() if k[0] == sid), summary["by_history"][sid][arm])
        if split == "test":
            published = paper["reported_f1"]["Llama-3.1-8B-Instruct"]["locomo"]
            check(percentages[arm] - published, summary["difference_from_paper_by_arm"][arm])
    for other in controls:
        deltas = [v - scores[other][k] for k, v in scores[candidate].items()]
        paired = summary["paired"][other]
        check(100 * mean(deltas), paired["difference_points"])
        if (sum(x > 0 for x in deltas), sum(x == 0 for x in deltas), sum(x < 0 for x in deltas)) != (
                paired["higher"], paired["equal"], paired["lower"]):
            raise ValueError("Paired question counts differ from the author's scores")
    return {"dataset": "locomo", "split": split, "scorer_commit": revision, "questions": len(questions),
            "candidate_changes": verified_policy_changes(directory, summary, candidate), "frozen_sha256": frozen,
            "protocol_sha256": hashlib.sha256(settings_file.read_bytes()).hexdigest(), "arm_scores_sha256": files,
            "f1_percent": percentages, "comparison_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest()}


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
    mode.add_argument("--verify-credit", nargs=2, type=Path, metavar=("POLICY_DIR", "NATIVE_DIR"),
                      help="Rescore the completed four-arm LoCoMo comparison using the original MemSkill commit")
    parser.add_argument("--validation-dir", type=Path, help="Verified validation outputs for a held-out comparison")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    spec = protocol["datasets"][args.dataset]
    try:
        if args.verify_credit:
            if args.dataset != "locomo":
                parser.error("--verify-credit requires LoCoMo")
            print(json.dumps(verify_credit_scores(args.data, *args.verify_credit, protocol, args.validation_dir), indent=2))
            return
        if args.validation_dir:
            parser.error("--validation-dir requires --verify-credit")
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
