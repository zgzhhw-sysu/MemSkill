"""Compare trace-only and verified-feedback Designers, with independent revision checks."""
import copy
import json
import os
from pathlib import Path

from experiments.designer.designer_pilot import prefix, save
from main import load_dataset, split_data, set_seed
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.designer import DesignerCase
from src.episode_trace import evidence_turns
from src.operation_bank import OperationBank
from src.replay import reader_probes, restore_text
from src.revision_validation import check_exposure, compare_revisions, evaluate_bank, validate_proposal
from src.trainer import get_trainer


def prepare_cases(trainer, source, fit):
    by_id = {c["sample_id"]: c for c in fit}
    trace_cases, verified_cases = [], []
    for filename in sorted(source.glob("*.jsonl")):
        records = [json.loads(line) for line in filename.read_text().splitlines()]
        sample = by_id[records[0]["sample"]]
        events = [row for row in records if row["kind"] == "span"]
        live = restore_text(events)
        queries = {r["query_id"]: r for r in records if r["kind"] == "qa"}
        for replay in (r for r in records if r["kind"] == "replay"):
            qid = replay["query_id"]
            q = queries[qid]
            qa = sample["qa"][int(qid.rsplit("_", 1)[1])]
            if q["question"] != qa["question"]:
                raise ValueError(f"Trace and dataset disagree for {qid}")
            ids = [mid for mid, _ in q["refs"]]
            case = DesignerCase(query_id=qid, question=q["question"], prediction=q["prediction"],
                                ground_truth=str(trainer.evaluator.get_ground_truth(qa)), evidence=qa["evidence"],
                                conversation_id=sample["sample_id"], f1_score=q["f1"], category=qa["category"],
                                retrieved_indices=[list(live).index(mid) for mid in ids],
                                retrieved_memories=[live[mid] for mid in ids])
            event = next(e for e in events if e["span_id"] == replay["span_id"])
            turns = evidence_turns(sample["conversation"], qa["evidence"])
            case.replay = {"span_id": event["span_id"], "trace": filename.name,
                           "source_turns": turns, "selected_skills": [(s["name"], s["version"]) for s in event["skills"]],
                           "actual_writes": [c for c in event["changes"] if c["after"]][:3]}
            trace_cases.append(copy.deepcopy(case))
            result = reader_probes(trainer, case, qa, events, turns)
            case.replay.update(rule_control_write=replay["unchanged_write_reproduced"],
                               tested_rule=replay["intervention"], tested_skill=replay["skill"],
                               rule_effect_on_final_f1=replay["f1_delta"], reader_probes=result)
            verified_cases.append(case)
    if {c.conversation_id for c in verified_cases} != set(by_id):
        raise ValueError("Missing training traces")
    return trace_cases, verified_cases


def study():
    args = get_agentic_memory_args()
    out = Path(args.out_file).parent
    out.mkdir(exist_ok=True, parents=True)
    protocol = out / "protocol.json"
    resume = os.environ.get("MEMSKILL_RESUME_VALIDATION") == "1"
    if protocol.exists() and not resume:
        raise FileExistsError("Choose a fresh PILOT_DIR for an independent revision study")
    train, _, _ = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    fit = [prefix(c, 2) for c in train[:2]]
    calibration, audit = ([prefix(c, 3) for c in part] for part in (train[2:4], train[4:6]))
    source_trial = args.designer_dated_source_rule
    if source_trial:
        _, validation, _ = split_data(load_dataset(args.data_file, "locomo"), "locomo")
        audit = []
        for sample in validation:
            earlier = {q["original_index"] for q in prefix(sample, 2)["qa"]}
            held = prefix(sample, 4)
            held["qa"] = [q for q in held["qa"] if q["original_index"] not in earlier]
            audit.append(held)
    source = Path(os.environ.get("MEMSKILL_FIT_TRACES", out.parent / "pilot_suffix/traces"))
    archive = os.environ.get("MEMSKILL_ARCHIVED_REVISIONS") == "1"
    specification = {"design": "one proposal per arm, native checkpoint, deterministic policy, no PPO during comparison",
                    "fit_trace_source": str(source), "fit": [c["sample_id"] for c in fit], "seed": args.seed,
                    "calibration": {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in calibration},
                    "audit": {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in audit},
                    "decision": "positive mean paired F1 and nonnegative mean change in each calibration conversation",
                    "primary": "question-level mean F1 on both audit prefixes; show ungated and deployed results"}
    if archive:
        specification["proposal_files"] = [str(out.parent / study / (arm + "_edit.json"))
                                            for study in ("pilot", "pilot_suffix")
                                            for arm in ("original_designer", "replay_designer")]
        specification["control_repeat"] = True
    if source_trial:
        specification["writer_modes"] = ["calendar instruction", "retain dated source turns"]
        specification["probe_file"] = str(out.parent / "calendar_check/diagnoses.json")
        specification["api"] = args.api_base
    if resume:
        if json.loads(protocol.read_text()) != specification:
            raise ValueError("Cannot resume with a different dataset or protocol")
    else:
        save(protocol, specification)
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    trainer.controller.eval()
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "additional_probe_usage.jsonl")
    if resume or archive:
        raw = verified = []
    elif source_trial:
        verified = [DesignerCase.from_dict(c) for c in json.loads(Path(specification["probe_file"]).read_text())]
        raw = []
    else:
        raw, verified = prepare_cases(trainer, source, fit)
        save(out / "diagnoses.json", [c.to_dict() for c in verified])
    arms = {"calendar": verified} if args.designer_calendar_proposal else {"trace_only": raw, "verified": verified}
    if source_trial:
        arms = {"text_instruction": verified, "dated_source": verified}
    if archive:
        arms = {Path(f).parent.name + "_" + Path(f).stem: [] for f in specification["proposal_files"]}
    proposals = {}
    for arm, cases in arms.items():
        if resume:
            proposals[arm] = json.loads((out / (arm + "_proposal.json")).read_text())["proposal"]
            continue
        if archive:
            file = next(Path(f) for f in specification["proposal_files"] if Path(f).parent.name + "_" + Path(f).stem == arm)
            original = json.loads(file.read_text())
            proposals[arm] = original["proposal"]
            save(out / (arm + "_proposal.json"), {"source": str(file), "proposal": original["proposal"]})
            continue
        if source_trial:
            args.designer_dated_source_rule = arm == "dated_source"
            proposal = trainer.designer.run_evolution(trainer.operation_bank, {"analysis_prompt": "", "analysis_cases": cases})
            proposals[arm] = proposal
            save(out / (arm + "_proposal.json"), {"source": specification["probe_file"], "proposal": proposal})
            continue
        os.environ["MEMSKILL_USAGE_LOG"] = str(out / (arm + "_design_usage.jsonl"))
        # Identical case IDs and common factual trace. Only this field lists interventions and their outcomes.
        case_text = []
        for c in cases:
            case_text.append(json.dumps({"question": c.question, "prediction": c.prediction,
                                         "expected": c.ground_truth, "trace": c.replay}, ensure_ascii=False))
        prompt = ("Review the following failed memory questions and trace evidence. Identify a specific existing "
                  "skill whose writing instruction can be improved. Explain which source detail is lost and how "
                  "an executor should preserve it. Distinguish demonstrated effects from hypotheses. Propose one "
                  "reusable revision without any sample names, dates or answers.\nCurrent skills:\n"
                  + trainer.designer._format_operation_bank_description(trainer.operation_bank)
                  + "\nCases:\n" + "\n".join(case_text))
        # Run the standard analysis and refinement stages with identical limits.
        proposal = trainer.designer.run_evolution(trainer.operation_bank, {"analysis_prompt": prompt,
                                                                           "analysis_cases": cases})
        proposals[arm] = proposal
        save(out / (arm + "_proposal.json"), {"prompt": prompt, "proposal": proposal})

    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "calibration_usage.jsonl")
    def scores_for(bank, conversations, directory):
        saved = directory / "scores.json"
        if resume and saved.is_file():
            scores = json.loads(saved.read_text())
            expected = {(c["sample_id"], i) for c in conversations for i, _ in enumerate(c["qa"])}
            actual = [(r["sample_id"], r["qa_index"]) for r in scores["answers"]]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError(f"Incomplete saved scores in {directory}")
            return scores
        return evaluate_bank(trainer, bank, conversations, directory)
    control = scores_for(trainer.operation_bank, calibration, out / "calibration/checkpoint")
    if archive:
        repeat = scores_for(trainer.operation_bank, calibration, out / "calibration/checkpoint_repeat")
        save(out / "calibration_noise.json", compare_revisions(control, repeat))
    decisions = {}
    for arm, p in proposals.items():
        folder = out / "calibration" / arm
        if resume and (folder / "candidate/scores.json").is_file():
            scores = scores_for(None, calibration, folder / "candidate")
            trial = OperationBank.from_dict(copy.deepcopy(trainer.operation_bank.to_dict()), encoder=trainer.op_encoder)
            if not trainer.designer.apply_evolution(trial, copy.deepcopy(p)):
                raise ValueError("Saved scores refer to an inapplicable proposal")
            decisions[arm] = check_exposure(trainer.operation_bank, trial, folder / "candidate/traces",
                                            compare_revisions(control, scores))
            save(folder / "decision.json", decisions[arm])
        else:
            decisions[arm] = validate_proposal(trainer, p, calibration, folder, control)
    save(out / "decisions.json", decisions)
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "audit_usage.jsonl")
    base = scores_for(trainer.operation_bank, audit, out / "audit/checkpoint")
    summary = {"checkpoint": base["f1"]}
    if archive:
        repeat = scores_for(trainer.operation_bank, audit, out / "audit/checkpoint_repeat")
        summary["control_repeat"] = compare_revisions(base, repeat)
    for arm, p in proposals.items():
        trial = OperationBank.from_dict(copy.deepcopy(trainer.operation_bank.to_dict()), encoder=trainer.op_encoder)
        scores = scores_for(trial, audit, out / "audit" / arm) if (
            trainer.designer.apply_evolution(trial, copy.deepcopy(p))) else base
        accepted = decisions[arm]["accept"]
        summary[arm] = {"accepted": accepted, "ungated_f1": scores["f1"],
                        "deployed_f1": scores["f1"] if accepted else base["f1"],
                        "audit_pairing": compare_revisions(base, scores)}
    save(out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    study()
