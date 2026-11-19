"""Screen actual failed questions with matched writer interventions; share a suffix only across its questions."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.designer.designer_pilot import prefix, save
from main import evaluate_text_dataset_queries, load_dataset, set_seed, split_data
from experiments.designer.replay_revision import normalized
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.designer import DesignerCase
from src.episode_trace import DETAIL_INSTRUCTION, confirmed_proposal
from src.operation_bank import OperationBank
from src.replay import rebuild_suffix, restore_text
from src.revision_validation import compare_revisions, evaluate_bank, validate_proposal
from src.trainer import get_trainer


def run():
    args = get_agentic_memory_args()
    out = Path(args.out_file).parent
    out.mkdir(parents=True, exist_ok=False)
    train, val, _ = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    fit = [prefix(train[i]) for i in (0, 1, 4, 5)]
    calibration = [prefix(c, 3) for c in train[2:4]]
    audit = []
    for c in val:
        old_ids = {q["original_index"] for q in prefix(c, 2)["qa"]}
        held = prefix(c, 4)
        held["qa"] = [q for q in held["qa"] if q["original_index"] not in old_ids]
        audit.append(held)
    groups = {"fit": fit, "calibration": calibration, "audit": audit}
    save(out / "protocol.json", {"samples": {k: {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in v}
                                              for k, v in groups.items()}, "model": args.model, "api": args.api_base,
                                "writer_change": DETAIL_INSTRUCTION, "seed": args.seed, "failure_threshold": .5,
                                "selection": "first complete evidence span, first selected INSERT/UPDATE rule",
                                "replication": "unchanged vs observed; revised bank and reader reproduced twice",
                                "gate": "positive mean verified effect then positive calibration without per-conversation loss"})
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    trainer.controller.eval()
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "screen_usage.jsonl")
    observations = evaluate_bank(trainer, trainer.operation_bank, fit, out / "observed")
    failures = {(r["sample_id"], r["qa_index"]): r for r in observations["answers"] if r["f1"] < .5}
    by_sample = {c["sample_id"]: c for c in fit}
    diagnoses = []
    for tracefile in sorted((out / "observed/traces").glob("*.jsonl")):
        events = [json.loads(line) for line in tracefile.read_text().splitlines()]
        events = [e for e in events if e["kind"] == "span"]
        sid = events[0]["sample"]
        sample = by_sample[sid]
        buckets = {}
        for (sample_id, i), r in failures.items():
            if sample_id != sid:
                continue
            evidence = set(sample["qa"][i]["evidence"])
            target = next((j for j, e in enumerate(events) if evidence.intersection(e["source_turns"])), None)
            if target is not None:
                buckets.setdefault(target, []).append(i)
        for start, indices in sorted(buckets.items()):
            event = events[start]
            idx = next((i for i, s in enumerate(event["skills"]) if s["update_type"].lower() in ("insert", "update")), None)
            if idx is None:
                continue
            skill = event["skills"][idx]
            trials = {}
            for arm in ("unchanged", "edited", "repeat_edit"):
                ops = [SimpleNamespace(**s) for s in event["skills"]]
                if arm != "unchanged":
                    ops[idx].instruction_template += DETAIL_INSTRUCTION
                actions = trainer.executor.execute_operation(ops, event["source"], event["retrieved"])
                bank, failed_span = rebuild_suffix(trainer, events, start, actions)
                if bank is None:
                    trials[arm] = {"valid": False, "failed_span": failed_span}
                    continue
                qa_subset = dict(sample, qa=[sample["qa"][i] for i in indices])
                args.out_file = str(out / f"{sid}_{start}_{arm}.json")
                scores = evaluate_text_dataset_queries(trainer, [qa_subset], {sid: bank}, args)
                trials[arm] = {"valid": True, "actions": normalized([vars(a) for a in actions]),
                               "memories": bank.get_all_contents(), "rows": scores["answers"]}
            save(out / f"{sid}_{start}_branches.json", trials)
            a, b, b2 = [trials[x] for x in ("unchanged", "edited", "repeat_edit")]
            original_write_matched = a["valid"] and a["actions"] == normalized(event["parsed_actions"])
            original_bank_matched = original_write_matched and a["memories"] == list(restore_text(events).values())
            edited_bank_repeated = (b["valid"] and b2["valid"] and b["actions"] == b2["actions"]
                                    and b["memories"] == b2["memories"])
            for j, i in enumerate(indices):
                obs = failures[sid, i]
                stable = (original_bank_matched and edited_bank_repeated
                          and obs["retrieved_indices"] == a["rows"][j]["retrieved_indices"]
                          and b["rows"][j]["retrieved_indices"] == b2["rows"][j]["retrieved_indices"]
                          and obs["prediction"] == a["rows"][j]["prediction"]
                          and b["rows"][j]["prediction"] == b2["rows"][j]["prediction"])
                delta = b["rows"][j]["f1"] - obs["f1"] if stable else None
                diagnoses.append({"query_id": f"{sid}_{i}", "question": sample["qa"][i]["question"],
                                  "ground_truth": str(trainer.evaluator.get_ground_truth(sample["qa"][i])), "replay": {
                                    "trace": tracefile.name, "span_id": start, "scope_kind": "suffix",
                                    "skill": skill["name"], "skill_version": skill["version"], "intervention": DETAIL_INSTRUCTION,
                                    "paired_answers": {"unchanged": a["rows"][j]["prediction"],
                                                       "edited": b["rows"][j]["prediction"]} if stable else {},
                                    "unchanged_write_reproduced": bool(original_write_matched),
                                    "unchanged_final_bank_reproduced": bool(original_bank_matched),
                                    "edited_bank_reproduced": bool(edited_bank_repeated), "f1_delta": delta}})
            save(out / "diagnoses.json", diagnoses)
    proposal = confirmed_proposal(trainer.operation_bank, [DesignerCase.from_dict(d) for d in diagnoses])
    save(out / "proposal.json", proposal)
    usable = [float(d["replay"]["f1_delta"]) for d in diagnoses if d["replay"]["f1_delta"] is not None]
    summary = {"observed": observations["count"], "failures": len(failures), "attempted": len(diagnoses),
               "verified": len(usable), "positive": sum(x > 0 for x in usable), "negative": sum(x < 0 for x in usable),
               "neutral": sum(x == 0 for x in usable), "proposal_action": proposal["action"]}
    if proposal["action"] != "no_change":
        os.environ["MEMSKILL_USAGE_LOG"] = str(out / "calibration_usage.jsonl")
        summary["calibration"] = validate_proposal(trainer, proposal, calibration, out / "calibration")
        os.environ["MEMSKILL_USAGE_LOG"] = str(out / "audit_usage.jsonl")
        control = evaluate_bank(trainer, trainer.operation_bank, audit, out / "audit/control")
        candidate = OperationBank.from_dict(copy.deepcopy(trainer.operation_bank.to_dict()), encoder=trainer.op_encoder)
        if not trainer.designer.apply_evolution(candidate, copy.deepcopy(proposal)):
            raise ValueError("Confirmed revision failed to apply")
        scores = evaluate_bank(trainer, candidate, audit, out / "audit/candidate")
        summary["audit"] = compare_revisions(control, scores)
    save(out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    run()
