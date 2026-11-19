"""Repeat a proposed instruction edit at a single historical write; keep all other skills fixed."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.designer.designer_pilot import prefix, save
from main import set_seed
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.episode_trace import evidence_turns
from src.replay import anchor_relative_time, replay_suffix, restore_text
from src.trainer import get_trainer


def normalized(actions):
    return [[a[k] for k in ("action_type", "success", "memory_index", "memory_content")] for a in actions]


def run():
    args = get_agentic_memory_args()
    out = Path(args.out_file).parent
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "usage.jsonl")
    base = out.parent
    fit = {c["sample_id"]: prefix(c, 2) for c in json.loads(Path(args.data_file).read_text())[:2]}
    proposal_file = Path(os.environ.get("MEMSKILL_PROPOSAL_FILE", base / "calendar_check/calendar_proposal.json"))
    proposal = json.loads(proposal_file.read_text())["proposal"]
    save(out / "protocol.json", {"proposal_file": str(proposal_file), "model": args.model, "api": args.api_base,
                                 "replication": ["original", "revision", "revision", "original"]})
    if proposal["action"] != "refine_existing":
        save(out / "results.json", {"reason": "No candidate revision", "cases": []})
        return
    edit = proposal["changes"][0]["refined_operation"]
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    trainer.controller.eval()
    records = []
    for path in sorted((base / "pilot_suffix/traces").glob("*.jsonl")):
        log = [json.loads(s) for s in path.read_text().splitlines()]
        events = [e for e in log if e["kind"] == "span"]
        sample = fit[events[0]["sample"]]
        for q in (e for e in log if e["kind"] == "qa" and e["f1"] < 0.5):
            qa = sample["qa"][int(q["query_id"].rsplit("_", 1)[1])]
            if int(qa["category"]) != 2:
                continue
            turns = evidence_turns(sample["conversation"], qa["evidence"])
            targets = [(i, e) for i, e in enumerate(events) if any(
                t["text"] in e["source"] and anchor_relative_time(t) != t["text"] for t in turns)
                and any(s["name"] == edit["name"] and s["version"] == proposal["base_version"] for s in e["skills"])]
            if not targets:
                continue
            start, event = targets[0]
            live = restore_text(events)
            case = SimpleNamespace(question=qa["question"], prediction=q["prediction"],
                                   retrieved_memories=[live[mid] for mid, _ in q["refs"]])
            trials = {"original": [], "revision": []}
            # Counterbalance order. A fresh pair with changed control has no historical attribution.
            for arm in ("original", "revision", "revision", "original"):
                ops = [SimpleNamespace(**s) for s in event["skills"]]
                if arm == "revision":
                    target = next(s for s in ops if s.name == edit["name"])
                    for field, value in edit["changes"].items():
                        if field not in ("instruction_template", "source_rule"):
                            raise ValueError(f"Unsupported isolated writer change {field}")
                        setattr(target, field, value)
                actions = trainer.executor.execute_operation(ops, event["source"], event["retrieved"])
                parsed = normalized([vars(a) for a in actions])
                result = replay_suffix(trainer, events, start, actions, qa, case)
                result["parsed"] = parsed
                trials[arm].append(result)
            def stable(pair):
                return (all(x["valid"] for x in pair) and pair[0]["parsed"] == pair[1]["parsed"]
                        and pair[0]["final_text_sha256"] == pair[1]["final_text_sha256"]
                        and pair[0]["answer"] == pair[1]["answer"])
            reproducible = all(stable(pair) for pair in trials.values())
            historical = all(x["valid"] and x["parsed"] == normalized(event["parsed_actions"])
                             and x["final_bank_matches_observed"] and x["context_matches_observed"]
                             and x["answer"].strip() == case.prediction.strip() for x in trials["original"])
            delta = trials["revision"][0]["f1"] - trials["original"][0]["f1"] if reproducible else None
            records.append({"query_id": q["query_id"], "trace": path.name, "span": event["span_id"],
                            "skill": edit["name"], "original_version": proposal["base_version"],
                            "historical_match": historical, "fresh_pair_stable": reproducible,
                            "fresh_pair_delta": delta, "historical_delta": delta if historical else None,
                            "trials": trials})
            save(out / "results.json", {"cases": records})
            print(q["query_id"], historical, reproducible, delta, flush=True)
    save(out / "results.json", {"cases": records, "complete": True})


if __name__ == "__main__":
    run()
