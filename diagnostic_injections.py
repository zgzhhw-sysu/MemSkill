"""Check localization against single faults injected into real, originally successful training cases."""
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from designer_pilot import prefix, save
from llm_utils import request_task
from src.config import get_agentic_memory_args
from src.episode_trace import evidence_turns
from src.eval import get_evaluator
from src.replay import reader_probes, restore_text


def run():
    args = get_agentic_memory_args()
    out = Path(args.out_file).parent
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "reader_usage.jsonl")
    evaluator = get_evaluator("locomo", args)
    trainer = SimpleNamespace(evaluator=evaluator)
    data = {c["sample_id"]: prefix(c, 2) for c in json.loads(Path(args.data_file).read_text())[:2]}
    labels = {"overwrite": "prior_version", "delete": "deleted_version", "retrieval": "existing_unretrieved_item"}
    tests = []
    for path in sorted((out.parent / "pilot_suffix/traces").glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        events = [r for r in records if r["kind"] == "span"]
        live = restore_text(events)
        sample = data[records[0]["sample"]]
        versions = {c["memory_id"]: c["version"] for e in events for c in e["changes"]}
        for q in (r for r in records if r["kind"] == "qa" and r["f1"] >= 0.5):
            qa = sample["qa"][int(q["query_id"].rsplit("_", 1)[1])]
            turns = evidence_turns(sample["conversation"], qa["evidence"])
            refs = [mid for mid, _ in q["refs"]]
            if not refs or not turns:
                continue
            control_prompt = evaluator.build_prompt(q["question"], [live[mid] for mid in refs], qa)
            _, clean_prediction, _, ok = request_task((0, control_prompt, evaluator.prepare_eval_args()))
            if not ok:
                raise RuntimeError("Clean reader control failed")
            stable = clean_prediction.strip() == q["prediction"].strip()
            def words(s):
                return set(re.findall(r"\w+", s.lower()))
            evidence = words(" ".join(t["text"] for t in turns))
            def overlap(mid):
                other = words(live[mid])
                return len(evidence & other) / max(1, len(evidence | other))
            target = max(refs, key=overlap)
            distractor = min((mid for mid in live if mid != target), key=overlap)
            for fault, label in labels.items():
                changed_refs = list(refs)
                changes = []
                if fault == "overwrite":
                    changes = [{"action": "UPDATE", "memory_id": target, "version": versions[target]+1,
                                "before": live[target], "after": live[distractor]}]
                else:
                    changed_refs.remove(target)
                    if fault == "delete":
                        changes = [{"action": "DELETE", "memory_id": target, "version": versions[target]+1,
                                    "before": live[target], "after": None}]
                span = events[-1]["span_id"] + 1
                altered = [*events, {"span_id": span, "refs": [], "retrieved": [], "changes": changes}]
                state = restore_text(altered)
                context = [state[mid] for mid in changed_refs]
                prompt = evaluator.build_prompt(q["question"], context, qa)
                _, pred, _, ok = request_task((0, prompt, evaluator.prepare_eval_args()))
                if not ok:
                    raise RuntimeError("Reader unavailable during injection")
                f1 = evaluator.compute_f1(pred, evaluator.get_ground_truth(qa), qa)
                case = SimpleNamespace(question=q["question"], prediction=pred, retrieved_memories=context,
                                       retrieved_indices=[list(state).index(mid) for mid in changed_refs])
                probe = reader_probes(trainer, case, qa, altered, turns)
                recovered = [p for p in probe["probes"] if p["type"] == label and p["ok"] and p["f1"] > f1]
                localized = any(p["memory_id"] == target and (fault == "retrieval" or p["span"] == span)
                                for p in recovered)
                tests.append({"query_id": q["query_id"], "clean_reproduced": stable, "fault": fault, "known_id": target, "known_span": span
                              if changes else None, "clean_f1": q["f1"], "fault_f1": f1, "localized": localized,
                              "finding": probe})
    counts = {}
    for fault in labels:
        rows = [t for t in tests if t["fault"] == fault]
        damaged = [t for t in rows if t["clean_reproduced"] and t["fault_f1"] < t["clean_f1"]]
        counts[fault] = {"injected": len(rows), "damaged": len(damaged),
                         "localized_on_damaged": sum(t["localized"] for t in damaged)}
    save(out / "results.json", {"counts": counts, "cases": tests})
    print(counts, flush=True)


if __name__ == "__main__":
    run()
