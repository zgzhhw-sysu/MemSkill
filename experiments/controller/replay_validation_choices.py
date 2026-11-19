"""Cross the observed validation choice schedule with a second frozen encoder backend."""
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from experiments.controller.audit_paired import probability, reader_inputs
from experiments.designer.designer_pilot import save
from src.revision_validation import evaluate_bank


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_schedule(directory, weights, steps):
    rows = read(directory / "decisions.json")
    if len(rows) != steps or any(row["use"] != (row["prob"] >= .5) or not math.isclose(
            row["prob"], probability(row, weights), abs_tol=2e-6) for row in rows):
        raise ValueError("Recorded development schedule differs from its frozen policy")
    return [r["use"] for r in rows]


def verify_rollout(trainer, path, convo, decisions, versions):
    trace, = (path / "traces").glob("*.jsonl")
    events = list(map(json.loads, trace.read_bytes().splitlines()))
    spans = [e for e in events if e["kind"] == "span"]
    for span, row in zip(spans, decisions, strict=True):
        chosen = row["grounded"] if row["use"] else row["plain"]
        if ([s["version"] for s in span["selected_skills"]] != [versions[i] for i in chosen]
                or any(s["name"] == "capture_activity_details" for s in span["skills"]) != row["use"]):
            raise ValueError("Writer received a different skill choice or version")
    records = read(path / "scores.json")["answers"]
    evaluator = trainer.evaluator
    refs = dict(evaluator.filter_qa_list(trainer.data_processor.get_qa_list(convo)))
    if (len(records) != len(refs) or {r["qa_index"] for r in records} != refs.keys()
            or any(min(r["tokens"]) <= 0 for r in records)):
        raise ValueError("Incomplete fixed-choice development rollout")
    reader_inputs(events, records, refs, evaluator)
    for row in records:
        question = refs[row["qa_index"]]
        score = evaluator.compute_f1(row["prediction"], evaluator.get_ground_truth(question), question)
        if not math.isclose(score, row["f1"]):
            raise ValueError("Development F1 did not reproduce")
    return len(records), mean(r["f1"] for r in records)


def replay(trainer, study, reference, conversations, weights):
    files = ("protocol.json", "terminal.json", "credit.json", "rate.json")
    frozen = {name: digest(study / name) for name in files}
    if frozen != {name: digest(reference / name) for name in files}:
        raise ValueError("Schedule replay requires identical frozen policies and protocols")
    if [c["sample_id"] for c in conversations] != read(study / "protocol.json")["validation"]:
        raise ValueError("Replay requires the original development dialogues in order")
    for convo in conversations:
        sid = convo["sample_id"]
        local, donor = [p / "validation/credit" / sid for p in (study, reference)]
        target, source = [read(p / "inputs.json") for p in (local, donor)]
        if (target["backend"] != os.environ["MEMSKILL_BACKEND_ID"] or target["backend"] == source["backend"]
                or dict(target, backend=source["backend"]) != source or target["arm"] != "credit"
                or target["weights"] != hashlib.sha256(weights.numpy().tobytes()).hexdigest()
                or target["conversation"] != hashlib.sha256(json.dumps(convo, sort_keys=True).encode()).hexdigest()):
            raise ValueError("Inputs differ from the frozen development runs")
        count = sum(k.startswith("session_") and not k.endswith("_date_time") for k in convo["conversation"])
        native, schedule = [verified_schedule(p, weights.numpy(), count) for p in (local, donor)]
        changes = [i for i, (a, b) in enumerate(zip(native, schedule, strict=True)) if a != b]
        inputs = {"local": target, "donor": source, "frozen": frozen,
                  "decisions": [digest(p / "decisions.json") for p in (local, donor)],
                  "scores": [digest(p / "scores.json") for p in (local, donor)]}
        key = hashlib.sha256(source["backend"].encode()).hexdigest()[:16]
        root = study / "diagnostics/validation-schedule" / key
        root.mkdir(parents=True, exist_ok=True)
        dest = root / sid
        if dest.exists():
            if read(dest / "inputs.json") != inputs:
                raise ValueError("Frozen schedule inputs changed")
        else:
            with TemporaryDirectory(dir=root) as tmp:
                work = Path(tmp) / "result"
                work.mkdir()
                save(work / "inputs.json", inputs)
                actual, decisions = local, read(local / "decisions.json")
                if changes:
                    actual = work / "rollout"
                    policy = trainer.controller
                    policy.mode, policy.weights, policy.schedule = "count", weights, schedule
                    policy.decisions.clear()
                    os.environ["MEMSKILL_USAGE_LOG"] = str(work / "usage.jsonl")
                    evaluate_bank(trainer, trainer.operation_bank, [convo], actual)
                    if [r["use"] for r in policy.decisions] != schedule:
                        raise ValueError("Replay did not execute the assigned choices")
                    save(work / "decisions.json", policy.decisions)
                    decisions = policy.decisions
                n, f1 = verify_rollout(trainer, actual, convo, decisions, target["skills"])
                save(work / "summary.json", {"sample": sid, "questions": n, "switched_zero_based": changes,
                     "local_f1": read(local / "scores.json")["f1"], "donor_f1": read(donor / "scores.json")["f1"],
                     "replayed_f1": f1, "reused_local": not changes})
                work.rename(dest)
        print(read(dest / "summary.json"), flush=True)
