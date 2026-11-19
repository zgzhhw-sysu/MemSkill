"""Inspect completed paired training episodes without a model or GPU."""
import argparse
import json
import math
from pathlib import Path


def _writer_skills(selected):
    return [skill for skill in selected if skill["name"].lower() != "noop" and skill["update_type"].lower() != "noop"]


def load_episode(directory, sample_id, span_count, question_ids=None, *, deployment_writer=None):
    """Verify an atomically published episode before using its rows for fitting."""
    rows = json.loads((directory / "rows.json").read_text())
    questions = json.loads((directory / "questions.json").read_text())
    if question_ids is not None and questions != question_ids:
        raise ValueError(f"Training questions differ at {directory}")
    trace_files = list((directory / "traces").glob("*.jsonl"))
    if len(trace_files) != 1 or not (directory / "usage.jsonl").is_file():
        raise ValueError(f"Incomplete episode files at {directory}")
    events = [json.loads(line) for line in trace_files[0].read_text().splitlines()]
    spans = [e for e in events if e["kind"] == "span"]
    diagnoses = [e["diagnosis"] for e in events if e["kind"] == "span_credit"]
    if (len(rows) != span_count or len(spans) != span_count or len(diagnoses) != span_count
            or any(e["sample"] != sample_id for e in events)
            or sum(e["kind"] == "snapshot" for e in events) != 1):
        raise ValueError(f"Incomplete or mismatched span trace at {directory}")
    for i, (row, span, diagnosis) in enumerate(zip(rows, spans, diagnoses, strict=True)):
        if deployment_writer is not None and deployment_writer != ("selected_skills" in span):
            raise ValueError(f"Writer protocol differs in span {i} at {directory}")
        if "selected_skills" in span:
            selected = span["selected_skills"]
            expected = _writer_skills(selected)
            if selected != diagnosis["sampled_skills"] or expected != span["skills"]:
                raise ValueError(f"Sampled skills and deployed writer differ in span {i} at {directory}")
            for key, branch_skills in (("control", expected), ("execution_repair", expected),
                                       ("repair_repeat", expected), ("swap", None), ("swap_repeat", None)):
                if key not in diagnosis:
                    continue
                if branch_skills is None:
                    branch_skills = _writer_skills(diagnosis["alternative"])
                if diagnosis[key].get("writer_skills") != branch_skills:
                    raise ValueError(f"Writer protocol differs in {key} at {directory}")
        chosen = row["grounded"] if row["use"] else row["plain"]
        recorded = [skill["version"] for skill in diagnosis["sampled_skills"]]
        versions = span["candidate_versions"]
        if (row["episode"] != directory.name or (row["span"], span["span_id"], diagnosis["span_id"]) != (i, i, i)
                or len(set(chosen)) != 3 or any(index < 0 or index >= len(versions) for index in chosen)
                or recorded != [versions[index] for index in chosen]
                or row["delta"] != diagnosis["mean_controller_delta"] or row["prob"] != 0.5
                or not math.isclose(row["old_log_probability"], math.log(0.5))
                or not row["state"] or not all(map(math.isfinite, row["state"]))
                or not all(map(math.isfinite, (row["terminal"], row["base_logit"], row["delta"])))):
            raise ValueError(f"Saved training row disagrees with span {i} at {directory}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    study = parser.parse_args().study
    protocol = json.loads((study / "protocol.json").read_text())
    complete, pending = [], []
    for window in protocol["train_sessions"]:
        for sample in protocol["train"]:
            path = study / "train" / f"{sample}_s{window[0]}"
            if path.exists():
                complete.extend(load_episode(path, sample, len(window),
                                             deployment_writer=protocol.get("deployment_writer", False)))
                print(f"ready    {path.name}")
            else:
                pending.append(path.name)
                print(f"pending  {path.name}")
    total = len(protocol["train"]) * len(protocol["train_sessions"])
    print(f"{total-len(pending)}/{total} episodes, {len(complete)} spans, "
          f"{sum(r['delta'] != 0 for r in complete)} nonzero span credits")


if __name__ == "__main__":
    main()
