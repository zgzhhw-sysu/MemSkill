"""Use the same activity-writer grouping intervention across historical skill comparisons."""

from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.controller.factorial_rollout import read
from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash, register
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest, read_response

PLAN = Path(__file__).with_name("protocols") / "writer_format_transfer.json"


def consistent_credit(original, activity, temporal):
    if not original:
        return 0.0
    sign = 1 if original > 0 else -1
    return sign * max(0.0, min(abs(original), sign * activity, sign * temporal))


def archive(base):
    older = base / "write-components"
    s = base / "source-compression"
    audit = s / "legacy-terminal-symmetry"
    out = s / "writer-format-credit/history.json"
    paths = {
        "plan": PLAN,
        "jobs": older / "jobs.json",
        "summary": older / "summary.json",
        "reader": older / "reader.json",
        "audit": audit / "registration.json",
        "complete": audit / "outcomes-complete.json",
        "original": s / "earliest-three-skills/forecast.json",
        "eligibility": s / "earlier-terminal-returns/prepared-complete.json",
    }
    inputs = {k: file_hash(p) for k, p in paths.items()}
    if out.exists():
        if read(out)["inputs"] != inputs:
            raise ValueError("Changed historical component archive")
        return out
    jobs = read(paths["jobs"])
    summary = read(paths["summary"])
    reader = read(paths["reader"])
    completed = read(paths["complete"])
    if reader != summary["manifest"] or reader["jobs"] != inputs["jobs"]:
        raise ValueError("Different historical reader")
    lookup = {(r["episode"], r["original_index"]): r for r in jobs["jobs"]}
    orig = read(paths["original"])["windows"]
    groups = defaultdict(list)
    flags = {}
    for j in read(paths["audit"])["jobs"]:
        f = audit / "outcomes" / (j["id"] + ".json")
        r = read(f)
        if (
            completed[j["id"]] != file_hash(f)
            or r["inputs"]["registration"] != inputs["audit"]
            or r["credit"] != (j["signed_credit"] if r["status"] == "confirmed" else 0)
        ):
            raise ValueError("Unbound symmetrical weaker-skill audit")
        flags[j["window"], j["original_index"]] = (r["credit"], j["signed_credit"])
    ev = LoCoMoEvaluator(SimpleNamespace())
    seen = set()
    for r in summary["rows"]:
        if not r["endpoints_reproduced"]:
            continue
        k = r["episode"]
        i = r["original_index"]
        j = lookup[k, i]
        f = older / "answers" / f"{k}-q{i}.json"
        saved = read(f)
        if (
            (k, i) in seen
            or i not in orig[k]["questions"]
            or r["dialogue"] != orig[k]["history"]
            or j["dialogue"] != r["dialogue"]
        ):
            raise ValueError("Incorrect historical history or question")
        seen.add((k, i))
        if (
            saved["manifest"] != reader
            or not j["control_matched"]
            or set(j["archived"]) != {"00", "11"}
            or set(saved["answers"]) != {"00", "01", "10", "11"}
        ):
            raise ValueError("Unmatched historical endpoint")
        for tag, a in saved["answers"].items():
            prompt = j["prompts"][tag]
            params = {
                "model": jobs["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "max_tokens": 32,
            }
            key = digest(
                {
                    "backend": reader["backend"],
                    "request": params,
                    "kind": "reader",
                    "format": 1,
                }
            )
            packet = read_response(older / "responses", key)
            if (
                not packet
                or packet["payload"]["prompt_sha256"] != sha(prompt)
                or packet["payload"]["response"] != a["answer"]
                or packet["payload"]["tokens"] != a["tokens"]
                or a["f1"] != r["f1"][tag]
                or abs(
                    ev.compute_f1(
                        a["answer"], ev.get_ground_truth(j["question"]), j["question"]
                    )
                    - a["f1"]
                )
                > 1e-10
            ):
                raise ValueError("Changed historical physical reader")
            if tag in j["archived"] and (
                not j["archived"][tag]["context_reproduced"]
                or j["archived"][tag]["answer"] != a["answer"].strip()
            ):
                raise ValueError("Earlier pure reader did not repeat")
        confirmed, reported = flags.get((k, i), (0.0, 0.0))
        if abs(reported - r["strict_credit"]) > 1e-10:
            raise ValueError("Changed sign of earlier choice credit")
        v = r["f1"]
        activity = v["10"] - v["11"]
        temporal = v["00"] - v["01"]
        # The historical label supports activity/pairs minus temporal/single.
        credit = consistent_credit(-confirmed, activity, temporal)
        groups[k].append(
            {
                "index": i,
                "response_set": file_hash(f),
                "activity_grouping": activity,
                "temporal_grouping": temporal,
                "credit": credit,
            }
        )
    if len(seen) != 41 or len(groups) != 11:
        raise ValueError("Different historical deployment support")
    rows = []
    for k, obs in sorted(groups.items()):
        ep = s / "earlier-terminal-returns/prepared" / (k + ".json")
        if read(paths["eligibility"])[k] != file_hash(ep):
            raise ValueError("Changed whole-window eligibility")
        rows.append(
            {
                "history": orig[k]["history"],
                "node": k,
                "features": orig[k]["features"],
                "points": obs,
                "whole_window_status": read(ep)["status"],
                "raw": mean(q["activity_grouping"] for q in obs),
                "credit": mean(q["credit"] for q in obs),
            }
        )
    register(out, {"inputs": inputs, "backend": reader["backend"], "rows": rows})
    return out
