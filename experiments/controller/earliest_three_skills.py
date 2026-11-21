"""Measure a third writer on the original pre-action state and adjudicate its local credit."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.capped_source_repair import diagnose, writer
from experiments.controller.earliest_skill_forecast import PLAN
from experiments.controller.exchange_paths import Ranking
from experiments.controller.exchange_questions import call
from experiments.controller.factorial_rollout import read
from experiments.controller.legacy_terminal_symmetry import (
    make_bank,
    make_writer,
    measure,
    ops,
    prepare,
)
from experiments.controller.relation_skill_repairs import candidates
from experiments.controller.resume_status import check_equal_source_choice
from experiments.controller.semantic_writer import REL, TEMP, sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_writer_credit import ACTIVITY
from src.credit_assignment import _digest, execution_fault
from src.episode_trace import skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings, read_response

PHASES = ("writers", "prepared", "readers", "pairs", "repairs", "judged")
ACCEPTED = {
    "writers": "matched_writer",
    "prepared": "prepared",
    "readers": "answered",
    "pairs": "matched",
    "repairs": "prepared_repairs",
}


def alternate_slots(row, skills, event, alternate):
    selected = [skills[i] for i in row["plain" if row["use"] else "grounded"]]
    check_equal_source_choice(row, skills, event["selected_skills"], selected)
    if [skill_spec(s) for s in ops(selected)] != alternate["writer_skills"]:
        raise ValueError(
            "Recorded controller slots and deployed writer operations differ"
        )
    return selected


def write_relation(out, event, forecast, proto):
    selected = forecast["selected"]
    base = event["selected_skills"]
    if (
        selected[:-1] != base[:-1]
        or selected[-1]["name"] != REL
        or selected[-1]["source_rule"] != "dated_single_turns"
        or base[-1]["name"] != (ACTIVITY if forecast["native_activity"] else TEMP)
        or sha(event["source"]) != forecast["source"]
    ):
        raise ValueError("Terminal intervention is not on the registered frozen choice")
    ex = make_writer(proto)
    actions = ops(selected)
    prompt = ex._build_executor_prompt(actions, event["source"], event["retrieved"])
    resp = call(out, prompt, "executor", proto["writer_tokens"])
    receipt = {
        "prompt_sha256": sha(prompt),
        "response": resp["text"],
        "input_tokens": resp["tokens"][0],
        "output_tokens": resp["tokens"][1],
    }
    replay = dict(event, executor_responses=[receipt])
    parsed = writer(ex, actions, replay, repair=False)
    deployed = [skill_spec(s) for s in actions]
    fault = execution_fault(parsed, True, deployed)
    changed = parsed != writer(ex, actions, replay, repair=True)
    return {
        "status": "relation_write_fault" if fault or changed else "matched_writer",
        "selected": selected,
        "actions": parsed,
        "writer_skills": deployed,
        "fault": fault,
        "cap_changed": changed,
        "prompt_hash": sha(prompt),
        "response": resp,
    }


def prepare_relation(rank, spans, questions, old, measurements, proto):
    prefix, event = spans
    bank = make_bank(rank, make_writer(proto), prefix, event, measurements, [])
    contents = bank.get_all_contents()
    vecs = [m.embedding for m in bank.memories]
    ev = LoCoMoEvaluator(SimpleNamespace())
    packets = []
    for q, p, v in zip(
        questions,
        old["questions"],
        rank.encode(tuple(q["question"] for q in questions)),
        strict=True,
    ):
        if (
            q != p["qa"]
            or bool(set(q["evidence"]) & set(event["source_turns"])) != p["linked"]
        ):
            raise ValueError("Wrong original question or focal evidence")
        idx, ctx = rank.rank(v, contents, vecs)
        prompt = ev.build_prompt(q["question"], ctx, q)
        packets.append(
            {
                "index": q["original_index"],
                "indices": idx,
                "context_hash": _digest(ctx),
                "prompt": prompt,
                "hash": sha(prompt),
            }
        )
    return {"status": "prepared", "bank": _digest(contents), "questions": packets}


def physical_read(out, earlier, packet, qa):
    ev = LoCoMoEvaluator(SimpleNamespace())
    if (
        sha(packet["prompt"]) != packet["hash"]
        or packet["index"] != qa["original_index"]
    ):
        raise ValueError("Changed relation reader input")
    resp = call(out, packet["prompt"], "reader", 32, reuse_from=earlier / "responses")
    answer = resp["text"].strip()
    return {
        "index": qa["original_index"],
        "answer": answer,
        "f1": ev.compute_f1(answer, ev.get_ground_truth(qa), qa),
        "ok": True,
        "receipt": resp,
        "indices": packet["indices"],
        "context_hash": packet["context_hash"],
    }


def pair_rows(forecast, questions, spans, prep, old_relation, old_two):
    f1 = []
    packets = []
    auth = []
    for q, p, r, t in zip(
        questions, prep["questions"], old_relation["rows"], old_two["rows"], strict=True
    ):
        idx = q["original_index"]
        if (
            idx != r["index"]
            or idx != t["original_index"]
            or p["qa"] != q
            or p["linked"] != t["linked"]
            or t["category"] != q["category"]
        ):
            raise ValueError("Unaligned original question or exposure")
        scores = {"relation": r}
        for name, a in t["arms"].items():
            skill = (
                "activity"
                if (name == "native") == forecast["native_activity"]
                else "temporal"
            )
            scores[skill] = {**a, "ok": True, "indices": p[name]["indices"]}
            auth.append((p[name]["hash"], a["receipt"]))
        expected = [scores[key]["f1"] for key in ("temporal", "activity", "relation")]
        if abs(t["delta"] - (expected[1] - expected[0])) > 1e-10:
            raise ValueError("Old two-action return changed")
        f1.append(expected)
        packets.append(dict(scores, original_index=idx, category=q["category"]))
    source = spans[-1]["source_turns"]
    jobs = {}
    linked = sum(bool(set(q["evidence"]) & set(source)) for q in questions)
    for baseline in ("temporal", "activity"):
        n, jobs[baseline] = candidates(questions, packets, source, baseline)
        if n != linked:
            raise ValueError("Changed local span evidence")
    return {
        "status": "matched",
        "linked": linked,
        "jobs": jobs,
        "rows": packets,
        "f1": f1,
    }, auth


def repair_jobs(
    rank,
    spans,
    questions,
    pair,
    measured,
    relation_prep,
    old,
    alternate,
    forecast,
    proto,
    history,
    *,
    comparison="relation",
):
    if not any(pair["jobs"].values()):
        return {"status": "prepared_repairs", "jobs": []}
    prefix, event = spans
    factual = {"actions": event["parsed_actions"], "writer_skills": event["skills"]}
    branches = {
        "relation": {
            **measured,
            "bank_hash": relation_prep["bank"],
            "qa": {str(i): r["relation"] for i, r in enumerate(pair["rows"])},
        }
    }
    ev = LoCoMoEvaluator(SimpleNamespace())
    for i, (tag, branch) in enumerate(
        (("native", factual), ("alternative", alternate))
    ):
        name = (
            "activity"
            if forecast["native_activity"] == (tag == "native")
            else "temporal"
        )
        bank = make_bank(rank, make_writer(proto), prefix, event, branch, [])
        contents = bank.get_all_contents()
        hm = _digest(contents)
        if hm != old["banks"][i]:
            raise ValueError("Changed historical bank for repair")
        responses = {}
        for j, (q, b, r) in enumerate(
            zip(questions, old["questions"], pair["rows"], strict=True)
        ):
            ix = b[tag]["indices"]
            ctx = [contents[k] for k in ix]
            if b[tag]["hash"] != sha(ev.build_prompt(q["question"], ctx, q)):
                raise ValueError("Changed ordered historical context")
            responses[str(j)] = dict(r[name], context_hash=_digest(ctx))
        branches[name] = {**branch, "bank_hash": hm, "qa": responses}
    jobs = []
    for kind, items in pair["jobs"].items():
        diag = {
            "control": branches[kind],
            "swap": branches[comparison],
            "questions": [
                {"qa_index": i, "original_index": q["original_index"]}
                for i, q in enumerate(questions)
            ],
        }
        for item in items:
            jobs.append(
                {
                    "baseline": kind,
                    "job": item,
                    "prepared": prepare(
                        item, rank, spans, diag, proto, history, strict_indices=True
                    ),
                }
            )
    return {"status": "prepared_repairs", "jobs": jobs}


def verify_response(cache, receipt, prompt):
    p = read_response(cache, receipt["key"])
    if (
        not p
        or p["sha256"] != receipt["sha256"]
        or p["payload"]["prompt_sha256"] != prompt
        or p["payload"]["response"] != receipt["text"]
        or p["payload"]["tokens"] != receipt["tokens"]
    ):
        raise ValueError("Unbound measured response")


def run(base, study, corpus, phase):
    src = base / "source-compression"
    earlier = src / "earlier-terminal-returns"
    out = src / "earliest-three-skills"
    dest = out / phase
    dest.mkdir(parents=True, exist_ok=True)
    model = out / "forecast.json"
    if not model.exists():
        return False
    p = read(model)
    proto = read(study / "protocol.json")
    cohort = read(earlier / "registration.json")
    if (
        file_hash(PLAN) != p["inputs"]["plan"]
        or file_hash(study / "protocol.json") != p["inputs"]["protocol"]
        or proto["input_sha256"] != file_hash(corpus)
        or p["inputs"]["cohort"] != file_hash(earlier / "registration.json")
        or p["inputs"]["eligible"] != file_hash(earlier / "prepared-complete.json")
    ):
        raise ValueError("Changed study")
    data = {d["sample_id"]: d for d in read(corpus) if d["sample_id"] in proto["train"]}
    eligibility = read(earlier / "prepared-complete.json")
    for name in cohort["order"]:
        f = dest / (name + ".json")
        pred = p["windows"][name]
        parent = (
            model
            if phase == "writers"
            else out / PHASES[PHASES.index(phase) - 1] / f.name
        )
        if not parent.exists():
            continue
        old = earlier / "prepared" / f.name
        if eligibility[name] != file_hash(old):
            raise ValueError("Changed earlier retrieval receipt")
        inputs = {
            "forecast": file_hash(model),
            "parent": file_hash(parent),
            "early_prepared": eligibility[name],
        }
        previous = read(parent) if phase != "writers" else None
        if phase == "pairs" and previous["status"] == "answered":
            two = earlier / "outcomes" / f.name
            if not two.exists():
                continue
            inputs["early_outcomes"] = file_hash(two)
        if f.exists():
            if read(f)["inputs"] != inputs:
                raise ValueError("Stage dependency changed")
            continue
        ep = study / "train" / name
        (tr,) = (ep / "traces").glob("*.jsonl")
        if pred["trace"] != file_hash(tr) or pred["rows"] != file_hash(
            ep / "rows.json"
        ):
            raise ValueError("Changed source or archived action")
        spans = [
            e
            for line in tr.read_bytes().splitlines()
            if (e := json.loads(line))["kind"] == "span"
        ]
        history = data[pred["history"]]
        qa = [dict(history["qa"][i], original_index=i) for i in pred["questions"]]
        prep = read(old)
        if prep["status"] != "prepared":
            result = {"status": prep["status"], "origin": "two_skill_retrieval"}
        elif (
            previous is not None
            and previous["status"] != ACCEPTED[PHASES[PHASES.index(phase) - 1]]
        ):
            result = {"status": previous["status"]}
        elif phase == "writers":
            if any(diagnose(make_writer(proto), e) for e in spans):
                raise ValueError("Changed factual writer")
            result = write_relation(out, spans[-1], pred, proto)
        elif phase == "prepared":
            verify_response(
                out / "responses", previous["response"], previous["prompt_hash"]
            )
            with history_embeddings():
                result = prepare_relation(
                    Ranking(proto["retriever"]), spans, qa, prep, previous, proto
                )
        elif phase == "readers":
            answers = [
                physical_read(out, earlier, q, r)
                for q, r in zip(previous["questions"], qa, strict=True)
            ]
            result = {"status": "answered", "rows": answers}
        elif phase == "pairs":
            prior = read(two)
            if prior["inputs"]["parent"] != file_hash(old):
                raise ValueError("Different two-skill outcome")
            if prior["status"] != "matched":
                result = {"status": prior["status"], "origin": "two_skill_readers"}
            else:
                result, checks = pair_rows(pred, qa, spans, prep, previous, prior)
                for prompt, r in checks:
                    verify_response(earlier / "responses", r, prompt)
                for spec, r in zip(
                    read(out / "prepared" / f.name)["questions"],
                    previous["rows"],
                    strict=True,
                ):
                    verify_response(out / "responses", r["receipt"], spec["hash"])
        elif phase == "repairs":
            with history_embeddings():
                rank = (
                    Ranking(proto["retriever"])
                    if any(previous["jobs"].values())
                    else None
                )
                result = repair_jobs(
                    rank,
                    spans,
                    qa,
                    previous,
                    read(out / "writers" / f.name),
                    read(out / "prepared" / f.name),
                    prep,
                    read(earlier / "writers" / f.name),
                    pred,
                    proto,
                    history,
                )
        else:
            adjudicated = []
            for r in previous["jobs"]:
                item = r["job"]
                v = r["prepared"]
                a = (
                    measure(out, item, v)
                    if v["status"] == "prepared"
                    else {"status": v["status"], "credit": 0.0}
                )
                adjudicated.append(
                    {
                        "baseline": r["baseline"],
                        "index": item["qa"]["original_index"],
                        "weak": item["weak"],
                        "raw": item["signed_credit"],
                        **a,
                    }
                )
            result = {"status": "judged", "cases": adjudicated}
        register(f, dict(result, inputs=inputs))
        print(name, phase, result["status"], flush=True)
        return True
    files = {
        k: file_hash(dest / (k + ".json"))
        for k in cohort["order"]
        if (dest / (k + ".json")).exists()
    }
    if len(files) == len(cohort["order"]):
        register(out / (phase + "-complete.json"), files)
    return False


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        cli.add_argument(k, type=Path)
    cli.add_argument("phase", choices=PHASES)
    a = cli.parse_args()
    run(a.base, a.study, a.corpus, a.phase)
