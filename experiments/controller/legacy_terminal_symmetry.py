"""Check the weaker writer action in earlier terminal credit examples."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.capped_source_repair import diagnose, writer
from experiments.controller.exchange_paths import Ranking, clean_marker
from experiments.controller.exchange_questions import call
from experiments.controller.resume_status import check_equal_source_choice, load_episode
from experiments.controller.semantic_writer import sha
from src.credit_assignment import (
    _digest,
    check_competing_evidence,
    execution_fault,
    hidden_retrieval_candidate,
    possible_recovery,
    recovered,
)
from src.episode_trace import evidence_turns
from src.eval.locomo import LoCoMoEvaluator
from src.executor import ExecutionResult, Executor, deployment_operations
from src.memory_bank import MemoryBank
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "legacy_terminal_symmetry.json"


def census(base, study, corpus):
    out = base / "source-compression/legacy-terminal-symmetry"
    out.mkdir(exist_ok=True)
    p = read(study / "protocol.json")
    ids = set(p["train"])
    if (
        len(ids) != 6
        or ids & set(p["validation"] + p["test"])
        or file_hash(corpus) != p["input_sha256"]
        or p["rule"] != 2
        or not p["all_linked_questions"]
    ):
        raise ValueError("Historical population changed")
    ev = LoCoMoEvaluator(SimpleNamespace())
    data = {r["sample_id"]: r for r in read(corpus) if r["sample_id"] in ids}
    jobs = []
    traces = {}
    questions = set()
    nlinked = 0
    roots = []
    for h in sorted(ids):
        for start in (1, 3):
            ep = study / "train" / f"{h}_s{start}"
            idx = read(ep / "questions.json")
            rows = load_episode(ep, h, 2, idx, deployment_writer=True)
            (trace,) = (ep / "traces").glob("*.jsonl")
            events = [json.loads(l) for l in trace.read_bytes().splitlines()]
            ds = [e["diagnosis"] for e in events if e["kind"] == "span_credit"]
            spans = [e for e in events if e["kind"] == "span"]
            traces[ep.name] = {
                "trace": file_hash(trace),
                "rows": file_hash(ep / "rows.json"),
                "questions": file_hash(ep / "questions.json"),
                "backend": read(ep / "backend.json")["inference"],
            }
            questions.update((h, i) for i in idx)
            for pivot, (d, s, row) in enumerate(zip(ds, spans, rows, strict=True)):
                check_equal_source_choice(
                    row, p["skills"], d["sampled_skills"], d.get("alternative")
                )
                if {q["original_index"] for q in d["questions"]} != {
                    i
                    for i in idx
                    if set(s["source_turns"]) & set(data[h]["qa"][i]["evidence"])
                }:
                    raise ValueError("Missing linked question in census")
                for q in d["questions"]:
                    nlinked += 1
                    if not q["controller_delta"]:
                        continue
                    item = data[h]["qa"][q["original_index"]]
                    if (
                        q["original_index"] != idx[q["qa_index"]]
                        or check_competing_evidence(d, q, ev.get_ground_truth(item))[
                            "controller_delta"
                        ]
                        != q["controller_delta"]
                    ):
                        raise ValueError("Wrong historical credit")
                    i = str(q["qa_index"])
                    low = (
                        "swap"
                        if d["control"]["qa"][i]["f1"] > d["swap"]["qa"][i]["f1"]
                        else "control"
                    )
                    hi = "control" if low == "swap" else "swap"
                    if not recovered(
                        d[low]["qa"][i], d[hi]["qa"][i], ev.get_ground_truth(item)
                    ):
                        raise ValueError("Historical effect fails strict recovery")
                    job = {
                        "id": f"{ep.name}_p{pivot}_q{q['original_index']}",
                        "window": ep.name,
                        "history": h,
                        "pivot": pivot,
                        "qa": item,
                        "question_index": q["qa_index"],
                        "original_index": q["original_index"],
                        "weak": low,
                        "use": row["use"],
                        "signed_credit": q["controller_delta"]
                        * (1 if row["use"] else -1),
                    }
                    (roots if pivot == 0 else jobs).append(job)
    if (len(questions), nlinked, len(jobs), len(roots)) != (123, 131, 5, 2):
        raise ValueError("Wrong earlier cohort")
    reg = out / "registration.json"
    register(
        reg,
        {
            "plan": file_hash(PLAN),
            "protocol": file_hash(study / "protocol.json"),
            "corpus": p["input_sha256"],
            "questions": len(questions),
            "linked": nlinked,
            "windows": traces,
            "jobs": jobs,
            "root_separate": roots,
        },
    )
    return reg


def source(job, reg, study):
    ep = study / "train" / job["window"]
    (trace,) = (ep / "traces").glob("*.jsonl")
    if file_hash(trace) != reg["windows"][ep.name]["trace"]:
        raise ValueError("Trace changed")
    events = [json.loads(l) for l in trace.read_bytes().splitlines()]
    spans = [e for e in events if e["kind"] == "span"]
    diag = [e["diagnosis"] for e in events if e["kind"] == "span_credit"][job["pivot"]]
    return spans, diag


def ops(skills):
    return deployment_operations([SimpleNamespace(**sk) for sk in skills])


def make_writer(p):
    return Executor(
        SimpleNamespace(
            model=p["model"],
            max_new_tokens=p["writer_tokens"],
            temperature=0,
            api_base="local",
            api_key="local",
            retriever=p["retriever"],
        )
    )


def verify_writer(out, job, spans, d, p):
    ex = make_writer(p)
    event = spans[job["pivot"]]
    for e in spans:
        if diagnose(ex, e) or execution_fault(
            e["parsed_actions"], e["applied"], e["skills"]
        ):
            return {"status": "factual_write_fault"}
    if d["control"]["actions"] != event["parsed_actions"] or d["control"][
        "writer_prompt_hashes"
    ] != [r["prompt_sha256"] for r in event["executor_responses"]]:
        raise ValueError("Factual writer did not reproduce")
    alternative = ops(d["alternative"])
    prompt = ex._build_executor_prompt(alternative, event["source"], event["retrieved"])
    if [sha(prompt)] != d["swap"]["writer_prompt_hashes"]:
        return {"status": "different_alternative_prompt"}
    replay = call(out, prompt, "executor", p["writer_tokens"])
    receipt = {
        "prompt_sha256": sha(prompt),
        "response": replay["text"],
        "input_tokens": replay["tokens"][0],
        "output_tokens": replay["tokens"][1],
    }
    event = dict(event, executor_responses=[receipt])
    if writer(ex, alternative, event, repair=False) != d["swap"]["actions"]:
        return {"status": "alternative_writer_did_not_repeat", "writer": replay}
    if (
        writer(ex, alternative, event, repair=True) != d["swap"]["actions"]
        or d["swap"]["contract_fault"]
    ):
        return {"status": "alternative_write_fault", "writer": replay}
    return {"status": "matched_writer", "writer": replay}


def make_bank(ranker, ex, prefix, event, branch, extra):
    bank = MemoryBank(ex.retriever_name, top_k=5)
    live = restore_text([prefix])
    contents = tuple(live.values())
    if contents:
        for t, v in zip(contents, ranker.encode(contents), strict=True):
            bank.add_memory(t, v)
    ids = {k: i for i, k in enumerate(live)}
    loc = [ids[k] for k, _ in event["refs"]]
    selected = branch["writer_skills"]
    actions = [ExecutionResult(**r) for r in branch["actions"]]
    written = set(event["retrieved"]) | {r.memory_content for r in actions}
    actions.extend(
        ExecutionResult("INSERT", True, memory_content=t)
        for t in extra
        if t not in written
    )
    if not ex.apply_to_memory_bank(
        actions, bank, loc, operation_name=[s["name"] for s in selected]
    ):
        raise ValueError("Could not apply archived writer actions")
    return bank


def prepare(job, ranker, spans, d, p, data, *, strict_indices=False):
    prefix, event = spans
    ex = make_writer(p)
    idx = job["question_index"]
    queries = [q["qa_index"] for q in d["questions"]]
    texts = tuple(data["qa"][q["original_index"]]["question"] for q in d["questions"])
    qvec = ranker.encode(texts)[queries.index(idx)]
    turns = [
        t
        for t in evidence_turns(data["conversation"], job["qa"]["evidence"])
        if t["dia_id"] in event["source_turns"]
    ]
    if not turns:
        raise ValueError("No evidence in the pivot")
    extra = [f"{t['session_date']} | {t['speaker']}: {t['text']}" for t in turns]
    packets = {}
    for k in ("control", "swap"):
        branch = d[k]
        bank = make_bank(ranker, ex, prefix, event, branch, [])
        contents = bank.get_all_contents()
        original = branch["qa"][str(idx)]
        if _digest(contents) != branch["bank_hash"]:
            return {"status": "changed_bank_hash"}
        vectors = [m.embedding for m in bank.memories]
        indices, context = ranker.rank(qvec, contents, vectors)
        if (
            (strict_indices and indices != original["indices"])
            or _digest(context) != original["context_hash"]
            or [contents[i] for i in original["indices"]] != context
        ):
            return {"status": "control_ranking_did_not_repeat"}
        packets[k] = {"native": context, "original": original}
        if k != job["weak"]:
            continue
        augmented = make_bank(ranker, ex, prefix, event, branch, extra)
        _, ctx = ranker.rank(
            qvec,
            augmented.get_all_contents(),
            [m.embedding for m in augmented.memories],
        )
        mid = hidden_retrieval_candidate(
            job["qa"], turns, indices, dict(enumerate(contents))
        )
        cleaned = [clean_marker(t) for t in contents]
        changed = [
            i for i, (a, b) in enumerate(zip(contents, cleaned, strict=True)) if a != b
        ]
        cleanvec = vectors.copy()
        if changed:
            for i, v in zip(
                changed, ranker.encode(tuple(cleaned[i] for i in changed)), strict=True
            ):
                cleanvec[i] = v
        packets[k].update(
            execution=ctx,
            retrieval=context[:-1] + [contents[mid]] if mid is not None else None,
            format=ranker.rank(qvec, cleaned, cleanvec)[1] if changed else context,
        )
    return {"status": "prepared", "branches": packets}


def measure(out, job, prepared):
    ev = LoCoMoEvaluator(SimpleNamespace())
    qa = job["qa"]
    truth = ev.get_ground_truth(qa)

    def ask(ctx):
        if ctx is None:
            return None
        v = call(out, ev.build_prompt(qa["question"], ctx, qa), "reader", 32)
        ans = v["text"].strip()
        return {
            "answer": ans,
            "ok": True,
            "f1": ev.compute_f1(ans, truth, qa),
            "receipt": v,
        }

    branches = prepared["branches"]
    native = {k: ask(b["native"]) for k, b in branches.items()}
    if any(
        v["answer"] != branches[k]["original"]["answer"]
        or abs(v["f1"] - branches[k]["original"]["f1"]) > 1e-12
        for k, v in native.items()
    ):
        return {"status": "native_answers_changed", "reads": native, "credit": 0.0}
    k = job["weak"]
    high = "swap" if k == "control" else "control"
    if not recovered(native[k], native[high], truth):
        raise ValueError("Non-recovery on exact repeated answers")
    repairs = {
        name: ask(branches[k][name]) for name in ("execution", "retrieval", "format")
    }
    veto = [
        name for name, v in repairs.items() if possible_recovery(native[k], v, truth)
    ]
    return {
        "status": "veto" if veto else "confirmed",
        "veto": veto,
        "reads": native,
        "repairs": repairs,
        "credit": 0.0 if veto else job["signed_credit"],
    }


def report(out, packet):
    marker = out / "outcomes-complete.json"
    if not marker.is_file():
        raise ValueError("The five terminal comparisons are still incomplete")
    hashes = read(marker)
    if set(hashes) != {job["id"] for job in packet["jobs"]}:
        raise ValueError("Different terminal population")
    cases = []
    for job in packet["jobs"]:
        path = out / "outcomes" / (job["id"] + ".json")
        if file_hash(path) != hashes[job["id"]]:
            raise ValueError("Changed terminal result")
        result = read(path)
        cases.append(
            {
                "id": job["id"],
                "history": job["history"],
                "weak": job["weak"],
                "prior_repair_on_weaker": job["weak"] == "control",
                "status": result["status"],
                "veto": result.get("veto", []),
                "credit": result.get("credit", 0.0),
            }
        )
    histories = {name.rsplit("_s", 1)[0] for name in packet["windows"]}
    counts = {}
    for h in sorted(histories):
        rows = [row for row in cases if row["history"] == h]
        counts[h] = {
            "candidates": len(rows),
            "confirmed": sum(r["status"] == "confirmed" for r in rows),
            "veto": sum(r["status"] == "veto" for r in rows),
            "failed_to_reproduce": sum(
                r["status"] not in ("confirmed", "veto") for r in rows
            ),
        }
    register(
        out / "summary.json",
        {
            "registration": file_hash(out / "registration.json"),
            "outcomes": file_hash(marker),
            "questions_in_census": packet["questions"],
            "linked_comparisons": packet["linked"],
            "root_cases_outside_terminal_audit": [
                job["id"] for job in packet["root_separate"]
            ],
            "cases": cases,
            "by_history": counts,
        },
    )


def run(base, study, corpus, phase):
    reg = census(base, study, corpus)
    out = reg.parent
    packet = read(reg)
    if phase == "report":
        return report(out, packet)
    p = read(study / "protocol.json")
    ranker = None
    data = {r["sample_id"]: r for r in read(corpus) if r["sample_id"] in p["train"]}
    dest = out / phase
    dest.mkdir(exist_ok=True)
    for job in packet["jobs"]:
        f = dest / (job["id"] + ".json")
        if f.exists():
            continue
        spans, d = source(job, packet, study)
        provenance = {"registration": file_hash(reg)}
        if phase == "writers":
            v = verify_writer(out, job, spans, d, p)
        else:
            prev = out / ("writers" if phase == "prepared" else "prepared") / f.name
            if not prev.is_file():
                continue
            a = read(prev)
            provenance["parent"] = file_hash(prev)
            if a["status"] != ("matched_writer" if phase == "prepared" else "prepared"):
                v = {"status": a["status"], "credit": 0.0}
            elif phase == "prepared":
                if ranker is None:
                    ranker = Ranking(p["retriever"])
                v = prepare(job, ranker, spans, d, p, data[job["history"]])
            else:
                v = measure(out, job, a)
        register(f, dict(v, inputs=provenance))
        print(phase, job["id"], v["status"], flush=True)
    if len(list(dest.glob("*.json"))) == len(packet["jobs"]):
        register(
            out / (phase + "-complete.json"),
            {f.stem: file_hash(f) for f in sorted(dest.glob("*.json"))},
        )
        if phase == "outcomes":
            report(out, packet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "study", "corpus"):
        parser.add_argument(name, type=Path)
    parser.add_argument("phase", choices=("writers", "prepared", "outcomes", "report"))
    args = parser.parse_args()
    run(args.base, args.study, args.corpus, args.phase)
