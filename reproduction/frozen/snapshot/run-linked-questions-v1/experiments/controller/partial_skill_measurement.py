"""Recover question-local temporal/activity comparisons with an authenticated weak repair."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.earliest_three_skills import verify_response
from experiments.controller.exchange_paths import Ranking
from experiments.controller.exchange_questions import call
from experiments.controller.factorial_rollout import read
from experiments.controller.legacy_terminal_symmetry import (
    make_bank,
    make_writer,
    ops,
    source,
)
from experiments.controller.resume_status import check_equal_source_choice
from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.writer_format_trial import PAIRS, SINGLE, reparse, shares
from src.credit_assignment import _digest, possible_recovery, recovered
from src.episode_trace import skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings

PLAN = Path(__file__).with_name("protocols") / "partial_skill_transport.json"


def registration(base, study, corpus):
    src = base / "source-compression"
    legacy = src / "legacy-terminal-symmetry"
    shared = src / "earliest-three-skills/shared-weak-credit"
    out = src / "partial-skill-transport"
    out.mkdir(exist_ok=True)
    f = out / "registration.json"
    inputs = {
        k: file_hash(p)
        for k, p in {
            "plan": PLAN,
            "legacy": legacy / "registration.json",
            "legacy_audit": legacy / "summary.json",
            "initial": shared / "initial.json",
            "cohort": shared / "registration.json",
            "old": src / "writer-format-credit/shared/outcomes-complete.json",
            "protocol": study / "protocol.json",
        }.items()
    }
    if f.exists():
        if read(f)["inputs"] != inputs:
            raise ValueError("Changed partial edge registration")
        return f
    reg = read(legacy / "registration.json")
    proto = read(study / "protocol.json")
    if (
        reg["protocol"] != inputs["protocol"]
        or file_hash(corpus) != reg["corpus"]
        or reg["corpus"] != proto["input_sha256"]
        or proto["rule"] != 2
        or set(proto["train"]) & set(proto["validation"] + proto["test"])
    ):
        raise ValueError("Different original native split")
    summary = read(legacy / "summary.json")
    if summary["registration"] != inputs["legacy"] or summary["outcomes"] != file_hash(
        legacy / "outcomes-complete.json"
    ):
        raise ValueError("Unbound legacy audits")
    cases = {r["id"]: r for r in summary["cases"]}
    if set(cases) != {j["id"] for j in reg["jobs"]}:
        raise ValueError("Incomplete historical candidate census")
    panel = base / "late-pairs/semantic-writer/three-writer"
    register(
        f,
        {
            "inputs": inputs,
            "cases": cases,
            "existing": {p.name: file_hash(p) for p in panel.glob("conv-*.json")},
            "early": {
                p.stem: file_hash(p) for p in (shared / "outcomes").glob("conv-*.json")
            },
        },
    )
    return f


def verify_read(ev, cache, qa, ctx, answer):
    if ctx is None and answer is None:
        return
    if ctx is None or answer is None or not answer["ok"]:
        raise ValueError("Unmatched physical read")
    rr = answer["receipt"]
    verify_response(cache, rr, sha(ev.build_prompt(qa["question"], ctx, qa)))
    if (
        rr["text"].strip() != answer["answer"]
        or abs(
            ev.compute_f1(answer["answer"], ev.get_ground_truth(qa), qa) - answer["f1"]
        )
        > 1e-10
    ):
        raise ValueError("Changed physical F1")


def prepare(rank, study, folder, reg, j, proto, history):
    legacy = folder.parent / "legacy-terminal-symmetry"
    name = j["id"] + ".json"
    files = [legacy / k / name for k in ("writers", "prepared", "outcomes")]
    writer, prep, result = map(read, files)
    parent = None
    for k, p, d in zip(
        ("writers", "prepared", "outcomes"), files, (writer, prep, result), strict=True
    ):
        expected = {"registration": file_hash(legacy / "registration.json")}
        if parent is not None:
            expected["parent"] = file_hash(parent)
        if d["inputs"] != expected or read(legacy / (k + "-complete.json"))[
            j["id"]
        ] != file_hash(p):
            raise ValueError("Changed original weak audit")
        parent = p
    ev = LoCoMoEvaluator(SimpleNamespace())
    qa = j["qa"]
    idx = j["original_index"]
    truth = ev.get_ground_truth(qa)
    ep = study / "train" / j["window"]
    qs = read(ep / "questions.json")
    row_file = ep / "rows.json"
    if (
        history["qa"][idx] != qa
        or qs[j["question_index"]] != idx
        or file_hash(row_file) != reg["windows"][ep.name]["rows"]
        or file_hash(ep / "questions.json") != reg["windows"][ep.name]["questions"]
    ):
        raise ValueError("Different original question")
    spans, d = source(j, reg, study)
    prefix, event = spans
    if j["pivot"] != 1 or not set(qa["evidence"]).intersection(event["source_turns"]):
        raise ValueError("Different focal decision")
    row = read(row_file)[1]
    check_equal_source_choice(
        row, proto["skills"], event["selected_skills"], d["alternative"]
    )
    if row["use"] != j["use"]:
        raise ValueError("Wrong native skill")
    if result["status"] not in ("confirmed", "veto"):
        return {"status": result["status"]}
    if prep["status"] != "prepared" or writer["status"] != "matched_writer":
        raise ValueError("Incomplete weaker action audit")
    weak = j["weak"]
    high = "swap" if weak == "control" else "control"
    rr = result["reads"]
    for k in ("control", "swap"):
        v = prep["branches"][k]
        orig = d[k]["qa"][str(j["question_index"])]
        verify_read(ev, legacy / "responses", qa, v["native"], rr[k])
        if (
            v["original"] != orig
            or _digest(v["native"]) != orig["context_hash"]
            or rr[k]["answer"] != orig["answer"]
            or rr[k]["receipt"]["tokens"] != orig["tokens"]
            or abs(rr[k]["f1"] - orig["f1"]) > 1e-10
        ):
            raise ValueError("Different original action answer")
    veto = []
    for k in ("execution", "retrieval", "format"):
        v = result["repairs"][k]
        verify_read(ev, legacy / "responses", qa, prep["branches"][weak][k], v)
        if possible_recovery(rr[weak], v, truth):
            veto.append(k)
    a, t = ("control", "swap") if j["use"] else ("swap", "control")
    delta = rr[a]["f1"] - rr[t]["f1"]
    if (
        not recovered(rr[weak], rr[high], truth)
        or result["veto"] != veto
        or (result["status"] == "veto") != bool(veto)
        or abs(delta - j["signed_credit"]) > 1e-10
        or abs(result["credit"] - (0 if veto else delta)) > 1e-10
    ):
        raise ValueError("Different audited causal contrast")
    if veto:
        return {"status": "veto", "reasons": veto}
    ex = make_writer(proto)
    raw = writer["writer"]
    selected = d["alternative"]
    request = ex._build_executor_prompt(
        ops(selected), event["source"], event["retrieved"]
    )
    verify_response(legacy / "responses", raw, sha(request))
    writes = {
        "control": (event["selected_skills"], event["executor_responses"]),
        "swap": (
            selected,
            [
                {
                    "prompt_sha256": sha(request),
                    "response": raw["text"],
                    "input_tokens": raw["tokens"][0],
                    "output_tokens": raw["tokens"][1],
                }
            ],
        ),
    }
    qvec = rank.encode(tuple(history["qa"][i]["question"] for i in qs))[
        j["question_index"]
    ]
    arms = {}
    mismatch = []
    for k, who, canonical in ((a, "activity", PAIRS), (t, "temporal", SINGLE)):
        sel, receipts = writes[k]
        if (
            sel[-1]["source_rule"] != canonical
            or [skill_spec(op) for op in ops(sel)] != d[k]["writer_skills"]
        ):
            raise ValueError("Changed writer choice")
        for label, rule in (("single", SINGLE), ("pairs", PAIRS)):
            key = who + "_" + label
            branch = reparse(ex, event, sel, rule, receipts, d[k]["actions"])
            bank = make_bank(rank, ex, prefix, event, branch, [])
            contents = bank.get_all_contents()
            ix, ctx = rank.rank(qvec, contents, [m.embedding for m in bank.memories])
            prompt = ev.build_prompt(qa["question"], ctx, qa)
            arm = {
                "hash": sha(prompt),
                "prompt": prompt,
                "bank": _digest(contents),
                "indices": ix,
                "context_hash": _digest(ctx),
                "writer": receipts[0]["prompt_sha256"],
                "selected": sel,
                "deployed": branch["writer_skills"],
            }
            if rule == canonical:
                orig = prep["branches"][k]
                if (
                    d[k]["bank_hash"] != _digest(contents)
                    or orig["original"]["indices"] != ix
                    or orig["native"] != ctx
                ):
                    mismatch.append(key)
                arm["old"] = rr[k]
            arms[key] = arm
    return {
        "status": "changed_original_ranking" if mismatch else "prepared",
        "mismatch": mismatch,
        "qa": qa,
        "delta": delta,
        "arms": arms,
    }


def measure(folder, packet):
    if packet["status"] != "prepared":
        return {"status": packet["status"]}
    ev = LoCoMoEvaluator(SimpleNamespace())
    qa = packet["qa"]
    reads = {}
    y = {}
    for k, arm in packet["arms"].items():
        if sha(arm["prompt"]) != arm["hash"]:
            raise ValueError("Different crossed writer input")
        r = call(
            folder,
            arm["prompt"],
            "reader",
            32,
            reuse_from=folder.parent / "legacy-terminal-symmetry/responses",
        )
        answer = r["text"].strip()
        y[k] = ev.compute_f1(answer, ev.get_ground_truth(qa), qa)
        reads[k] = {"answer": answer, "receipt": r, "f1": y[k]}
        old = arm.get("old")
        if old and (
            answer != old["answer"]
            or r["tokens"] != old["receipt"]["tokens"]
            or abs(y[k] - old["f1"]) > 1e-10
        ):
            return {"status": "different_native_answer", "arm": k, "reads": reads}
    template = (
        (y["activity_pairs"] - y["temporal_pairs"])
        + (y["activity_single"] - y["temporal_single"])
    ) / 2
    grouping = (
        (y["activity_pairs"] - y["activity_single"])
        + (y["temporal_pairs"] - y["temporal_single"])
    ) / 2
    parts = {
        "raw": y["activity_pairs"] - y["temporal_single"],
        "template": template,
        "grouping": grouping,
    }
    return {
        "status": "matched",
        "parts": parts,
        "shares": shares(parts, packet["delta"]),
        "reads": reads,
    }


def run(base, study, corpus, phase):
    f = registration(base, study, corpus)
    if phase == "register":
        return
    folder = f.parent
    reg = read(folder.parent / "legacy-terminal-symmetry/registration.json")
    data = {
        d["sample_id"]: d
        for d in read(corpus)
        if d["sample_id"] in read(study / "protocol.json")["train"]
    }
    dest = folder / phase
    dest.mkdir(exist_ok=True)
    rank = None
    marks = {}
    for j in reg["jobs"]:
        name = j["id"] + ".json"
        path = dest / name
        prev = f if phase == "prepared" else folder / "prepared" / name
        inputs = {"registration": file_hash(f), "parent": file_hash(prev)}
        if not path.exists():
            if phase == "prepared":
                with history_embeddings():
                    if rank is None:
                        rank = Ranking(read(study / "protocol.json")["retriever"])
                    result = prepare(
                        rank,
                        study,
                        folder,
                        reg,
                        j,
                        read(study / "protocol.json"),
                        data[j["history"]],
                    )
            else:
                result = measure(folder, read(prev))
            register(path, dict(result, inputs=inputs))
            print(phase, j["id"], result["status"], flush=True)
        if read(path)["inputs"] != inputs:
            raise ValueError("Changed partial experiment")
        marks[j["id"]] = file_hash(path)
    register(folder / (phase + "-complete.json"), marks)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        cli.add_argument(k, type=Path)
    cli.add_argument("phase", choices=("register", "prepared", "outcomes"))
    a = cli.parse_args()
    run(a.base, a.study, a.corpus, a.phase)
