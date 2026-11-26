"""Cross two real writer responses with two source rules at the same archived decision."""

import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.capped_source_repair import writer
from experiments.controller.exchange_paths import Ranking
from experiments.controller.exchange_questions import call
from experiments.controller.legacy_terminal_symmetry import make_bank, make_writer, ops
from experiments.controller.paired_grounding import window
from experiments.controller.relation_skill_repairs import activity_rows, check_response
from experiments.controller.semantic_writer import REL, sha
from experiments.controller.skill_graph_data import training_rows
from experiments.controller.three_writer_credit import ACTIVITY
from src.credit_assignment import _digest, execution_fault
from src.episode_trace import skill_spec
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings
from src.replay import restore_text

PLAN = Path(__file__).with_name("protocols") / "writer_format_credit.json"
SINGLE = "dated_single_turns"
PAIRS = "dated_adjacent_turns"
CELLS = ("activity_pairs", "activity_single", "relation_pairs", "relation_single")
NATIVE = ("activity_pairs", "relation_single")


def effects(v):
    ap, a1, rp, r1 = (v[k] for k in CELLS)
    template = ((rp - ap) + (r1 - a1)) / 2
    grouping = ((a1 - ap) + (r1 - rp)) / 2
    return {
        "template": template,
        "grouping": grouping,
        "interaction": r1 - rp - a1 + ap,
        "raw": r1 - ap,
        "template_at_pairs": rp - ap,
        "template_at_single": r1 - a1,
        "grouping_at_activity": a1 - ap,
        "grouping_at_relation": r1 - rp,
    }


def shares(parts, verified):
    if not verified:
        return [0.0, 0.0]
    if (
        abs(parts["raw"] - verified) > 1e-10
        or abs(parts["template"] + parts["grouping"] - verified) > 1e-10
    ):
        raise ValueError("Crossed responses do not recover the verified contrast")
    values = [
        max(0.0, (1 if verified > 0 else -1) * parts[k])
        for k in ("template", "grouping")
    ]
    return [verified * v / sum(values) for v in values]


def registration(base, study):
    src = base / "source-compression"
    out = src / "writer-format-credit"
    out.mkdir(exist_ok=True)
    graph = (
        base
        / "late-pairs/semantic-writer/three-writer/skill-graph/complete/models.json"
    )
    audit = src / "activity-relation-repairs/0/outcomes-complete.json"
    f = out / "registration.json"
    inputs = {
        "plan": file_hash(PLAN),
        "graph": file_hash(graph),
        "audit": file_hash(audit),
        "protocol": file_hash(study / "protocol.json"),
    }
    if f.exists():
        if read(f)["inputs"] != inputs:
            raise ValueError("Changed component audit")
        return f
    rows, excluded = training_rows(base, True)
    if rows != read(graph)["donors"]:
        raise ValueError("Different original donors")
    flags = {
        w["window"]: {
            str(q["index"]): q["edges"][2] for q in w["points"] if q["edges"][2]
        }
        for w in rows
    }
    ids = {w["window"]: [q["index"] for q in w["points"]] for w in rows}
    if len(set(ids) | set(excluded)) != 12 or sum(map(len, ids.values())) != 74:
        raise ValueError("Changed direct-edge cohort")
    register(
        f,
        {
            "inputs": inputs,
            "flags": flags,
            "questions": ids,
            "excluded": excluded,
            "before": {
                k: {p.stem: file_hash(p) for p in (out / k).glob("conv-*.json")}
                for k in ("prepared", "verified", "readers")
            },
        },
    )
    return f


def revise(selected, rule):
    op = SimpleNamespace(**{**selected[-1], "source_rule": rule})
    return [*selected[:-1], skill_spec(op)]


def reparse(ex, event, selected, rule, receipts, expected):
    deployed = ops(selected)
    modified = ops(revise(selected, rule))
    request = ex._build_executor_prompt(deployed, event["source"], event["retrieved"])
    if (
        len(receipts) != 1
        or receipts[0]["prompt_sha256"] != sha(request)
        or ex._build_executor_prompt(modified, event["source"], event["retrieved"])
        != request
    ):
        raise ValueError("Formatting changed the archived writer request")
    replay = dict(event, executor_responses=receipts)
    if writer(ex, deployed, replay, repair=False) != expected:
        raise ValueError("Original parsing did not repeat")
    actions = writer(ex, modified, replay, repair=False)
    skills = [skill_spec(op) for op in modified]
    if execution_fault(actions, True, skills) or actions != writer(
        ex, modified, replay, repair=True
    ):
        raise ValueError("Changed parser contract")
    return {"actions": actions, "writer_skills": skills}


def prepare(base, study, data, rank, name, reg, *, temporal_only=False):
    src = base / "source-compression"
    archive = src / "factorial"
    sem = src / "semantic-writer"
    fold = sem / name
    proto = read(study / "protocol.json")
    ex = make_writer(proto)
    sid, n = name.rsplit("_s", 1)
    qs = window(data[sid], int(n), 2)["qa"]
    truth = {q["original_index"]: q for q in qs}
    if list(truth) != reg["questions"][name]:
        raise ValueError("Different question cohort")
    cell = archive / name / "00"
    (trace,) = (cell / "traces").glob("*.jsonl")
    temporal = [
        e
        for line in trace.read_bytes().splitlines()
        if (e := json.loads(line))["kind"] == "span"
    ]
    old = read(fold / "pair.json")
    rel = read(fold / "write.json")
    choice = read(fold / "choice.json")
    if (
        not old["matched"]
        or old["source_trace"] != file_hash(trace)
        or choice["control_trace"] != file_hash(trace)
        or choice["fixed_prefix"] != _digest(list(restore_text([temporal[0]]).items()))
        or old["choice"] != choice
        or rel["choice"] != file_hash(fold / "choice.json")
    ):
        raise ValueError("Changed relation writer")
    check_response(rel["writer"], sem)
    activity = archive / name / "01"
    spans, rows = activity_rows(
        activity, archive / "responses", temporal, qs, old["rows"], proto
    )
    if rows is None:
        raise ValueError("Previously matched control is faulty")
    prefix, end = spans
    check_response(end["executor_responses"], activity, cache=archive / "responses")
    if (
        choice["selected"][:-1] != end["selected_skills"][:-1]
        or choice["selected"][-1]["name"] != REL
        or choice["selected"][-1]["source_rule"] != SINGLE
        or end["selected_skills"][-1]["name"] != ACTIVITY
        or end["selected_skills"][-1]["source_rule"] != PAIRS
    ):
        raise ValueError("Different deployed skill pair")
    expected = {
        "activity_pairs": list(restore_text(spans).values()),
        "relation_single": rel["contents"],
    }
    ev = LoCoMoEvaluator(SimpleNamespace())
    queries = rank.encode(tuple(q["question"] for q in qs))
    packets = {
        str(i): {
            "qa": q,
            "flag": reg["flags"][name].get(str(i), 0.0),
            "arms": {},
            "mismatch": [],
        }
        for i, q in truth.items()
    }
    writes = {
        "activity": (
            end["selected_skills"],
            end["executor_responses"],
            end["parsed_actions"],
        ),
        "relation": (choice["selected"], rel["writer"], rel["actions"]),
    }
    cells = CELLS
    if temporal_only:
        orig = temporal[-1]
        if (
            orig["selected_skills"][-1]["name"] != "capture_temporal_context"
            or orig["selected_skills"][-1]["source_rule"] != SINGLE
        ):
            raise ValueError("Different original temporal write")
        check_response(orig["executor_responses"], cell, cache=archive / "responses")
        writes["temporal"] = (
            orig["selected_skills"],
            orig["executor_responses"],
            orig["parsed_actions"],
        )
        expected = {"temporal_single": list(restore_text(temporal).values())}
        cells = ("temporal_single", "temporal_pairs")
    banks = {}
    for key in cells:
        who, format_name = key.split("_")
        sel, wr, acts = writes[who]
        branch = reparse(
            ex, end, sel, SINGLE if format_name == "single" else PAIRS, wr, acts
        )
        bank = make_bank(rank, ex, prefix, end, branch, [])
        contents = bank.get_all_contents()
        if key in expected and contents != expected[key]:
            return {"status": "different_pure_bank", "arm": key}
        banks[key] = {
            "hash": _digest(contents),
            "writer_prompt": wr[0]["prompt_sha256"],
            "skills": branch["writer_skills"],
        }
        vec = [m.embedding for m in bank.memories]
        for v, q, orig in zip(queries, qs, rows, strict=True):
            idx, ctx = rank.rank(v, contents, vec)
            prompt = ev.build_prompt(q["question"], ctx, q)
            record = packets[str(q["original_index"])]
            arm = {
                "prompt": prompt,
                "hash": sha(prompt),
                "indices": idx,
                "context_hash": _digest(ctx),
            }
            if key in expected:
                previous = orig[who]
                if (
                    previous["indices"] != idx
                    or previous["context_hash"] != arm["context_hash"]
                ):
                    record["mismatch"].append(key)
                arm["old"] = {k: previous[k] for k in ("answer", "f1", "tokens")}
            record["arms"][key] = arm
    return {"status": "prepared", "banks": banks, "questions": packets}


def measure(out, pkt, cache):
    if pkt["mismatch"]:
        return {"status": "different_pure_ranking", "arms": pkt["mismatch"]}
    ev = LoCoMoEvaluator(SimpleNamespace())
    qa = pkt["qa"]
    rows = {}
    for k in (*NATIVE, "activity_single", "relation_pairs"):
        arm = pkt["arms"][k]
        if sha(arm["prompt"]) != arm["hash"]:
            raise ValueError("Changed physical reader prompt")
        resp = call(out, arm["prompt"], "reader", 32, reuse_from=cache)
        answer = resp["text"].strip()
        f1 = ev.compute_f1(answer, ev.get_ground_truth(qa), qa)
        rows[k] = {
            "answer": answer,
            "f1": f1,
            "receipt": resp,
            "context_hash": arm["context_hash"],
        }
        if k in NATIVE and (
            answer != arm["old"]["answer"]
            or abs(f1 - arm["old"]["f1"]) > 1e-10
            or resp["tokens"] != arm["old"]["tokens"]
        ):
            return {"status": "different_pure_answer", "arm": k, "reads": rows}
    parts = effects({k: v["f1"] for k, v in rows.items()})
    return {
        "status": "matched",
        "reads": rows,
        "effects": parts,
        "credit": shares(parts, pkt["flag"]),
    }


def question_receipt(out, name, i, root, inputs):
    qfile = out / "questions" / name / (str(i) + ".json")
    qfile.parent.mkdir(exist_ok=True, parents=True)
    with qfile.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not qfile.exists():
            record = measure(
                out,
                root["questions"][str(i)],
                out.parent / "activity-relation-repairs/0/responses",
            )
            register(qfile, dict(record, inputs=inputs))
        if read(qfile)["inputs"] != inputs:
            raise ValueError("Unbound question return")
    return qfile


def run(base, study, corpus, phase):
    f = registration(base, study)
    reg = read(f)
    out = f.parent
    if phase == "register":
        return
    flags = reg["flags"]
    names = sorted(flags, key=lambda k: (not flags[k], k))
    if phase == "verified":
        names = [k for k in names if flags[k]]
    dest = out / phase
    dest.mkdir(exist_ok=True)
    rank = None
    batch = 0
    for name in names:
        p = dest / (name + ".json")
        prior = out / "prepared" / p.name
        if phase != "prepared" and not prior.is_file():
            continue
        inputs = {
            "registration": file_hash(f),
            "parent": file_hash(f if phase == "prepared" else prior),
        }
        if p.exists():
            if read(p)["inputs"] != inputs:
                raise ValueError("Measurement inputs changed")
            continue
        if phase == "prepared":
            proto = read(study / "protocol.json")
            if file_hash(corpus) != proto["input_sha256"] or set(proto["train"]) & set(
                proto["validation"] + proto["test"]
            ):
                raise ValueError("Changed training corpus")
            data = {
                d["sample_id"]: d
                for d in read(corpus)
                if d["sample_id"] in proto["train"]
            }
            with history_embeddings():
                if rank is None:
                    rank = Ranking(proto["retriever"])
                v = prepare(base, study, data, rank, name, reg)
        else:
            root = read(prior)
            if root["status"] != "prepared":
                v = {"status": root["status"], "rows": {}}
            else:
                ids = (
                    flags[name]
                    if phase == "verified"
                    else [str(i) for i in reg["questions"][name]]
                )
                answers = {}
                for i in ids:
                    qfile = question_receipt(out, name, i, root, inputs)
                    answers[i] = file_hash(qfile)
                v = {"status": "answered", "rows": answers}
        register(p, dict(v, inputs=inputs))
        print(phase, name, v["status"], flush=True)
        batch += 1
        if phase != "prepared" or batch == 3:
            return
    if all((dest / (k + ".json")).exists() for k in names):
        register(
            out / (phase + "-complete.json"),
            {k: file_hash(dest / (k + ".json")) for k in names},
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        cli.add_argument(k, type=Path)
    cli.add_argument("phase", choices=("register", "prepared", "verified", "readers"))
    a = cli.parse_args()
    run(a.base, a.study, a.corpus, a.phase)
