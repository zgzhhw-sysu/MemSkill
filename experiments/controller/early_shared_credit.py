"""Adjudicate a unique early skill using one physical repair per distinct weak action."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.controller.earliest_skill_evaluation import arm_data, available
from experiments.controller.earliest_three_skills import (
    alternate_slots,
    repair_jobs,
    verify_response,
)
from experiments.controller.exchange_paths import Ranking
from experiments.controller.exchange_questions import call
from experiments.controller.factorial_rollout import read
from experiments.controller.legacy_terminal_symmetry import (
    make_bank,
    make_writer,
    measure,
    ops,
)
from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash, register
from experiments.controller.three_skill_shared_credit import (
    EDGES,
    RULES,
    SKILLS,
    decompose,
)
from experiments.controller.writer_format_trial import PAIRS, SINGLE, reparse
from src.credit_assignment import _digest, possible_recovery, recovered
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings

PLAN = Path(__file__).with_name("protocols") / "shared_weak_repair.json"
REPAIRS = ("execution", "retrieval", "format")
PHASES = ("audit", "additional", "certified", "prepared", "outcomes")


def registration(base, study, corpus):
    src = base / "source-compression"
    early = src / "earliest-three-skills"
    folder = early / "shared-weak-credit"
    folder.mkdir(exist_ok=True)
    dest = folder / "registration.json"
    inputs = {
        "plan": file_hash(PLAN),
        "forecast": file_hash(early / "forecast.json"),
        "cohort": file_hash(src / "earlier-terminal-returns/registration.json"),
        "earlier_six": file_hash(
            src / "writer-format-credit/shared/outcomes-complete.json"
        ),
        "protocol": file_hash(study / "protocol.json"),
    }
    if dest.exists():
        if read(dest)["inputs"] != inputs:
            raise ValueError("Changed early unique-question registration")
        return dest
    p = read(study / "protocol.json")
    pred = read(early / "forecast.json")
    ws = pred["windows"]
    if (
        p["rule"] != 2
        or p["input_sha256"] != file_hash(corpus)
        or set(p["train"]) & set(p["validation"] + p["test"])
        or len(ws) != 12
        or sum(len(w["questions"]) for w in ws.values()) != 123
        or any(w["history"] not in p["train"] for w in ws.values())
        or any(pred["inputs"][k] != inputs[k] for k in ("cohort", "protocol"))
    ):
        raise ValueError("Different training census")
    cut = {f.stem: file_hash(f) for f in sorted((early / "judged").glob("conv-*.json"))}
    if not cut or not set(cut) <= set(ws):
        raise ValueError("Invalid initial cutoff")
    register(
        dest,
        {
            "inputs": inputs,
            "initial": cut,
            "windows": {n: w["questions"] for n, w in ws.items()},
        },
    )
    return dest


def candidate(row, linked, truth):
    if not linked:
        return None
    scores = [row[k]["f1"] for k in SKILLS]
    for i, name in enumerate(SKILLS):
        peers = [j for j in range(3) if j != i]
        a, b = (scores[j] for j in peers)
        if abs(a - b) > 1e-10 or abs(scores[i] - a) < 1e-10:
            continue
        edges = [scores[y] - scores[x] if i in (x, y) else 0.0 for x, y in EDGES]
        weak = {}
        for j in peers:
            lo, hi = (i, j) if scores[i] < scores[j] else (j, i)
            if not recovered(row[SKILLS[lo]], row[SKILLS[hi]], truth):
                return None
            weak.setdefault(SKILLS[lo], []).append(SKILLS[hi])
        return {"outlier": name, "edges": edges, "weak": weak}
    return None


def authenticate(ev, cache, q, job, prepared, result, row, left, right):
    weak = job["weak"]
    if result["status"] not in ("confirmed", "veto"):
        return result["status"]
    if prepared["status"] != "prepared" or job["qa"] != q:
        raise ValueError("Repair belongs to another question")
    branches = prepared["branches"]
    truth = ev.get_ground_truth(q)

    def check(ctx, answer):
        if ctx is None or answer is None:
            if ctx is not None or answer is not None:
                raise ValueError("Missing repair response")
            return
        receipt = answer["receipt"]
        verify_response(cache, receipt, sha(ev.build_prompt(q["question"], ctx, q)))
        if (
            not answer["ok"]
            or answer["answer"] != receipt["text"].strip()
            or abs(answer["f1"] - ev.compute_f1(answer["answer"], truth, q)) > 1e-10
        ):
            raise ValueError("Wrong physical answer or score")

    for branch, skill in (("control", left), ("swap", right)):
        a = result["reads"][branch]
        native = branches[branch]
        orig = row[skill]
        check(native["native"], a)
        if (
            a["answer"] != orig["answer"]
            or abs(a["f1"] - orig["f1"]) > 1e-10
            or a["receipt"]["tokens"] != orig["receipt"]["tokens"]
            or native["original"]["receipt"] != orig["receipt"]
            or native["original"]["context_hash"] != _digest(native["native"])
        ):
            raise ValueError("Weak audit and deployed action differ")
    loser = result["reads"][weak]
    high = result["reads"]["swap" if weak == "control" else "control"]
    if not recovered(loser, high, truth):
        raise ValueError("Changed original weak comparison")
    delta = row[right]["f1"] - row[left]["f1"]
    if abs(delta - job["signed_credit"]) > 1e-10:
        raise ValueError("Changed comparison direction")
    veto = []
    for k in REPAIRS:
        answer = result["repairs"][k]
        check(branches[weak][k], answer)
        if possible_recovery(loser, answer, truth):
            veto.append(k)
    status = "veto" if veto else "confirmed"
    if (
        status != result["status"]
        or veto != result["veto"]
        or abs(result["credit"] - (0 if veto else delta)) > 1e-10
    ):
        raise ValueError("Wrong competing explanation decision")
    return status


def audit(early, earlier, study, name, pred, spans, questions):
    file = name + ".json"
    judge = read(early / "judged" / file)
    pair = read(early / "pairs" / file)
    packets = read(early / "repairs" / file)
    if judge["inputs"]["parent"] != file_hash(early / "repairs" / file) or packets[
        "inputs"
    ]["parent"] != file_hash(early / "pairs" / file):
        raise ValueError("Changed repair ancestry")
    if judge["status"] != "judged":
        return {"status": judge["status"]}
    scores, levels, *_ = arm_data(
        early, earlier, study, name, pred, available(early, earlier)
    )
    native = "activity" if pred["native_activity"] else "temporal"
    ev = LoCoMoEvaluator(SimpleNamespace())
    audited = {}
    for v, result in zip(packets["jobs"], judge["cases"], strict=True):
        job = v["job"]
        j = job["question_index"]
        idx = questions[j]["original_index"]
        left = v["baseline"]
        if (
            left != result["baseline"]
            or idx != result["index"]
            or job["weak"] != result["weak"]
            or job["qa"] != questions[j]
            or abs(result["raw"] - job["signed_credit"]) > 1e-10
        ):
            raise ValueError("Unaligned repair audit")
        status = authenticate(
            ev,
            early / "responses",
            questions[j],
            job,
            v["prepared"],
            result,
            pair["rows"][j],
            left,
            "relation",
        )
        weak = left if job["weak"] == "control" else "relation"
        audited.setdefault(idx, {}).setdefault(weak, []).append(status)
    candidates = {}
    for j, (q, row) in enumerate(zip(questions, pair["rows"], strict=True)):
        idx = q["original_index"]
        if row["original_index"] != idx:
            raise ValueError("Wrong ordered question")
        truth = ev.get_ground_truth(q)
        for k, score in zip(SKILLS, scores[j], strict=True):
            r = row[k]
            original = (
                levels["readers"]["rows"][j]
                if k == "relation"
                else levels["binary"]["rows"][j]["arms"][
                    "native" if k == native else "alternative"
                ]
            )
            if r["receipt"] != original["receipt"] or r["f1"] != score or not r["ok"]:
                raise ValueError(
                    "Deployed reader is not bound to this question and skill"
                )
            if (
                r["answer"] != r["receipt"]["text"].strip()
                or abs(r["f1"] - ev.compute_f1(r["answer"], truth, q)) > 1e-10
            ):
                raise ValueError("Wrong deployed F1")
        linked = bool(set(q["evidence"]) & set(spans[-1]["source_turns"]))
        c = candidate(row, linked, truth)
        if c:
            statuses = {k: audited.get(idx, {}).get(k, []) for k in c["weak"]}
            candidates[str(idx)] = {**c, "j": j, "audits": statuses}
    return {"status": "matched", "candidates": candidates, "audited": audited}


def missing_comparisons(audits):
    if any("veto" in v for v in audits.values()):
        return []
    return [k for k, statuses in audits.items() if not statuses]


def additional(early, earlier, file, pred, spans, questions, history, proto, packet):
    groups = {}
    pair = read(early / "pairs" / file)
    for c in packet["candidates"].values():
        for weak in missing_comparisons(c["audits"]):
            high = c["weak"][weak][0]
            j = c["j"]
            row = pair["rows"][j]
            groups.setdefault((weak, high), []).append(
                {
                    "qa": questions[j],
                    "question_index": j,
                    "weak": "control",
                    "signed_credit": row[high]["f1"] - row[weak]["f1"],
                }
            )
    jobs = []
    if groups:
        with history_embeddings():
            rank = Ranking(proto["retriever"])
            for (weak, high), items in groups.items():
                result = repair_jobs(
                    rank,
                    spans,
                    questions,
                    dict(pair, jobs={weak: items}),
                    read(early / "writers" / file),
                    read(early / "prepared" / file),
                    read(earlier / "prepared" / file),
                    read(earlier / "writers" / file),
                    pred,
                    proto,
                    history,
                    comparison=high,
                )
                jobs.extend(dict(v, high=high) for v in result["jobs"])
    return {"status": "matched", "candidates": packet["candidates"], "jobs": jobs}


def certify(folder, file, questions, packet):
    ev = LoCoMoEvaluator(SimpleNamespace())
    pair = read(folder.parent / "pairs" / file)
    candidates = packet["candidates"]
    results = []
    for v in packet["jobs"]:
        job = v["job"]
        prepared = v["prepared"]
        j = job["question_index"]
        r = (
            measure(folder, job, prepared)
            if prepared["status"] == "prepared"
            else {"status": prepared["status"]}
        )
        status = authenticate(
            ev,
            folder / "responses",
            questions[j],
            job,
            prepared,
            r,
            pair["rows"][j],
            v["baseline"],
            v["high"],
        )
        idx = str(questions[j]["original_index"])
        candidates[idx]["audits"][v["baseline"]].append(status)
        results.append(
            {"index": idx, "weak": v["baseline"], "high": v["high"], "result": r}
        )
    for c in candidates.values():
        audits = list(c["audits"].values())
        if any("veto" in x for x in audits):
            c["status"] = "veto"
        elif all("confirmed" in x for x in audits):
            c["status"] = "certified"
        else:
            c["status"] = "unresolved"
    return {"status": "matched", "candidates": candidates, "additional": results}


def six_cells(early, earlier, file, spans, questions, pred, proto, packet, decision):
    selected = {
        k: c for k, c in packet["candidates"].items() if c["status"] == "certified"
    }
    result = {
        "status": "matched",
        "candidates": packet["candidates"],
        "questions": {},
        "banks": {},
    }
    if not selected:
        return result
    prefix, event = spans
    ex = make_writer(proto)
    ev = LoCoMoEvaluator(SimpleNamespace())
    two = read(earlier / "prepared" / file)
    rel = read(early / "prepared" / file)
    pair = read(early / "pairs" / file)
    native, alt = (
        ("activity", "temporal")
        if pred["native_activity"]
        else ("temporal", "activity")
    )
    writes = {
        native: (
            event["selected_skills"],
            event["executor_responses"],
            event["parsed_actions"],
        )
    }
    for k, root in ((alt, earlier), ("relation", early)):
        w = read(root / "writers" / file)
        s = (
            alternate_slots(decision, proto["skills"], event, w)
            if k == alt
            else w["selected"]
        )
        prompt = ex._build_executor_prompt(ops(s), event["source"], event["retrieved"])
        resp = w["response"]
        verify_response(root / "responses", resp, sha(prompt))
        writes[k] = (
            s,
            [
                {
                    "prompt_sha256": sha(prompt),
                    "response": resp["text"],
                    "input_tokens": resp["tokens"][0],
                    "output_tokens": resp["tokens"][1],
                }
            ],
            w["actions"],
        )
    if sha(event["source"]) != pred["source"]:
        raise ValueError("Changed predecision source")
    expected = {native: two["banks"][0], alt: two["banks"][1], "relation": rel["bank"]}
    rank = Ranking(proto["retriever"])
    queries = rank.encode(tuple(q["question"] for q in questions))
    for skill, original_rule in zip(SKILLS, RULES, strict=True):
        s, wr, acts = writes[skill]
        canonical = PAIRS if original_rule == "pairs" else SINGLE
        if s[:-1] != event["selected_skills"][:-1] or s[-1]["source_rule"] != canonical:
            raise ValueError("Different original controller slots or source rule")
        for key, rule in (("single", SINGLE), ("pairs", PAIRS)):
            arm = skill + "_" + key
            branch = reparse(ex, event, s, rule, wr, acts)
            bank = make_bank(rank, ex, prefix, event, branch, [])
            contents = bank.get_all_contents()
            bank_hash = _digest(contents)
            vecs = [m.embedding for m in bank.memories]
            pure = key == original_rule
            result["banks"][arm] = {
                "hash": bank_hash,
                "prompt": wr[0]["prompt_sha256"],
                "writer_skills": branch["writer_skills"],
            }
            for i, c in selected.items():
                j = c["j"]
                q = questions[j]
                rec = result["questions"].setdefault(
                    i, {"qa": q, "arms": {}, "mismatch": []}
                )
                ix, ctx = rank.rank(queries[j], contents, vecs)
                prompt = ev.build_prompt(q["question"], ctx, q)
                a = {
                    "indices": ix,
                    "context_hash": _digest(ctx),
                    "hash": sha(prompt),
                    "prompt": prompt,
                }
                if pure:
                    orig = (
                        rel["questions"][j]
                        if skill == "relation"
                        else two["questions"][j][
                            "native" if skill == native else "alternative"
                        ]
                    )
                    if (
                        bank_hash != expected[skill]
                        or orig["indices"] != ix
                        or orig["hash"] != sha(prompt)
                        or orig["prompt"] != prompt
                    ):
                        rec["mismatch"].append(arm)
                    a["old"] = pair["rows"][j][skill]
                rec["arms"][arm] = a
    return result


def outcomes(folder, earlier, packet):
    rows = {}
    ev = LoCoMoEvaluator(SimpleNamespace())
    for i, q in packet["questions"].items():
        if q["mismatch"]:
            rows[i] = {"status": "different_original_context", "arms": q["mismatch"]}
            continue
        qa = q["qa"]
        values, reads = {}, {}
        mismatch = []
        for arm, spec in q["arms"].items():
            if sha(spec["prompt"]) != spec["hash"]:
                raise ValueError("Different registered reader")
            root = folder.parent if arm.startswith("relation_") else earlier
            r = call(
                folder, spec["prompt"], "reader", 32, reuse_from=root / "responses"
            )
            answer = r["text"].strip()
            f1 = ev.compute_f1(answer, ev.get_ground_truth(qa), qa)
            reads[arm] = {"receipt": r, "answer": answer, "f1": f1}
            values[arm] = f1
            old = spec.get("old")
            if old and (
                old["answer"] != answer
                or old["receipt"]["tokens"] != r["tokens"]
                or abs(old["f1"] - f1) > 1e-10
            ):
                mismatch.append(arm)
        part = (
            {"status": "different_original_answer", "arms": mismatch}
            if mismatch
            else decompose(values, packet["candidates"][i]["edges"])
        )
        rows[i] = {**part, "reads": reads, "f1": values}
    return {"status": "matched", "rows": rows, "candidates": packet["candidates"]}


def run(base, study, corpus, phase, *, recheck=False):
    reg = registration(base, study, corpus)
    if phase == "register":
        return
    roster = read(reg)
    folder = reg.parent
    early = folder.parent
    earlier = early.parent / "earlier-terminal-returns"
    if phase == "markers":
        for wave in ("initial", "final"):
            names = roster["initial"] if wave == "initial" else roster["windows"]
            paths = {n: folder / "outcomes" / (n + ".json") for n in names}
            if all(p.exists() for p in paths.values()):
                marker = folder / (wave + ".json")
                register(
                    marker,
                    {
                        "registration": file_hash(reg),
                        "outcomes": {n: file_hash(p) for n, p in paths.items()},
                    },
                )
        return
    proto = read(study / "protocol.json")
    if file_hash(corpus) != proto["input_sha256"]:
        raise ValueError("Different cohort source")
    data = {d["sample_id"]: d for d in read(corpus) if d["sample_id"] in proto["train"]}
    dest = folder / phase
    dest.mkdir(exist_ok=True)
    for name in sorted(roster["windows"], key=lambda n: n not in roster["initial"]):
        file = name + ".json"
        f = dest / file
        pred = read(early / "forecast.json")["windows"][name]
        previous = (
            early / "judged" / file
            if phase == "audit"
            else folder / PHASES[PHASES.index(phase) - 1] / file
        )
        if not previous.exists():
            continue
        inputs = {"registration": file_hash(reg), "parent": file_hash(previous)}
        if (
            name in roster["initial"]
            and file_hash(early / "judged" / file) != roster["initial"][name]
        ):
            raise ValueError("Changed initial adjudication")
        if f.exists():
            if read(f)["inputs"] != inputs:
                raise ValueError("Changed early comparison")
            if not recheck:
                continue
        packet = read(previous)
        (trace,) = (study / "train" / name / "traces").glob("*.jsonl")
        if pred["trace"] != file_hash(trace) or pred["rows"] != file_hash(
            trace.parent.parent / "rows.json"
        ):
            raise ValueError("Different native decision")
        spans = [
            e
            for line in trace.read_bytes().splitlines()
            if (e := json.loads(line))["kind"] == "span"
        ]
        history = data[pred["history"]]
        qs = [dict(history["qa"][i], original_index=i) for i in roster["windows"][name]]
        if phase == "audit":
            result = audit(early, earlier, study, name, pred, spans, qs)
        elif packet["status"] != "matched":
            result = {"status": packet["status"]}
        elif phase == "additional":
            result = additional(
                early, earlier, file, pred, spans, qs, history, proto, packet
            )
        elif phase == "certified":
            result = certify(folder, file, qs, packet)
        elif phase == "prepared":
            with history_embeddings():
                result = six_cells(
                    early,
                    earlier,
                    file,
                    spans,
                    qs,
                    pred,
                    proto,
                    packet,
                    read(trace.parent.parent / "rows.json")[1],
                )
        else:
            result = outcomes(folder, earlier, packet)
        register(f, dict(result, inputs=inputs))
        print(phase, name, result["status"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        cli.add_argument(k, type=Path)
    cli.add_argument("phase", choices=("register", "markers", *PHASES))
    cli.add_argument("--recheck-audit", action="store_true")
    a = cli.parse_args()
    if a.recheck_audit and a.phase != "audit":
        cli.error("--recheck-audit requires the audit phase")
    run(a.base, a.study, a.corpus, a.phase, recheck=a.recheck_audit)
