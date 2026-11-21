"""Reconstruct the six crossed paths for the same question's verified skill comparisons."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.exchange_paths import Ranking
from experiments.controller.exchange_questions import call
from experiments.controller.semantic_writer import sha
from experiments.controller.writer_format_trial import prepare
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import history_embeddings, read_response

PLAN = Path(__file__).with_name("protocols") / "three_skill_shared_credit.json"
SKILLS = ("temporal", "activity", "relation")
RULES = ("single", "pairs", "single")
EDGES = ((0, 1), (0, 2), (1, 2))


def decompose(values, signed):
    pairs = [edge for edge, c in zip(EDGES, signed, strict=True) if c]
    if len(pairs) != 2:
        return {"status": "no_unique_two_edge_comparison"}
    (outlier,) = set(pairs[0]).intersection(pairs[1])

    def y(skill, rule):
        return values[SKILLS[skill] + "_" + rule]

    peers = [i for i in range(3) if i != outlier]
    if abs(y(peers[0], RULES[peers[0]]) - y(peers[1], RULES[peers[1]])) > 1e-10:
        return {"status": "different_peer_outcomes"}
    comparisons = {}
    for j in peers:
        a, b = RULES[outlier], RULES[j]
        raw = y(outlier, a) - y(j, b)
        edge = tuple(sorted((j, outlier)))
        confirmed = signed[EDGES.index(edge)] * (1 if outlier == edge[1] else -1)
        if abs(raw - confirmed) > 1e-10:
            raise ValueError("Original verified edge changed")
        template_p = y(outlier, b) - y(j, b)
        template_q = y(outlier, a) - y(j, a)
        fmt_p = y(j, a) - y(j, b)
        fmt_q = y(outlier, a) - y(outlier, b)
        comparisons[SKILLS[j]] = {
            "template": (template_p + template_q) / 2,
            "grouping": (fmt_p + fmt_q) / 2,
            "template_at_peer_rule": template_p,
            "template_at_outlier_rule": template_q,
            "grouping_at_peer_writer": fmt_p,
            "grouping_at_outlier_writer": fmt_q,
            "interaction": template_q - template_p,
            "raw": raw,
        }
    avg = {
        k: sum(v[k] for v in comparisons.values()) / 2
        for k in ("raw", "template", "grouping")
    }
    raw = avg["raw"]
    if abs(avg["template"] + avg["grouping"] - raw) > 1e-10 or not raw:
        raise ValueError("Unbalanced physical credit")
    aligned = [
        max(v * (1 if raw > 0 else -1), 0.0) for v in (avg["template"], avg["grouping"])
    ]
    return {
        "status": "unique",
        "outlier": SKILLS[outlier],
        "comparisons": comparisons,
        "mean": avg,
        "credit": [raw * v / sum(aligned) for v in aligned],
    }


def registration(base):
    src = base / "source-compression/writer-format-credit"
    dest = src / "shared"
    dest.mkdir(exist_ok=True)
    f = dest / "registration.json"
    donors = (
        base
        / "late-pairs/semantic-writer/three-writer/skill-graph/complete/models.json"
    )
    inputs = {
        "plan": file_hash(PLAN),
        "study": file_hash(src / "registration.json"),
        "verified": file_hash(src / "verified-complete.json"),
        "donors": file_hash(donors),
    }
    if f.exists():
        if read(f)["inputs"] != inputs:
            raise ValueError("Different shared credit census")
        return f
    jobs = {}
    for w in read(donors)["donors"]:
        name = w["window"]
        for q in w["points"]:
            if sum(v != 0 for v in q["edges"]) < 2:
                continue
            i = str(q["index"])
            pure = src / "questions" / name / (i + ".json")
            h = read(src / "verified" / (name + ".json"))
            if read(src / "verified-complete.json")[name] != file_hash(
                src / "verified" / (name + ".json")
            ) or h["rows"][i] != file_hash(pure):
                raise ValueError("Unauthenticated previously verified cells")
            jobs.setdefault(name, {})[i] = {
                "scores": q["scores"],
                "edges": q["edges"],
                "four": file_hash(pure),
            }
    register(
        f,
        {
            "inputs": inputs,
            "jobs": jobs,
            "population": sum(len(w["points"]) for w in read(donors)["donors"]),
        },
    )
    return f


def run(base, study, corpus, phase):
    f = registration(base)
    dest = f.parent
    src = dest.parent
    reg = read(f)
    if phase == "register":
        return
    rank = None
    for name, jobs in reg["jobs"].items():
        folder = dest / phase
        folder.mkdir(exist_ok=True)
        out = folder / (name + ".json")
        pf = dest / "prepared" / out.name
        parent = file_hash(f) if phase == "prepared" else file_hash(pf)
        inputs = {"registration": file_hash(f), "parent": parent}
        if out.exists():
            if read(out)["inputs"] != inputs:
                raise ValueError("Changed shared result")
            continue
        if phase == "prepared":
            proto = read(study / "protocol.json")
            reg0 = read(src / "registration.json")
            if proto["input_sha256"] != file_hash(corpus) or reg0["inputs"][
                "protocol"
            ] != file_hash(study / "protocol.json"):
                raise ValueError("Wrong registered training corpus")
            data = {
                d["sample_id"]: d
                for d in read(corpus)
                if d["sample_id"] in proto["train"]
            }
            with history_embeddings():
                if rank is None:
                    rank = Ranking(proto["retriever"])
                saved = prepare(base, study, data, rank, name, reg0, temporal_only=True)
            if saved["status"] == "prepared":
                saved["questions"] = {i: saved["questions"][i] for i in jobs}
        else:
            pack = read(pf)
            if pack["inputs"] != {"registration": file_hash(f), "parent": file_hash(f)}:
                raise ValueError("Unbound temporal rankings")
            rows = {}
            if pack["status"] == "prepared":
                ev = LoCoMoEvaluator(SimpleNamespace())
                for i, job in jobs.items():
                    q = pack["questions"][i]
                    prior = src / "questions" / name / (i + ".json")
                    old = read(prior)
                    if file_hash(prior) != job["four"]:
                        raise ValueError("Four-arm receipt changed")
                    if q["mismatch"] or old["status"] != "matched":
                        rows[i] = {
                            "status": "different_prior_reader_or_temporal_ranking"
                        }
                        continue
                    qa = q["qa"]
                    reads = {}
                    for k, a in old["reads"].items():
                        receipt = a["receipt"]
                        archive = read_response(src / "responses", receipt["key"])
                        if (
                            not archive
                            or archive["sha256"] != receipt["sha256"]
                            or archive["payload"]["response"].strip() != a["answer"]
                        ):
                            raise ValueError("Missing four-arm GPU response")
                    values = {k: v["f1"] for k, v in old["reads"].items()}
                    for k, a in q["arms"].items():
                        if sha(a["prompt"]) != a["hash"]:
                            raise ValueError("Changed temporal reader prompt")
                        rr = call(
                            dest,
                            a["prompt"],
                            "reader",
                            32,
                            reuse_from=src / "responses",
                        )
                        answer = rr["text"].strip()
                        score = ev.compute_f1(answer, ev.get_ground_truth(qa), qa)
                        values[k] = score
                        reads[k] = {"answer": answer, "f1": score, "receipt": rr}
                    t = q["arms"]["temporal_single"]["old"]
                    result = reads["temporal_single"]
                    if (
                        result["answer"] != t["answer"]
                        or result["receipt"]["tokens"] != t["tokens"]
                        or abs(result["f1"] - t["f1"]) > 1e-10
                    ):
                        rows[i] = {
                            "status": "different_original_temporal_answer",
                            "reads": reads,
                        }
                        continue
                    if any(
                        abs(values[s + "_" + rule] - y) > 1e-10
                        for s, rule, y in zip(SKILLS, RULES, job["scores"], strict=True)
                    ):
                        raise ValueError("Different deployed triangle")
                    rows[i] = {
                        **decompose(values, job["edges"]),
                        "f1": values,
                        "reads": reads,
                        "four": job["four"],
                    }
            saved = {"status": pack["status"], "rows": rows}
        register(out, dict(saved, inputs=inputs))
        print(phase, name, saved["status"], flush=True)
    register(
        dest / (phase + "-complete.json"),
        {k: file_hash(dest / phase / (k + ".json")) for k in reg["jobs"]},
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        cli.add_argument(k, type=Path)
    cli.add_argument("phase", choices=("register", "prepared", "outcomes"))
    a = cli.parse_args()
    run(a.base, a.study, a.corpus, a.phase)
