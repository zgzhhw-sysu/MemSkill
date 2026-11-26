"""Replay one fixed draw on measured three-action paths and account for terminal costs."""

import hashlib
import json

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.witness_passage_policy import limited
from src.replay import restore_text

SALT = "memskill-three-skill-replay-v1"
NAMES = (
    "capture_temporal_context",
    "capture_activity_details",
    "handle_entity_relationships",
)


def uniform(panel_hash, name):
    key = f"{SALT}/{panel_hash}/{name}".encode()
    # 53-bit mantissa, half-open [0,1).
    return (int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11) / 2**53


def decisions(probabilities, raw, draw):
    high = probabilities.cumsum(dim=1)
    if (
        not 0 <= draw < 1
        or not probabilities.isfinite().all()
        or ((high[:, -1] - 1).abs() > 1e-10).any()
        or (probabilities < 0).any()
    ):
        raise ValueError("Invalid action distribution")
    low = torch.cat((torch.zeros_like(high[:, :1]), high[:, :-1]), dim=1)
    overlap = (torch.minimum(high, high[raw]) - torch.maximum(low, low[raw])).clamp_min(
        0
    )
    return (high[:, :-1] <= draw).sum(1), (1 - overlap.sum(1)).clamp(0, 1)


def branches(base, w, scores):
    late = base / "late-pairs"
    name = scores["window"]
    sem = late / "semantic-writer" / name
    att = read(late / "semantic-writer/three-writer" / (name + ".json"))
    pair = read(sem / "pair.json")
    choice = read(sem / "choice.json")
    write = read(sem / "write.json")
    audit = base / "source-compression/retrieval-competition/pairs" / (name + ".json")
    if (
        att["semantic"] != file_hash(sem / "pair.json")
        or att["activity"] != file_hash(audit)
        or write["choice"] != file_hash(sem / "choice.json")
        or pair["choice"] != choice
    ):
        raise ValueError("Unbound measured write or third skill")
    physical = read(audit)
    result = []
    digest = {}
    for a, skill in enumerate(NAMES):
        if a < 2:
            bits = f"{w['background']}{a}"
            cell = late / "outcomes" / name / bits
            (trace,) = (cell / "traces").glob("*.jsonl")
            h = file_hash(trace)
            f = cell / "scores.json"
            digest[bits] = {"trace": h, "scores": file_hash(f)}
            if digest[bits]["scores"] != physical["paths"][bits] or (
                a == 0 and h != pair["source_trace"]
            ):
                raise ValueError("Changed physical branch")
            ep = [
                e
                for line in trace.read_bytes().splitlines()
                if (e := json.loads(line))["kind"] == "span"
            ]
            prefix, end = ep
            state = (
                prefix["selected_skills"],
                list(restore_text([prefix]).items()),
                end["source"],
                end["retrieved"],
                end["selected_skills"][:-1],
            )
            if a == 0:
                temporal_state = state
            elif state != temporal_state:
                raise ValueError(
                    "Different pre-action state for measured activity write"
                )
            selected = end["selected_skills"]
            receipts = end["executor_responses"]
            answers = read(f)["answers"]
            if [q["qa_index"] for q in answers] != list(range(len(w["questions"]))):
                raise ValueError("Incomplete reader answers")
        else:
            selected = choice["selected"]
            receipts = write["writer"]
            answers = [r["relation"] for r in pair["rows"]]
        if (
            selected[-1]["name"] != skill
            or not receipts
            or len(answers) != len(w["questions"])
            or [q["index"] for q in att["rows"]] != w["questions"]
            or any(
                abs(q["f1"] - r["f1"][a]) > 1e-10
                for q, r in zip(answers, att["rows"], strict=True)
            )
        ):
            raise ValueError("Changed terminal selection or measured scores")
        result.append(
            {
                "skills": [
                    {"name": v["name"], "version": v["version"]} for v in selected
                ],
                "writer": [
                    sum(r[k] for r in receipts)
                    for k in ("input_tokens", "output_tokens")
                ],
                "reader": [q["tokens"] for q in answers],
                "f1": [q["f1"] for q in answers],
            }
        )
    if any(r["skills"][:-1] != result[0]["skills"][:-1] for r in result):
        raise ValueError("Fixed skill slots changed")
    digest["relation_write"] = file_hash(sem / "write.json")
    return result, digest


def record(base, path, result):
    out = path.parent / "decisions"
    out.mkdir(exist_ok=True)
    late = base / "late-pairs/semantic-writer"
    panel = late / "panel.json"
    h = file_hash(panel)
    specs = {f"{w['dialogue']}_s{w['start']}": w for w in read(panel)["windows"]}
    records = []
    for row in result["rows"]:
        name = row["window"]
        dest = out / (name + ".json")
        inputs = {
            "forecast": file_hash(path),
            "triplet": result["sources"][name],
            "panel": h,
            "draw_salt": SALT,
        }
        arms, digests = branches(base, specs[name], row)
        keys = list(row["p"])
        u = uniform(h, name)
        with limited():
            p = torch.tensor(
                [row["p"][k] for k in keys], device="cuda", dtype=torch.float64
            )
            ids, chances = decisions(p, keys.index("raw"), u)
            y = p.new_tensor([r["f1"] for r in arms])
            expectations = p @ y
            if any(
                abs(float(yhat) - row["f1"][k]) > 1e-10
                for k, yhat in zip(keys, expectations.mean(1), strict=True)
            ):
                raise ValueError("Replay and expected F1 differ")
            reports = {}
            for key, i, chance, probs in zip(
                keys, ids.tolist(), chances.tolist(), p.tolist(), strict=True
            ):
                reports[key] = {
                    "action": i,
                    "p": probs,
                    "chance_different_from_raw": chance,
                    "draw_changed_from_raw": i != int(ids[keys.index("raw")]),
                    "f1_sum": sum(arms[i]["f1"]),
                }
        data = {
            "inputs": inputs,
            "paths": digests,
            "uniform": u,
            "history": row["history"],
            "partition": row["partition"],
            "questions": specs[name]["questions"],
            "branches": arms,
            "policies": reports,
        }
        register(dest, data)
        records.append(data)
    for partition in ("donor", "evaluation"):
        cases = [v for v in records if v["partition"] == partition]
        if not cases:
            continue
        n = sum(len(v["questions"]) for v in cases)
        totals = {}
        for k in cases[0]["policies"]:
            sampled_f1 = 0.0
            changed = 0
            expected_changes = 0.0
            all_same = 1.0
            writer = [0, 0]
            reader = [0, 0]
            expected_writer = [0.0, 0.0]
            expected_reader = [0.0, 0.0]
            for v in cases:
                a = v["policies"][k]
                sample = v["branches"][a["action"]]
                sampled_f1 += a["f1_sum"]
                changed += a["draw_changed_from_raw"]
                expected_changes += a["chance_different_from_raw"]
                all_same *= 1 - a["chance_different_from_raw"]
                for j in range(2):
                    writer[j] += sample["writer"][j]
                    reader[j] += sum(t[j] for t in sample["reader"])
                    expected_writer[j] += sum(
                        p * r["writer"][j]
                        for p, r in zip(a["p"], v["branches"], strict=True)
                    )
                    expected_reader[j] += sum(
                        p * sum(t[j] for t in r["reader"])
                        for p, r in zip(a["p"], v["branches"], strict=True)
                    )
            totals[k] = {
                "draw_f1": sampled_f1 / n,
                "changed_windows": changed,
                "expected_changed_windows": expected_changes,
                "probability_any_change": 1 - all_same,
                "writer_tokens": writer,
                "reader_tokens": reader,
                "expected_writer_tokens": expected_writer,
                "expected_reader_tokens": expected_reader,
            }
        register(
            out / f"{partition}-{len(result['sources']):02d}.json",
            {
                "inputs": {
                    "forecast": file_hash(path),
                    "sources": result["sources"],
                    "salt": SALT,
                },
                "windows": len(cases),
                "questions": n,
                "policies": totals,
            },
        )
