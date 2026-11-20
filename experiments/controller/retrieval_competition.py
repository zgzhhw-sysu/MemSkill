"""Price both possible writes against the actual prefix bank before seeing the question."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.factorial_rollout import read
from experiments.controller.late_pair_trial import paths
from experiments.controller.outcome_credit import matched_offset
from experiments.controller.source_compression import file_hash, register
from experiments.controller.turn_key_credit import fit, offset, paired_interval
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn
from src.replay import dated_group_parts, restore_text

PLAN = Path(__file__).with_name("protocols") / "retrieval_competition.json"
PATHS = ("00", "01", "10", "11")
ARMS = (
    "native",
    "terminal",
    "credit",
    "return_only",
    "rate",
    "root_only",
    "last_only",
    "temporal",
    "activity",
)


def simulation(query, vectors, memory, records, probe_ids, adjacent_ids):
    """Count a source turn only once, even if multiple of its chunks were retrieved."""
    n = len(probe_ids)
    texts = [*memory, *(r[0] for r in records)]
    scores = query @ torch.stack([vectors[t] for t in texts]).T
    ranks = scores.argsort(dim=1, descending=True, stable=True)[:, :5]
    # Source ID -1 marks a prior item or a missing second turn.
    ids = [[-1, -1] for _ in memory]
    for _, parts in records:
        keys = [i for i, _ in parts]
        ids.append(keys + [-1] * (2 - len(keys)))
    sources = torch.as_tensor(ids, device=query.device)[ranks[:n]].flatten(1)
    own = (sources == probe_ids[:, None]).any(1).float().mean()
    has_partner = adjacent_ids >= 0
    companion = (
        (sources == adjacent_ids[:, None]).any(1) & has_partner
    ).sum() / has_partner.sum().clamp_min(1)
    retained = (
        (ranks[n:] < len(memory)).float().mean() if memory else query.new_zeros(())
    )
    words = (
        query.new_tensor([len(t.split()) for t in texts])[ranks[:n]].sum(1).mean() / 100
    )
    return torch.stack((own, companion, retained, words))


def features(source, memory, encoder):
    single, paired = [dated_group_parts(source, n) for n in (1, 2)]
    if not single or any(len(parts) != 1 for _, parts in single):
        raise ValueError("Expected dated source chunks and their visible turn IDs")
    probes, ids = [], []
    for text, ((idx, _),) in single:
        body = text.split(" | ", 1)[1]
        _, sep, rest = body.partition(": ")
        # Continuation chunks have no leading speaker marker.
        # The first chunk of any source turn includes its speaker even for one-off speakers.
        probes.append(rest if sep and idx not in ids else body)
        ids.append(idx)
    texts = list(dict.fromkeys([*memory, *probes, *(t for t, _ in single + paired)]))
    enc = encode_batch(encoder, texts, 2, normalize=True)
    emb = F.normalize(torch.as_tensor(enc, device="cuda").float(), dim=1)
    vectors = dict(zip(texts, emb, strict=True))
    query = torch.stack([vectors[t] for t in probes + memory])
    turn_ids = torch.tensor(ids, device=query.device)
    available = set(ids)
    neighbor_ids = torch.tensor(
        [i ^ 1 if (i ^ 1) in available else -1 for i in ids], device=query.device
    )
    a, b = [
        simulation(query, vectors, memory, items, turn_ids, neighbor_ids)
        for items in (single, paired)
    ]
    return (b - a).tolist()


def cell_states(folder, bits, encoder, cache):
    (trace,) = (folder / "traces").glob("*.jsonl")
    spans = [
        v
        for l in trace.read_bytes().splitlines()
        if (v := json.loads(l))["kind"] == "span"
    ]
    dec = read(folder / "decisions.json")
    if (
        len(spans) != 2
        or [int(v["use"]) for v in dec] != list(map(int, bits))
        or [v["span_id"] for v in spans] != [0, 1]
    ):
        raise ValueError("Wrong real trajectory or selected slot")
    out = []
    for j, s in enumerate(spans):
        selected = s["selected_skills"]
        expected = (
            ("capture_temporal_context", "dated_single_turns")
            if bits[j] == "0"
            else ("capture_activity_details", "dated_adjacent_turns")
        )
        if (
            len(selected) != 3
            or (selected[-1]["name"], selected[-1].get("source_rule")) != expected
        ):
            raise ValueError(
                "Recorded ordered selection disagrees with the assigned third slot"
            )
        memory = list(restore_text(spans[:j]).values())
        key = digest([s["source"], memory])
        slot = cache / (key + ".json")
        if not slot.exists():
            x = features(s["source"], memory, encoder)
            register(slot, {"state": key, "features": x, "prior_items": len(memory)})
        entry = read(slot)
        if entry["state"] != key:
            raise ValueError("Mismatched state receipt")
        out.append(dict(entry, base=dec[j]["base_logit"]))
    return out


def probability(state, model):
    z = state["base"] + offset(model, state["features"])
    return 1 / (1 + math.exp(-z))


def regression(states, targets):
    return fit(
        [
            {"features": s["features"], "target": v}
            for s, v in zip(states, targets, strict=True)
        ]
    )


def terminal_fit(windows, reward):
    states = []
    targets = []
    for w in windows:
        for a in range(2):
            states.append(w["states"][1 + a])
            targets.append(
                w["values"][2 * a + 1]
                - w["values"][2 * a]
                + (w["signed"][a] if reward else 0)
            )
    return regression(states, targets)


def common_suffix_effect(probabilities, effects):
    after_temporal, after_activity = probabilities
    return (1 - max(after_temporal, after_activity)) * effects[0] + min(
        after_temporal, after_activity
    ) * effects[1]


def learn(windows, reward, *, root_effects=None):
    terminal = terminal_fit(windows, reward)
    ids = sorted({w["dialogue"] for w in windows})
    inner = {
        sid: terminal_fit([w for w in windows if w["dialogue"] != sid], reward)
        for sid in ids
    }
    targets = []
    for w in windows:
        y = w["values"]
        ps = [probability(s, inner[w["dialogue"]]) for s in w["states"][1:]]
        low, high = [y[2 * a] + ps[a] * (y[2 * a + 1] - y[2 * a]) for a in range(2)]
        bonus = (
            common_suffix_effect(ps, root_effects[w["window"]])
            if root_effects is not None
            else 0.0
        )
        targets.append(high - low + bonus)
    root = regression([w["states"][0] for w in windows], targets)
    shifts = []
    for j, model in enumerate((root, terminal, terminal)):
        states = [w["states"][j] for w in windows]
        base = torch.tensor([s["base"] for s in states], dtype=torch.float64)
        probs = base.new_tensor([probability(s, model) for s in states])
        shifts.append(matched_offset(base, probs, torch.ones_like(base)))
    return {"root": root, "terminal": terminal, "shifts": shifts, "donors": ids}


def probabilities(state, j, models, terminal):
    base = state["base"]
    native = 1 / (1 + math.exp(-base))
    stage = "root" if j == 0 else "terminal"
    credit = probability(state, models["credit"][stage])
    return {
        "native": native,
        "terminal": terminal,
        "credit": credit,
        "return_only": probability(state, models["return_only"][stage]),
        "rate": 1 / (1 + math.exp(-(base + models["credit"]["shifts"][j]))),
        "root_only": credit if j == 0 else native,
        "last_only": credit if j else native,
        "temporal": 0.0,
        "activity": 1.0,
    }


def training_window(base, folder, encoder, cache):
    measured = read(folder / "measurement.json")
    repair_path = base / "factorial-repairs" / (folder.name + ".json")
    repair = read(repair_path)
    if repair["inputs"]["window"] != file_hash(folder / "measurement.json"):
        raise ValueError("Training labels changed since the repair probes")
    sid = repair["dialogue"]
    order = [v["original_index"] for v in measured["per_question"]]
    cases = {}
    states = {}
    for bits in PATHS:
        path = folder / bits
        seen = cell_states(path, bits, encoder, cache)
        for j, s in zip((0, 1 + int(bits[0])), seen, strict=True):
            if states.setdefault(j, s) != s:
                raise ValueError("Unchanged decision state differs across paths")
        responses = read(path / "scores.json")["answers"]
        if [v["qa_index"] for v in responses] != list(range(len(order))):
            raise ValueError("Unaligned calibration questions")
        cases[bits] = [v["f1"] for v in responses]
    signed = []
    types = []
    for a in range(2):
        rows = [
            r for r in repair["rows"] if r["position"] == 1 and r["background"] == a
        ]
        if [r["original_index"] for r in rows] != order:
            raise ValueError("Unaligned repair probes")
        types = [r["category"] for r in rows]
        vals = []
        for r in rows:
            v = r.get("probe", {}).get("signed_effect", 0.0)
            if v and (
                not r["linked"]
                or r["probe"]["status"] != "survived_probes"
                or v != r["delta"]
            ):
                raise ValueError("Unverified extra selection credit")
            vals.append(v)
        signed.append(mean(vals))
    root = measured["path_mass"]
    terminal = [sum(root["1" + str(b)]["terminal"] for b in range(2))]
    terminal += [
        root[f"{a}1"]["terminal"] / sum(root[f"{a}{b}"]["terminal"] for b in range(2))
        for a in range(2)
    ]
    return {
        "dialogue": sid,
        "window": folder.name,
        "states": [states[j] for j in range(3)],
        "values": [mean(cases[b]) for b in PATHS],
        "signed": signed,
        "cases": cases,
        "questions": order,
        "types": types,
        "terminal": terminal,
        "repair": file_hash(repair_path),
    }


def mass(bits, probs, arm):
    a, b = map(int, bits)
    u, v = probs[0][arm], probs[1 + a][arm]
    return (u if a else 1 - u) * (v if b else 1 - v)


def calibration(w, model):
    ps = {
        j: probabilities(s, j, model, w["terminal"][j])
        for j, s in enumerate(w["states"])
    }
    rows = []
    for i, idx in enumerate(w["questions"]):
        rows.append(
            {
                "index": idx,
                "type": w["types"][i],
                "f1": {
                    arm: sum(mass(b, ps, arm) * w["cases"][b][i] for b in PATHS)
                    for arm in ARMS
                },
            }
        )
    return {
        "dialogue": w["dialogue"],
        "window": w["window"],
        "probabilities": ps,
        "rows": rows,
    }


def budgets(folder, bits):
    usage = [
        json.loads(l) for l in (folder / bits / "usage.jsonl").read_bytes().splitlines()
    ]
    return {
        op: sum(v["input"] + v["output"] for v in usage if v["kind"] == op)
        for op in ("reader", "executor")
    }


def later(folder, spec, models, weights, encoder, cache):
    report = read(folder / "pair.json")
    pivot = spec["pivot"]
    background = spec["background"]
    chosen = paths(pivot, background)
    if (report["dialogue"], report["start"], report["pivot"], report["background"]) != (
        spec["dialogue"],
        spec["start"],
        pivot,
        background,
    ) or report["paths"] != [list(b) for b in chosen]:
        raise ValueError("Paired trial changed assignment")
    if [(r["original_index"], r["type"]) for r in report["questions"]] != list(
        zip(spec["questions"], spec["types"], strict=True)
    ):
        raise ValueError("Paired questions differ from the frozen cohort")
    ps = {}
    evidence = {}
    costs = {}
    states = {}
    branches = []
    for bits in chosen:
        label = "".join(map(str, bits))
        cell = folder / label
        dec = read(cell / "decisions.json")
        ss = cell_states(cell, label, encoder, cache)
        for j, s, d in zip((0, 1 + bits[0]), ss, dec, strict=True):
            v = torch.as_tensor(d["state"], dtype=torch.float64, device="cuda")
            w = v.new_tensor(weights)
            terminal = float(
                (s["base"] + w[0] + w[1:] @ F.normalize(v, dim=0)).sigmoid()
            )
            probabilities_here = probabilities(s, j, models, terminal)
            if (
                ps.setdefault(j, probabilities_here) != probabilities_here
                or states.setdefault(j, s) != s
            ):
                raise ValueError(
                    "Decision state changed before the paired intervention"
                )
        evidence[label] = file_hash(cell / "scores.json")
        costs[label] = budgets(folder, label)
        answers = read(cell / "scores.json")["answers"]
        if [r["qa_index"] for r in answers] != list(range(len(spec["questions"]))):
            raise ValueError("Paired score order changed")
        branches.append([r["f1"] for r in answers])
    focus = ps[(1 + background) if pivot == 1 else 0]
    rows = []
    for i, r in enumerate(report["questions"]):
        low, high = (branch[i] for branch in branches)
        if (low, high) != (r["temporal"], r["activity"]):
            raise ValueError("Branch F1 changed")
        focal = {k: low + p * (high - low) for k, p in focus.items()}
        ht = {
            k: 2 * (mass(chosen[0], ps, k) * low + mass(chosen[1], ps, k) * high)
            for k in ARMS
        }
        for k in ("native", "terminal"):
            if not math.isclose(
                focal[k], r["focal"][k], abs_tol=1e-10
            ) or not math.isclose(ht[k], r["ht"][k], abs_tol=1e-10):
                raise ValueError("Published randomized estimands did not reproduce")
        rows.append(
            {"index": r["original_index"], "type": r["type"], "f1": focal, "ht": ht}
        )
    return {
        "dialogue": spec["dialogue"],
        "window": folder.name,
        "probabilities": ps,
        "focal_probabilities": focus,
        "rows": rows,
        "paths": evidence,
        "tokens": costs,
        "state_hashes": {k: v["state"] for k, v in states.items()},
        "pivot": pivot,
    }


def summarize(
    windows,
    arms=ARMS,
    candidate="credit",
    benchmarks=("native", "terminal", "return_only", "rate"),
):
    if not windows:
        return {"windows": 0, "questions": 0}
    rows = [(w, r) for w in windows for r in w["rows"]]

    def average(items):
        result = {"questions": len(items)}
        for metric in ("f1", "ht"):
            if metric in items[0][1]:
                result[metric] = {k: mean(r[metric][k] for _, r in items) for k in arms}
        return result

    groups = {
        sid: average([(w, r) for w, r in rows if w["dialogue"] == sid])
        for sid in sorted({w["dialogue"] for w, _ in rows})
    }
    ps = [
        p
        for w in windows
        for p in (
            [w["focal_probabilities"]]
            if "focal_probabilities" in w
            else w["probabilities"].values()
        )
    ]
    out = dict(
        windows=len(windows),
        **average(rows),
        by_dialogue=groups,
        by_type={
            str(cat): average([(w, r) for w, r in rows if r["type"] == cat])
            for cat in sorted({r["type"] for _, r in rows})
        },
        mean_abs_probability_shift=mean(abs(p[candidate] - p["native"]) for p in ps),
    )
    if len(groups) >= 2:
        out["dialogue_95pct"] = {
            metric: {
                k: paired_interval(
                    [v["questions"] for v in groups.values()],
                    [v[metric][candidate] - v[metric][k] for v in groups.values()],
                )
                for k in benchmarks
            }
            for metric in ("f1", "ht")
            if metric in out
        }
    return out


def run(root, study):
    src = root / "source-compression"
    late = root / "late-pairs"
    dest = src / "retrieval-competition"
    cache = dest / "states"
    cache.mkdir(parents=True, exist_ok=True)
    spec = read(study / "protocol.json")
    panel = read(src / "factorial/manifest.json")
    cohort = read(late / "cohort.json")
    history_ids = set(spec["train"])
    assignments = cohort["assignments"]
    if (
        len(history_ids) != 6
        or history_ids & set(spec["validation"] + spec["test"])
        or not spec["equal_sources"]
        or panel["protocol"] != file_hash(study / "protocol.json")
        or cohort["corpus"] != spec["input_sha256"]
        or {(sid, n) for sid in history_ids for n in (5, 7)}
        != {tuple(x) for x in panel["windows"]}
        or len(assignments) != 52
        or sum(len(a["questions"]) for a in assignments) != 466
        or {a["dialogue"] for a in assignments} != history_ids
    ):
        raise ValueError("Incorrect frozen data or skills")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= 0.30
    ):
        raise RuntimeError("Use a single bounded CUDA lease")
    free, total = torch.cuda.mem_get_info()
    budget = min(free - 2048 * 2**20, int(total * 0.65))
    if budget < 1800 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    set_retriever_flash_attn(False)
    _, encoder = init_context_model(spec["retriever"])
    if not all(p.is_cuda for p in encoder.parameters()):
        raise ValueError("Encoder failed CUDA placement")
    identity = {
        k: file_hash(p)
        for k, p in {
            "plan": PLAN,
            "panel": src / "factorial/manifest.json",
            "repairs": src / "factorial-repairs/summary.json",
            "study": study / "protocol.json",
            "later": late / "cohort.json",
            "reference": late / "frozen.json",
        }.items()
    }
    identity["encoder"] = os.environ["MEMSKILL_BACKEND_ID"]
    register(dest / "manifest.json", identity)
    archive = late / "outcomes"
    frozen = dest / "fit.json"
    if not frozen.exists():
        train = []
        for sid, start in panel["windows"]:
            folder = src / "factorial" / f"{sid}_s{start}"
            t = training_window(src, folder, encoder, cache)
            t["tokens"] = {b: budgets(folder, b) for b in PATHS}
            train.append(t)
            print("Calibration", folder.name, flush=True)
        with torch.device("cuda"):
            models = {}
            for sid in sorted(history_ids | {"deployment"}):
                donors = [w for w in train if w["dialogue"] != sid]
                models[sid] = {
                    k: learn(donors, k == "credit") for k in ("credit", "return_only")
                }
            register(
                frozen,
                {
                    "manifest": file_hash(dest / "manifest.json"),
                    "models": models,
                    "training": train,
                },
            )
    f = read(frozen)
    if f["manifest"] != file_hash(dest / "manifest.json"):
        raise ValueError("Frozen fit changed inputs")
    existing = scored_at(archive, frozen.stat().st_mtime_ns)
    register(dest / "preexisting.json", {"fit": file_hash(frozen), "scores": existing})
    prior = {p.split("/")[0] for p in existing}
    if not (dest / "calibration.json").exists():
        ws = [calibration(w, f["models"][w["dialogue"]]) for w in f["training"]]
        for w, t in zip(ws, f["training"], strict=True):
            w["tokens"] = t["tokens"]
            archived = read(src / "factorial" / w["window"] / "measurement.json")[
                "per_question"
            ]
            if any(
                not math.isclose(r["f1"][k], q["f1"][k], abs_tol=1e-10)
                for r, q in zip(w["rows"], archived, strict=True)
                for k in ("native", "terminal")
            ):
                raise ValueError("Native/terminal four-path policy failed reproduction")
        register(
            dest / "calibration.json",
            {"fit": file_hash(frozen), "summary": summarize(ws), "windows": ws},
        )
    weights = read(late / "frozen.json")["terminal"]
    done = []
    out = dest / "pairs"
    out.mkdir(exist_ok=True)
    for a in assignments:
        sid = a["dialogue"]
        key = f"{sid}_s{a['start']}"
        folder = archive / key
        if not (folder / "pair.json").is_file():
            continue
        p = out / (key + ".json")
        stamp = {"pair": file_hash(folder / "pair.json"), "fit": file_hash(frozen)}
        if not p.exists():
            if sid in f["models"][sid]["credit"]["donors"]:
                raise ValueError("Target trained on its own labels")
            table = later(folder, a, f["models"][sid], weights[sid], encoder, cache)
            register(p, dict(table, inputs=stamp))
        v = read(p)
        if v["inputs"] != stamp:
            raise ValueError("Pair or fit changed")
        done.append(v)
    stats = summarize(done)
    fresh = summarize([w for w in done if w["window"] not in prior])
    report = {
        "fit": file_hash(frozen),
        "cohort_questions": 466,
        "all": stats,
        "prospective": fresh,
    }
    register(dest / f"through-{len(done):02d}.json", report)
    if len(done) == len(assignments):
        register(dest / "summary.json", report)
    print(
        "Paired randomized windows",
        len(done),
        "prospective questions",
        fresh["questions"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("study", type=Path)
    run(**vars(parser.parse_args()))
