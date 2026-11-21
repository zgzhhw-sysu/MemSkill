"""Change the final writer instruction while retaining the same dated source turns."""

import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller.factorial_repairs import answer, checked_cell, replay_writes
from experiments.controller.factorial_rollout import read
from experiments.controller.paired_grounding import window
from experiments.controller.source_compression import file_hash, register
from src.credit_assignment import _digest, execution_fault
from src.episode_trace import skill_spec
from src.evaluation_cache import history_embeddings, read_response
from src.executor import deployment_operations
from src.rag_utils import get_embeddings
from src.replay import dated_turn_groups, restore_text

PLAN = Path(__file__).with_name("protocols") / "semantic_writer.json"
TEMP, REL = "capture_temporal_context", "handle_entity_relationships"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_offsets(calibration, target):
    quartiles = torch.quantile(calibration, calibration.new_tensor([0.25, 0.5, 0.75]))
    width = quartiles[2] - quartiles[0]
    if not width:
        return torch.zeros_like(target)
    return 2 * ((target - quartiles[1]) / width).clamp(-1, 1)


def freeze(trainer, train, specs, dest, identity):
    path = dest / "source-forecast.json"
    if not path.exists():

        def encode(text):
            vec = trainer.state_encoder._encode_texts(text).reshape(-1)
            return F.normalize(
                torch.as_tensor(vec, device=trainer.device).double(), dim=0
            )

        lookup = {s["name"]: s for s in specs}
        direction = encode(lookup[REL]["description"]) - encode(
            lookup[TEMP]["description"]
        )
        rows = []
        for c in train:
            for n in (1, 2, 3, 4, 6, 8):
                k = f"session_{n}"
                (src,) = trainer.data_processor._extract_session_chunks(
                    c["conversation"][k], c["conversation"][k + "_date_time"]
                )
                rows.append(
                    {
                        "dialogue": c["sample_id"],
                        "session": n,
                        "source": sha(src),
                        "affinity": float(encode(src) @ direction),
                    }
                )
        calibration = direction.new_tensor(
            [r["affinity"] for r in rows if r["session"] <= 4]
        )
        targets = [r for r in rows if r["session"] > 4]
        if len(calibration) != 24 or len(targets) != 12:
            raise ValueError("Unexpected source calibration cohort")
        for r, v in zip(
            targets,
            source_offsets(
                calibration, direction.new_tensor([r["affinity"] for r in targets])
            ).tolist(),
            strict=True,
        ):
            r["offset"] = v
        register(path, {"identity": identity, "rows": rows})
    result = read(path)
    if result["identity"] != identity:
        raise ValueError("Source encoding inputs changed")
    return {(r["dialogue"], r["session"]): r for r in result["rows"] if "offset" in r}


def probabilities(trainer, state, indices, offset, *, alternative=REL):
    native = trainer.controller.native
    ops = trainer.operation_bank.get_candidate_operations()
    with torch.no_grad():
        x = torch.as_tensor(state, device=trainer.device).float()
        vec = torch.as_tensor(
            np.stack([op.embedding for op in ops]), device=trainer.device
        ).float()
        logits = native.get_action_logits(
            native.encode_state(x[None]), native.encode_ops(vec[None])
        )[0]
        logits = native._apply_new_action_bias(logits, None)
        base = logits[indices[alternative]] - logits[indices[TEMP]]
        return {
            "base_logit": float(base),
            "native": float(base.sigmoid()),
            "proposed": float((base + offset).sigmoid()),
        }


def measured_readers(folder, cache):
    result = {}
    for row in map(json.loads, (folder / "usage.jsonl").read_bytes().splitlines()):
        if row["kind"] == "reader":
            saved = read_response(cache, row["cache_key"])
            if not saved or saved["sha256"] != row["response_sha256"]:
                raise ValueError("Missing measured temporal reader response")
            payload = saved["payload"]
            result[payload["prompt_sha256"]] = payload
    return result


def trial(trainer, convo, folder, cache, specs, forecast, out, *, prefix_action=0):
    control = checked_cell(trainer, convo, folder, (prefix_action, 0), specs, cache)
    prefix, event = control["spans"]
    original = event["selected_skills"]
    receipt = {
        "dialogue": convo["sample_id"],
        "source_trace": control["trace"],
        "source_forecast": forecast,
        "questions": len(convo["qa"]),
        "matched": False,
    }
    if forecast["source"] != sha(event["source"]) or original[-1]["name"] != TEMP:
        raise ValueError("Unexpected control source or skill")
    if REL in {s["name"] for s in original[:-1]}:
        return dict(receipt, reason="relation_already_in_fixed_slots")
    if any(
        execution_fault(s["parsed_actions"], s["applied"], s["skills"])
        for s in control["spans"]
    ):
        return dict(receipt, reason="control_execution_fault")
    bank = replay_writes(trainer, [prefix])
    session = trainer.state_encoder._encode_texts(event["source"]).reshape(-1)
    ctx, idx, mem = bank.retrieve(
        session, use_state_encoder=True, return_embeddings=True
    )
    prefix_ids = list(restore_text([prefix]))
    if ctx != event["retrieved"] or [prefix_ids[j] for j in idx] != [
        mid for mid, _ in event["refs"]
    ]:
        return dict(receipt, reason="unreproduced_prefix_read")
    state = trainer.state_encoder.encode(
        event["source"], ctx, session_embedding=session, memory_embeddings=mem
    )
    indices = {s["name"]: i for i, s in enumerate(specs)}
    relation = SimpleNamespace(
        **dict(specs[indices[REL]], source_rule="dated_single_turns")
    )
    selected = [*(SimpleNamespace(**s) for s in original[:-1]), relation]
    if original[-1]["source_rule"] != relation.source_rule:
        raise ValueError("Source rules differ")
    choice = dict(
        probabilities(trainer, state, indices, forecast["offset"]),
        selected=[skill_spec(op) for op in selected],
        fixed_prefix=_digest(list(restore_text([prefix]).items())),
        source=forecast,
        state_sha256=hashlib.sha256(np.asarray(state).tobytes()).hexdigest(),
        control_trace=control["trace"],
    )
    register(out / "choice.json", choice)
    recorded = measured_readers(folder, cache)
    factual = replay_writes(trainer, control["spans"])
    queries = get_embeddings(
        trainer.args.retriever, [q["question"] for q in convo["qa"]], "query"
    )
    baseline = []
    for i, (q, v) in enumerate(zip(convo["qa"], queries, strict=True)):
        old, read_idx = factual.retrieve(v, top_k=trainer.config.mem_top_k_eval)
        prompt = trainer.evaluator.build_prompt(q["question"], old, q)
        saved = recorded.get(sha(prompt))
        arch = control["scores"][i]
        if (
            saved is None
            or list(map(int, read_idx)) != arch["indices"]
            or saved["response"].strip() != arch["answer"]
            or saved["tokens"] != arch["tokens"]
        ):
            return dict(receipt, reason="unreproduced_control_reader")
        baseline.append(dict(arch, context_hash=_digest(old)))
    audit = []
    writer = deployment_operations(selected)
    acts = trainer.executor.execute_operation(writer, event["source"], ctx, audit=audit)
    if any("API call failed" in a.reasoning for a in acts):
        raise RuntimeError("Relation writer request failed")
    parsed = [vars(a) for a in acts]
    applied = trainer.executor.apply_to_memory_bank(
        acts, bank, idx, operation_name=[op.name for op in selected]
    )
    bank.step()
    contents = bank.get_all_contents()
    write = {
        "choice": file_hash(out / "choice.json"),
        "writer": audit,
        "actions": parsed,
        "applied": applied,
        "contents": contents,
    }
    register(out / "write.json", write)
    if execution_fault(parsed, applied, [skill_spec(op) for op in writer]):
        return dict(receipt, reason="relation_execution_fault")
    raw = set(dated_turn_groups(event["source"], 1))
    if (
        not raw
        or not raw.issubset(contents)
        or not raw.issubset(factual.get_all_contents())
    ):
        return dict(receipt, reason="incomplete_dated_source_coverage")
    rows = []
    for q, v, old in zip(convo["qa"], queries, baseline, strict=True):
        new, ix = bank.retrieve(v, top_k=trainer.config.mem_top_k_eval)
        same = _digest(new) == old["context_hash"]
        observed = dict(old) if same else answer(trainer, q, new)
        observed["indices"] = list(map(int, ix))
        rows.append(
            {
                "original_index": q["original_index"],
                "category": q["category"],
                "temporal": old,
                "relation": observed,
                "same_prompt": same,
            }
        )
    return dict(
        receipt, matched=True, raw_turn_items=len(raw), choice=choice, rows=rows
    )


def summarize(reports):
    matched = [w for w in reports if w["matched"]]
    if not matched:
        return {"windows": len(reports), "matched": 0}
    rate = mean(w["choice"]["proposed"] for w in matched)

    def scores(group):
        values = {k: [] for k in ("temporal", "relation", "native", "proposed", "rate")}
        for w in group:
            choice = w["choice"]
            probabilities = {
                "temporal": 0.0,
                "relation": 1.0,
                "native": choice["native"],
                "proposed": choice["proposed"],
                "rate": rate,
            }
            for q in w["rows"]:
                low = q["temporal"]["f1"]
                delta = q["relation"]["f1"] - low
                for name, probability in probabilities.items():
                    values[name].append(low + probability * delta)
        return {
            "questions": len(values["native"]),
            "f1": {k: mean(v) for k, v in values.items()},
        }

    return {
        "windows": len(reports),
        "matched": len(matched),
        "matched_rate": rate,
        **scores(matched),
        "per_dialogue": {
            sid: scores([w for w in matched if w["dialogue"] == sid])
            for sid in sorted({w["dialogue"] for w in matched})
        },
    }


def run(trainer, study, forecast, train, specs, limit, *, prefix_action=0):
    root = forecast.parent
    archive = root / "factorial"
    dest = root / "semantic-writer"
    dest.mkdir(exist_ok=True)
    manifest = read(archive / "manifest.json")
    if manifest["protocol"] != file_hash(study / "protocol.json") or manifest[
        "forecast"
    ] != file_hash(forecast):
        raise ValueError("Different factorial protocol or frozen source")
    identity = {
        "plan": file_hash(PLAN),
        "archive": file_hash(archive / "manifest.json"),
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
    }
    if prefix_action and not (dest / "source-forecast.json").is_file():
        raise ValueError("The temporal-prefix source forecast must already be frozen")
    offsets = freeze(trainer, train, specs, dest, identity)
    source_hash = file_hash(dest / "source-forecast.json")
    parent = dest
    if prefix_action == 1:
        from experiments.controller.relation_prefix_interaction import preregister

        dest = parent / "activity-prefix"
        dest.mkdir(exist_ok=True)
        binding = preregister(dest, parent, manifest, identity, source_hash)
    elif prefix_action != 0:
        raise ValueError("Unknown prefix action")
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(dest / "responses")
    os.environ["MEMSKILL_USAGE_LOG"] = str(dest / "usage.jsonl")
    data = {c["sample_id"]: c for c in train}
    reports = []
    for sid, start in manifest["windows"][:limit]:
        folder = archive / f"{sid}_s{start}" / f"{prefix_action}0"
        if not (folder / "scores.json").exists():
            break
        out = dest / f"{sid}_s{start}"
        out.mkdir(exist_ok=True)
        receipt = out / "pair.json"
        if not receipt.exists():
            with history_embeddings():
                v = trial(
                    trainer,
                    window(data[sid], start, 2),
                    folder,
                    archive / "responses",
                    specs,
                    offsets[sid, start + 1],
                    out,
                    prefix_action=prefix_action,
                )
            if prefix_action:
                v.update(prefix_action=prefix_action, registration=binding)
            register(receipt, dict(v, forecast=source_hash))
        v = read(receipt)
        (trace,) = (folder / "traces").glob("*.jsonl")
        if v["forecast"] != source_hash or v[
            "source_trace"
        ] != file_hash(trace):
            raise ValueError("Published control or source forecast changed")
        if prefix_action and (v["registration"] != binding or v["prefix_action"] != 1):
            raise ValueError("Changed activity prefix registration")
        reports.append(v)
        print(sid, start, v["matched"], v.get("reason", ""), flush=True)
    if reports:
        if prefix_action:
            from experiments.controller.relation_prefix_interaction import report

            report(parent, dest, archive, manifest, data, len(reports))
        register(dest / f"through-{len(reports)}.json", summarize(reports))
    if len(reports) == len(manifest["windows"]):
        register(dest / "summary.json", summarize(reports))
