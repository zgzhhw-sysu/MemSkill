"""Prospective relation-versus-temporal writer trial under randomized earlier actions."""

import os
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.late_pair_policy import assignments
from experiments.controller.paired_grounding import window
from experiments.controller.semantic_writer import REL, TEMP, sha, source_offsets, trial
from src.evaluation_cache import history_embeddings
from src.replay import dated_turn_groups

PLAN = Path(__file__).with_name("protocols") / "late_semantic_writer.json"


def source_scores(trainer, source, direction):
    def contrast(texts):
        values = trainer.state_encoder._encode_texts(texts)
        tensor = torch.as_tensor(values, device=trainer.device).reshape(
            -1, direction.numel()
        )
        return F.normalize(tensor.double(), dim=1) @ direction

    turns = [text.split(" | ", 1)[1] for text in dated_turn_groups(source, 1)]
    if not turns:
        raise ValueError("No dated source items for the terminal decision")
    return [
        float(contrast(source).item()),
        float(torch.quantile(contrast(turns), 0.75)),
    ]


def freeze(trainer, specs, data, panel, out, archive, identity):
    path = out / "forecast.json"
    if not path.exists():
        if any(out.glob("conv-*/*.json")):
            raise ValueError(
                "Relation branch already exists without its frozen forecast"
            )
        lookup = {s["name"]: s["description"] for s in specs}
        # Encode the two descriptions separately to match the earlier whole-source actor.
        vectors = [
            torch.as_tensor(
                trainer.state_encoder._encode_texts(lookup[name]), device=trainer.device
            )
            .reshape(-1)
            .double()
            for name in (REL, TEMP)
        ]
        direction = F.normalize(vectors[0], dim=0) - F.normalize(vectors[1], dim=0)

        def measure(sid, session):
            history = data[sid]["conversation"]
            key = f"session_{session}"
            (source,) = trainer.data_processor._extract_session_chunks(
                history[key], history[key + "_date_time"]
            )
            return {
                "dialogue": sid,
                "session": session,
                "source": sha(source),
                "scores": source_scores(trainer, source, direction),
            }

        calibration = [measure(sid, i) for sid in sorted(data) for i in range(1, 5)]
        targets = [measure(w["dialogue"], w["start"] + 1) for w in panel]
        x, y = [
            direction.new_tensor([r["scores"] for r in group])
            for group in (calibration, targets)
        ]
        whole, tail = [source_offsets(x[:, i], y[:, i]) for i in (0, 1)]
        for row, offset, prob, tail_prob in zip(
            targets,
            whole.tolist(),
            whole.sigmoid().tolist(),
            tail.sigmoid().tolist(),
            strict=True,
        ):
            row.update(offset=offset, whole=prob, tail=tail_prob)
        known = {}
        for w in panel:
            f = (
                archive
                / f"{w['dialogue']}_s{w['start']}"
                / f"{w['background']}0"
                / "scores.json"
            )
            if f.is_file():
                known[str(f.relative_to(archive))] = file_hash(f)
        register(
            path,
            {
                "identity": identity,
                "calibration": calibration,
                "targets": targets,
                "temporal_at_freeze": known,
            },
        )
    frozen = read(path)
    if frozen["identity"] != identity or any(
        file_hash(archive / k) != v for k, v in frozen["temporal_at_freeze"].items()
    ):
        raise ValueError("Forecast or an archived control changed")
    return frozen


def summarize(panel, reports):
    strata = {}
    missing = Counter()
    matched = [r for r in reports if r["matched"]]
    rates = (
        {
            actor: mean(r["source_forecast"][actor] for r in matched)
            for actor in ("whole", "tail")
        }
        if matched
        else {}
    )
    for w, r in zip(panel, reports):
        if not r["matched"]:
            missing[r["reason"]] += len(w["questions"])
            continue
        probs = {
            "native": r["choice"]["native"],
            "temporal": 0.0,
            "relation": 1.0,
            "uniform": 0.5,
        }
        for actor, rate in rates.items():
            probs[actor] = r["source_forecast"][actor]
            probs[actor + "_rate"] = rate
        for q in r["rows"]:
            low = q["temporal"]["f1"]
            delta = q["relation"]["f1"] - low
            for group in (
                "all",
                r["dialogue"],
                "category_" + str(q["category"]),
                "prefix_" + str(w["background"]),
            ):
                scores = strata.setdefault(group, {key: [] for key in probs})
                for key, prob in probs.items():
                    scores[key].append(low + prob * delta)
    return {
        "registered_windows": len(panel),
        "registered_questions": sum(len(w["questions"]) for w in panel),
        "completed_windows": len(reports),
        "matched_windows": len(matched),
        "unmatched_questions": dict(missing),
        "pending_questions": sum(len(w["questions"]) for w in panel[len(reports) :]),
        "matched_rates": rates,
        "groups": {
            g: {"questions": len(v["native"]), "f1": {k: mean(s) for k, s in v.items()}}
            for g, v in strata.items()
        },
    }


def run(trainer, study, locked, train, specs, limit):
    base = locked.parent
    archive, out = base / "outcomes", base / "semantic-writer"
    out.mkdir(exist_ok=True)
    data = {c["sample_id"]: c for c in train}
    cohort = read(base / "cohort.json")
    manifest = read(archive / "manifest.json")
    versions = [s["version"] for s in specs]
    if (
        assignments(data) != cohort["assignments"]
        or cohort["corpus"] != file_hash(Path(trainer.args.data_file))
        or manifest["forecast"] != file_hash(locked)
        or manifest["protocol"] != file_hash(study / "protocol.json")
        or manifest["skill_versions"] != versions
        or read(locked)["inputs"]["cohort"] != file_hash(base / "cohort.json")
    ):
        raise ValueError("Different randomized control, dataset, skills or study")
    panel = [w for w in cohort["assignments"] if w["pivot"] == 1]
    if (len(data), len(panel), sum(len(w["questions"]) for w in panel)) != (6, 25, 209):
        raise ValueError("Incomplete registered semantic writer cohort")
    register(
        out / "panel.json",
        {
            "plan": file_hash(PLAN),
            "cohort": file_hash(base / "cohort.json"),
            "windows": panel,
        },
    )
    identity = {
        "panel": file_hash(out / "panel.json"),
        "archive": file_hash(archive / "manifest.json"),
        "backend": os.environ["MEMSKILL_BACKEND_ID"],
    }
    frozen = freeze(trainer, specs, data, panel, out, archive, identity)
    digest = file_hash(out / "forecast.json")
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(out / "responses")
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "usage.jsonl")
    reports = []
    for w, forecast in zip(panel, frozen["targets"], strict=True):
        if limit is not None and len(reports) >= limit:
            break
        sid, start = w["dialogue"], w["start"]
        if (forecast["dialogue"], forecast["session"]) != (sid, start + 1):
            raise ValueError("Forecast order changed")
        name = f"{sid}_s{start}"
        folder, episode = archive / name / f"{w['background']}0", out / name
        if not (folder / "scores.json").is_file():
            break
        episode.mkdir(exist_ok=True)
        receipt = episode / "pair.json"
        if not receipt.exists():
            convo = window(data[sid], start, 2)
            if [q["original_index"] for q in convo["qa"]] != w["questions"]:
                raise ValueError("Question population changed")
            with history_embeddings():
                result = trial(
                    trainer,
                    convo,
                    folder,
                    archive / "responses",
                    specs,
                    forecast,
                    episode,
                    prefix_action=w["background"],
                )
            register(
                receipt, dict(result, forecast=digest, prefix_action=w["background"])
            )
        result = read(receipt)
        (trace,) = (folder / "traces").glob("*.jsonl")
        if (
            result["forecast"] != digest
            or result["source_trace"] != file_hash(trace)
            or result["source_forecast"] != forecast
            or result["prefix_action"] != w["background"]
            or result["questions"] != len(w["questions"])
            or (
                result["matched"]
                and [(q["original_index"], int(q["category"])) for q in result["rows"]]
                != list(zip(w["questions"], w["types"], strict=True))
            )
        ):
            raise ValueError(
                "Reported pair differs from its frozen forecast or archive"
            )
        reports.append(result)
        print(name, result["matched"], result.get("reason", ""), flush=True)
    if reports:
        report = dict(summarize(panel, reports), forecast=digest)
        register(out / f"through-{len(reports):02d}.json", report)
        if len(reports) == len(panel):
            register(out / "summary.json", report)
