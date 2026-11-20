"""Learn a terminal action update from repaired outcomes and evaluate it on measured pairs."""

import argparse
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "outcome_credit.json"


def action_advantages(p, d):
    return -p * d, (1 - p) * d


def state_vectors(states):
    x = torch.as_tensor(states, dtype=torch.float64)
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError(
            "Expected a concatenation of equal-width source and memory vectors"
        )
    src, mem = x.chunk(2, dim=-1)
    return torch.cat([F.normalize(src, dim=1), F.normalize(mem, dim=1)], dim=1)


def estimate(source, target, reach, questions, effects):
    weights = torch.exp((target @ source.T - 2.0) / 0.25) * (reach * questions)
    return (weights @ effects) / weights.sum(dim=1, keepdim=True)


def matched_offset(logits, probs, weights):
    low, high = -2.0, 2.0
    target = (weights @ probs) / weights.sum()
    for _ in range(48):
        mid = (low + high) / 2
        if (weights @ (logits + mid).sigmoid()) / weights.sum() < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def donors(root):
    source = root / "source-compression"
    specs = read(source / "factorial/manifest.json")
    rows, hashes = [], {}
    for sid, start in specs["windows"]:
        key = f"{sid}_s{start}"
        probe = source / "factorial-repairs" / (key + ".json")
        if not probe.is_file():
            continue
        report = read(probe)
        folder = source / "factorial" / key
        measured = read(folder / "measurement.json")
        if report["inputs"]["window"] != file_hash(folder / "measurement.json"):
            raise ValueError("Different underlying four-path measurements")
        ids = [q["original_index"] for q in measured["per_question"]]
        masses = measured["path_mass"]
        hashes[key] = file_hash(probe)
        for a in (0, 1):
            cell = folder / f"{a}0"
            d = read(cell / "decisions.json")[1]
            qs = [
                q for q in report["rows"] if q["position"] == 1 and q["background"] == a
            ]
            if [q["original_index"] for q in qs] != ids:
                raise ValueError(
                    "Original questions or ordering differ between probes and writer"
                )
            reach = sum(masses[f"{a}{b}"]["native"] for b in (0, 1))
            p = masses[f"{a}1"]["native"] / reach
            if not math.isclose(p, 1 / (1 + math.exp(-d["base_logit"])), abs_tol=1e-10):
                raise ValueError("Conditional action propensity changed")
            selected = [q.get("probe", {}).get("signed_effect", 0.0) for q in qs]
            if any(
                v
                and (
                    not q["linked"]
                    or q["probe"]["status"] != "survived_probes"
                    or v != q["delta"]
                )
                for q, v in zip(qs, selected, strict=True)
            ):
                raise ValueError("Unverified selection effect entered the learner")
            effects = torch.tensor(selected, dtype=torch.float64)
            at, aa = action_advantages(p, effects)
            if not torch.allclose(
                (1 - p) * at + p * aa, torch.zeros_like(at), atol=1e-12
            ):
                raise ValueError(
                    "Action baseline did not center at the measured propensity"
                )
            rows.append(
                {
                    "window": key,
                    "dialogue": sid,
                    "prefix": a,
                    "reach": reach,
                    "base": d["base_logit"],
                    "state": d["state"],
                    "n": len(qs),
                    "signed": float((aa - at).mean()),
                    "raw": mean(q["delta"] for q in qs),
                    "supported": sum(v != 0 for v in selected),
                }
            )
    return rows, hashes, specs


def rates(rows, targets):
    x = state_vectors([r["state"] for r in rows])
    y = state_vectors([r["state"] for r in targets])
    reach, n, base = [
        x.new_tensor([r[k] for r in rows]) for k in ("reach", "n", "base")
    ]
    effects = x.new_tensor([[r["signed"], r["raw"]] for r in rows])
    correction = (estimate(x, y, reach, n, effects) / 0.05).clamp(-2, 2)
    training = (estimate(x, x, reach, n, effects)[:, 0] / 0.05).clamp(-2, 2)
    shift = matched_offset(base, (base + training).sigmoid(), reach)
    intercept = float(((reach * n) @ effects[:, 0] / (reach @ n) / 0.05).clamp(-2, 2))
    z = x.new_tensor([r["base"] for r in targets])
    preds = torch.stack(
        [z, z + correction[:, 0], z + correction[:, 1], z + shift, z + intercept], dim=1
    ).sigmoid()
    return [
        dict(
            zip(("native", "outcome", "raw", "rate", "constant_update"), v, strict=True)
        )
        for v in preds.tolist()
    ]


def late_rows(root, spec):
    late = root / "late-pairs"
    assignments = {
        (w["dialogue"], w["start"]): w
        for w in read(late / "cohort.json")["assignments"]
    }
    out = []
    for path in sorted((late / "outcomes").glob("*/pair.json")):
        data = read(path)
        if data["pivot"] != 1:
            continue
        sid, start, a = data["dialogue"], data["start"], data["background"]
        assigned = assignments[sid, start]
        if (
            sid not in {s for s, _ in spec["windows"]}
            or data["paths"] != [[a, 0], [a, 1]]
            or (assigned["pivot"], assigned["background"]) != (1, a)
            or [q["original_index"] for q in data["questions"]] != assigned["questions"]
        ):
            raise ValueError("Randomized target or physical paths changed")
        cell = path.parent / f"{a}0"
        decisions = read(cell / "decisions.json")
        other = read(path.parent / f"{a}1" / "decisions.json")[1]
        d = decisions[1]
        if any(d[k] != other[k] for k in ("state", "base_logit", "plain", "grounded")):
            raise ValueError("Terminal state changed before taking the focal action")
        out.append(
            {
                "window": path.parent.name,
                "dialogue": sid,
                "state": d["state"],
                "base": d["base_logit"],
                "questions": data["questions"],
                "terminal": data["focal_probabilities"]["terminal"],
            }
        )
    return out


def report(windows):
    if not windows:
        return {"windows": 0, "questions": 0}
    scores = [q["f1"] for w in windows for q in w["questions"]]
    return {
        "windows": len(windows),
        "questions": len(scores),
        "f1": {k: mean(q[k] for q in scores) for k in scores[0]},
        "probability_shift": mean(
            abs(w["probabilities"]["outcome"] - w["probabilities"]["native"])
            for w in windows
        ),
    }


def run(root):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use one leased GPU for outcome fitting")
    torch.cuda.set_per_process_memory_fraction(0.06)
    with torch.device("cuda"):
        train, hashes, spec = donors(root)
        if not train:
            return
        stage = (
            "main" if len(hashes) == len(spec["windows"]) else f"pilot-{len(hashes)}"
        )
        dest = root / "source-compression" / "outcome-credit" / stage
        dest.mkdir(exist_ok=True, parents=True)
        groups = sorted({s for s, _ in spec["windows"]})
        # Atomic registration binds every fold's full sufficient statistics, including the state vectors.
        freeze = dest / "fit.json"
        register(
            freeze,
            {
                "plan": file_hash(PLAN),
                "probes": hashes,
                "rows": train,
                "folds": {
                    s: [i for i, r in enumerate(train) if r["dialogue"] != s]
                    for s in groups
                },
            },
        )
        scored = dest / "preexisting.json"
        if not scored.exists():
            archive = root / "late-pairs/outcomes"
            cutoff = freeze.stat().st_mtime_ns
            previous = {
                str(p.relative_to(archive)): file_hash(p)
                for p in archive.glob("*/*/scores.json")
                if p.stat().st_mtime_ns <= cutoff
            }
            register(scored, {"fit": file_hash(freeze), "paths": previous})
        saved = read(scored)
        archive = root / "late-pairs/outcomes"
        if saved["fit"] != file_hash(freeze) or any(
            file_hash(archive / k) != h for k, h in saved["paths"].items()
        ):
            raise ValueError("Prior published paths differ from numeric fit")
        existing = {Path(k).parts[0] for k in saved["paths"]}
        targets = late_rows(root, spec)
        windows = []
        for sid in groups:
            held = [r for r in targets if r["dialogue"] == sid]
            fold = [r for r in train if r["dialogue"] != sid]
            if not (fold and held):
                continue
            for target, probs in zip(held, rates(fold, held), strict=True):
                probs.update(terminal=target["terminal"], temporal=0.0, activity=1.0)
                questions = []
                for q in target["questions"]:
                    delta = q["activity"] - q["temporal"]
                    f1 = {k: q["temporal"] + p * delta for k, p in probs.items()}
                    if not math.isclose(
                        f1["native"], q["focal"]["native"], abs_tol=1e-10
                    ):
                        raise ValueError("Randomized native return did not reproduce")
                    questions.append(
                        {
                            "original_index": q["original_index"],
                            "category": q["type"],
                            "f1": f1,
                        }
                    )
                windows.append(
                    {
                        "window": target["window"],
                        "dialogue": sid,
                        "probabilities": probs,
                        "training_questions_with_credit": sum(
                            r["supported"] for r in fold
                        ),
                        "questions": questions,
                    }
                )
        result = {
            "fit": file_hash(freeze),
            "all": report(windows),
            "subsequent": report([w for w in windows if w["window"] not in existing]),
            "per_dialogue": {
                s: report([w for w in windows if w["dialogue"] == s]) for s in groups
            },
            "rows": windows,
        }
        register(dest / f"terminal-pairs-{len(targets)}.json", result)
        print(stage, result["all"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
