"""Learn a root-only addition from repaired outcomes on jointly possible suffix decisions."""

import argparse
import math
import time
from pathlib import Path
from statistics import mean

import torch

from experiments.controller.audit_prefix_return import scored_at
from experiments.controller.factorial_rollout import read
from experiments.controller.retrieval_competition import (
    learn,
    mass,
    probability,
    summarize,
)
from experiments.controller.source_compression import file_hash, register

PLAN = Path(__file__).with_name("protocols") / "root_overlap_credit.json"
ARMS = ("overlap", "prior", "rate", "return_only", "native", "terminal")
CONTROLS = ("prior", "rate", "return_only", "native", "terminal")


def registration(base):
    dest = base / "root-overlap"
    dest.mkdir(exist_ok=True)
    path = dest / "registration.json"
    if not path.exists():
        register(
            path,
            {
                "utc_ns": time.time_ns(),
                "plan": file_hash(PLAN),
                "earlier_probes": sorted(
                    str(p.relative_to(base / "root-write-repairs"))
                    for p in (base / "root-write-repairs").glob("*/q*.json")
                ),
            },
        )
    if read(path)["plan"] != file_hash(PLAN):
        raise ValueError("Registration changed")
    return dest


def gather(base, train, omit=None):
    root = base / "root-write-repairs"
    order = {}
    for w in train:
        p = root / w["window"]
        v = read(p / "window.json")
        if v["questions"] != len(w["questions"]) or v["dialogue"] != w["dialogue"]:
            raise ValueError("Earlier-span population changed")
        effects = [0.0, 0.0]
        seen = set()
        for row in v["rows"]:
            key = row["index"], row["suffix"]
            if key in seen or key[0] not in w["questions"]:
                raise ValueError("Duplicate or unregistered root question")
            seen.add(key)
            path = p / f"q{key[0]}-suffix{key[1]}.json"
            r = read(path)
            probe = r["probe"]
            orig = r["original"]
            if (
                file_hash(path) != row["receipt"]
                or orig["position"] != 0
                or not orig["linked"]
                or not orig["recovery_sign"]
                or (orig["original_index"], orig["background"]) != key
                or row["signed_effect"] != probe["signed_effect"]
            ):
                raise ValueError("Root-repair provenance changed")
            if probe["signed_effect"] and (
                probe["status"] != "survived_probes"
                or probe["competing"]
                or probe["signed_effect"] != orig["delta"]
            ):
                raise ValueError("Competing or unverified root feedback")
            if (w["dialogue"], key[0]) != omit:
                effects[key[1]] += probe["signed_effect"] / v["questions"]
        order[w["window"]] = effects
    return order


def root_shift(logits, target):
    # Both root policies have offsets in [-2,2] from the identical native logit.
    low, high = -4.0, 4.0
    for _ in range(52):
        mid = (low + high) / 2
        if float((logits + mid).sigmoid().mean()) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def policy_states(old, state, model):
    result = {}
    for j, prior in old.items():
        j = int(j)
        if j == 0:
            candidate = probability(state, model["root"])
            rate = 1 / (
                1
                + math.exp(
                    -(
                        math.log(prior["credit"] / (1 - prior["credit"]))
                        + model["shift"]
                    )
                )
            )
        else:
            candidate = rate = prior["credit"]
        result[j] = dict(
            overlap=candidate,
            prior=prior["credit"],
            rate=rate,
            **{k: prior[k] for k in ("return_only", "native", "terminal")},
        )
    return result


def score(original, states, cases, bits, costs, pivot=None, background=None):
    partial = pivot is not None
    focus = states[0 if pivot == 0 else 1 + background] if partial else None
    out = []
    for i, r in enumerate(original["rows"]):
        obs = [cases[b][i] for b in bits]
        expected = {
            k: (2 if partial else 1)
            * sum(mass(b, states, k) * x for b, x in zip(bits, obs, strict=True))
            for k in ARMS
        }
        if partial:
            focal = {k: obs[0] + v * (obs[1] - obs[0]) for k, v in focus.items()}
            vals = {"f1": focal, "ht": expected}
        else:
            vals = {"f1": expected}
        for metric, values in vals.items():
            for arm, old in (
                ("prior", "credit"),
                ("return_only", "return_only"),
                ("native", "native"),
                ("terminal", "terminal"),
            ):
                if not math.isclose(values[arm], r[metric][old], abs_tol=1e-10):
                    raise ValueError("Published policy score did not reproduce")
        out.append(dict(index=r["index"], type=r["type"], **vals))
    logical = {
        k: {
            op: sum(
                (2 if partial else 1) * mass(b, states, k) * costs[b][op] for b in bits
            )
            for op in ("executor", "reader")
        }
        for k in ARMS
    }
    result = {
        "dialogue": original["dialogue"],
        "window": original["window"],
        "probabilities": states,
        "rows": out,
        "expected_tokens": logical,
    }
    if partial:
        result["focal_probabilities"] = focus
    return result


def report(rows):
    v = summarize(rows, ARMS, "overlap", CONTROLS)
    if rows:
        v["root_shift_vs_prior"] = mean(
            abs(w["probabilities"][0]["overlap"] - w["probabilities"][0]["prior"])
            for w in rows
        )
        v["expected_tokens"] = {
            k: {
                "writer_per_window": mean(
                    w["expected_tokens"][k]["executor"] for w in rows
                ),
                "reader_per_question": sum(
                    w["expected_tokens"][k]["reader"] for w in rows
                )
                / v["questions"],
            }
            for k in ARMS
        }
    return v


def question_influence(base, dest, previous, train, fit, frozen):
    path = dest / "question-influence.json"
    if path.exists():
        return
    plan = PLAN.with_name("root_question_influence.json")
    full = read(dest / "calibration.json")
    baseline = {w["window"]: w for w in full["windows"]}
    old = {w["window"]: w for w in read(previous / "calibration.json")["windows"]}
    ids = sorted(
        {
            (w["dialogue"], row["index"])
            for w in train
            for row in read(base / "root-write-repairs" / w["window"] / "window.json")[
                "rows"
            ]
            if row["signed_effect"]
        }
    )
    diagnostics = []
    with torch.device("cuda"):
        for sid, idx in ids:
            effects = gather(base, train, omit=(sid, idx))
            predictors = {}
            for target, original in fit["models"].items():
                if target == "deployment":
                    continue
                donors = [w for w in train if w["dialogue"] != target]
                predictors[target] = learn(donors, True, root_effects=effects)["root"]
                if target == sid and predictors[target] != original["root"]:
                    raise ValueError("Removing the held-out dialogue changed its model")
            scores = []
            changes = []
            for w in train:
                o = old[w["window"]]
                model = {"root": predictors[w["dialogue"]], "shift": 0.0}
                ps = policy_states(o["probabilities"], w["states"][0], model)
                changes.append(
                    abs(
                        ps[0]["overlap"]
                        - baseline[w["window"]]["probabilities"]["0"]["overlap"]
                    )
                )
                for i in range(len(w["questions"])):
                    scores.append(
                        sum(
                            mass(b, ps, "overlap") * q[i] for b, q in w["cases"].items()
                        )
                    )
            diagnostics.append(
                {
                    "omitted_dialogue": sid,
                    "omitted_original_index": idx,
                    "f1": mean(scores),
                    "mean_absolute_root_shift_vs_full": mean(changes),
                }
            )
    register(
        path,
        {
            "plan": file_hash(plan),
            "fit": file_hash(frozen),
            "full_f1": full["summary"]["f1"]["overlap"],
            "omissions": diagnostics,
        },
    )


def run(base):
    dest = registration(base)
    previous = base / "retrieval-competition"
    late = base.parent / "late-pairs/outcomes"
    f = read(previous / "fit.json")
    train = f["training"]
    fh = dest / "fit.json"
    identity = {
        "plan": file_hash(PLAN),
        "registration": file_hash(dest / "registration.json"),
        "original_fit": file_hash(previous / "fit.json"),
        "repairs": file_hash(base / "root-write-repairs/summary.json"),
        "windows": {
            w["window"]: file_hash(
                base / "root-write-repairs" / w["window"] / "window.json"
            )
            for w in train
        },
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use a CUDA lease")
    torch.cuda.set_per_process_memory_fraction(0.06)
    if not fh.exists():
        effects = gather(base, train)
        models = {}
        with torch.device("cuda"):
            for sid, priors in f["models"].items():
                donors = [w for w in train if w["dialogue"] != sid]
                m = learn(donors, True, root_effects=effects)
                if (
                    m["terminal"] != priors["credit"]["terminal"]
                    or m["donors"] != priors["credit"]["donors"]
                ):
                    raise ValueError("Frozen terminal policy changed")
                x = [w["states"][0] for w in donors]
                prior = torch.tensor(
                    [probability(s, priors["credit"]["root"]) for s in x],
                    dtype=torch.float64,
                )
                shift = root_shift(
                    prior.logit(), mean(probability(s, m["root"]) for s in x)
                )
                models[sid] = {"root": m["root"], "shift": shift, "donors": m["donors"]}
        register(fh, {"identity": identity, "effects": effects, "models": models})
    fit = read(fh)
    if fit["identity"] != identity:
        raise ValueError("Different root credit evidence")
    frozen = scored_at(late, fh.stat().st_mtime_ns)
    register(dest / "preexisting.json", {"fit": file_hash(fh), "scores": frozen})
    occupied = {k.split("/")[0] for k in frozen}
    if not (dest / "calibration.json").exists():
        old = {w["window"]: w for w in read(previous / "calibration.json")["windows"]}
        results = []
        for w in train:
            o = old[w["window"]]
            m = fit["models"][w["dialogue"]]
            ss = policy_states(o["probabilities"], w["states"][0], m)
            results.append(score(o, ss, w["cases"], list(w["cases"]), w["tokens"]))
        register(
            dest / "calibration.json",
            {"fit": file_hash(fh), "summary": report(results), "windows": results},
        )
    question_influence(base, dest, previous, train, fit, fh)
    panels = []
    for path in sorted((previous / "pairs").glob("*.json")):
        o = read(path)
        w = o["window"]
        sampled = read(late / w / "pair.json")
        raw = read(previous / "states" / (o["state_hashes"]["0"] + ".json"))
        if (
            o["inputs"]["pair"] != file_hash(late / w / "pair.json")
            or raw["state"] != o["state_hashes"]["0"]
        ):
            raise ValueError("Randomized panel changed")
        bits = ["".join(map(str, b)) for b in sampled["paths"]]
        dec = read(late / w / bits[0] / "decisions.json")[0]
        x = dict(raw, base=dec["base_logit"])
        ss = policy_states(o["probabilities"], x, fit["models"][o["dialogue"]])
        cases = {
            b: [r[arm] for r in sampled["questions"]]
            for b, arm in zip(bits, ("temporal", "activity"), strict=True)
        }
        panels.append(
            score(
                o, ss, cases, bits, o["tokens"], sampled["pivot"], sampled["background"]
            )
        )
    subsequent = report([w for w in panels if w["window"] not in occupied])
    eligible = False
    if (
        len(panels) == 52
        and subsequent["questions"] >= 0.9 * 466
        and len(subsequent["by_dialogue"]) == 6
    ):
        focal = subsequent["f1"]
        ht = subsequent["ht"]
        eligible = (
            all(
                focal["overlap"] >= focal[k] + 0.02 and ht["overlap"] > ht[k]
                for k in ("native", "terminal")
            )
            and all(
                focal["overlap"] > focal[k] for k in ("prior", "rate", "return_only")
            )
            and sum(
                v["f1"]["overlap"] > v["f1"]["rate"]
                for v in subsequent["by_dialogue"].values()
            )
            >= 4
        )
    result = {
        "fit": file_hash(fh),
        "all": report(panels),
        "prospective": subsequent,
        "eligible_for_independent_development": eligible,
        "windows": panels,
    }
    register(dest / f"through-{len(panels):02d}.json", result)
    if len(panels) == 52:
        register(dest / "summary.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("--register", action="store_true")
    args = p.parse_args()
    if args.register:
        registration(args.base)
    else:
        run(args.base)
