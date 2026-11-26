"""Cross-fit the usefulness of each donor case separately for the root and last write."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch

from experiments.controller.factorial_rollout import CELLS, read
from experiments.controller.passage_advantage import contributions, evaluate, key
from experiments.controller.passage_evidence_actor import exposure, gpu
from experiments.controller.passage_evidence_prefix import evaluate as evaluate_prefix
from experiments.controller.passage_evidence_prefix import extend
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited

PLAN = Path(__file__).with_name("protocols") / "passage_case_reliability.json"
ARMS = (
    "credit",
    "raw",
    "unweighted",
    "uniform",
    "reassigned",
    "parent",
    "native",
    "terminal",
)


def reliability(votes):
    positive = sum(x > 1e-10 for x in votes)
    negative = sum(x < -1e-10 for x in votes)
    return max(0, positive - negative) / (2 + positive + negative)


def combine(terms):
    return 2 * (terms.amax(-1).clamp_min(0) + terms.amin(-1).clamp_max(0))


def fit_cases(packet, panels, device):
    raw = {}
    for name, rows in packet["matches"].items():
        for r in rows:
            raw[name, tuple(r["case"])] = r["scores"][0]
    labels = packet["labels"]
    corpus = sorted({w["dialogue"] for w in panels})
    cases = sorted({k for n, k in raw})
    if len(corpus) != 6 or any(g not in corpus for g, i in cases):
        raise ValueError("Different six-history case bank")
    result = {}

    def t(x):
        return torch.tensor(x, device=device, dtype=torch.float64)

    for held in corpus:
        votes = defaultdict(lambda: defaultdict(list))
        for w in panels:
            inner = w["dialogue"]
            if inner == held:
                continue
            donors = [k for k in cases if k[0] not in (held, inner)]
            if not donors:
                continue
            n = int(w["window"].rsplit("_s", 1)[1])
            if n not in (5, 7):
                raise ValueError("A later question entered the reliability fit")
            scores = []
            for g in donors:
                ref = [
                    raw[f"{sid}_s{j}", g]
                    for sid in corpus
                    if sid not in (held, inner, g[0])
                    for j in range(5, 13)
                ]
                if len(ref) != 24:
                    raise ValueError(
                        "Nested reference contains a held or donor history"
                    )
                observed = t([raw[f"{inner}_s{n + j}", g] for j in (0, 1)])
                strength = (
                    1
                    - 2
                    * (1 + (t(ref)[:, None] >= observed[None, :]).sum(0).double())
                    / 25
                ).clamp_min(0)
                scores.append(strength * labels[g])
            # Row zero uses all donors. Each other row omits exactly one case.
            terms = torch.stack(scores, dim=-1)
            masks = torch.cat(
                (
                    terms.new_ones((1, len(donors))),
                    1 - torch.eye(len(donors), device=device, dtype=terms.dtype),
                )
            )
            logit = combine(terms[None, :, :] * masks[:, None, :])
            p = t([w["probabilities"][str(i)]["native"] for i in range(3)])
            prop = (p.logit() + logit[:, [0, 1, 1]]).sigmoid()
            table = t([w["cells"][key(b)] for b in CELLS]).T.reshape(-1, 2, 2)
            # Native continuation and native occupancy isolate the reliability of each stage.
            adv, _ = contributions(table, p, p)
            change = prop[0] - prop[1:]
            root = change[:, 0, None] * adv[:, 0]
            suffix = (1 - p[0]) * change[:, 1, None] * adv[:, 1] + p[0] * change[
                :, 2, None
            ] * adv[:, 2]
            for j, values in enumerate((root, suffix)):
                for g, ys in zip(donors, values.tolist(), strict=True):
                    votes[g, j][inner].extend(ys)
        folded = []
        for g in cases:
            if g[0] == held:
                continue
            result_row = {"case": list(g), "signed_credit": labels[g], "stages": []}
            for j in (0, 1):
                ys = {sid: sum(y) / len(y) for sid, y in votes[g, j].items()}
                if set(ys) != set(corpus) - {held, g[0]}:
                    raise ValueError("Expected four other histories for each case")
                result_row["stages"].append(
                    {"history_gains": ys, "weight": reliability(ys.values())}
                )
            folded.append(result_row)
        result[held] = folded
    return result


def apply_weights(history, matches, fits, original, stage):
    registry = {tuple(r["case"]): r for r in fits[history]}
    evidence = []
    for r in matches:
        k = tuple(r["case"])
        entry = registry[k]
        if k[0] == history:
            raise ValueError("Held history voted on its own sample")
        evidence.append(
            (
                r["strength"][0] * entry["signed_credit"],
                entry["stages"][stage]["weight"],
            )
        )
    amounts = [x for x, w in evidence]
    weights = [w for x, w in evidence]

    def correction(ws):
        vals = [x * y for x, y in zip(amounts, ws, strict=True)]
        return 2 * (max([0.0, *vals]) + min([0.0, *vals]))

    unit = correction([1.0] * len(weights))
    if not math.isclose(unit, original["credit"], abs_tol=1e-10):
        raise ValueError("Unweighted source offset not reproduced")
    return {
        "credit": correction(weights),
        "uniform": correction([sum(weights) / len(weights)] * len(weights))
        if weights
        else 0.0,
        "reassigned": correction(weights[1:] + weights[:1]),
        "unweighted": unit,
        "parent": original["parent"],
    }


def prepare(base, stage, device):
    src = base / "source-compression/witness-passages"
    primary = src / "evidence-advantage" / stage
    local = primary / "forecast.json"
    # The early physical outcomes originate from the group archive for both stages.
    early = src / "evidence-advantage/group/early"
    if not local.is_file() or not (early / "summary.json").is_file():
        return None
    out = primary.parent / "case-reliability" / stage
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "forecast.json"
    fit = out / "fit.json"
    files = sorted(early.glob("conv-*.json"))
    if len(files) != 12:
        raise ValueError("Expected every early physical window")
    identity = {
        "actor": file_hash(local),
        "plan": file_hash(PLAN),
        "early": {f.stem: file_hash(f) for f in files},
        "donors": file_hash(src / "registration.json"),
    }
    if dest.exists():
        if read(dest)["inputs"] != {"fit": file_hash(fit), **identity}:
            raise ValueError("Changed frozen reliability actor")
        return dest
    pkt = read(local)
    labels = {
        (r["dialogue"], r["index"]): r["credit"]
        for r in read(src / "registration.json")["records"]
    }
    if pkt["inputs"]["ranks"] != identity["donors"]:
        raise ValueError("Different passage credit bank")
    with limited():
        fits = fit_cases({**pkt, "labels": labels}, [read(f) for f in files], device)
    register(fit, {"inputs": identity, "folds": fits})
    stages = {
        str(j): {
            n: apply_weights(
                n.rsplit("_s", 1)[0], pkt["matches"].get(n, []), fits, v, j
            )
            for n, v in pkt["offsets"].items()
        }
        for j in (0, 1)
    }
    register(
        dest,
        {
            "inputs": {"fit": file_hash(fit), **identity},
            "sources": pkt["sources"],
            "by_stage": stages,
            "arms": list(ARMS),
        },
    )
    return dest


def run(base, study, corpus):
    for stage in ("group", "witness"):
        local = prepare(base, stage, "cuda")
        if not local:
            continue
        exposure(base, local)
        evaluate(base, study, corpus, local)
        src = base / "source-compression/witness-passages/evidence-advantage"
        if not (src / "prefix-sources/complete.json").is_file():
            continue
        prefix = extend(base, study, corpus, stage)
        if not prefix:
            continue
        entry = read(prefix)
        mr = prefix.parent / "case-matches.json"
        details = read(mr)
        if details["forecast"] != file_hash(prefix):
            raise ValueError("Different prefix matches")
        fits = read(local.parent / "fit.json")["folds"]
        folder = local.parent / "long-prefix"
        folder.mkdir(exist_ok=True)
        dest = folder / "forecast.json"
        params = {
            n: apply_weights(n.rsplit("_s", 1)[0], details["matches"][n], fits, v, 1)
            for n, v in entry["offsets"].items()
        }
        register(
            dest,
            {
                "inputs": {
                    **entry["inputs"],
                    "local": file_hash(local),
                    "prefix": file_hash(prefix),
                    "matches": file_hash(mr),
                },
                "sources": entry["sources"],
                "offsets": params,
                "arms": list(ARMS),
            },
        )
        evaluate_prefix(base, dest)
        print("scored", local.parent, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    gpu()
    run(args.base, args.study, args.corpus)
