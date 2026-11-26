"""Route verified span preferences through separately measured writer and grouping effects."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from experiments.controller import file_hash, read, register
from experiments.controller.direct_skill_allocation import direct_distributions
from experiments.controller.passage_evidence_actor import gpu
from experiments.controller.selective_span_policy import weights
from experiments.controller.skill_graph_data import training_rows
from experiments.controller.three_skill_margin import margin_fit, score
from experiments.controller.witness_passage_policy import limited
from experiments.controller.writer_format_history import archive as history_archive
from experiments.controller.writer_format_sampling import SAMPLED
from experiments.controller.writer_format_trial import CELLS, effects, shares

SENSITIVITY = Path(__file__).with_name("protocols") / "writer_format_sensitivity.json"
MODES = ("sensitivity", "verified", "dense", "history", "transfer", *SAMPLED)


def collect(base, src, mode):
    original, _ = training_rows(base, True)
    reg = read(src / "registration.json")
    if mode in ("sensitivity", "history"):
        marks = {}
    else:
        phase = "readers" if mode == "dense" else "verified"
        marks = read(src / (phase + "-complete.json"))
        required = (
            set(reg["questions"])
            if mode == "dense"
            else {k for k, v in reg["flags"].items() if v}
        )
        if set(marks) != required:
            raise ValueError("Incomplete component measurement census")
    samples = []
    rejected = {}
    for w in original:
        name = w["window"]
        qids = reg["questions"][name]
        flag = reg["flags"][name]
        physical = {}
        if name in marks:
            parent = src / phase / (name + ".json")
            obs = read(parent)
            prepared = src / "prepared" / parent.name
            expected = {
                "registration": file_hash(src / "registration.json"),
                "parent": file_hash(prepared),
            }
            if file_hash(parent) != marks[name] or obs["inputs"] != expected:
                raise ValueError("Unbound component measurements")
            if obs["status"] == "answered":
                ids = {str(i) for i in qids} if mode == "dense" else set(flag)
                if set(obs["rows"]) != ids:
                    raise ValueError("Missing component question")
                for i, h in obs["rows"].items():
                    f = src / "questions" / name / (i + ".json")
                    r = read(f)
                    if file_hash(f) != h or r["inputs"] != expected:
                        raise ValueError("Unbound crossed question")
                    if r["status"] == "matched":
                        parts = effects({k: r["reads"][k]["f1"] for k in CELLS})
                        if (
                            parts != r["effects"]
                            or shares(parts, flag.get(i, 0.0)) != r["credit"]
                        ):
                            raise ValueError("Changed decomposed credit")
                        physical[int(i)] = r
                    else:
                        rejected[f"{name}/{i}"] = r["status"]
            else:
                rejected[name] = obs["status"]
        if mode == "dense" and len(physical) != len(qids):
            rejected.setdefault(name, "incomplete_matched_questions")
            continue
        c = [0.0, 0.0]
        for q in w["points"]:
            idx = q["index"]
            y = q["scores"][2] - q["scores"][1]
            v = q["edges"][2]
            if v != flag.get(str(idx), 0.0):
                raise ValueError("Changed original direct credit")
            if idx in physical and abs(physical[idx]["effects"]["raw"] - y) > 1e-10:
                raise ValueError("Original raw return changed")
            assigned = (
                [v / 2, v / 2]
                if mode == "sensitivity"
                else physical.get(idx, {}).get("credit", [0.0, 0.0])
            )
            for j in range(2):
                c[j] += assigned[j] / len(qids)
        samples.append(
            {
                "history": w["history"],
                "node": name,
                "features": w["features"],
                "component_credit": c,
                "raw": mean(q["scores"][2] - q["scores"][1] for q in w["points"]),
                "component_raw": [
                    mean(r["effects"][k] for r in physical.values())
                    for k in ("template", "grouping")
                ]
                if mode == "dense"
                else None,
            }
        )
    return samples, rejected


def rotation(rows):
    perm = list(range(len(rows)))
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["history"]].append(i)
    for group in by.values():
        group.sort(key=lambda i: rows[i]["node"])
        for a, b in zip(group, group[1:] + group[:1], strict=True):
            perm[a] = b
    return perm


def qualified_weights(rows, ids, targets, tensor):
    active = [
        dict(r, qualified=True) if any(v) else r
        for r, v in zip(rows, targets.tolist(), strict=True)
    ]
    return weights(active, ids, "qualified", tensor)


def fit(samples, h, device, *, dense=False, history=()):
    rows = [r for r in samples if r["history"] != h]
    ids = {r["history"] for r in rows}
    if not ids:
        raise ValueError("No out-of-history training observations")

    def t(v):
        return torch.as_tensor(v, device=device, dtype=torch.float64)

    w = weights(rows, ids, "raw", t)
    off, g = t([r["features"] for r in rows]).T
    center = w @ g
    sd = (w @ (g - center).square()).sqrt().clamp_min(0.02)
    z = torch.zeros_like(g)
    one = torch.ones_like(g)
    template = torch.stack([one, off / 2, z, z], dim=1)
    grouping = torch.stack([z, z, one, ((g - center) / sd).clamp(-3, 3)], dim=1)
    whole = template + grouping
    cx = torch.cat([template, grouping])
    if dense:
        x = cx
        y = t([r["component_raw"] for r in rows]).T.reshape(-1)
        rw = w.repeat(2)
    else:
        x = whole
        y = t([r["raw"] for r in rows])
        rw = w
    c = t([r["component_credit"] for r in rows])
    perm = rotation(rows)
    older = [r for r in history if r["history"] != h]
    if older:
        g_old = t([r["features"][1] for r in older])
        z_old = torch.zeros_like(g_old)
        x_old = torch.stack(
            (
                z_old,
                z_old,
                torch.ones_like(g_old),
                ((g_old - center) / sd).clamp(-3, 3),
            ),
            dim=1,
        )
        old_ids = {r["history"] for r in older}
        x = torch.cat((x, x_old))
        y = torch.cat((y, t([r["raw"] for r in older])))
        rw = torch.cat((rw, weights(older, old_ids, "raw", t)))
        old_credit = t([[r["credit"]] for r in older])
    total = c.sum(1)
    equal = total[:, None].expand(-1, 2) / 2
    choices = {
        "credit": c,
        "swapped": c.flip((1,)),
        "equal": equal,
        "rotated": c[perm],
        "writer_only": torch.stack([total, z], 1),
        "grouping_only": torch.stack([z, total], 1),
        "undivided": c,
    }
    if older:
        choices.update(zero_history=c, zero_current=torch.zeros_like(c))
    coef = {}
    for k, targets in choices.items():
        v = qualified_weights(rows, ids, targets, t)
        joint = k == "undivided"
        target_y = targets.sum(1) if joint else targets.T.reshape(-1)
        target_w = v if joint else v.repeat(2)
        target_x = whole if joint else cx
        if older:
            oc = old_credit[rotation(older)] if k == "rotated" else old_credit
            if k == "zero_history":
                oc = torch.zeros_like(oc)
            target_x = torch.cat((target_x, x_old))
            target_y = torch.cat((target_y, oc.flatten()))
            target_w = torch.cat((target_w, qualified_weights(older, old_ids, oc, t)))
        raw, b = margin_fit(
            x,
            y,
            rw,
            target_y,
            target_w,
            constraint_x=target_x,
            penalty=t([0.04, 0.25, 0.04, 0.25]),
        )
        coef[k] = b.tolist()
        coef["raw"] = raw.tolist()
    return {
        "center": float(center),
        "sd": float(sd),
        "coef": coef,
        "histories": sorted(ids),
        "windows": [r["node"] for r in rows],
        **({"historical_windows": [r["node"] for r in older]} if older else {}),
    }


def scores_at(model, source, device):
    t = lambda v: torch.as_tensor(v, dtype=torch.float64, device=device)
    x = t(
        [
            1.0,
            source[0] / 2,
            1.0,
            max(-3.0, min(3.0, (source[1] - model["center"]) / model["sd"])),
        ]
    )
    return {k: float(x @ t(v)) for k, v in model["coef"].items()}


def probabilities(native, temporal, scores):
    prior = native[2] / (native[1] + native[2])
    result = {}
    for k, v in scores.items():
        q = 0.2 * prior + 0.8 * (native.new_tensor(v) / 0.05).sigmoid()
        result[k] = torch.stack(
            (temporal, (1 - temporal) * (1 - q), (1 - temporal) * q)
        )
    return result


def allocate(native, packet):
    prior = direct_distributions(native, packet)
    return {
        **probabilities(native, prior["raw"][0], packet["factors"]),
        "native": native,
        "uniform": prior["uniform"],
        "parent_credit": prior["credit"],
        "parent_raw": prior["raw"],
    }


def predict(model, comparisons, source):
    pred = scores_at(model, source, "cuda")
    for k, c in comparisons.items():
        s = scores_at(c, source, "cuda")
        for arm in ("credit", "raw"):
            pred[k + "_" + arm] = s[arm]
    return pred


def surveyed_samples(rows, survey, key):
    result = []
    for r in rows:
        v = survey["windows"][r["node"]]
        if "excluded" in v:
            continue
        if v["n"] <= 0 or v[key] is None:
            raise ValueError("Incomplete window estimate")
        total, difference = r["raw"], v[key]
        result.append(
            {**r, "component_raw": [(total + difference) / 2, (total - difference) / 2]}
        )
    return result


def freeze(base, mode):
    src = base / "source-compression/writer-format-credit"
    three = base / "late-pairs/semantic-writer/three-writer"
    parent = three / "checkpoint-odds/forecast.json"
    early = src.parent / "earliest-three-skills/forecast.json"
    out = three / ("writer-format-" + mode)
    out.mkdir(exist_ok=True)
    earlyout = early.parent / out.name
    earlyout.mkdir(exist_ok=True)
    f = out / "forecast.json"
    e = earlyout / "forecast.json"
    historic = history_archive(base) if mode in ("history", "transfer") else None
    sampled = mode in SAMPLED
    survey_path = src / "sampling" / (mode + ".json")
    identity = {
        "parent": file_hash(parent),
        "early": file_hash(early),
        "registration": file_hash(src / "registration.json"),
        "stage": file_hash(
            survey_path
            if sampled
            else historic
            if mode == "history"
            else SENSITIVITY
            if mode == "sensitivity"
            else src
            / (("readers" if mode == "dense" else "verified") + "-complete.json")
        ),
    }
    if sampled:
        identity["verified"] = file_hash(src / "verified-complete.json")
        identity["frame"] = file_hash(src / "sampling/frame.json")
    if historic is not None:
        identity["historical"] = file_hash(historic)
    original_factor = three / "writer-format-verified/forecast.json"
    if mode == "transfer":
        identity["no_raw_transport"] = file_hash(original_factor)
    if f.exists() and e.exists():
        if (
            read(f)["inputs"] != identity
            or read(f)["models"] != file_hash(out / "models.json")
            or read(e)["parent"] != file_hash(f)
        ):
            raise ValueError("Changed writer/format forecast")
        return f
    rows, excluded = collect(base, src, "verified" if sampled else mode)
    controls = {}
    if sampled:
        survey = read(survey_path)
        if survey["frame"] != identity["frame"] or set(survey["windows"]) != {
            r["node"] for r in rows
        }:
            raise ValueError("Changed sampled-difference population")
        excluded.update(
            {k: v["excluded"] for k, v in survey["windows"].items() if "excluded" in v}
        )
        proxy = surveyed_samples(rows, survey, "proxy")
        if mode != "proxy":
            controls = {
                "proxy": proxy,
                "sample_only": surveyed_samples(rows, survey, "sample_only"),
            }
        rows = proxy if mode == "proxy" else surveyed_samples(rows, survey, "corrected")
        controls["whole"] = rows
    history_rows = read(historic)["rows"] if historic is not None else ()
    earlier_without = early.parent / "writer-format-verified/forecast.json"
    without = read(original_factor)["windows"] if mode == "transfer" else {}
    without_early = read(earlier_without)["windows"] if without else {}
    if without and read(earlier_without)["parent"] != identity["no_raw_transport"]:
        raise ValueError("Changed no-transport early model")
    forecasts = {}
    earlies = {}
    models = {}
    later = read(parent)["windows"]
    original = read(early)
    if (mode == "dense" or sampled) and len({r["history"] for r in rows}) < 2:
        register(
            out / "unavailable.json",
            {
                "inputs": identity,
                "reason": "fewer_than_two_matched_donor_histories",
                "excluded": excluded,
            },
        )
        return None
    for h in sorted({w["history"] for w in original["windows"].values()}):
        with limited():
            m = fit(
                rows, h, "cuda", dense=mode == "dense" or sampled, history=history_rows
            )
            counter = {
                k: fit(v, h, "cuda", dense=k != "whole") for k, v in controls.items()
            }
            models[h] = {**m, "comparison_models": counter} if sampled else m

            for k, w in later.items():
                if k.rsplit("_s", 1)[0] != h:
                    continue
                pred = predict(m, counter, w["scores"]["source_features"])
                if (
                    mode in ("sensitivity", "verified")
                    and abs(pred["raw"] - w["scores"]["direct"]["raw"]) > 1e-10
                ):
                    raise ValueError(
                        "Two additive heads changed the source-only raw baseline"
                    )
                if without:
                    pred["no_transport"] = without[k]["scores"]["factors"]["credit"]
                forecasts[k] = {**w, "scores": {**w["scores"], "factors": pred}}
            for k, w in original["windows"].items():
                if w["history"] != h:
                    continue
                p = w["probabilities"]
                native = torch.tensor(p["native"], dtype=torch.float64, device="cuda")
                ps = probabilities(
                    native,
                    native.new_tensor(p["direct_raw"][0]),
                    predict(m, counter, w["features"]),
                )
                if (
                    mode in ("sensitivity", "verified")
                    and max(
                        abs(a - b)
                        for a, b in zip(
                            ps["raw"].tolist(), p["direct_raw"], strict=True
                        )
                    )
                    > 1e-10
                ):
                    raise ValueError("Early raw probability differs from the original")
                if without_early:
                    ps["no_transport"] = native.new_tensor(
                        without_early[k]["probabilities"]["factor_credit"]
                    )
                earlies[k] = {
                    **w,
                    "probabilities": {
                        **p,
                        **{"factor_" + j: v.tolist() for j, v in ps.items()},
                    },
                }
    if len(forecasts) != 25 or len(earlies) != 12:
        raise ValueError("Incomplete issued forecasts")
    register(
        out / "models.json",
        {"inputs": identity, "donors": rows, "excluded": excluded, "models": models},
    )
    register(
        f,
        {
            "inputs": identity,
            "models": file_hash(out / "models.json"),
            "windows": forecasts,
        },
    )
    x = earlyout / "exposure.json"
    if not x.exists():
        register(
            x,
            {
                phase: {
                    p.stem: file_hash(p)
                    for p in (early.parent / phase).glob("conv-*.json")
                }
                for phase in ("writers", "readers", "pairs", "judged")
            },
        )
    register(
        e,
        {
            "inputs": original["inputs"],
            "parent": file_hash(f),
            "exposure": file_hash(x),
            "windows": earlies,
        },
    )
    return f


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("mode", choices=MODES)
    a = cli.parse_args()
    gpu()
    f = freeze(a.base, a.mode)
    if f is not None:
        score(
            a.base,
            f,
            allocate,
            direct=True,
            eligible=a.mode not in ("sensitivity", "history", "proxy", "sample1"),
            extra_controls=("swapped", "equal", "undivided")
            + (
                ("zero_history", "zero_current", "no_transport")
                if a.mode == "transfer"
                else ("proxy_credit", "sample_only_credit", "whole_credit")
                if a.mode == "sample2"
                else ()
            ),
            replay_choices=True,
        )
