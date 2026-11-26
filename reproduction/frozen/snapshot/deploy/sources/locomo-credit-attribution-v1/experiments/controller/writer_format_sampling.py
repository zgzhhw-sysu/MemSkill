"""Estimate the difference of two crossed reader returns by stratified probability sampling."""

import argparse
import fcntl
import secrets
import time
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, variance

from experiments.controller import file_hash, read, register
from experiments.controller.writer_format_trial import NATIVE, question_receipt
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "writer_format_sample.json"
HYBRIDS = ("relation_pairs", "activity_single")
SAMPLED = ("proxy", "sample1", "sample2")


def auxiliary(arms):
    def predict(hybrid):
        prompt = arms[hybrid]["prompt"].split()
        sims = [
            SequenceMatcher(
                None, prompt, arms[k]["prompt"].split(), autojunk=False
            ).ratio()
            for k in NATIVE
        ]
        return mean(
            arms[k]["old"]["f1"]
            for k, s in zip(NATIVE, sims, strict=True)
            if s == max(sims)
        )

    return predict(HYBRIDS[0]) - predict(HYBRIDS[1])


def measured(packet):
    a, b = (packet["reads"][k]["f1"] for k in HYBRIDS)
    return a - b


def frame(src):
    dest = src / "sampling"
    dest.mkdir(exist_ok=True)
    path = dest / "frame.json"
    prep = src / "prepared-complete.json"
    reg = src / "registration.json"
    identity = {
        "plan": file_hash(PLAN),
        "prepared": file_hash(prep),
        "registration": file_hash(reg),
    }
    with (dest / "frame.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        draw = dest / "draw.json"
        if path.exists():
            saved = read(path)
            if saved["inputs"] != identity or saved["draw"] != file_hash(draw):
                raise ValueError("Changed sampling frame")
            return path
        # Record the seed and cutoff once, even if interrupted while assembling the frame.
        if not draw.exists():
            cutoff = time.time_ns()
            register(
                draw,
                {
                    "inputs": identity,
                    "seed": secrets.token_hex(32),
                    "cutoff_ns": cutoff,
                    "existing": {
                        str(f.relative_to(src)): file_hash(f)
                        for f in sorted((src / "questions").glob("*/*.json"))
                        if f.stat().st_mtime_ns <= cutoff
                    },
                },
            )
        d = read(draw)
        if d["inputs"] != identity:
            raise ValueError("Changed preregistered random draw")
        regdata = read(reg)
        windows = {}
        prepared = read(prep)
        if set(prepared) != set(regdata["questions"]):
            raise ValueError("Incomplete sampling population")
        for name, h in prepared.items():
            pf = src / "prepared" / (name + ".json")
            p = read(pf)
            if (
                file_hash(pf) != h
                or p["inputs"]["registration"] != identity["registration"]
            ):
                raise ValueError("Unbound prepared window")
            ids = list(map(str, regdata["questions"][name]))
            if p["status"] != "prepared":
                windows[name] = {"n": len(ids), "excluded": p["status"]}
                continue
            if set(p["questions"]) != set(ids) or len(ids) != len(set(ids)):
                raise ValueError("Changed question census")
            by = {}
            for i in ids:
                q = p["questions"][i]
                f = src / "questions" / name / (i + ".json")
                rel = str(f.relative_to(src))
                if q["mismatch"]:
                    by[i] = {"excluded": "different_pure_ranking"}
                elif rel in d["existing"]:
                    obs = read(f)
                    if file_hash(f) != d["existing"][rel] or obs["inputs"] != {
                        "parent": h,
                        "registration": identity["registration"],
                    }:
                        raise ValueError("Unbound certainty question")
                    by[i] = (
                        {"exact": measured(obs), "source": "physical"}
                        if obs["status"] == "matched"
                        else {"excluded": obs["status"]}
                    )
                elif q["arms"][HYBRIDS[0]]["hash"] == q["arms"][HYBRIDS[1]]["hash"]:
                    by[i] = {"exact": 0.0, "source": "identical_requests"}
                else:
                    by[i] = {"proxy": auxiliary(q["arms"])}
            unknown = [i for i, v in by.items() if "proxy" in v]
            sampled = sorted(unknown, key=lambda i: digest([d["seed"], name, i]))[:2]
            windows[name] = {
                "prepared": h,
                "rows": by,
                "n": len(ids),
                "sample": sampled,
                "inclusion_probability": [
                    min(k, len(unknown)) / len(unknown) if unknown else None
                    for k in (1, 2)
                ],
            }
        register(
            path, {"inputs": identity, "draw": file_hash(draw), "windows": windows}
        )
    return path


def estimate(rows, sample, values):
    """Mean and design variance under SRSWOR within the remaining stratum."""
    n = len(rows)
    m = sum("proxy" in v for v in rows.values())
    certainty = sum(v["exact"] for v in rows.values() if "exact" in v)
    proxy = (certainty + sum(v["proxy"] for v in rows.values() if "proxy" in v)) / n
    if len(sample) != len(set(sample)) or any(
        i not in values or "proxy" not in rows[i] for i in sample
    ):
        raise ValueError("Invalid residual sample")
    if not sample:
        return {
            "proxy": proxy,
            "corrected": None if m else proxy,
            "sample_only": None if m else proxy,
            "variance": None if m else 0.0,
        }
    residuals = [values[i] - rows[i]["proxy"] for i in sample]
    k = len(sample)
    var = (
        0.0
        if k == m
        else (m / n) ** 2 * (1 - k / m) * variance(residuals) / k
        if k > 1
        else None
    )
    return {
        "proxy": proxy,
        "corrected": proxy + m / n * mean(residuals),
        "sample_only": certainty / n + m / n * mean(values[i] for i in sample),
        "variance": var,
    }


def report(src, path, stage):
    manifest = read(path)
    windows = {}
    sources = {}
    for name, w in manifest["windows"].items():
        bad = w.get("excluded") or {
            i: r["excluded"] for i, r in w["rows"].items() if "excluded" in r
        }
        if bad:
            windows[name] = {"excluded": bad}
            continue
        choice = w["sample"][:stage]
        values = {}
        for i in choice:
            f = src / "questions" / name / (i + ".json")
            if not f.is_file():
                return None
            j = read(f)
            if j["inputs"] != {
                "parent": w["prepared"],
                "registration": manifest["inputs"]["registration"],
            }:
                raise ValueError("Different physical survey answer")
            sources[str(f.relative_to(src))] = file_hash(f)
            if j["status"] != "matched":
                bad[i] = j["status"]
            else:
                values[i] = measured(j)
        windows[name] = (
            {"excluded": bad}
            if bad
            else {"n": w["n"], "sample": values, **estimate(w["rows"], choice, values)}
        )
    target = path.parent / (SAMPLED[stage] + ".json")
    register(target, {"frame": file_hash(path), "sources": sources, "windows": windows})
    return target


def advance(src):
    f = frame(src)
    data = read(f)
    for stage in range(3):
        if (f.parent / (SAMPLED[stage] + ".json")).is_file():
            continue
        if report(src, f, stage):
            continue
        for name, w in data["windows"].items():
            if w.get("excluded") or any("excluded" in r for r in w["rows"].values()):
                continue
            for i in w["sample"][:stage]:
                if (src / "questions" / name / (i + ".json")).is_file():
                    continue
                p = src / "prepared" / (name + ".json")
                if file_hash(p) != w["prepared"]:
                    raise ValueError("Changed prepared reader")
                question_receipt(
                    src,
                    name,
                    i,
                    read(p),
                    {
                        "parent": w["prepared"],
                        "registration": data["inputs"]["registration"],
                    },
                )
                return


def compare(src):
    f = frame(src)
    full = src / "readers-complete.json"
    final = f.parent / "sample2.json"
    if not final.is_file() or not full.is_file():
        return
    survey = read(final)
    if survey["frame"] != file_hash(f):
        raise ValueError("Changed completed survey")
    results = {}
    errors = {k: [] for k in ("proxy", "corrected", "sample_only")}
    for name, w in read(f)["windows"].items():
        ep = src / "readers" / (name + ".json")
        er = read(ep)
        if file_hash(ep) != read(full)[name] or er["inputs"] != {
            "parent": read(src / "prepared-complete.json")[name],
            "registration": file_hash(src / "registration.json"),
        }:
            raise ValueError("Changed physical census")
        if er["status"] == "answered" and set(er["rows"]) != set(w["rows"]):
            raise ValueError("Different full-census questions")
        reasons = {}
        obs = []
        for i, h in er["rows"].items():
            q = src / "questions" / name / (i + ".json")
            v = read(q)
            if file_hash(q) != h or v["inputs"] != er["inputs"]:
                raise ValueError("Different physical endpoint")
            if v["status"] == "matched":
                obs.append(measured(v))
            else:
                reasons[i] = v["status"]
        b = survey["windows"][name]
        if "excluded" in b or reasons or len(obs) != w["n"]:
            results[name] = {
                "excluded": b.get("excluded") or reasons or er["status"],
                "matched": len(obs),
                "n": w["n"],
            }
            continue
        y = mean(obs)
        es = {k: b[k] - y for k in errors}
        for k, z in es.items():
            errors[k].append(abs(z))
        results[name] = {
            "n": len(obs),
            "census": y,
            "error": es,
            "reported_variance": b["variance"],
        }
    register(
        f.parent / "census.json",
        {
            "sample": file_hash(final),
            "census": file_hash(full),
            "windows": results,
            "mae": {k: mean(v) if v else None for k, v in errors.items()},
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("phase", choices=("register", "read", "compare"))
    a = cli.parse_args()
    s = a.base / "source-compression/writer-format-credit"
    if a.phase == "register":
        report(s, frame(s), 0)
    elif a.phase == "read":
        advance(s)
    else:
        compare(s)
