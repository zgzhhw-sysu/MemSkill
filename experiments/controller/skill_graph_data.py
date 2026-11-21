"""Join authenticated fixed-prefix skill comparisons by original question."""

from collections import defaultdict
from pathlib import Path

from experiments.controller import file_hash, read, register


PLAN = Path(__file__).with_name("protocols") / "skill_graph_credit.json"


def registration(base):
    src = base / "source-compression"
    semantic = base / "late-pairs/semantic-writer"
    dest = semantic / "three-writer/skill-graph"
    dest.mkdir(exist_ok=True)
    path = dest / "registration.json"
    identity = {
        "plan": file_hash(PLAN),
        "margin": file_hash(semantic / "three-writer/margin/forecast.json"),
        "questions": file_hash(src / "query-displacement/registration.json"),
        "prepared": file_hash(
            src / "activity-relation-repairs/0/prepared-complete.json"
        ),
        "time_relation": file_hash(
            src / "relation-skill-repairs/0/outcomes-complete.json"
        ),
        "geometry": file_hash(src / "matched-span-rank/features.json"),
        "panel": file_hash(semantic / "panel.json"),
        "affinity": file_hash(semantic / "forecast.json"),
    }
    prior = read(src / "matched-span-rank/root-augmented/forecast.json")["inputs"]
    if (
        identity["questions"] != prior["registration"]
        or identity["geometry"] != prior["features"]
    ):
        raise ValueError("Changed physical action archive or pre-action geometry")
    if not path.exists():
        register(
            path,
            {
                "inputs": identity,
                "existing": {
                    str(f.relative_to(base)): file_hash(f)
                    for directory, pattern in (
                        (src / "activity-relation-repairs/0/outcomes", "conv-*.json"),
                        (semantic, "conv-*/write.json"),
                        (semantic, "conv-*/pair.json"),
                        (base / "late-pairs/outcomes", "conv-*/*/scores.json"),
                    )
                    for f in directory.glob(pattern)
                },
            },
        )
    receipt = read(path)
    if receipt["inputs"] != identity or any(
        file_hash(base / rel) != digest for rel, digest in receipt["existing"].items()
    ):
        raise ValueError("Changed three-skill comparison inputs")
    return path


def audit(folder, name, phase, parent, hashes):
    path = folder / phase / (name + ".json")
    row = read(path)
    if (
        file_hash(path) != hashes[name]
        or row["inputs"]["parent"] != file_hash(parent)
        or row["inputs"]["registration"] != file_hash(folder / "registration.json")
    ):
        raise ValueError("Changed skill audit")
    return row


def credits(receipt):
    out = {}
    for j in receipt["jobs"]:
        idx = j["original_index"]
        if idx in out or j["credit"] != (j["raw"] if j["status"] == "confirmed" else 0):
            raise ValueError("Ambiguous verified outcome")
        out[idx] = j["credit"]
    return out


def training_rows(base, complete):
    src = base / "source-compression"
    direct = src / "activity-relation-repairs/0"
    relation = src / "relation-skill-repairs/0"
    pre = read(direct / "prepared-complete.json")
    rt = read(relation / "outcomes-complete.json")
    ar = read(direct / "outcomes-complete.json") if complete else None
    if (
        len(pre) != 12
        or pre.keys() != rt.keys()
        or (complete and pre.keys() != ar.keys())
    ):
        raise ValueError("Changed source census")
    activity = defaultdict(dict)
    archive = src / "query-displacement/registration.json"
    for r in read(archive)["rows"]:
        if r["stage"] == 1 and r["bits"] == ["00", "01"]:
            group = activity[r["window"]]
            if r["index"] in group:
                raise ValueError("Repeated physical question at decision")
            group[r["index"]] = r
    geo = read(src / "matched-span-rank/features.json")["features"]
    prior = file_hash(src / "matched-span-rank/root-augmented/forecast.json")
    windows, excluded = [], {}
    for name in sorted(pre):
        pair = src / "semantic-writer" / name / "pair.json"
        old = read(pair)
        d = audit(direct, name, "prepared", pair, pre)
        r = audit(
            relation, name, "outcomes", relation / "prepared" / (name + ".json"), rt
        )
        if read(relation / "prepared" / (name + ".json"))["inputs"][
            "parent"
        ] != file_hash(pair):
            raise ValueError("Changed relation source")
        final = (
            audit(direct, name, "outcomes", direct / "prepared" / (name + ".json"), ar)
            if complete
            else None
        )
        if final and any(
            final.get(k) != d.get(k) for k in ("matched", "questions", "raw_pairs")
        ):
            raise ValueError("Direct audit changed original contrast")
        if not (d["matched"] and r["matched"] and old["matched"]):
            excluded[name] = d.get("reason") or r.get("reason") or old.get("reason")
            continue
        cr = credits(r)
        ca = credits(final) if final else {}
        h, n = name.rsplit("_s", 1)
        node = f"{h}_s{int(n) + 1}"
        early = read(src / "matched-span-rank/root-augmented/early" / (name + ".json"))
        if (
            early["inputs"]["forecast"] != prior
            or early["dialogue"] != h
            or early["window"] != name
        ):
            raise ValueError("Changed early decision probabilities")
        (trace,) = [
            v
            for k, v in early["inputs"]["paths"]["00"].items()
            if k.startswith("traces/")
        ]
        if trace != old["source_trace"] or old["choice"]["control_trace"] != trace:
            raise ValueError("Different original terminal decision")
        points = []
        for i, (q, p, original) in enumerate(
            zip(old["rows"], d["raw_pairs"], early["rows"], strict=True)
        ):
            idx = q["original_index"]
            a = activity[name][idx]
            scores = [q["temporal"]["f1"], p["activity"], p["relation"]]
            edges = [a["credit"], cr.get(idx, 0), ca.get(idx, 0)]
            deltas = [scores[j] - scores[k] for k, j in ((0, 1), (0, 2), (1, 2))]
            if (
                idx != p["original_index"]
                or idx != original["index"]
                or a["node"] != node
                or q["relation"]["f1"] != scores[2]
                or abs(deltas[0] - a["raw"]) > 1e-10
                or any(
                    abs(scores[j] - early["cells"][bit][i]) > 1e-10
                    for j, bit in enumerate(("00", "01"))
                )
                or any(
                    c and abs(c - delta) > 1e-10
                    for c, delta in zip(edges, deltas, strict=True)
                )
            ):
                raise ValueError(
                    "Skill outcomes or signed verified effects do not align"
                )
            points.append(
                {
                    "index": idx,
                    "category": q["category"],
                    "scores": scores,
                    "edges": edges,
                }
            )
        if (
            len(points) != old["questions"]
            or len(points) != d["questions"]
            or not (cr.keys() | ca.keys()) <= {p["index"] for p in points}
        ):
            raise ValueError("Incomplete physical window")
        sf = old["source_forecast"]
        if sf["dialogue"] != h or sf["session"] != int(n) + 1:
            raise ValueError("Misaligned pre-action feature")
        windows.append(
            {
                "window": name,
                "history": h,
                "features": [sf["offset"], geo[node][5]],
                "native_pair": [
                    early["probabilities"]["1"]["native"],
                    old["choice"]["native"],
                ],
                "points": points,
            }
        )
    if len(windows) != 10 or sum(len(w["points"]) for w in windows) != 74:
        raise ValueError("Incomplete registered time-prefix triplets")
    return windows, excluded
