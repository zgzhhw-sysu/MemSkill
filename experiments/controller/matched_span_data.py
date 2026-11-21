"""Source-only regrouping features and physically matched donor decisions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller.exchange_paths import Ranking
from experiments.controller.factorial_rollout import read
from experiments.controller.legacy_terminal_symmetry import census
from experiments.controller.query_displacement_data import (
    encode_groups,
    registration,
    store_vectors,
)
from experiments.controller.source_compression import file_hash, register
from experiments.controller.witness_passage_policy import limited
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "matched_span_rank.json"


def structure(singles, pairs):
    single, member = singles
    pair, joined = pairs
    if member.shape[1] != joined.shape[1] or not (member.any(0) & joined.any(0)).all():
        raise ValueError("The actions must cover the same turns")
    single, pair = F.normalize(single), F.normalize(pair)
    turns = F.normalize(member.T.to(single.dtype) @ single)

    def scores(v, membership):
        sim = turns @ v.T
        own = sim.masked_fill(~membership.T, -torch.inf).amax(1)
        other = sim.masked_fill(membership.T, -torch.inf).amax(1)
        return own, own - torch.where(other.isfinite(), other, 0)

    lo, lm = scores(single, member)
    hi, hm = scores(pair, joined)
    idx = torch.arange(len(turns), device=turns.device)
    mate = idx ^ 1
    affinities = turns @ turns.T
    adjacent = torch.where(
        mate < len(turns), affinities[idx, mate.clamp_max(len(turns) - 1)], 0
    )
    available = (idx[:, None] != idx) & (mate[:, None] != idx)
    outsider = affinities.masked_fill(~available, -torch.inf).amax(1)
    outsider = torch.where(outsider.isfinite(), outsider, 0)
    values = torch.stack([hi - lo, hm - lm, adjacent, adjacent - outsider], dim=1)
    return torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]), dim=0).flatten()


def early_sources(base, study, corpus, *, pivot=1):
    reg = census(base, study, corpus)
    known = read(reg)
    proto = read(study / "protocol.json")
    data = {
        x["sample_id"]: x["conversation"]
        for x in read(corpus)
        if x["sample_id"] in proto["train"]
    }
    proc = LoCoMoProcessor(chunk_mode="full-session")
    nodes = {}
    for window, receipt in sorted(known["windows"].items()):
        h, first = window.rsplit("_s", 1)
        n = int(first) + pivot
        sk = f"session_{n}"
        (source,) = proc._extract_session_chunks(
            data[h][sk], data[h][sk + "_date_time"]
        )
        root = study / "train" / window
        (trace,) = (root / "traces").glob("*.jsonl")
        archive = [
            e
            for line in trace.read_bytes().splitlines()
            if (e := json.loads(line))["kind"] == "span"
        ]
        if (
            file_hash(trace) != receipt["trace"]
            or len(archive) != 2
            or archive[pivot]["source"] != source
        ):
            raise ValueError("Earlier predecision source did not match the old rollout")
        rows = read(root / "rows.json")
        if (
            file_hash(root / "rows.json") != receipt["rows"]
            or file_hash(root / "questions.json") != receipt["questions"]
        ):
            raise ValueError("Different earlier decision or question count")
        nodes[f"{h}_s{n}"] = (
            source,
            data[h],
            sk,
            window,
            int(rows[1 - pivot]["use"]),
            len(read(root / "questions.json")),
        )
    return reg, nodes


def encode_extra(base, study, corpus, *, pivot=1):
    qreg = registration(base, study, corpus)
    root = qreg.parent.parent / "matched-span-rank"
    out = root / ("earlier-sources" if pivot else "earlier-root-sources")
    out.mkdir(parents=True, exist_ok=True)
    earlier, nodes = early_sources(base, study, corpus, pivot=pivot)
    encoded = qreg.parent / "encoded.json"
    receipts = read(encoded)["sources"]
    identity = {
        "earlier": file_hash(earlier),
        "encoded": file_hash(encoded),
        "plan": file_hash(PLAN),
    }
    manifest = out / "complete.json"
    if manifest.is_file():
        if read(manifest)["inputs"] != identity:
            raise ValueError("Different earlier source inputs")
        return
    model = Ranking(read(study / "protocol.json")["retriever"])
    files = {}
    for node, (src, conv, sk, *_) in sorted(
        nodes.items(), key=lambda t: (t[0] not in receipts, t[0])
    ):
        f = out / (node + ".npz")
        if not f.exists():
            store_vectors(f, **encode_groups(src, digest(src), conv, sk, model))
        if node in receipts:
            old = qreg.parent / "sources" / f.name
            if file_hash(old) != receipts[node]:
                raise ValueError("Original source archive changed")
            with (
                np.load(old, allow_pickle=False) as a,
                np.load(f, allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files) or any(
                    not np.array_equal(a[k], b[k]) for k in a.files
                ):
                    raise ValueError(
                        "Earlier group embeddings did not reproduce on the overlap"
                    )
        files[node] = file_hash(f)
        print("source", node, flush=True)
    register(manifest, {"inputs": identity, "sources": files})


def features(root, source_dir, manifest):
    dest = root / "features.json"
    identity = {"source": file_hash(manifest), "plan": file_hash(PLAN)}
    if dest.is_file():
        packet = read(dest)
        if packet["inputs"] != identity:
            raise ValueError("Changed structural inputs")
        return packet["features"]
    values = {}
    for node, h in read(manifest)["sources"].items():
        path = source_dir / (node + ".npz")
        if file_hash(path) != h:
            raise ValueError("Changed group embedding archive")
        with np.load(path, allow_pickle=False) as q, limited():
            groups = [
                (
                    torch.as_tensor(q[f"vec{i}"], device="cuda", dtype=torch.float64),
                    torch.as_tensor(q[f"map{i}"], device="cuda", dtype=torch.bool),
                )
                for i in (1, 2)
            ]
            values[node] = structure(*groups).tolist()
    register(dest, {"inputs": identity, "features": values})
    return values


def original_units(packet):
    bins = defaultdict(list)
    for r in packet["rows"]:
        j = r["stage"]
        a, b = r["bits"]
        if a[j] != "0" or b[j] != "1" or a[1 - j] != b[1 - j]:
            raise ValueError("Did not isolate a single ordered action")
        bins[r["history"], j, int(a[1 - j]), r["node"]].append(r)
    units = [
        {
            "history": h,
            "stage": j,
            "background": b,
            "node": n,
            "questions": len(v),
            "raw": mean(x["raw"] for x in v),
            "credit": mean(x["credit"] for x in v),
        }
        for (h, j, b, n), v in sorted(bins.items())
    ]
    if len(units) != 60 or sum(x["questions"] for x in units) != 452:
        raise ValueError("Missing original physical contrasts")
    return units


def extra_units(base, study, corpus):
    earlier, nodes = early_sources(base, study, corpus)
    summary = earlier.parent / "summary.json"
    checked = read(summary)
    if checked["registration"] != file_hash(earlier) or checked[
        "outcomes"
    ] != file_hash(earlier.parent / "outcomes-complete.json"):
        raise ValueError("Different weaker-action audit")
    jobs = {j["id"]: j for j in read(earlier)["jobs"]}
    cases = checked["cases"]
    if len(cases) != len(jobs) or {c["id"] for c in cases} != set(jobs):
        raise ValueError("Incomplete weaker-action cases")
    by_window = defaultdict(float)
    for c in cases:
        j = jobs[c["id"]]
        wanted = j["signed_credit"] if c["status"] == "confirmed" else 0.0
        if c["credit"] != wanted or c["history"] != j["history"] or j["pivot"] != 1:
            raise ValueError("Incorrect earlier terminal credit")
        by_window[j["window"]] += wanted
    return [
        {
            "history": n.rsplit("_s", 1)[0],
            "stage": 1,
            "background": b,
            "node": n,
            "questions": count,
            "credit": by_window[w] / count,
        }
        for n, (_, _, _, w, b, count) in sorted(nodes.items())
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("--pivot", type=int, choices=(0, 1), default=1)
    a = p.parse_args()
    encode_extra(a.base, a.study, a.corpus, pivot=a.pivot)
