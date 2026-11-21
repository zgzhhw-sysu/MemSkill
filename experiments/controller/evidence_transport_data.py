"""Build donor-only evidence keys and outcomes from verified, measured paths."""

import math
from statistics import mean

import numpy as np
import torch

from experiments.controller import file_hash, read


def sources(dest, device):
    result = {}
    for path in sorted((dest / "turns").glob("*.npz")):
        with np.load(path, allow_pickle=False) as arr:
            ids = arr["ids"].tolist()
            matrix = torch.as_tensor(arr["vectors"], device=device)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(ids)
                or len(ids) != len(set(ids))
                or not torch.isfinite(matrix).all()
            ):
                raise ValueError("Invalid saved source vectors")
            result[path.stem] = (ids, matrix, str(arr["source"]), file_hash(path))
    return result


def source_keys(name):
    sid, n = name.rsplit("_s", 1)
    return name, f"{sid}_s{int(n) + 1}"


def evidence_links(name, questions, data, turns):
    sid = name.rsplit("_s", 1)[0]
    maps = [{turn: i for i, turn in enumerate(turns[k][0])} for k in source_keys(name)]
    rows = []
    for q in questions:
        gold = set(data[sid]["qa"][q]["evidence"])
        if not gold or gold - maps[0].keys() - maps[1].keys():
            raise ValueError("Donor evidence falls outside its assigned window")
        rows.append(tuple([m[t] for t in sorted(gold & m.keys())] for m in maps))
    return rows


def factorial(base, turns, data):
    source = base / "source-compression"
    early = read(source / "retrieval-competition/fit.json")["training"]
    root = read(source / "root-overlap/fit.json")["effects"]
    revised = read(source / "writer-block-repair/fit.json")["revised"]
    checked = read(source / "writer-block-repair/credit/summary.json")["files"]
    donors = {}
    for w in early:
        name = w["window"]
        q = w["questions"]
        y = w["cases"]
        root_file = source / "root-write-repairs" / name / "window.json"
        term_file = source / "writer-block-repair/credit" / (name + ".json")
        if file_hash(term_file) != checked[name]:
            raise ValueError("Changed rechecked terminal credit")
        root_bonus = [[0.0] * len(q) for _ in range(2)]
        term_bonus = [[0.0] * len(q) for _ in range(2)]
        for r in read(root_file)["rows"]:
            v = r["signed_effect"]
            if v and r["status"] != "survived_probes":
                raise ValueError("Unverified root credit")
            root_bonus[r["suffix"]][q.index(r["index"])] = v
        for g in read(term_file)["groups"]:
            for r in g["questions"]:
                if not r["matched"] or q[r["qa_index"]] != r["original_index"]:
                    raise ValueError("Mismatched corrected question")
                term_bonus[g["prefix"]][r["qa_index"]] = r["credit"]
        for bonus, ref in ((root_bonus, root[name]), (term_bonus, revised[name])):
            if not all(
                math.isclose(mean(vals), v, abs_tol=1e-12)
                for vals, v in zip(bonus, ref, strict=True)
            ):
                raise ValueError(
                    "Individual credits no longer reproduce verified aggregates"
                )
        donors[name] = {
            "sid": w["dialogue"],
            "links": evidence_links(name, q, data, turns),
            "ids": q,
            "y": y,
            "root": root_bonus,
            "last": term_bonus,
            "base": [s["base"] for s in w["states"]],
            "pivot": None,
            "bg": None,
            "inputs": [file_hash(root_file), file_hash(term_file), w["repair"]],
        }
    return donors


def randomized(base, turns, data, *, first_session_only=False):
    late = base / "late-pairs"
    folder = late / "selection-labels"
    split = read(folder / "split.json")["windows"]["donor"]
    if len(split) != 12 or sum(map(len, split.values())) != 88:
        raise ValueError("Different expanded donor population")
    if first_session_only:
        split = {n: qs for n, qs in split.items() if n.endswith("_s9")}
        if len(split) != 6 or sum(map(len, split.values())) != 45:
            raise ValueError("Different first chronological donor block")
    if not all((folder / (name + ".json")).exists() for name in split):
        return None
    donors = {}
    for name, qs in split.items():
        label_path = folder / (name + ".json")
        lab = read(label_path)
        f = late / "outcomes" / name
        pair = read(f / "pair.json")
        if lab["inputs"]["pair"] != file_hash(f / "pair.json") or qs != [
            r["original_index"] for r in lab["rows"]
        ]:
            raise ValueError("Unaligned randomized donor evidence")
        pivot, bg = lab["pivot"], lab["background"]
        if (pivot, bg) != (pair["pivot"], pair["background"]):
            raise ValueError("Mismatched pivot/background")
        base_logits = [None] * 3
        y = {}
        credits = []
        for bits, side in zip(pair["paths"], ("temporal", "activity"), strict=True):
            key = "".join(map(str, bits))
            d = read(f / key / "decisions.json")
            for j, row in zip((0, 1 + bits[0]), d, strict=True):
                if base_logits[j] is not None and base_logits[j] != row["base_logit"]:
                    raise ValueError("Different predecision state")
                base_logits[j] = row["base_logit"]
            y[key] = [q[side] for q in pair["questions"]]
        for r, q in zip(lab["rows"], pair["questions"], strict=True):
            if (
                r["original_index"] != q["original_index"]
                or not math.isclose(
                    r["delta"], q["activity"] - q["temporal"], abs_tol=1e-12
                )
                or r["credit"]
                and (r["status"] != "survived_probes" or r["credit"] != r["delta"])
            ):
                raise ValueError("Unverified donor label")
            credits.append(r["credit"])
        bonuses = [None, None]
        bonuses[bg] = credits
        donors[name] = {
            "sid": lab["dialogue"],
            "links": evidence_links(name, qs, data, turns),
            "ids": qs,
            "y": y,
            "root": bonuses if pivot == 0 else None,
            "last": bonuses if pivot == 1 else None,
            "base": base_logits,
            "pivot": pivot,
            "bg": bg,
            "inputs": [file_hash(label_path), file_hash(f / "pair.json")],
        }
    return donors
