"""Apply a frozen Llama selector to held-dialogue examples of verified span credit."""

import argparse
import json
import os
import re
from itertools import product
from pathlib import Path
from urllib.request import Request, urlopen

from experiments.controller.factorial_rollout import read
from experiments.controller.query_prototypes import mask_speakers
from experiments.controller.source_compression import file_hash, register
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest

PLAN = Path(__file__).with_name("protocols") / "case_credit.json"
KINDS = ("true", "rotated", "empty")
SKILLS = (
    "Capture dates, timing and event order. Store each dated turn separately.",
    "Capture activities, participants and concrete details. Store adjacent dated turns together.",
)
URL = "http://127.0.0.1:18774/v1/chat/completions"


def excerpt(history, session):
    turns = history[f"session_{session}"]
    if not turns:
        raise ValueError("Empty target session")
    names = {history["speaker_a"]: "Speaker 1", history["speaker_b"]: "Speaker 2"}
    positions = (
        [i * (len(turns) - 1) // 11 for i in range(12)]
        if len(turns) > 12
        else range(len(turns))
    )
    pattern = re.compile(
        r"\b(?:"
        + "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
        + r")\b",
        re.IGNORECASE,
    )
    by_case = {name.casefold(): role for name, role in names.items()}
    lines = []
    for i in positions:
        line = pattern.sub(lambda m: by_case[m[0].casefold()], turns[i]["text"])
        lines.append(names[turns[i]["speaker"]] + ": " + " ".join(line.split()[:20]))
    return "\n".join(lines)


def examples(data, witnesses, held):
    out = []
    for w in sorted(
        witnesses, key=lambda x: (x["dialogue"], x["session"], x["original_index"])
    ):
        if w["dialogue"] == held:
            continue
        donor = data[w["dialogue"]]
        history = donor["conversation"]
        question = donor["qa"][w["original_index"]]
        linked = set(question["evidence"])
        turn = next(
            t for t in history[f"session_{w['session']}"] if t["dia_id"] in linked
        )
        if (
            not w["effect"]
            or w["query"] != mask_speakers(question["question"], history)
            or w["session"] not in range(1, 5)
        ):
            raise ValueError("Invalid donor credit or masked question")
        out.append(
            {
                "donor": w["dialogue"],
                "question": w["query"],
                "clue": " ".join(mask_speakers(turn["text"], history).split()[:24]),
                "winner": int(w["effect"] > 0),
            }
        )
    return out


def prompt(text, donors, kind, reverse):
    order = (1, 0) if reverse else (0, 1)
    letters = {a: letter for a, letter in zip(order, "AB", strict=True)}
    options = [letters[a] + ". " + SKILLS[a] for a in order]
    labels = [d["winner"] for d in donors]
    if kind == "rotated":
        labels = labels[1:] + labels[:1]
    header = [
        "Choose the memory-writing method most likely to help answer later questions about the new conversation.",
        "Both methods retain the original words and create generated memories.",
        *options,
    ]
    if kind != "empty":
        header.append(
            "Earlier examples from other conversations, with the better measured writing method:"
        )
        for d, label in zip(donors, labels, strict=True):
            header.append(
                f"Conversation clue: {d['clue']}\nLater question: {d['question']}\nBetter: {letters[label]}"
            )
    return "\n".join(
        [
            *header,
            "New conversation:",
            text,
            "Choose one writing method. Reply with A or B only.",
        ]
    )


def vote(reply, reverse):
    m = re.fullmatch(r"([AB])[.!]?", reply.strip(), re.IGNORECASE)
    if not m:
        return 0
    return 1 if ((m[1].upper() == "A") == reverse) else -1


def corrections(votes):
    v = {k: sum(votes[k]) / 2 for k in KINDS}
    return {
        "credit": max(-2.0, min(2.0, v["empty"] + v["true"] - v["rotated"])),
        "semantic": v["empty"],
        "verified": v["true"],
        "permuted": v["rotated"],
    }


def vote_request(text):
    """Construct the request shared by measurement and both receipt audits."""
    return {
        "model": "llama3-8b",
        "temperature": 0,
        "max_tokens": 3,
        "messages": [{"role": "user", "content": text}],
    }


def possible_offsets(target, dest):
    keys = []
    observed = {}
    for kind in KINDS:
        for text in target["prompts"][kind]:
            key = digest(vote_request(text))
            keys.append(key)
            path = dest / "responses" / (key + ".json")
            if path.exists() and key not in observed:
                record = read(path)
                if key != record["request"]:
                    raise ValueError("Wrong source prompt response")
                observed[key] = vote(record["reply"], False)
    missing = sorted(set(keys) - observed.keys())
    offsets = set()
    for completion in product((-1, 0, 1), repeat=len(missing)):
        votes = observed | dict(zip(missing, completion, strict=True))
        by_kind = {
            kind: [votes[keys[2 * j]], -votes[keys[2 * j + 1]]]
            for j, kind in enumerate(KINDS)
        }
        offsets.add(corrections(by_kind)["credit"])
    return sorted(offsets), observed


def registration(base, study, corpus):
    source, late = base / "source-compression", base / "late-pairs"
    out = source / "case-credit"
    out.mkdir(exist_ok=True)
    corpus_hash = file_hash(corpus)
    proto = read(study / "protocol.json")
    train = set(proto["train"])
    if (
        corpus_hash != proto["input_sha256"]
        or len(train) != 6
        or train & set(proto["validation"] + proto["test"])
    ):
        raise ValueError("Unexpected corpus or held-out split")
    witness_file = source / "query-prototypes/frozen.json"
    witnesses = read(witness_file)["witnesses"]
    if len(witnesses) != 7 or not {w["dialogue"] for w in witnesses} <= train:
        raise ValueError("Expected the previously verified training witnesses")
    data = {d["sample_id"]: d for d in read(corpus) if d["sample_id"] in train}
    panel = [
        {"dialogue": sid, "start": n, "panel": "factorial"}
        for sid, n in read(source / "factorial/manifest.json")["windows"]
    ]
    later = [
        dict(w, panel="later")
        for w in read(late / "cohort.json")["assignments"]
        if w["pivot"] == 1
    ]
    if (
        len(panel) != 12
        or len(later) != 25
        or sum(len(w["questions"]) for w in later) != 209
    ):
        raise ValueError("Different terminal decision panels")
    processor = LoCoMoProcessor(chunk_mode="full-session")
    targets = {}
    # One chronological early case then one later case keeps both panels making progress.
    for i in range(max(len(panel), len(later))):
        for block in (panel, later):
            if i >= len(block):
                continue
            w = block[i]
            sid, start = w["dialogue"], w["start"]
            history = data[sid]["conversation"]
            session = start + 1
            k = f"session_{session}"
            (source_text,) = processor._extract_session_chunks(
                history[k], history[k + "_date_time"]
            )
            donors = examples(data, witnesses, sid)
            text = excerpt(history, session)
            targets[f"{sid}_s{start}"] = dict(
                w,
                source=digest(source_text),
                donors=[d["donor"] for d in donors],
                prompts={
                    kind: [
                        prompt(text, donors, kind, reverse) for reverse in (False, True)
                    ]
                    for kind in KINDS
                },
            )
    path = out / "prompts.json"
    register(
        path,
        {
            "code": file_hash(Path(__file__)),
            "plan": file_hash(PLAN),
            "corpus": corpus_hash,
            "witnesses": file_hash(witness_file),
            "study": file_hash(study / "protocol.json"),
            "backend": os.environ["MEMSKILL_BACKEND_ID"],
            "targets": targets,
        },
    )
    prior = {
        str(f.relative_to(late)): file_hash(f)
        for f in sorted((late / "outcomes").glob("*/*/scores.json"))
        if f.stat().st_mtime_ns <= path.stat().st_mtime_ns
    }
    register(
        out / "preexisting.json", {"registration": file_hash(path), "scores": prior}
    )
    return out, targets


def measure(out, targets):
    responses = out / "responses"
    choices = out / "choices"
    for folder in (responses, choices):
        folder.mkdir(exist_ok=True)
    frozen = file_hash(out / "prompts.json")
    for name, target in targets.items():
        path = choices / (name + ".json")
        if path.exists():
            if read(path)["registration"] != frozen:
                raise ValueError("Choice registration changed")
            continue
        votes, receipts = {}, []
        for kind, strings in target["prompts"].items():
            votes[kind] = []
            for reverse, text in enumerate(strings):
                payload = vote_request(text)
                key = digest(payload)
                saved = responses / (key + ".json")
                if not saved.exists():
                    req = Request(
                        URL,
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urlopen(req, timeout=300) as http:
                        observed = json.load(http)
                    reply = observed["choices"][0]["message"]["content"]
                    usage = observed["usage"]
                    if usage["prompt_tokens"] <= 0 or usage["completion_tokens"] < 0:
                        raise ValueError("Missing measured inference usage")
                    register(saved, {"request": key, "reply": reply, "usage": usage})
                rec = read(saved)
                if rec["request"] != key:
                    raise ValueError("Mismatched decision response")
                votes[kind].append(vote(rec["reply"], bool(reverse)))
                receipts.append(key)
        register(
            path,
            {
                "registration": frozen,
                "source": target["source"],
                "votes": votes,
                "offsets": corrections(votes),
                "responses": receipts,
            },
        )
        print(name, votes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "study", "corpus"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    measure(*registration(args.base, args.study, args.corpus))
