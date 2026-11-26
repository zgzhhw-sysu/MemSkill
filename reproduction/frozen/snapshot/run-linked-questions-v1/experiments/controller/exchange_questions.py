"""Create source-grounded auxiliary questions from natural, first-six-history replies."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from experiments.controller import file_hash, read, register
from experiments.controller.late_pair_trial import paths
from experiments.controller.reply_probes import exchanges
from src.episode_trace import reference_match
from src.eval.locomo import LoCoMoEvaluator
from src.eval_utils import normalize_answer, ps
from src.evaluation_cache import digest, publish_response, read_response, request_cache
from src.llm_utils import get_llm_response_via_api

PLAN = Path(__file__).with_name("protocols") / "exchange_credit.json"
BACKEND = "251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30"
MODEL = "llama3-8b"


def retain_phrase(text, phrase):
    seq = [ps.stem(t) for t in normalize_answer(text).split()]
    ref = [ps.stem(t) for t in normalize_answer(phrase).split()]
    return bool(ref) and any(
        seq[i : i + len(ref)] == ref for i in range(len(seq) - len(ref) + 1)
    )


def teacher_prompt(asker, question, responder, reply):
    exchange = json.dumps(
        {"asker": asker, "asked": question, "respondent": responder, "replied": reply},
        ensure_ascii=False,
    )
    return (
        "Create one short factual reading-comprehension item about the respondent from this dialogue exchange. "
        "Write a standalone English question in the third person naming the respondent. Ask about one concrete "
        "fact actually given in the reply that answers the speaker's question. The answer must be a contiguous "
        "verbatim phrase of one to eight words in the reply. Do not reveal the answer in the question. "
        "Avoid yes/no, greetings and opinions without a specific detail. Return exactly one JSON object with "
        'string fields "question" and "answer", or null if no such item exists.\nExchange:\n'
        + exchange
    )


def parse(raw, obs, framed=False):
    stripped = raw.strip()
    if framed:
        prefix, separator, rest = stripped.partition("\n")
        if separator and re.fullmatch(r"[^{}\[\]?`\n]{1,95}:", prefix):
            stripped = rest.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    try:
        obj = json.loads(fenced[1] if fenced else stripped)
    except ValueError:
        return None, "malformed_json"
    if obj is None:
        return None, "teacher_abstained"
    if (
        not isinstance(obj, dict)
        or set(obj) != {"question", "answer"}
        or not all(isinstance(v, str) for v in obj.values())
    ):
        return None, "invalid_fields"
    q, a = (obj[k].strip() for k in ("question", "answer"))
    if (
        not 4 <= len(q.split()) <= 36
        or q.count("?") != 1
        or not q.endswith("?")
        or not re.search(
            r"(?<!\w)" + re.escape(obs["responder"]) + r"(?!\w)", q, re.IGNORECASE
        )
        or re.search(r"\b(?:I|we|you|your|my|our)\b", q, re.IGNORECASE)
    ):
        return None, "not_standalone"
    if not 1 <= len(a.split()) <= 8 or not re.search(
        r"(?<!\w)" + re.escape(a) + r"(?!\w)", obs["reply"], re.IGNORECASE
    ):
        return None, "reference_not_in_reply"
    if not normalize_answer(a) or normalize_answer(a) in {
        "yes",
        "no",
        "thanks",
        "thank you",
        "yeah",
        "sure",
        "ok",
        "okay",
        "it",
        "that",
        "he",
        "she",
        "they",
        "me",
        "i",
        "you",
        "nothing",
        "something",
    }:
        return None, "generic_reference"
    if retain_phrase(q, a) or retain_phrase(obs["source_question"], a):
        return None, "answer_in_question"
    return {
        "question": q,
        "answer": a,
        "category": 4,
        "evidence": [obs["ask_id"], obs["reply_id"]],
    }, "candidate"


def register_population(base, study, corpus):
    out = base / "source-compression/exchange-credit"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "registration.json"
    late = base / "late-pairs"
    early = base / "source-compression/factorial/manifest.json"
    cap = base / "source-compression/capped-source-credit/registration.json"
    cohort = late / "cohort.json"
    inputs = {
        "plan": file_hash(PLAN),
        "study": file_hash(study / "protocol.json"),
        "corpus": file_hash(corpus),
        "early": file_hash(early),
        "cohort": file_hash(cohort),
        "capped": file_hash(cap),
        "backend": BACKEND,
    }
    if path.exists():
        if read(path)["inputs"] != inputs:
            raise ValueError("Different registered source population")
        return path
    proto = read(study / "protocol.json")
    train = set(proto["train"])
    if (
        len(train) != 6
        or train & set(proto["validation"] + proto["test"])
        or proto["input_sha256"] != inputs["corpus"]
        or proto["model"] != MODEL
    ):
        raise ValueError("Expected frozen training corpus")
    data = {
        c["sample_id"]: c["conversation"]
        for c in read(corpus)
        if c["sample_id"] in train
    }
    windows = [(sid, n, j, None) for sid, n in read(early)["windows"] for j in (0, 1)]
    windows.extend(
        (w["dialogue"], w["start"], w["pivot"], w["background"])
        for w in read(cohort)["assignments"]
        if w["start"] in (9, 11)
    )
    if len(windows) != 36:
        raise ValueError("Wrong early and donor sessions")
    jobs = []
    coverage = {}
    for sid, start, pivot, background in windows:
        n = start + pivot
        h = data[sid]
        sk = f"session_{n}"
        turns = h[sk]
        ex = exchanges(turns)
        name = f"{sid}_s{n}"
        coverage[name] = len(ex)
        supplies = sorted(
            ex, key=lambda x: digest(["exchange-order-1", sid, n, x[1], x[2]])
        )
        for slot, (question, ask, ans, _, peers) in enumerate(supplies):
            ident = f"{name}_t{ask:03d}"
            a, b = turns[ask], turns[ans]
            coin = (
                int(digest(["exchange-background-1", ident])[:8], 16) % 2
                if background is None
                else background
            )
            family = "early" if background is None else "donor"
            obs = {
                "id": ident,
                "dialogue": sid,
                "session": n,
                "start": start,
                "pivot": pivot,
                "slot": slot,
                "background": coin,
                "bits": ["".join(map(str, v)) for v in paths(pivot, coin)],
                "panel": family,
                "date": h[sk + "_date_time"],
                "asker": a["speaker"],
                "responder": b["speaker"],
                "ask_id": a["dia_id"],
                "reply_id": b["dia_id"],
                "source_question": question,
                "ask_text": a["text"],
                "reply": b["text"],
            }
            order = peers[peers.index(ans) + 1 :] + peers[: peers.index(ans)]
            obs["other_replies"] = [
                {"id": turns[i]["dia_id"], "text": turns[i]["text"]} for i in order
            ]
            jobs.append(obs)
    # Breadth-first across sessions; all first session-5 attempts from the six histories come first.
    jobs.sort(key=lambda r: (r["slot"], r["session"], r["dialogue"]))
    register(path, {"inputs": inputs, "coverage": coverage, "jobs": jobs})
    return path


def population(base, study, corpus):
    original = register_population(base, study, corpus)
    if os.environ.get("MEMSKILL_EXCHANGE_FORMAT", "strict") == "strict":
        return original
    if os.environ["MEMSKILL_EXCHANGE_FORMAT"] != "framed":
        raise ValueError("Unknown teacher format")
    dest = original.parent / "framed"
    dest.mkdir(exist_ok=True)
    body = read(original)
    path = dest / "registration.json"
    identity = {
        **body["inputs"],
        "strict": file_hash(original),
        "framing": file_hash(PLAN.with_name("exchange_framing.json")),
    }
    if path.exists():
        if read(path)["inputs"] != identity:
            raise ValueError("Different framing study")
    else:
        register(path, {**body, "inputs": identity})
    return path


def call(out, prompt, kind, limit, *, reuse_from=None):
    if (
        os.environ.get("MEMSKILL_BACKEND_ID") != BACKEND
        or os.environ.get("MEMSKILL_API_BASE") != "http://127.0.0.1:18773/v1"
    ):
        raise ValueError("Unexpected auxiliary reader/teacher backend")
    cache = (out.parent if out.name == "framed" else out) / "responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    params = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": limit,
    }
    _, key = request_cache(os.environ["MEMSKILL_API_BASE"], params, kind)
    v = read_response(cache, key)
    if not v and reuse_from is not None:
        previous = read_response(reuse_from, key)
        if previous:
            payload = previous["payload"]
            if payload["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
                raise ValueError("Different archived GPU prompt")
            publish_response(
                cache,
                key,
                payload["prompt_sha256"],
                payload["response"],
                *payload["tokens"],
            )
            v = read_response(cache, key)
            if v["sha256"] != previous["sha256"]:
                raise ValueError("Different response for the same GPU request")
    if not v:
        subprocess.run(
            [
                "bash",
                "scripts/experiments/run_witness_four_paths_251.sh",
                "check-writer",
            ],
            check=True,
        )
        get_llm_response_via_api(
            prompt,
            LLM_MODEL=MODEL,
            base_url=os.environ["MEMSKILL_API_BASE"],
            api_key="local-only",
            MAX_TOKENS=limit,
            TAU=0.0,
            TOP_P=1.0,
            SEED=42,
            usage_kind=kind,
        )
        v = read_response(cache, key)
    p = v["payload"]
    if p["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("Changed physical prompt")
    return {
        "text": p["response"],
        "tokens": p["tokens"],
        "key": key,
        "sha256": v["sha256"],
    }


def quote(obs, speaker, text):
    return f"{obs['date']} | {speaker}: {text}"


def screen(out, job, ev):
    raw = call(
        out,
        teacher_prompt(
            job["asker"], job["source_question"], job["responder"], job["reply"]
        ),
        "exchange-teacher",
        96,
    )
    qa, status = parse(raw["text"], job, framed=out.name == "framed")
    result = {"status": status, "teacher": raw, "qa": qa}
    if not qa:
        return result
    decoy = next(
        (r for r in job["other_replies"] if not retain_phrase(r["text"], qa["answer"])),
        None,
    )
    if decoy is None:
        return dict(result, status="no_matched_decoy")
    result["decoy_id"] = decoy["id"]

    def ask(text):
        context = [
            quote(job, job["asker"], job["ask_text"]),
            quote(job, job["responder"], text),
        ]
        return call(out, ev.build_prompt(qa["question"], context, qa), "reader", 32)

    real = ask(job["reply"])
    result["source"] = real
    if not reference_match(real["text"].strip(), qa["answer"]):
        return dict(result, status="source_not_recovered")
    sham = ask(decoy["text"])
    result["decoy"] = sham
    return dict(
        result,
        status="decoy_recovered"
        if retain_phrase(sham["text"], qa["answer"])
        else "accepted",
    )


def verify_question(reg, job, packet):
    if packet["population"] != file_hash(reg):
        raise ValueError("Unexpected exchange population")
    identity = packet.get("job_sha256", packet.get("source"))
    if isinstance(identity, str):
        if identity != digest(job):
            raise ValueError("Wrong source coordinates")
        return
    # Earlier screening receipts used "source" for the real reader response.
    # Verify both physical prompts against the content-addressed response archive.
    out = reg.parent
    cache = (out.parent if out.name == "framed" else out) / "responses"
    ev = LoCoMoEvaluator(SimpleNamespace())
    texts = [
        quote(job, job["asker"], job["ask_text"]),
        quote(job, job["responder"], job["reply"]),
    ]
    checks = [
        (
            "teacher",
            teacher_prompt(
                job["asker"], job["source_question"], job["responder"], job["reply"]
            ),
        ),
        ("source", ev.build_prompt(packet["qa"]["question"], texts, packet["qa"])),
    ]
    if "decoy" in packet:
        alt = next(
            v["text"] for v in job["other_replies"] if v["id"] == packet["decoy_id"]
        )
        checks.append(
            (
                "decoy",
                ev.build_prompt(
                    packet["qa"]["question"],
                    [texts[0], quote(job, job["responder"], alt)],
                    packet["qa"],
                ),
            )
        )
    for key, prompt in checks:
        r = packet[key]
        saved = read_response(cache, r["key"])
        if (
            not saved
            or saved["sha256"] != r["sha256"]
            or saved["payload"]["response"] != r["text"]
            or saved["payload"]["prompt_sha256"]
            != hashlib.sha256(prompt.encode()).hexdigest()
        ):
            raise ValueError("Different source or physical response")


def generate(reg, limit):
    out = reg.parent
    dest = out / "questions"
    dest.mkdir(exist_ok=True)
    ev = LoCoMoEvaluator(SimpleNamespace())
    done = 0
    for job in read(reg)["jobs"]:
        f = dest / (job["id"] + ".json")
        if f.exists():
            continue
        strict = {}
        if out.name == "framed":
            original = out.parent / "questions" / f.name
            if not original.exists():
                break
            v0 = read(original)
            strict = {"strict": file_hash(original), "strict_status": v0["status"]}
        v = screen(out, job, ev)
        if strict and v["teacher"] != v0["teacher"]:
            raise ValueError("Framing sensitivity comparison changed teacher response")
        register(
            f, {"population": file_hash(reg), "job_sha256": digest(job), **strict, **v}
        )
        print(job["id"], v["status"], flush=True)
        done += 1
        if done == limit:
            break
    if len(list(dest.glob("*.json"))) == len(read(reg)["jobs"]):
        register(
            out / "questions-complete.json",
            {
                "registration": file_hash(reg),
                "counts": dict(Counter(read(f)["status"] for f in dest.glob("*.json"))),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("--generate", type=int, default=0)
    a = p.parse_args()
    reg = population(a.base, a.study, a.corpus)
    if a.generate:
        generate(reg, a.generate)
    else:
        print(reg)
