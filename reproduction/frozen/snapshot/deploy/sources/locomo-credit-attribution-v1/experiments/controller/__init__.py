"""Controller credit assignment studies and fault checks."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def read(path):
    return json.loads(path.read_bytes())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def measured_answer(prompt, model, q, ev, cache, max_tokens=32):
    from src.llm_utils import get_llm_response_via_api

    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    answer, n_in, n_out = get_llm_response_via_api(
        prompt, LLM_MODEL=model, MAX_TOKENS=max_tokens, TAU=0.0, base_url=os.environ["MEMSKILL_API_BASE"],
        api_key=os.environ.get("MEMSKILL_API_KEY", "local-only"))
    return {"answer": answer, "tokens": [n_in, n_out], "f1": ev.compute_f1(answer, ev.get_ground_truth(q), q)}


def register(path, report):
    """Atomically create a receipt. Concurrent repeat runs must agree."""
    with TemporaryDirectory(dir=path.parent) as temp:
        fresh = Path(temp) / path.name
        fresh.write_text(json.dumps(report, indent=2))
        try:
            os.link(fresh, path)
        except FileExistsError:
            if read(path) != read(fresh):
                raise ValueError(f"Different registration already exists at {path}")


def measured_cost(folder, spans, questions):
    usage = [
        json.loads(line) for line in (folder / "usage.jsonl").read_bytes().splitlines()
    ]
    by_kind = {}
    for kind, expected in (("executor", len(spans)), ("reader", questions)):
        rows = [r for r in usage if r["kind"] == kind]
        if len(rows) != expected or any(
            min(r["input"], r["output"]) <= 0 for r in rows
        ):
            raise ValueError("Incomplete model usage for full-prefix rollout")
        by_kind[kind] = {
            "requests": len(rows),
            "cache_hits": sum(r["cache_hit"] for r in rows),
            "input_tokens": sum(r["input"] for r in rows),
            "output_tokens": sum(r["output"] for r in rows),
        }
    return {
        "model": by_kind,
        "implemented_writes": sum(len(e["changes"]) for e in spans),
    }


def repair_competition(native, swapped, current, prior):
    """Signed repeat-stable switch advantage and the amount a native-skill write can recover."""
    if any(len(pair) != 2 or any(not 0 <= x <= 1 for x in pair) for pair in (native, swapped, current, prior)):
        raise ValueError("Expected two complete F1 observations per branch")
    positive = max(0., min(swapped) - max(native))
    negative = max(0., min(native) - max(swapped))
    repaired = min(positive, max(0., min(current) - max(*native, *prior)))
    return positive - negative, positive - negative - repaired
