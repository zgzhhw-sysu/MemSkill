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
