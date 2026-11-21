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
