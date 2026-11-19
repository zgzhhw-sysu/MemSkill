"""Content-addressed responses for deterministic, local paired evaluation."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_cache(base_url, params, kind):
    root = os.environ.get("MEMSKILL_GREEDY_EVAL_CACHE")
    backend = os.environ.get("MEMSKILL_BACKEND_ID")
    if not root or not backend or params["temperature"] != 0 or urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    key = digest({"backend": backend, "request": params, "kind": kind, "format": 1})
    return Path(root), key


def response_path(root, key):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid evaluation request digest")
    return root / key[:2] / (key + ".json")


def read_response(root, key):
    path = response_path(root, key)
    try:
        row = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    checksum = row["sha256"]
    payload = row["payload"]
    if (payload["key"] != key or digest(payload) != checksum or min(payload["tokens"]) <= 0
            or len(payload["tokens"]) != 2 or not isinstance(payload["response"], str)):
        raise ValueError(f"Invalid measured response at {path}")
    return row


def publish_response(root, key, prompt_hash, response, input_tokens, output_tokens):
    if not isinstance(response, str) or min(input_tokens, output_tokens) <= 0:
        raise ValueError("Local greedy evaluation requires one measured response")
    payload = {"key": key, "prompt_sha256": prompt_hash, "response": response, "tokens": [input_tokens, output_tokens]}
    row = {"sha256": digest(payload), "payload": payload}
    path = response_path(root, key)
    path.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        file = Path(tmp) / "response.json"
        file.write_text(json.dumps(row, ensure_ascii=False))
        try:
            os.link(file, path)
        except FileExistsError:
            if read_response(root, key) != row:
                raise ValueError("Identical evaluation requests returned different responses or token counts")
    return row
