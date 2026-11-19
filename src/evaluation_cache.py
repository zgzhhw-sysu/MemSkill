"""Reuse measured model outputs during deterministic paired evaluation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


_embedding_batches = ContextVar("paired_history_embedding_batches", default=None)


def _encode(model, texts, batch_size, normalize):
    options = {"normalize_embeddings": True} if normalize else {}
    return model.encode(texts, convert_to_numpy=True, batch_size=batch_size, **options)


@contextmanager
def history_embeddings():
    """Reuse exact ordered batches within one history and frozen encoder instance."""
    @lru_cache(maxsize=384)
    def measured(model, texts, batch_size, normalize):
        return _encode(model, list(texts), batch_size, normalize)

    token = _embedding_batches.set(measured)
    try:
        yield measured.cache_info
    finally:
        _embedding_batches.reset(token)
        measured.cache_clear()


def encode_batch(model, texts, batch_size, *, normalize=False):
    measured = _embedding_batches.get()
    if measured is None or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return _encode(model, texts, batch_size, normalize)
    # Callers may keep or mutate their own vectors when assembling the memory bank.
    return measured(model, tuple(texts), batch_size, normalize).copy()


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
