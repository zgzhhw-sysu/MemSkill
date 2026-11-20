"""Reuse measured model outputs during deterministic paired evaluation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import time
from urllib.parse import urlsplit


_embedding_batches = ContextVar("paired_history_embedding_batches", default=None)
_encoder_lock = Lock()


def _encode(model, texts, batch_size, normalize):
    options = {"normalize_embeddings": True} if normalize else {}
    options.update(convert_to_numpy=True, batch_size=batch_size)
    value = os.environ.get("MEMSKILL_ENCODER_DUTY")
    if value is None:
        return model.encode(texts, **options)
    duty = float(value)
    if not 0 < duty <= 1:
        raise ValueError("MEMSKILL_ENCODER_DUTY must be in (0, 1]")
    if model.device.type != "cuda" or duty == 1 or not len(texts):
        return model.encode(texts, **options)

    # Keep the original encoder's sorting, padding, batch sizes and choices of kernels.
    import torch
    device = model.device
    transformer = next(iter(model.children()))
    started = 0.
    batches = 0

    def begin(module, args):
        nonlocal started
        torch.cuda.synchronize(device)
        started = time.monotonic()

    def rest(module, args, result):
        nonlocal batches
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        time.sleep(elapsed * (1 / duty - 1))
        batches += 1

    with _encoder_lock:
        before = transformer.register_forward_pre_hook(begin)
        after = transformer.register_forward_hook(rest)
        try:
            embeddings = model.encode(texts, **options)
            if not batches:
                raise RuntimeError("No GPU encoder batch reached the configured duty limiter")
            return embeddings
        finally:
            after.remove()
            before.remove()


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
    limit = os.environ.get("MEMSKILL_ENCODER_MAX_BATCH")
    if limit is not None and model.device.type == "cuda":
        limit = int(limit)
        if limit < 1:
            raise ValueError("MEMSKILL_ENCODER_MAX_BATCH must be positive")
        batch_size = min(batch_size, limit)
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
