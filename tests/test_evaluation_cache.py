import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluation_cache import publish_response, read_response, response_path
from src.llm_utils import get_llm_response_via_api


class EvaluationCacheTest(unittest.TestCase):
    def test_frozen_local_requests_reuse_measured_tokens_and_check_identity(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "MEMSKILL_GREEDY_EVAL_CACHE": tmp, "MEMSKILL_BACKEND_ID": "fixed",
            "MEMSKILL_USAGE_LOG": str(Path(tmp) / "usage.jsonl"),
        }), patch("src.llm_utils._get_client_round_robin") as client:
            request = client.return_value.chat.completions.create
            request.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))],
                                                    usage=SimpleNamespace(prompt_tokens=25, completion_tokens=2))
            kwargs = dict(prompt="Q?", LLM_MODEL="test", api_key="local", base_url="http://127.0.0.1/v1", TAU=0)
            actual = get_llm_response_via_api(**kwargs)
            self.assertEqual(get_llm_response_via_api(**kwargs), actual)
            self.assertEqual(request.call_count, 1)
            for changes in ({"MAX_TOKENS": 9}, {"SEED": 3}, {"usage_kind": "executor"}, {"prompt": "Q!"},
                            {"response_format": {"type": "json_object"}}, {"LLM_MODEL": "other"}):
                get_llm_response_via_api(**dict(kwargs, **changes))
            with patch.dict(os.environ, MEMSKILL_BACKEND_ID="different"):
                get_llm_response_via_api(**kwargs)
            logs = [json.loads(s) for s in (Path(tmp) / "usage.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["cache_hit"] for r in logs), 1)
            self.assertEqual(logs[1]["attempts"], 0)
            self.assertEqual(logs[0]["response_sha256"], logs[1]["response_sha256"])
            self.assertEqual(request.call_count, len(logs) - 1)
            for changes in ({"TAU": .4}, {"base_url": "https://example.com"}):
                before = request.call_count
                for _ in range(2):
                    get_llm_response_via_api(**dict(kwargs, **changes))
                self.assertEqual(request.call_count, before + 2)

    def test_competing_results_and_corrupt_or_incomplete_entries_fail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = "d" * 64
            prompt_hash = hashlib.sha256(b"prompt").hexdigest()
            args = (root, key, prompt_hash, "A", 18, 1)
            self.assertIsNone(read_response(root, key))
            self.assertEqual(publish_response(*args), publish_response(*args))
            with self.assertRaises(ValueError):
                publish_response(root, key, prompt_hash, "B", 18, 1)
            row = read_response(root, key)
            row["payload"]["response"] = "B"
            response_path(root, key).write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                read_response(root, key)
            with self.assertRaises(ValueError):
                publish_response(root, "e" * 64, prompt_hash, "A", 0, 1)
