import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from experiments.controller.capped_source_policy import (
    probabilities,
    surviving_credit,
    targets,
)
from experiments.controller.capped_source_repair import suffix_provider, writer
from experiments.controller.late_credit_kernel import (
    probabilities as original_probabilities,
)
from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash, register
from src.evaluation_cache import publish_response
from src.executor import Executor

SOURCE = 'DATE: 1:00 pm on 9 March, 2023\nCONVERSATION:\nAnn said, "I walked home."\nLee said, "I slept."'
PREFIX = "ACTION: INSERT\nMEMORY_ITEM: Ann walked home.\n"
REPLY = PREFIX + "ACTION: INSERT"


class CappedSourceTests(unittest.TestCase):
    def execute(self, text, *, tokens=32, rule="dated_single_turns", repair=True):
        executor = Executor(
            SimpleNamespace(
                model="local",
                retriever="unused",
                max_new_tokens=32,
                temperature=0,
                api_base="local",
                api_key="unused",
            )
        )
        ops = [
            SimpleNamespace(
                name="facts",
                description="facts",
                update_type="insert",
                instruction_template="retain facts",
                source_rule=rule,
            )
        ]
        calls = []
        actions = executor.execute_operation(
            ops,
            SOURCE,
            [],
            audit=calls,
            repair_capped_source=repair,
            response_provider=lambda **_: (text, 27, tokens),
        )
        event = {"source": SOURCE, "retrieved": [], "executor_responses": calls}
        self.assertEqual(
            [vars(a) for a in actions], writer(executor, ops, event, repair=repair)
        )
        return [vars(a) for a in actions], calls

    def test_fixed_skill_replay_retains_generation_and_restores_source_records(self):
        for rule, extra in [("dated_single_turns", 2), ("dated_adjacent_turns", 1)]:
            with self.subTest(rule=rule):
                old, raw = self.execute(REPLY, rule=rule, repair=False)
                new, replay = self.execute(REPLY, rule=rule)
                self.assertEqual(raw, replay)
                self.assertEqual(old[0], new[0])
                self.assertFalse(old[-1]["success"])
                self.assertEqual([a["reasoning"] for a in new[1:]], [rule] * extra)

    def test_repair_requires_full_cap_valid_prefix_final_empty_block_and_source_rule(
        self,
    ):
        for text, tokens, rule in [
            (REPLY, 31, "dated_single_turns"),
            (REPLY, 32, None),
            (REPLY + "\nMEMORY_ITEM:", 32, "dated_adjacent_turns"),
            ("ACTION: INSERT\n" + REPLY, 32, "dated_single_turns"),
            ("ACTION: INSERT", 32, "dated_single_turns"),
        ]:
            with self.subTest(text=text, tokens=tokens, rule=rule):
                self.assertEqual(
                    self.execute(text, tokens=tokens, rule=rule),
                    self.execute(text, tokens=tokens, rule=rule, repair=False),
                )

    def test_changed_suffix_requires_its_own_measured_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dialogue" / "11"
            dest.mkdir(parents=True)
            request = {
                "prompt": "corrected memory",
                "cap": 32,
                "prefix": "updated first write",
            }
            self.assertIsNone(suffix_provider(dest, request))
            key = "a" * 64
            prompt = sha(request["prompt"])
            receipt = publish_response(
                root / "responses", key, prompt, "ACTION: NOOP", 18, 3
            )
            reply = {
                "request": file_hash(dest / "suffix-request.json"),
                "call": {
                    "prompt_sha256": prompt,
                    "response": "ACTION: NOOP",
                    "input_tokens": 18,
                    "output_tokens": 3,
                },
            }
            register(dest / "suffix-response.json", reply)
            usage = {
                "kind": "executor",
                "input": 18,
                "output": 3,
                "cache_key": key,
                "response_sha256": receipt["sha256"],
            }
            log = dest / "usage.jsonl"
            log.write_text(json.dumps(dict(usage, output=4)))
            with self.assertRaises(ValueError):
                suffix_provider(dest, request)
            log.write_text(json.dumps(usage))
            provider = suffix_provider(dest, request)
            self.assertEqual(
                provider(prompt=request["prompt"], MAX_TOKENS=32),
                ("ACTION: NOOP", 18, 3),
            )
            with self.assertRaises(ValueError):
                suffix_provider(dest, dict(request, prefix="different first write"))

    def test_no_extra_credit_after_weaker_answer_recovered_or_direction_flipped(self):
        good = {"answer": "A cozy restaurant.", "f1": 1.0, "ok": True}
        weak = {"answer": "Greenhouse.", "f1": 0.0, "ok": True}
        partial = dict(good, f1=0.75)
        truth = "cozy restaurant"
        self.assertEqual(surviving_credit(0.9, weak, partial, truth), 0.75)
        self.assertEqual(surviving_credit(-0.8, good, weak, truth), -0.8)
        self.assertEqual(surviving_credit(1.0, partial, good, truth), 0.0)
        self.assertEqual(surviving_credit(1.0, good, weak, truth), 0.0)
        self.assertEqual(surviving_credit(0.0, weak, good, truth), 0.0)

    def test_mask_preserves_sample_weight_and_other_targets(self):
        w = {
            "cells": {
                "00": {"triggers": [None, {}]},
                "10": {"triggers": [None, {"restored_items": 2}]},
                "01": {"triggers": [None, None]},
                "11": {"triggers": [None, None]},
            }
        }
        row = {"raw": 0.3, "credit": 0.2, "weight": 0.5}
        actual = targets(
            None, None, [row, row], [(w, ["00", "10"], []), (w, ["01", "11"], [])], {}
        )
        self.assertEqual(actual, [dict(row, raw=0.0, credit=0.0), row])

    def test_uncorrected_probabilities_match_existing_kernel(self):
        data = tuple(
            torch.tensor(x, dtype=torch.float64)
            for x in ([[1, 0], [0, 1]], [1, 1], [[0.2, 0.1], [-0.05, -0.03]])
        )
        d = {"state": [0.8, 0.2], "base_logit": -0.2}
        w = [0.01, 0.1, -0.1]
        actual = probabilities(d, [(data, 0.1)] * 3, w)
        old = original_probabilities(d, (data, data, 0.1), w)
        for a, b in [
            ("uncorrected", "credit"),
            ("uncorrected_raw", "raw"),
            ("rate", "rate"),
            ("native", "native"),
            ("terminal", "terminal"),
        ]:
            self.assertAlmostEqual(actual[a], old[b])
