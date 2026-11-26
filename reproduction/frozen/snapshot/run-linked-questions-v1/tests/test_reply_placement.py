import unittest

import torch

from experiments.controller.reply_placement import development_gate
from experiments.controller.reply_probes import (
    candidate_text,
    excess,
    exchanges,
    ranked_turns,
)
from experiments.controller.writer_block_policy import trajectory_table


class ReplyPlacementTests(unittest.TestCase):
    def test_weak_targets_and_role_matched_sham(self):
        turns = [
            {"speaker": "A", "text": "Old question? Fine. What changed?"},
            {"speaker": "B", "text": "A new job."},
            {"speaker": "A", "text": "Great."},
            {"speaker": "B", "text": "At school."},
        ]
        self.assertEqual(exchanges(turns), [("What changed?", 0, 1, 3, [1, 3])])
        self.assertEqual(exchanges(turns[:2]), [])
        turns[1]["speaker"] = "A"
        self.assertEqual(exchanges(turns), [])

    def test_redaction_uses_remaining_fragment_including_across_chunk_boundary(self):
        records = [
            ("ask", [(0, "ask")]),
            ("tail+reply", [(0, "tail"), (1, "reply")]),
            ("foil", [(2, "foil")]),
            ("reply2", [(1, "reply2")]),
        ]
        self.assertIsNone(candidate_text(*records[0], 0))
        self.assertEqual(candidate_text(*records[1], 0), "reply")
        self.assertEqual(candidate_text(*records[1], 2), "tail+reply")
        numbers = {
            "q": 1.0,
            "ask": 0.6,
            "tail+reply": 0.9,
            "reply": 0.1,
            "foil": 0.8,
            "reply2": 0.05,
        }
        vectors = {k: torch.tensor([x]) for k, x in numbers.items()}
        intact = ranked_turns(records, vectors, "q", None)
        redacted = ranked_turns(records, vectors, "q", 0)
        self.assertEqual(intact[1], 1.0)
        self.assertEqual(redacted, {2: 1.0, 1: 0.5})
        self.assertAlmostEqual(excess(redacted, 1, [1, 2, 3]), 0.0)

    def test_temporal_gate_requires_fresh_results_and_placement_benefit(self):
        controls = ("native", "terminal", "rate", "raw", "sham", "robust")
        block = {
            "questions": 350,
            "by_dialogue": dict.fromkeys("abcd"),
            "f1": dict.fromkeys(controls, 0.4) | {"reply": 0.43},
            "ht": {"reply": 0.42, "native": 0.41},
        }
        self.assertFalse(
            development_gate(block, {"questions": 0}, 378)[
                "eligible_for_independent_development"
            ]
        )
        self.assertTrue(
            development_gate(block, {"questions": 3}, 378)[
                "eligible_for_independent_development"
            ]
        )
        block["f1"]["rate"] = 0.43
        self.assertFalse(
            development_gate(block, {"questions": 3}, 378)[
                "eligible_for_independent_development"
            ]
        )

    def test_randomized_trajectory_reproduces_focal_and_joint_scores(self):
        probabilities = {
            0: {"native": 0.4, "reply": 0.8},
            1: {"native": 0.5, "reply": 0.2},
            2: {"native": 0.5, "reply": 0.2},
        }
        cases = {"01": [0.25], "11": [1.0]}
        costs = {bits: {"reader": 5, "executor": 3} for bits in cases}
        base = {
            "dialogue": "a",
            "window": "a_s9",
            "rows": [
                {"index": 3, "type": 2, "f1": {"native": 0.55}, "ht": {"native": 0.55}}
            ],
        }
        report = trajectory_table(
            base,
            probabilities,
            cases,
            costs,
            ("native", "reply"),
            ("native",),
            pivot=0,
            background=1,
        )
        self.assertAlmostEqual(report["rows"][0]["f1"]["reply"], 0.85)
        self.assertAlmostEqual(report["rows"][0]["ht"]["reply"], 0.34)
        self.assertAlmostEqual(report["expected_tokens"]["reply"]["reader"], 2.0)
        with self.assertRaises(ValueError):
            trajectory_table(
                base,
                probabilities,
                {"01": [0.0], "11": [1.0]},
                costs,
                ("native", "reply"),
                ("native",),
                pivot=0,
                background=1,
            )
