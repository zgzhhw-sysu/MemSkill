import unittest
from types import SimpleNamespace

import torch

from experiments.controller.late_credit_kernel import kernel_effect, support
from experiments.controller.late_selection_labels import labels


class LateCreditKernelTests(unittest.TestCase):
    def test_absent_similarity_gives_zero_and_matching_evidence_is_shrunk(self):
        x = torch.eye(2, dtype=torch.float64)
        bank = x[:1]
        result = kernel_effect(
            x, bank, x.new_tensor([2.0]), x.new_tensor([[0.6, -0.3]])
        )
        torch.testing.assert_close(result, x.new_tensor([[0.4, -0.2], [0.0, 0.0]]))

    def test_excludes_entire_target_dialogue_and_matches_equal_donor_history_weight(
        self,
    ):
        def row(sid, stage, weight, value):
            return {
                "dialogue": sid,
                "stage": stage,
                "weight": weight,
                "base": 0.0,
                "state": [1.0, 0.0],
                "raw": value,
                "credit": 0.0,
            }

        rows = [
            row("a", 0, 0.5, 0.2),
            row("a", 0, 0.5, 0.2),
            row("b", 0, 1.0, 0.4),
            row("b", 0, 1.0, 0.4),
            row("target", 0, 1.0, 100.0),
            row("a", 1, 1.0, 100.0),
        ]
        bank, shift = support(rows, "target", 0, "cpu")
        states, weights, _ = bank
        self.assertEqual(len(states), 4)
        torch.testing.assert_close(weights[:2].sum(), weights[2:].sum())
        self.assertAlmostEqual(float(weights.sum()), 3.0)
        self.assertAlmostEqual(float(kernel_effect(states[:1], *bank)[0, 0]), 0.225)
        self.assertAlmostEqual(shift, 2.0, places=10)

    def test_archive_only_pass_stops_before_required_model_probes(self):
        trainer = SimpleNamespace(
            evaluator=SimpleNamespace(get_ground_truth=lambda q: q["answer"])
        )
        q = {
            "question": "Which color?",
            "answer": "red",
            "original_index": 3,
            "category": 4,
            "evidence": ["D9:1"],
        }
        event = {
            "selected_skills": [{}, {}],
            "refs": [],
            "retrieved": [],
            "source": "text",
            "source_turns": ["D9:1"],
        }

        def cell(answer, score):
            return {
                "spans": [event, event],
                "scores": {0: {"answer": answer, "f1": score, "ok": True}},
            }

        cells = {(0, 0): cell("no idea", 0.0), (1, 0): cell("red", 1.0)}
        raw = {
            "questions": [
                {"original_index": 3, "type": 4, "temporal": 0.0, "activity": 1.0}
            ]
        }
        cfg = {"pivot": 0, "background": 0}
        self.assertIsNone(
            labels(
                trainer, {"qa": [q]}, cfg, cells, raw, None, None, receipts_only=True
            )
        )
        unrelated = dict(q, evidence=["D10:2"])
        result = labels(
            trainer,
            {"qa": [unrelated]},
            cfg,
            cells,
            raw,
            None,
            None,
            receipts_only=True,
        )
        self.assertEqual((result[0]["status"], result[0]["credit"]), ("unlinked", 0.0))
