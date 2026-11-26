import unittest

import torch

from experiments.controller.semantic_writer import source_offsets, summarize
from experiments.controller.semantic_writer_balanced import panels
from experiments.controller.terminal_capacity import optimal_rates


class SemanticWriterTests(unittest.TestCase):
    def test_source_forecast_uses_only_earlier_quantiles_and_caps_the_shift(self):
        x = torch.arange(5, dtype=torch.float64)
        target = x.new_tensor([-20, 1, 2, 3, 20])
        self.assertEqual(
            source_offsets(x, target).tolist(), [-2.0, -1.0, 0.0, 1.0, 2.0]
        )
        self.assertFalse(source_offsets(torch.ones_like(x), target).any())

    def test_question_counts_weight_returns_and_expected_visits_weight_usage(self):
        native = torch.tensor([0.5, 0.5], dtype=torch.float64)
        reach = native.new_tensor([0.75, 0.25])
        # The second span has a lower per-question effect but higher total marginal return per visit.
        p = optimal_rates(
            reach,
            native,
            native.new_tensor([0.2, 0.1]),
            native.new_tensor([1, 3]),
            None,
        )
        torch.testing.assert_close(p, native.new_tensor([1 / 3, 1.0]))
        self.assertAlmostEqual(float(reach @ p), float(reach @ native))
        bounded = optimal_rates(
            reach, native, native.new_tensor([0.2, 0.1]), native.new_tensor([1, 3]), 2.0
        )
        self.assertLess(float(bounded[1]), 1.0)
        self.assertAlmostEqual(float(reach @ bounded), 0.5)

    def test_equal_usage_per_window_including_a_duplicate_question_count(self):
        def window(name, p, low, high, n):
            return {
                "matched": True,
                "dialogue": name,
                "choice": {"native": 0.5, "proposed": p},
                "rows": [
                    {"temporal": {"f1": low}, "relation": {"f1": high}}
                    for _ in range(n)
                ],
            }

        a = window("a", 0.25, 1.0, 0.0, 3)
        b = window("b", 0.75, 0.0, 1.0, 1)
        s = summarize([a, b, {"matched": False}])
        self.assertEqual((s["windows"], s["matched"], s["questions"]), (3, 2, 4))
        self.assertEqual(s["matched_rate"], 0.5)
        self.assertEqual(s["f1"]["proposed"], 0.75)
        self.assertEqual(s["f1"]["rate"], 0.5)

    def test_balanced_prior_preserves_the_checkpoint_control_and_uses_one_rate_per_window(
        self,
    ):
        item = {
            "matched": True,
            "dialogue": "d",
            "choice": {"native": 0.005, "proposed": 0.01},
            "rows": [{"temporal": {"f1": 0.0}, "relation": {"f1": 1.0}}],
        }
        summary = panels([("w", item)], {"w": 0.75})
        self.assertEqual(summary["balanced"]["f1"]["uniform"], 0.5)
        self.assertEqual(summary["balanced"]["f1"]["source_only"], 0.75)
        self.assertEqual(summary["balanced"]["f1"]["rate"], 0.75)
        self.assertEqual(summary["checkpoint"]["f1"]["native"], 0.005)
        self.assertEqual(item["choice"]["proposed"], 0.01)
