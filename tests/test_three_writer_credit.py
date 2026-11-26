import unittest

import torch

from experiments.controller.late_semantic_writer import summarize
from experiments.controller.three_writer_credit import aggregate, three_probabilities


class ThreeWriterTests(unittest.TestCase):
    def test_three_arms_reproduce_both_input_conditional_distributions(self):
        activity = torch.tensor([0.1, 0.75], dtype=torch.float64)
        relation = torch.tensor([0.8, 0.01], dtype=torch.float64)
        temporal, act, rel = three_probabilities(activity, relation).T
        torch.testing.assert_close(temporal + act + rel, torch.ones_like(temporal))
        torch.testing.assert_close(act / (temporal + act), activity)
        torch.testing.assert_close(rel / (temporal + rel), relation)
        with self.assertRaises(ValueError):
            three_probabilities(activity, relation.new_tensor([1.0, float("nan")]))

    def test_matched_usage_is_per_span_but_f1_is_per_question(self):
        reports = []
        for name, n, weights, f1 in (
            ("a", 3, [0.25, 0.25, 0.5], [0, 0, 1]),
            ("b", 1, [0.5, 0.25, 0.25], [1, 0, 0]),
        ):
            reports.append(
                {
                    "matched": True,
                    "questions": n,
                    "dialogue": name,
                    "probabilities": {"native": [1, 0, 0], "credit_tail": weights},
                    "rows": [{"f1": f1, "category": 2, "linkage": "first_only"}] * n,
                }
            )
        reports.append({"matched": False, "questions": 2})
        report = aggregate(reports)
        self.assertEqual(report["matched_rate"], [0.375, 0.25, 0.375])
        self.assertEqual(report["mean_total_variation"], 0.625)
        self.assertEqual(report["unmatched_questions"], 2)
        comparison = report["comparisons"]["rate"]
        self.assertEqual(comparison["difference"], 0.125)
        self.assertEqual(comparison["interval"], [0.0, 0.125])
        scores = report["groups"]["all"]["f1"]
        self.assertEqual(
            (scores["native"], scores["credit_tail"], scores["rate"]),
            (0.25, 0.5, 0.375),
        )

    def test_semantic_denominator_keeps_missing_and_pending_questions(self):
        panel = [
            {"background": 0, "questions": [0, 1]},
            {"background": 1, "questions": [2]},
            {"background": 0, "questions": [3, 4, 5]},
        ]
        reports = [
            {
                "matched": True,
                "dialogue": "d",
                "choice": {"native": 0.05},
                "source_forecast": {"whole": 0.75, "tail": 0.25},
                "rows": [
                    {"temporal": {"f1": 1.0}, "relation": {"f1": 0.0}, "category": 1}
                ]
                * 2,
            },
            {"matched": False, "reason": "unreproduced_prefix_read"},
        ]
        stats = summarize(panel, reports)
        self.assertEqual(stats["registered_questions"], 6)
        self.assertEqual(stats["pending_questions"], 3)
        self.assertEqual(stats["unmatched_questions"], {"unreproduced_prefix_read": 1})
        self.assertEqual(stats["groups"]["all"]["questions"], 2)
        self.assertEqual(stats["groups"]["prefix_0"]["f1"]["tail_rate"], 0.75)
