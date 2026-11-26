import unittest
from unittest.mock import patch

import torch

from experiments.controller.direct_skill_allocation import direct_distributions
from experiments.controller.earlier_terminal_returns import read_window
from experiments.controller.passage_advantage import decision_probabilities
from experiments.controller.selective_span_bounds import state_ranges
from experiments.controller.selective_span_policy import allocations, fit_split
from experiments.controller.semantic_writer import sha
from experiments.controller.three_skill_margin import (
    aggregate,
    distributions,
    margin_fit,
)


class SelectiveSpanTests(unittest.TestCase):
    def tensor(self, data):
        return torch.tensor(
            data,
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def test_confirmation_requires_both_physical_return_and_independent_credit(self):
        histories = ["a", "b", "c", "d", "e"]
        rows = [{"history": h, "node": h, "raw": 0.0, "credit": 0.0} for h in histories]
        features = self.tensor([[0.0] * 13 for _ in rows])
        p = self.tensor([0.4] * 5)
        raw = self.tensor([1, 1, 1, -1, -1])
        c = self.tensor([-1] * 5)

        def fit(credit):
            return fit_split(
                rows, features, p, raw, credit, credit, {"a", "b", "c"}, {"d", "e"}, 0
            )

        full = fit(c)
        self.assertEqual(full["base"]["action"], 1)
        self.assertTrue(full["offers"]["credit"]["accepted"])
        self.assertGreater(full["offers"]["credit"]["confirmation_raw"], 0)
        absent = fit(self.tensor([-1, -1, -1, 0, 0]))
        self.assertFalse(absent["offers"]["credit"]["accepted"])
        self.assertEqual(
            absent["offers"]["credit"]["discovery_gain"],
            full["offers"]["credit"]["discovery_gain"],
        )
        self.assertFalse(fit(torch.zeros_like(c))["offers"]["credit"]["accepted"])

    def test_opposing_votes_average_distributions_and_preserve_native_when_abstaining(
        self,
    ):
        def model(sign):
            rule = {"region": (-1, 0.0, 1), "action": sign}
            return {
                "base": rule,
                "offers": {
                    k: dict(rule, accepted=False)
                    for k in ("credit", "rotated", "raw_rule")
                },
            }

        target = self.tensor([[0.0] * 13])
        mixed = allocations([model(1), model(-1)], target)[0]
        z = self.tensor(0.1).logit()
        result = decision_probabilities(
            ("native", "credit"), {"native": z}, {"credit": mixed["credit"]}
        )
        self.assertAlmostEqual(result["credit"], 0.35 * 0.1 + 0.65 * 0.5)
        abstain = allocations([model(0)], target)[0]
        self.assertEqual(abstain["credit"], [0.0, 0.0])

    def test_opposing_approved_changes_have_width_even_when_their_mean_cancels(self):
        folds = []
        for sign in (1, -1):
            base = {"region": (-1, 0.0, 1), "action": sign}
            offer = dict(base, action=-sign, accepted=True)
            folds.append({"base": base, "offers": {"credit": offer}})
        recorded = dict.fromkeys(("credit", "base_rule", "unscreened"), 0.5)
        intervals = state_ranges(self.tensor([0.0] * 13), 0.5, folds, recorded)
        for name in ("approved", "discovered", "unrestricted"):
            self.assertAlmostEqual(intervals[name][0], 0.175)
            self.assertAlmostEqual(intervals[name][1], 0.825)

    def test_margin_has_known_scalar_solution_and_does_not_penalize_satisfied_direction(
        self,
    ):
        one = self.tensor([1.0])
        x = one[:, None]
        y = self.tensor([-0.1])
        raw, b = margin_fit(x, y, one, self.tensor([0.5]), one)
        self.assertAlmostEqual(float(raw), -0.1 / 1.02)
        self.assertAlmostEqual(float(b), 0.4 / 2.02)
        for c, w in ((-0.05, 1.0), (0.5, 0.0)):
            _, exact = margin_fit(x, y, one, self.tensor([c]), self.tensor([w]))
            torch.testing.assert_close(exact, raw)

    def test_three_way_probability_and_ablations(self):
        old = self.tensor([0.8, 0.195, 0.005])
        p = distributions(old, {"raw": [0, 0], "credit": [0, 0.05], "rotated": [0, 0]})
        torch.testing.assert_close(p["raw"], 0.2 * old + 0.8 / 3)
        torch.testing.assert_close(p["credit"], p["relation_only"])
        torch.testing.assert_close(p["activity_only"], p["raw"])
        self.assertGreater(float(p["credit"][2]), float(p["raw"][2]))
        self.assertAlmostEqual(float(p["credit"].sum()), 1.0)

    def test_spillover_is_scored_in_action_order_and_changed_native_is_excluded(self):
        qa = {
            "original_index": 7,
            "category": 1,
            "question": "Where?",
            "answer": "Paris",
        }
        old = {"answer": "London", "f1": 0.0}
        q = {
            "qa": qa,
            "linked": False,
            "native": {"prompt": "A", "hash": sha("A"), "old": old},
            "alternative": {"prompt": "B", "hash": sha("B"), "old": None},
        }

        def measured(_, prompt, *args):
            return {"text": "London" if prompt == "A" else "Paris", "tokens": [2, 1]}

        with patch(
            "experiments.controller.earlier_terminal_returns.call", side_effect=measured
        ):
            result = read_window(None, {"questions": [q]}, False)
            self.assertEqual(result["mean_activity_minus_temporal"], 1.0)
            self.assertFalse(result["rows"][0]["linked"])
            self.assertNotIn("credit", result["rows"][0])
            q["native"]["old"] = {"answer": "Bath", "f1": 0.0}
            self.assertEqual(
                read_window(None, {"questions": [q]}, False)["status"],
                "changed_reader_answer",
            )


class DirectComparisonTests(unittest.TestCase):
    def test_temporal_mass_and_exact_reader_value_decomposition(self):
        native = torch.tensor(
            [0.78, 0.21, 0.01],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        scores = {
            "two_edge": {
                "raw": [0.01, 0.03],
                "credit": [-0.02, 0.03],
                "rotated": [0.0, 0.0],
            },
            "direct": {"raw": 0.02, "credit": -0.04, "rotated": 0.01},
        }
        p = direct_distributions(native, scores)
        self.assertEqual(float(p["raw"][0]), float(p["credit"][0]))
        self.assertEqual(float(p["parent_raw"][0]), float(p["credit"][0]))
        measured = native.new_tensor([[1, 0, 1], [0, 0.7, 0.3], [0.4, 0.2, 0.2]])
        torch.testing.assert_close(
            measured @ (p["credit"] - p["raw"]),
            (p["credit"][2] - p["raw"][2]) * (measured[:, 2] - measured[:, 1]),
        )
        self.assertLess(float(p["credit"][2]), float(p["raw"][2]))

    def test_conditional_rate_preserves_both_question_weighted_marginals(self):
        rows = [
            {
                "n": 1,
                "p": {"credit": [0.8, 0.1, 0.1]},
                "means": [1.0, 0.0, 1.0],
                "f1": {"credit": 0.9},
            },
            {
                "n": 3,
                "p": {"credit": [0.2, 0.2, 0.6]},
                "means": [0.0, 1.0, 0.0],
                "f1": {"credit": 0.2},
            },
        ]
        a = aggregate(rows, direct=True)
        non_temporal = 0.2 + 3 * 0.8
        conditional = (0.1 + 3 * 0.6) / non_temporal
        expected = (0.8 + 0.2 * conditional + 3 * 0.8 * (1 - conditional)) / 4
        self.assertAlmostEqual(a["conditional_matched_rate"], expected)
