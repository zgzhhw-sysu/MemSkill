import math
import unittest
from itertools import product

import torch

from experiments.controller.audit_prefix_return import BENCHMARKS, exact_report, verdict
from experiments.controller.late_pair_policy import STEPS
from experiments.controller.prefix_return import (
    attribute,
    budget_envelope,
    fit_strengths,
    masses,
    predicted_offsets,
)
from experiments.controller.prefix_history import measure, summarize
from experiments.controller.rank_return import fit_rules, rank_offsets


def direction():
    return dict(
        center=[0.0], scale=[1.0], slope=[1.0], shifts={str(s): 0.0 for s in STEPS}
    )


class PrefixReturnTests(unittest.TestCase):
    def test_oracle_allocation_obeys_probability_limits_and_shared_questions(self):
        # First state already uses the better skill 95% of the time. The remaining budget goes to state 2.
        a = budget_envelope([(1, 0.95, 0.75, 1.0), (1, 0.2, 0.2, 0.5)], target=0.04)
        self.assertAlmostEqual(a["actual_gain"], -0.1)
        self.assertAlmostEqual(a["same_magnitudes_best_sign"], 0.025)
        self.assertAlmostEqual(a["same_budget_reallocated"], 0.025 + 0.075 * 0.5)
        self.assertAlmostEqual(a["minimum_budget_for_target"], 0.025 + 0.015 / 0.5)
        b = budget_envelope([(2, 1.0, 1.0, 0.2)], target=0.02)
        self.assertIsNone(b["minimum_budget_for_target"])
        self.assertEqual(b["unconstrained_oracle_gain"], 0.0)

    def test_rank_rule_separates_sources_a_prefix_constant_cannot(self):
        windows = [
            dict(
                dialogue=str(i), n=5, features=[[x], [0.0]], base=[0.0, 0.0, 0.0], f1=y
            )
            for i, (x, y) in enumerate(
                [(-1.0, [1.0, 1.0, 0.0, 0.0]), (1.0, [0.0, 0.0, 1.0, 1.0])]
            )
        ]
        fit = fit_rules(windows, direction())
        self.assertGreaterEqual(fit["rules"][0], 3)
        self.assertEqual(fit["rules"][1:], [0, 0])
        self.assertAlmostEqual(fit["fit_f1"]["context"], 0.5)
        self.assertAlmostEqual(fit["fit_f1"]["joint"], 1 / (1 + math.exp(-2)))
        self.assertAlmostEqual(fit["rate_shifts"][0], 0.0)
        strength = fit_strengths(windows, direction())
        model = dict(direction=direction(), gate=fit, strength=strength)
        self.assertEqual(rank_offsets(model, [-1.0])["prefix"][0], -2.0)
        self.assertEqual(rank_offsets(model, [1.0])["prefix"][0], 2.0)
        at = fit["cuts"][0][fit["rules"][0] - 3]
        self.assertEqual(rank_offsets(model, [at])["prefix"][0], 2.0)

    def test_constant_returns_make_threshold_fit_identical_to_native(self):
        w = dict(
            dialogue="a",
            n=2,
            features=[[1.0], [0.0]],
            base=[-0.5, 0.5, -0.1],
            f1=[0.4] * 4,
        )
        fit = fit_rules([w], direction())
        self.assertEqual(fit["rules"], [0] * 3)
        self.assertEqual(fit["context_rules"], [0] * 3)

    def test_main_decision_requires_matched_control_and_prospective_coverage(self):
        names = {
            *BENCHMARKS,
            "prefix",
            "prefix_strength",
            "prefix_context",
            "calibrated",
            "calibrated_greedy",
            "source",
            "permuted",
            "unit",
        }
        scores = dict.fromkeys(names, 0.40)
        scores["prefix"] = 0.44
        scores["prefix_context"] = 0.44
        reports = [
            dict(
                dialogue=d,
                pivot=0,
                sampling_mass=dict.fromkeys(names, 1.0),
                focal_probabilities=dict(native=0.4, prefix=0.6),
                questions=[
                    dict(type=2, temporal=0.32, activity=0.52, focal=scores, ht=scores)
                ],
            )
            for d in "ABCDEF"
        ]
        self.assertFalse(verdict(reports, 6, True)["eligible_for_development"])
        scores["prefix_context"] = 0.41
        self.assertFalse(verdict(reports, 7, True)["eligible_for_development"])
        self.assertFalse(verdict(reports, 6, False)["eligible_for_development"])
        self.assertTrue(verdict(reports, 6, True)["eligible_for_development"])

    def test_advantage_attribution_with_a_different_second_state(self):
        x = torch.tensor(
            [[0.1, 0.9, 0.7, 0.3], [0.7, 0.1, 0.1, 0.9]], dtype=torch.float64
        )
        base = x.new_tensor([0.7, 0.1, 0.8])
        new = x.new_tensor([0.2, 0.8, 0.1])
        parts = attribute(x, base, new)
        torch.testing.assert_close(
            parts.sum(-1), (x * (masses(new) - masses(base))).sum(-1)
        )
        self.assertAlmostEqual(float(parts[0, 2]), 0.2 * (0.1 - 0.8) * (0.3 - 0.7))

    def test_full_return_learns_opposite_continuations_without_changing_input_features(
        self,
    ):
        w = dict(
            dialogue="a",
            n=3,
            base=[0.0, 0.0, 0.0],
            features=[[0.0], [1.0]],
            f1=[0.0, 1.0, 1.0, 0.0],
        )
        fit = fit_strengths([w, dict(w, dialogue="b", n=7)], direction())
        self.assertEqual(fit["steps"], [0.0, 2.0, 0.0])
        self.assertEqual(fit["shared"], 0.0)
        self.assertGreater(fit["conditional_fit_f1"], fit["shared_fit_f1"])
        for i in range(3):
            self.assertAlmostEqual(fit["rate_shifts"][i], fit["steps"][i])
        self.assertAlmostEqual(
            fit["conditional_fit_kl"],
            0.5 * (math.log(2 / (1 + math.exp(2))) + 2 / (1 + math.exp(-2))),
        )

    def test_tie_prefers_no_change_and_histories_have_equal_weight(self):
        a = dict(
            dialogue="a",
            n=300,
            base=[-1.0, -1.0, -1.0],
            features=[[1.0], [2.0]],
            f1=[0.75] * 4,
        )
        b = dict(a, dialogue="b", n=1, f1=[0.25] * 4)
        fit = fit_strengths([a, b], direction())
        self.assertEqual(fit["steps"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(fit["native_fit_f1"], 0.5)
        self.assertAlmostEqual(fit["conditional_fit_kl"], 0.0)
        with self.assertRaises(ValueError):
            fit_strengths([dict(a, f1=[2.0] * 4)], direction())

    def test_exact_and_randomized_paths_use_only_previous_action(self):
        get_dec = lambda x: dict(base_logit=x, state=[1.0], plain=[0], grounded=[1])
        cells = {
            (a, b): (
                [get_dec(-0.5), get_dec(a - 0.2)],
                {"first": a},
                {0: 0.1 + 0.2 * b, 1: 0.8 - 0.3 * a * b},
            )
            for a, b in product(range(2), repeat=2)
        }
        model = dict(
            direction=direction(),
            gate=dict(steps=[0.25, 1.0, 2.0], shared=1.0, rate_shifts=[0.2, 0.1, -0.2]),
        )
        preds = [
            dict(
                residual=0.0,
                unit_residual=0.2,
                source=0.6,
                component=0.2,
                prefix_offsets=predicted_offsets(model, [x]),
            )
            for x in [-0.4, 0.9]
        ]
        r = exact_report(
            dict(dialogue="x", start=5, questions=[23, 47], types=[2, 4]),
            cells,
            preds,
            [0.0, 0.0],
        )
        for q in r["questions"]:
            self.assertAlmostEqual(
                sum(q["contributions"]), q["returns"]["prefix"] - q["returns"]["native"]
            )
        self.assertGreater(r["states"]["prefix"][2], r["states"]["prefix"][1])
        self.assertFalse(verdict([], 466, False)["eligible_for_development"])

    def test_historical_focal_probabilities_use_executed_previous_skill(self):
        fit = dict(
            direction=direction(),
            gate=dict(
                steps=[0.0, 0.0, 2.0],
                shared=0.0,
                rate_shifts=[0.0, 0.0, 0.0],
                dialogues=["in_fit"],
            ),
        )
        rows = [
            dict(
                dialogue=d,
                session=s,
                base_logit=0.0,
                features=[1.0],
                state=[1.0],
                use=(d == "act"),
                scores=[dict(original_index=42, temporal=0.0, activity=1.0)],
            )
            for s in (1, 2)
            for d in ("in_fit", "act", "tmp")
        ]
        mix = dict(
            rows=[
                dict(dialogue=r["dialogue"], session=r["session"], offsets=[0.0] * 5)
                for r in rows
            ],
            weights={d: [1.0, 0.0, 0.0, 0.0, 0.0] for d in ("act", "tmp")},
        )
        groups = measure(rows, fit, mix, {"tmp": [0.0, 0.0], "act": [0.0, 0.0]})
        self.assertEqual(set(groups), {"tmp", "act"})
        self.assertAlmostEqual(groups["tmp"][-1]["returns"]["prefix"], 0.5)
        self.assertGreater(groups["act"][-1]["returns"]["prefix"], 0.85)
        self.assertEqual(summarize(groups["act"])["links"], 2)
