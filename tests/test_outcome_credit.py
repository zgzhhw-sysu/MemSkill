import unittest

import torch

from experiments.controller.outcome_credit import (
    action_advantages,
    estimate,
    rates,
    state_vectors,
)


class OutcomeCreditTests(unittest.TestCase):
    def test_binary_baseline_centers_without_losing_the_signed_contrast(self):
        d = torch.tensor([-0.8, 0.0, 0.4], dtype=torch.float64)
        temporal, activity = action_advantages(0.15, d)
        torch.testing.assert_close(
            0.85 * temporal + 0.15 * activity, torch.zeros_like(d)
        )
        torch.testing.assert_close(activity - temporal, d)

    def test_identical_states_average_by_questions_and_prefix_reach(self):
        x = state_vectors([[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 2.0, 0.0]])
        r = x.new_tensor([0.8, 0.2])
        n = x.new_tensor([1.0, 4.0])
        effects = x.new_tensor([[0.0, 0.1], [1.0, 0.3]])
        torch.testing.assert_close(
            estimate(x, x[:1], r, n, effects), x.new_tensor([[0.5, 0.2]])
        )

    def test_zero_verified_credit_keeps_native_probability_even_if_raw_f1_differs(self):
        data = [
            {
                "state": [1.0, 0.0, 0.0, 1.0],
                "n": 6,
                "reach": 0.5,
                "base": 0.0,
                "signed": 0.0,
                "raw": 0.5,
            }
        ]
        (p,) = rates(data, data)
        self.assertEqual(p["outcome"], p["native"])
        self.assertAlmostEqual(p["rate"], p["native"])
        self.assertEqual(p["constant_update"], p["native"])
        self.assertGreater(p["raw"], p["native"])
