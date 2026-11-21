import unittest

import torch

from experiments.controller.replicated_outlier_policy import intersect, support
from experiments.controller.shared_outlier_policy import fit_correction


class ReplicatedCreditTests(unittest.TestCase):
    def test_opposite_or_single_dialogue_effects_abstain(self):
        d = torch.tensor([[0.9, -0.7, 1.0], [0.4, -0.2, 0.5], [0.8, -0.4, -0.6]])
        result = intersect(d, [2, 2, 2])
        torch.testing.assert_close(result, torch.tensor([0.4, -0.2, 0.0]))
        torch.testing.assert_close(intersect(d, [1, 1, 2]), torch.zeros(3))
        self.assertEqual(float(intersect(torch.tensor([[0.8], [0.0]]), [2])[0]), 0.0)

    def test_multiple_labels_in_one_dialogue_count_once(self):
        rows = [
            {"history": h, "credit": [[0.0, 0.0], [0.0, v], [r, 0.0]]}
            for h, v, r in (("a", 0.1, 0.2), ("a", -0.4, -0.3), ("b", -0.5, 0.0))
        ]
        self.assertEqual(
            support(rows), {"temporal": [], "activity": ["a", "b"], "relation": ["a"]}
        )

    def test_deleting_the_only_history_with_a_credit_gives_zero_utility(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = [
            {
                "node": h + str(i),
                "history": h,
                "features": [0.2 * i, 0.11 * i],
                "raw": 0.0,
                "credit": [
                    [0.0, 0.0],
                    [0.0, -0.2 if h == "donor" else 0.0],
                    [0.0, 0.0],
                ],
            }
            for h in ("donor", "a", "b")
            for i in (1, 2)
        ]
        model = fit_correction(rows, "donor", device)
        self.assertNotIn("donor", model["histories"])
        self.assertEqual(model["coef"]["components"][1], [0.0, 0.0, 0.0, 0.0])

    def test_sign_only_bound_covers_unknown_returns_and_keeps_window_action_shared(
        self,
    ):
        from experiments.controller.three_action_bounds import signed_direction_bounds

        z = torch.zeros(3, dtype=torch.float64)
        y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=z.dtype)
        lo, hi = signed_direction_bounds(z, torch.tensor([0.0, -1.0, 0.0]), y)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0)
        y[1, 1] = torch.nan
        lo, hi = signed_direction_bounds(z, torch.tensor([0.0, -1.0, 0.0]), y)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.8 / 3)
        lo, hi = signed_direction_bounds(z, torch.tensor([1.0, 0.0, 0.0]), y[:1])
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.8 / 3)
