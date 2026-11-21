import itertools
import unittest

import torch

from experiments.controller.direct_credit_step import allocate as step_allocate
from experiments.controller.direct_credit_step import scale
from experiments.controller.direct_skill_allocation import direct_distributions
from experiments.controller.skill_choice_replay import decisions, uniform
from experiments.controller.skill_graph_attribution import gain_range, summarize
from experiments.controller.skill_graph_policy import allocate, project


class SkillGraphTests(unittest.TestCase):
    def tensor(self, x):
        return torch.tensor(
            x,
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def test_shared_weak_answer_counts_once_and_unconnected_skill_stays_neutral(self):
        y = self.tensor(
            [[1, 0, 1], [0.1, 0.9, 0.4], [0.25, 0.5, 0.625], [0.25, 0.5, 0.625]]
        )
        c = self.tensor([[-1, 0, 1], [0.8, 0, 0], [0.25, 0.375, 0.125], [0, 0, 0]])
        out = project(y, c)
        expected = self.tensor(
            [
                [1 / 3, -2 / 3, 1 / 3],
                [-0.4, 0.4, 0],
                [-5 / 24, 1 / 24, 1 / 6],
                [0, 0, 0],
            ]
        )
        torch.testing.assert_close(out, expected)
        c[2, 2] = 0
        torch.testing.assert_close(project(y, c), out)
        c[0, 0] = 1
        with self.assertRaises(ValueError):
            project(y, c)

    def test_unknown_answer_bound_includes_all_possible_physical_outcomes(self):
        graph = {"raw": [0.1, 0.05, -0.2], "credit": [0.06, 0.15, -0.15]}
        fixed = self.tensor(graph["credit"])
        bounds = []
        for old in ([0.8, 0.19, 0.01], [0.2, 0.2, 0.6]):
            policies = allocate(self.tensor(old), {"graph": graph})
            delta = policies["credit"] - policies["raw"]
            bounds.append(gain_range(delta, None))
            values = [
                gain_range(delta, fixed.new_tensor(v))[0]
                for v in itertools.product((0, 1), repeat=3)
            ]
            self.assertAlmostEqual(min(values), bounds[-1][0])
            self.assertAlmostEqual(max(values), bounds[-1][1])
        torch.testing.assert_close(self.tensor(bounds[0]), self.tensor(bounds[1]))

    def test_one_realization_applies_to_all_windows(self):
        windows = [
            {
                "n": 2,
                "matched": True,
                "gains": {"0": {"raw": [0.3, 0.3]}, "1": {"raw": [0, 0]}},
            },
            {
                "n": 1,
                "matched": False,
                "gains": {"0": {"raw": [-0.3, 0.3]}, "1": {"raw": [-0.8, 0.8]}},
            },
        ]
        uncertain = summarize(windows, ["0", "1"], [None], "raw")
        self.assertEqual(uncertain["questions"], 3)
        self.assertEqual(uncertain["matched_questions"], 2)
        self.assertAlmostEqual(uncertain["feasible_bounds"][1], 0.3)
        known = summarize(windows, ["0", "1"], [1], "raw")
        self.assertEqual(known["feasible_masks"], ["1"])
        self.assertAlmostEqual(known["feasible_bounds"][1], 0.8 / 3)

    def test_calibrated_move_preserves_time_and_soft_when_donors_disagree(self):
        self.assertEqual(scale([0.02, -0.02]), 0.0)
        self.assertAlmostEqual(scale([0.005, 0.005]), 0.5)
        native = self.tensor([0.8, 0.18, 0.02])
        scores = {
            "two_edge": {
                "raw": [0.01, 0.02],
                "credit": [0.02, 0.01],
                "rotated": [0, 0.01],
            },
            "direct": {"raw": 0.01, "credit": 0.012, "rotated": 0.01},
            "scale": {"credit": 1.0, "rotated": 0.0},
        }
        parent = direct_distributions(native, scores)
        p = step_allocate(native, scores)
        torch.testing.assert_close(p["soft"], parent["credit"])
        torch.testing.assert_close(p["raw"], parent["raw"])
        self.assertEqual(float(p["credit"][0]), float(p["raw"][0]))
        self.assertGreater(float(p["credit"][2]), float(p["soft"][2]))
        torch.testing.assert_close(p["rotated"], parent["raw"])
        scores["scale"]["credit"] = 0
        torch.testing.assert_close(
            step_allocate(native, scores)["credit"], parent["credit"]
        )

    def test_common_draw_respects_ordered_slots_and_quantifies_a_skill_switch(self):
        probabilities = self.tensor([[0.2, 0.7, 0.1], [0.2, 0.1, 0.7], [0.4, 0.3, 0.3]])
        selected, chances = decisions(probabilities, 0, 0.29)
        self.assertEqual(selected.tolist(), [1, 1, 0])
        torch.testing.assert_close(chances, self.tensor([0, 0.6, 0.4]))
        self.assertLess(uniform("panel", "conv-26_s9"), 1)
        self.assertNotEqual(
            uniform("panel", "conv-26_s9"), uniform("panel", "conv-26_s11")
        )
