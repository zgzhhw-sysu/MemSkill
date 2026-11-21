import unittest

import torch

from experiments.controller.shared_outlier_policy import (
    corrected_utilities,
    corrections,
    fit_correction,
    original,
)
from experiments.controller.three_skill_shared_credit import SKILLS, decompose


class SharedQuestionCreditTests(unittest.TestCase):
    def test_shared_grouping_evidence_has_one_unit_of_mass(self):
        table = {
            s + "_" + f: float(f == "single")
            for s in SKILLS
            for f in ("single", "pairs")
        }
        edges = [-1, 0, 1]
        self.assertEqual(original({"edges": edges, "scores": [1, 0, 1]}), (1, -1))
        r = decompose(table, edges)
        self.assertEqual(r["outlier"], "activity")
        self.assertEqual(r["credit"], [0, -1])
        self.assertEqual(sum(r["credit"]), r["mean"]["raw"])
        self.assertEqual([v["raw"] for v in r["comparisons"].values()], [-1, -1])

    def test_shared_deployed_rule_keeps_the_template_contrast(self):
        v = {
            s + "_" + f: (0.75 if s == "relation" else 1.0)
            for s in SKILLS
            for f in ("single", "pairs")
        }
        r = decompose(v, [0, -0.25, -0.25])
        self.assertEqual(r["credit"], [-0.25, 0])
        self.assertEqual(r["comparisons"]["temporal"]["grouping"], 0.0)
        v["activity_pairs"] = 0.9
        self.assertEqual(
            decompose(v, [0, -0.25, -0.15])["status"], "different_peer_outcomes"
        )
        self.assertIsNone(
            original({"scores": [1, 0.9, 0.75], "edges": [0, -0.25, -0.15]})
        )
        self.assertEqual(
            decompose(v, [0, 0, -0.15])["status"], "no_unique_two_edge_comparison"
        )

    def test_update_only_moves_the_supervised_logit_and_holds_out_the_dialogue(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = [
            {
                "node": f"{h}_{i}",
                "history": h,
                "features": [0.1 * i, 0.1 * i],
                "raw": 0,
                "credit": [[0.0, 0.0], [0.0, 0.08 * i], [0.0, 0.0]],
            }
            for h in ("a", "b", "c")
            for i in (1, 2)
        ]
        m = fit_correction(rows, "c", device)
        rows[-1]["credit"] = [[0.8, 1.0]] * 3
        self.assertEqual(m, fit_correction(rows, "c", device))
        d = corrections(m, [0.2, 0.2], device, False)
        self.assertEqual((d["credit"][0], d["credit"][2]), (0.0, 0.0))
        self.assertGreater(d["credit"][1], 0.0)
        y = corrected_utilities([0.0, -0.02, 0.04], d)
        p = (
            torch.as_tensor(
                [y[k] for k in ("raw", "credit")], device=device, dtype=torch.float64
            )
            .div(0.05)
            .softmax(1)
        )
        self.assertAlmostEqual(float(p[0, 0] / p[0, 2]), float(p[1, 0] / p[1, 2]))
