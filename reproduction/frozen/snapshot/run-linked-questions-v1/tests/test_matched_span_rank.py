import copy
import math
import unittest

import torch

from experiments.controller.matched_span_calibration import (
    calibrate,
    corrections,
    inner_fit,
)
from experiments.controller.matched_span_consensus import deleted_models, voting
from experiments.controller.matched_span_data import structure
from experiments.controller.matched_span_policy import (
    concordance,
    deploy,
    fit,
    paired_edges,
    predict,
)
from experiments.controller.third_skill_capacity import aggregate, bounds


class MatchedSpanRankTest(unittest.TestCase):
    def setUp(self):
        self.units = []
        self.x = {}
        for h in ("a", "b", "c"):
            for background in (0, 1):
                for index in (0, 1, 2):
                    node = f"{h}_s{2 * index + background + 1}"
                    x = torch.tensor(
                        [index + background / 2, background], dtype=torch.float64
                    )
                    self.x[node] = x
                    self.units.append(
                        {
                            "history": h,
                            "background": background,
                            "stage": 1,
                            "node": node,
                            "credit": index / 10,
                            "raw": -index / 10,
                        }
                    )

    def test_grouping_changes_distinguish_own_score_margin_and_partner(self):
        turns = torch.eye(3, dtype=torch.float64)
        single = turns, torch.eye(3, dtype=torch.bool)
        paired = (
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool),
        )
        z = structure(single, paired).reshape(3, 4)
        self.assertAlmostEqual(float(z[1, 0]), 2**-0.5 - 1)
        self.assertAlmostEqual(float(z[1, 1]), 2**-0.5 - 1)
        torch.testing.assert_close(z[:, 2:], torch.zeros_like(z[:, 2:]))
        alone = (turns[:1], torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(
            structure(alone, alone), torch.zeros(12, dtype=turns.dtype)
        )
        reversed_chunks = (paired[0].flip(0), paired[1].flip(0))
        torch.testing.assert_close(structure(single, reversed_chunks), z.flatten())

    def test_pairing_keeps_stage_history_and_other_action_fixed(self):
        extra = [dict(self.units[0], stage=0), dict(self.units[1], stage=0)]
        edges = paired_edges(self.units + extra)
        self.assertEqual(len(edges), 19)
        self.assertTrue(
            all(e["a"].split("_")[0] == e["b"].split("_")[0] for e in edges)
        )
        self.assertEqual(sum(e["stage"] == 0 for e in edges), 1)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            paired_edges(self.units + self.units[:1])

    def test_within_stratum_constants_and_held_labels_cannot_change_the_model(self):
        args = ("c", 1, ("credit", "rotated"))
        base = fit(self.units, self.x, *args)
        other = copy.deepcopy(self.units)
        x = dict(self.x)
        for u in other:
            if u["history"] == "c":
                x.pop(u["node"])
                u["credit"] = 888.0
            else:
                u["credit"] += (
                    9 + 7 * u["background"] + (5 if u["history"] == "a" else 0)
                )
        changed = fit(other, x, *args)
        for k in ("center", "scale", "width", "coef"):
            torch.testing.assert_close(base[k], changed[k])
        self.assertGreater(
            float(
                predict(base, self.x["c_s5"])[0][0]
                - predict(base, self.x["c_s1"])[0][0]
            ),
            0,
        )
        raw = fit(self.units, self.x, "c", 1, ("raw",))
        p, _ = deploy(self.x["c_s5"], raw, base)
        self.assertLess(p["raw_rank"], 0)
        self.assertGreater(p["credit"], 0)
        self.assertTrue(all(abs(v) <= 2 for v in p.values()))

    def test_calibrator_weights_histories_equally_and_shrinks_to_raw(self):
        rows = [
            {
                "history": "first",
                "actual": 0.2,
                "raw_rank": 0.0,
                "direct": 0.1,
                "placebo": -0.1,
            },
            {
                "history": "second",
                "actual": -0.1,
                "raw_rank": 0.0,
                "direct": 0.1,
                "placebo": -0.1,
            },
        ]
        g = calibrate(rows, "cpu")
        self.assertAlmostEqual(float(g[0]), 0.005 / (0.01 + 0.025**2))
        self.assertAlmostEqual(float(g[1]), -float(g[0]))
        torch.testing.assert_close(g, calibrate(rows + rows[:1] * 8, "cpu"))
        exact = [dict(row, raw_rank=row["actual"]) for row in rows]
        torch.testing.assert_close(calibrate(exact, "cpu"), torch.zeros_like(g))

    def test_inner_predictions_and_calibration_ignore_the_entire_outer_history(self):
        args = (self.units, self.units, self.x, "c", 1)
        g, trace = inner_fit(*args)
        changed = [
            dict(r, raw=-93, credit=12) if r["history"] == "c" else r
            for r in self.units
        ]
        x = {k: v for k, v in self.x.items() if not k.startswith("c_")}
        gg, second = inner_fit(changed, changed, x, "c", 1)
        self.assertEqual(trace, second)
        torch.testing.assert_close(g, gg)
        self.assertEqual({r["history"] for r in trace}, {"a", "b"})
        raw = fit(self.units, self.x, "c", 1, ("raw",))
        c = fit(self.units, self.x, "c", 1, ("credit", "rotated"))
        for v in self.x.values():
            arms, _ = corrections(raw, c, v, g.new_zeros(2))
            self.assertEqual(arms["credit"], arms["raw_rank"])
            self.assertEqual(arms["rotated"], arms["raw_rank"])
            self.assertAlmostEqual(arms["raw_rank"], deploy(v, raw, c)[0]["raw_rank"])

    def test_consensus_counts_agreement_abstention_and_full_steps(self):
        scores = torch.tensor(
            [
                [0.01, 0.04, 0.01],
                [-0.009, 0.03, -0.02],
                [0.6, 0.5, -0.7],
                [-0.1, 0.2, 0],
                [0.2, 0.3, 0],
            ],
            dtype=torch.float64,
        )
        v = voting(scores, scores.new_tensor(-0.01))
        self.assertAlmostEqual(v["credit"], 0.8)
        self.assertEqual(v["raw_rank"], 2.0)
        self.assertEqual(v["single"], -2.0)
        self.assertEqual(v["unanimous"], 0.0)
        self.assertEqual(
            voting(scores[:, [1, 0, 2]], scores.new_tensor(0.0))["unanimous"], 2.0
        )

    def test_deleted_history_stays_out_of_coefficients_and_calibration(self):
        data = list(self.units)
        x = dict(self.x)
        for h in ("d", "e"):
            for u in self.units:
                if u["history"] != "a":
                    continue
                node = h + u["node"][1:]
                x[node] = self.x[u["node"]] * 1.2
                data.append(dict(u, history=h, node=node))
        result = deleted_models(data, data, x, "c", 1)
        self.assertEqual({r[0] for r in result}, {"a", "b", "d", "e"})
        for h, rm, cm, g, edges in result:
            self.assertEqual(set(rm["histories"]), set(cm["histories"]))
            self.assertFalse({h, "c"} & set(rm["histories"]))
            self.assertEqual({r["history"] for r in edges}, set(rm["histories"]))
            self.assertTrue((g.abs() <= 1).all())
        # Tampering with the deleted donor cannot affect its deletion fit.
        changed = [
            dict(u, credit=999, raw=-777) if u["history"] == "a" else u for u in data
        ]
        other = deleted_models(changed, changed, x, "c", 1)
        self.assertEqual(result[0][0], "a")
        for i in (1, 2):
            torch.testing.assert_close(result[0][i]["coef"], other[0][i]["coef"])
        torch.testing.assert_close(result[0][3], other[0][3])
        self.assertEqual(result[0][4], other[0][4])

    def test_third_skill_bounds_use_shared_probabilities_and_include_the_two_arm_family(
        self,
    ):
        cases = [
            ([0.4, 0.5, 0.9], [0.5, 0.49, 0.01]),
            ([0.8, 0.6, 0.2], [0.2, 0.2, 0.6]),
        ]
        rows = []
        for n, (means, ps) in enumerate(cases, 1):
            values = bounds(means, ps)
            for i in range(-4, 5):
                for j in range(-4, 5):
                    a = ps[1] / ps[0] * math.exp(i / 2)
                    r = ps[2] / ps[0] * math.exp(j / 2)
                    measured = (means[0] + a * means[1] + r * means[2]) / (1 + a + r)
                    self.assertLessEqual(measured, values["bounded_three"] + 1e-12)
            self.assertEqual(
                values["bounded_optional_relation"],
                max(values["bounded_two"], values["bounded_three"]),
            )
            rows.append({"n": n, "means": means, "bounds": values})
        total = aggregate(rows)
        self.assertEqual(total["strict_relation_wins"], 1)
        self.assertAlmostEqual(total["free_three"], (0.9 + 2 * 0.8) / 3)
        self.assertLess(total["best_fixed_three"], total["free_three"])

    def test_expanded_credit_edges_cannot_impute_an_unobserved_raw_return(self):
        data = self.units[:2] + [dict(self.units[2])]
        del data[-1]["raw"]
        ed = paired_edges(data)
        self.assertEqual(sum(e["raw"] is not None for e in ed), 1)
        f = {
            "1": {
                r["node"]: dict.fromkeys(("raw_rank", "credit", "rotated"), r["credit"])
                for r in data
            }
        }
        s = concordance(ed, f)["all"]
        self.assertEqual(s["raw"]["edges"], 1)
        self.assertEqual(s["credit"]["edges"], 3)
        self.assertEqual(s["credit"]["predictors"]["credit"]["correct"], 3)


if __name__ == "__main__":
    unittest.main()
