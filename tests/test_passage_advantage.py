import unittest

import torch

from experiments.controller.passage_advantage import (
    contributions,
    direction_report,
    equal_rate_exact,
    exact,
    rank_correlation,
)
from experiments.controller.passage_case_reliability import (
    apply_weights,
    fit_cases,
    reliability,
)
from experiments.controller.passage_evidence_actor import (
    placebo_positions,
    score_sessions,
    support_margin,
)
from experiments.controller.within_session_opportunity import group_questions, mass


class PassageAdvantageTest(unittest.TestCase):
    def t(self, x):
        return torch.tensor(
            x,
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def test_root_backup_and_realized_suffix_visitation(self):
        # At root choosing activity permanently changes the value of a later activity choice.
        y = self.t([[[0, 0.8], [0.9, 0.1]], [[1, 1], [1, 1]]])
        a = self.t([0.2, 0.5, 0.25])
        b = self.t([0.7, 0.75, 0.75])
        advantage, delta = contributions(y, a, b)
        self.assertTrue(torch.allclose(advantage[0], self.t([0.3, 0.8, -0.8])))
        self.assertTrue(torch.allclose(delta[0], self.t([0.15, 0.06, -0.28])))
        self.assertTrue(torch.allclose(delta.sum(1), exact(y, b) - exact(y, a)))
        self.assertTrue(torch.equal(delta[1], self.t([0, 0, 0])))

    def test_evidence_position_has_no_access_to_mixed_path_return(self):
        hist = {"session_3": [{"dia_id": f"D3:{i}"} for i in range(6)]}
        qa = [
            {"evidence": ids}
            for ids in (["D3:0"], ["D3:2", "D2:1"], ["D3:1", "D3:4"], ["D1:1"])
        ]
        cut, rows = group_questions(
            hist, 3, [(0, 0.0, 1.0), (1, 0.7, 0.2), (2, 0.1, 0.1), (3, 0.0, 0.0)], qa
        )
        self.assertEqual(cut, 2)
        self.assertEqual(
            [r["group"] for r in rows], ["first", "second", "both", "elsewhere"]
        )
        self.assertAlmostEqual(mass(rows)["extra"], 0.5)
        self.assertFalse(mass(rows, require_whole=True)["supported"])
        self.assertEqual(mass(rows, require_whole=True)["extra"], 0.0)

    def test_zero_and_one_propensities_are_valid_path_values(self):
        y = self.t([[[0.1, 0.3], [0.7, 0.9]]])
        a = self.t([0, 1, 1])
        b = self.t([1, 0, 0])
        self.assertAlmostEqual(float(exact(y, b)), 0.7)
        self.assertAlmostEqual(float(contributions(y, a, b)[1].sum()), 0.4)

    def test_placebo_wraparound_cardinality_and_disjointness(self):
        self.assertEqual(placebo_positions([1, 4], 5), [0, 2])
        self.assertEqual(placebo_positions([0, 1, 2], 5), [])
        self.assertEqual(placebo_positions([], 4), [])

    def test_competing_turn_can_veto_a_perfect_topic_match(self):
        donor = self.t([[1, 0], [0, 1], [0.6, 0.8]])
        target = self.t([[0, 1], [1, 0]])
        margin, i = support_margin(target, donor, self.t([1, 0]), [0])
        self.assertEqual(i, 0)
        self.assertTrue(torch.allclose(margin, self.t([0, 0.4])))

    def test_all_reference_counts_exclude_donor_and_held(self):
        names = [f"{h}_s{i}" for h in "ABCDEF" for i in range(5, 13)]
        scores = self.t([[1] * 4 if n[0] in "AB" else [0] * 4 for n in names])
        fits, _ = score_sessions(
            dict.fromkeys(names),
            {("B", 42): {"scores": scores, "credit": 0.5, "hits": [None] * len(names)}},
            {n: {"credit": 0.2} for n in names},
        )
        self.assertAlmostEqual(fits["A_s6"]["credit"], 1 - 2 / 33)
        self.assertEqual(fits["B_s6"]["credit"], 0)
        self.assertEqual(fits["B_s6"]["parent"], 0.2)
        self.assertEqual(fits["C_s6"]["credit"], 0)

    def test_reliability_does_not_reverse_labels_or_count_empty_votes(self):
        self.assertEqual(reliability([0, 1e-12, -1.0, -0.3]), 0)
        self.assertAlmostEqual(reliability([1.0, 0.3, -0.4, 0.0]), 1 / 5)
        registry = {
            "A": [
                {
                    "case": ["B", 2],
                    "signed_credit": -1.0,
                    "stages": [{"weight": 0.0}, {"weight": 0.5}],
                }
            ]
        }
        match = [{"case": ["B", 2], "strength": [0.6]}]
        original = {"parent": 0.4, "credit": -1.2}
        self.assertEqual(
            apply_weights("A", match, registry, original, 0)["credit"], 0.0
        )
        self.assertAlmostEqual(
            apply_weights("A", match, registry, original, 1)["credit"], -0.6
        )

    def test_nested_fit_is_invariant_to_held_labels_calibration_and_outcomes(self):
        device = self.t([0]).device
        histories = "ABCDEF"

        def fit(alter):
            labels = {(h, 0): (-1 if h in "ACE" else 1) for h in histories}
            if alter:
                labels["A", 0] = 0.45
            matches = {}
            for i, x in enumerate(histories):
                for n in range(5, 13):
                    matches[f"{x}_s{n}"] = [
                        {
                            "case": [g, 0],
                            "scores": [
                                100.0
                                if alter and (x == "A" or g == "A")
                                else float((n + 3 * i + j) % 7)
                            ],
                        }
                        for j, g in enumerate(histories)
                        if g != x
                    ]
            panels = [
                {
                    "window": f"{x}_s{n}",
                    "dialogue": x,
                    "probabilities": {
                        str(j): {"native": 0.1 * (1 + j)} for j in range(3)
                    },
                    "cells": {
                        k: [
                            0.0
                            if x == "A" and alter
                            else float(k[0]) * 0.7 + float(k[1]) * 0.2
                        ]
                        for k in ("00", "01", "10", "11")
                    },
                }
                for x in histories
                for n in (5, 7)
            ]
            return fit_cases({"matches": matches, "labels": labels}, panels, device)[
                "A"
            ]

        original = fit(False)
        self.assertEqual(original, fit(True))
        self.assertEqual(len(original), 5)
        for r in original:
            self.assertEqual(
                set(r["stages"][0]["history_gains"]),
                set(histories) - {"A", r["case"][0]},
            )

    def test_direction_counts_ties_and_undefined_correlations(self):
        r = direction_report([(0.1, 0.5), (-0.2, 0.3), (0, -1), (0.1, 0)])
        self.assertEqual((r["states"], r["nonzero"], r["correct"]), (4, 2, 1))
        self.assertAlmostEqual(rank_correlation([2, 2, 1], [2, 2, 4]), -1)
        self.assertIsNone(rank_correlation([1, 1], [2, 3]))

    @unittest.skipUnless(
        torch.cuda.is_available(), "full-panel rate matching requires a guarded GPU"
    )
    def test_suffix_use_matches_at_actual_root_occupancy(self):
        def window(n, p):
            return {
                "rows": [{}] * n,
                "probabilities": {
                    str(i): {"credit": v, "native": 0.5} for i, v in enumerate(p)
                },
                "cells": {s: [int(s[-1])] * n for s in ("00", "01", "10", "11")},
            }

        panel = [window(2, [0.1, 0.9, 0.2]), window(1, [0.9, 0.3, 0.7])]
        self.assertAlmostEqual(
            equal_rate_exact(panel),
            (2 * (0.9 * 0.9 + 0.1 * 0.2) + 0.1 * 0.3 + 0.9 * 0.7) / 3,
        )
