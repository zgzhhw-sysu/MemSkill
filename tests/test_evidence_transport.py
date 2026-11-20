import unittest
from itertools import product

import torch

from experiments.controller.complete_donor_paths import (
    complement,
    decision_context,
    returns,
)
from experiments.controller.evidence_transport import (
    ARMS,
    Transport,
    aggregate,
    coefficients,
    query_similarity,
    root_targets,
)
from experiments.controller.evidence_transport_capacity import (
    compare_suffixes,
    envelope,
)
from experiments.controller.evidence_turns import texts
from src.evaluation_cache import digest


class EvidenceTransportTests(unittest.TestCase):
    def test_evidence_spans_are_anonymized_before_encoding(self):
        history = {"speaker_a": "Lee", "speaker_b": "Ann"}
        turns = [
            {
                "speaker": "Lee",
                "text": "ANN told Leela about Lee.",
                "blip_caption": "Ann on a bike",
            }
        ]
        self.assertEqual(
            texts(history, turns),
            [
                "Speaker 1: Speaker 2 told Leela about Speaker 1. [Image: Speaker 2 on a bike]"
            ],
        )

    def test_root_cannot_see_second_span_and_absent_evidence_is_constant(self):
        eye = torch.eye(2)
        donor = [eye, eye]
        links = [([1], [0]), ([], [0])]
        target = [eye[1:], eye[:1]]
        initial = query_similarity(target[:1], donor[:1], links, 0)
        self.assertEqual(initial.tolist(), [1.0, 0.0])
        self.assertEqual(query_similarity(target, donor, links, 1).tolist(), [1.0, 1.0])
        self.assertEqual(
            query_similarity([target[0], eye[1:]], donor, links, 1).tolist(), [0.5, 0.0]
        )

    def test_permutation_preserves_donor_and_linkage_totals(self):
        groups = [("a", 0), ("a", 0), ("a", 1), ("b", 0)]
        sim = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        w, blind, perm = coefficients(sim, groups, "cpu")
        self.assertEqual(perm.tolist(), [1, 0, 2, 3])
        self.assertAlmostEqual(float(w[:2].sum()), 1 / 3)
        self.assertAlmostEqual(float(w[2]), 1 / 6)
        self.assertAlmostEqual(float(w[3]), 0.5)
        self.assertAlmostEqual(float(blind[0]), 1 / 6)
        y = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float64).repeat(
            1, len(ARMS)
        )
        result = aggregate(sim, groups, y, "cpu")
        self.assertGreater(float(result[ARMS.index("credit")]), 0.33)
        self.assertLess(float(result[ARMS.index("sham")]), 0.001)

    def test_randomized_centered_backup_has_the_complete_four_path_expectation(self):
        values = {
            "00": [0.1, 0.4],
            "01": [0.8, 0.2],
            "10": [0.2, 0.1],
            "11": [0.4, 0.9],
        }
        bonuses = [[0.2, 0.0], [-0.3, 0.1]]
        full = {"pivot": None, "y": values, "root": bonuses}
        p = torch.tensor(
            [[0.8, 0.2, 0.6, 0.5, 0.4], [0.1, 0.3, 0.4, 0.5, 0.7]], dtype=torch.float64
        )
        expected = root_targets(full, p, "cpu")
        draws = [
            root_targets(
                {
                    "pivot": 0,
                    "bg": b,
                    "root": bonuses,
                    "y": {k: v for k, v in values.items() if k[1] == str(b)},
                },
                p,
                "cpu",
            )
            for b in (0, 1)
        ]
        self.assertTrue(torch.allclose(expected, (draws[0] + draws[1]) / 2))
        self.assertAlmostEqual(
            float(expected[0, ARMS.index("raw")]), 0.1 + 0.3 * 0.2 - 0.2 * 0.7
        )

    def test_nested_donor_backups_exclude_the_target_dialogue(self):
        def learner(held_return):
            turn = torch.ones((1, 2), dtype=torch.float64)
            turns = {
                f"{sid}_s{n}": (["id"], turn, "src", "file")
                for sid in "abcd"
                for n in (5, 6)
            }
            donors = {}
            for i, sid in enumerate("abcd"):
                value = held_return if sid == "d" else i / 3
                donors[f"{sid}_s5"] = {
                    "sid": sid,
                    "links": [([0], [])],
                    "pivot": None,
                    "last": [[0.0], [0.0]],
                    "root": [[0.0], [0.0]],
                    "y": {"00": [0.0], "01": [value], "10": [0.1], "11": [0.0]},
                    "base": [0.0, 0.0, 0.0],
                }
            return Transport(turns, donors, "cpu")

        a, b = (learner(v) for v in (0.0, 1.0))
        self.assertEqual(a.forecast("d_s5"), b.forecast("d_s5"))
        self.assertEqual(set(a.backups("d")), {"a_s5", "b_s5", "c_s5"})
        self.assertEqual(a.forecast("d_s5")["donors"], ["a", "b", "c"])

    def test_mixed_panel_uses_only_observed_pivots_for_training(self):
        turns = {}
        donors = {}
        for i, (sid, pivot, bg) in enumerate(
            [
                ("a", None, None),
                ("b", None, None),
                ("c", None, None),
                ("d", 0, 1),
                ("e", 1, 0),
                ("f", 1, 1),
            ]
        ):
            for j in (5, 6):
                turns[f"{sid}_s{j}"] = (["id"], torch.ones(1, 1), "", "")
            paths = (
                list(product((0, 1), repeat=2))
                if pivot is None
                else complement(pivot, bg)[0]
            )
            base = [0.0, 0.0, 0.0]
            root = [[0.01], [0.02]]
            last = [[0.03], [0.04]]
            if pivot == 0:
                last = None
                root[1 - bg] = None
            if pivot == 1:
                root = None
                last[1 - bg] = None
                base[1 + (1 - bg)] = None
            donors[f"{sid}_s5"] = {
                "sid": sid,
                "pivot": pivot,
                "bg": bg,
                "links": [([0], [0])],
                "y": {f"{a}{b}": [(a + 2 * b + i) / 9] for a, b in paths},
                "root": root,
                "last": last,
                "base": base,
            }
        model = Transport(turns, donors, "cpu")
        self.assertEqual(set(model.backups("a")), {"b_s5", "c_s5", "d_s5"})
        self.assertIn(("e_s5", 0), model.terminal_values)
        self.assertNotIn(("e_s5", 1), model.terminal_values)
        offsets = model.forecast("a_s5")["offsets"]
        self.assertEqual(len(offsets), 3)
        self.assertTrue(
            all(
                set(d) == set(ARMS) and all(-2 <= z <= 2 for z in d.values())
                for d in offsets
            )
        )

    def test_complement_recovers_joint_value_for_both_randomized_axes(self):
        cases = {(0, 0): [0.1], (0, 1): [0.9], (1, 0): [0.4], (1, 1): [0.2]}
        ps = {j: {"arm": v} for j, v in enumerate((0.7, 0.9, 0.1))}
        target = 0.3 * 0.82 + 0.7 * 0.38
        for pivot in (0, 1):
            estimates = []
            for bg in (0, 1):
                assigned, missing = complement(pivot, bg)
                self.assertEqual(set(assigned) | set(missing), set(cases))
                self.assertFalse(set(assigned) & set(missing))
                row = returns(cases, ps, assigned, pivot, bg, 0)["arm"]
                self.assertAlmostEqual(row["exact"], target)
                estimates.append(row["ht"])
            self.assertAlmostEqual(sum(estimates) / 2, target)

    def test_actual_span_source_and_order_are_checked(self):
        span = {
            "source": "observed",
            "selected_skills": [{}, {}, {"name": "capture_activity_details"}],
            "refs": [],
            "retrieved": [],
        }
        state = {
            "state": [1],
            "base_logit": 0,
            "plain": [0, 1, 2],
            "grounded": [0, 1, 3],
        }
        self.assertTrue(decision_context(span, state, digest("observed"), 1))
        for source, bit in ((digest("different"), 1), (digest("observed"), 0)):
            with self.assertRaises(ValueError):
                decision_context(span, state, source, bit)

    def test_sign_constrained_capacity_distinguishes_wrong_direction(self):
        z = torch.zeros((1, 3), dtype=torch.float64)
        values = z.new_tensor([[0.0, 0.0, 1.0, 1.0]])
        result = envelope(z, z.new_tensor([[-0.1, 0, 0.1]]), values, z.new_ones(1))
        self.assertEqual(result["sign_oracle"], 0.5)
        self.assertEqual(result["best_stage_grid"], 0.5)
        self.assertLess(result["learned"], 0.5)
        self.assertAlmostEqual(
            result["unrestricted"], float(torch.sigmoid(z.new_tensor(2)))
        )

    def test_suffix_diagnostic_counts_reversals_separately_from_zeros(self):
        exact = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64)
        draws = exact.new_tensor([[-0.1, -1.0, -0.2], [1.1, 0.0, 0.2]])
        counts = compare_suffixes(exact, draws)
        self.assertEqual(
            (
                counts["nonzero_draws"],
                counts["opposite_draws"],
                counts["zero_draws_from_nonzero"],
            ),
            (4, 1, 1),
        )
        with self.assertRaises(ValueError):
            compare_suffixes(exact + 0.01, draws)
