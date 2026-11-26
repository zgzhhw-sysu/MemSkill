import itertools
import unittest

import torch

from experiments.controller.reader_slot_credit import components, slots
from experiments.controller.reader_slot_policy import (
    direct_range,
    effects,
    possible_probability,
)


class ReaderSlotTests(unittest.TestCase):
    def test_writer_paths_follow_displayed_versions_and_take_the_shortest_route(self):
        from experiments.controller.reader_slot_credit import writer_read_paths
        spans = [{"span_id": i, "refs": refs, "changes": [{"memory_id": mid, "version": version}]}
                 for i, (refs, mid, version) in enumerate([
                     ([], 0, 1), ([], 1, 1), ([[0, 1]], 0, 2), ([[0, 2]], 2, 1),
                 ])]
        contexts = {5: [(3, "copy")], 6: [(1, "copy")], 7: [(2, "x"), (3, "copy")], 8: []}
        paths = writer_read_paths(spans, contexts, [0, 2], "cpu")
        self.assertEqual(paths["0"], {"5": 2, "6": None, "7": 1, "8": None})
        self.assertEqual(paths["2"], {"5": 1, "6": None, "7": 0, "8": None})

    def test_matched_route_coverage_equals_exhaustive_within_quarter_draws(self):
        from experiments.controller.reader_slot_credit import matched_path_placement
        hops = {str(i): {"1": (0 if i in (0, 2) else None), "2": (1 if i in (0, 3) else None)} for i in range(4)}
        panel = matched_path_placement(hops, list(range(4)), {0, 3}, [1, 2], total=8)
        self.assertEqual(panel["switches_per_quarter"], [1, 1, 0, 0])
        self.assertEqual(panel["coverage"]["direct"], {"observed_questions": [1], "expected_count": 0.75})
        self.assertEqual(panel["coverage"]["one_or_less"], {"observed_questions": [1, 2], "expected_count": 1.5})

    def test_exact_copy_policy_preserves_text_and_slot_identity(self):
        from experiments.controller.reader_slot_credit import focal_reader_slots
        ctx = [(2, "Same"), (1, "same"), (6, "Same"), (2, "Same "), (2, "Same"), (1, "Same ")]
        self.assertEqual(focal_reader_slots(ctx, 6), [2])
        self.assertEqual(focal_reader_slots(ctx, 6, True), [0, 2, 4])
        self.assertEqual(focal_reader_slots(ctx, 1, True), [1, 3, 5])
        self.assertEqual(focal_reader_slots(ctx, 5, True), [])

    def test_copy_credit_conditions_on_equal_removals(self):
        from experiments.controller.reader_slot_credit import copy_contrasts, matched_reader_slots

        self.assertEqual(matched_reader_slots([1, 4], [2, 2, 3, 2, 3, 2], removed=[0, 3]), [5, 2])
        self.assertIsNone(matched_reader_slots([1, 2], [1, 1, 1, 1], removed=[0]))
        scores = {"full": 1, "no_focal": 1, "no_copies": 1, "neither": 0, "no_control": 1, "no_focal_control": 1}
        values = copy_contrasts({arm: {"f1": v} for arm, v in scores.items()})
        self.assertEqual((values["focal_order_average"], values["copies_order_average"]), (.5, .5))
        self.assertEqual(values["interaction_vs_control"], 1)
        scores["no_focal_control"] = 0
        self.assertEqual(copy_contrasts({arm: {"f1": v} for arm, v in scores.items()})["interaction_vs_control"], 0)

    def test_sham_slots_match_length_then_rank_and_preserve_duplicates(self):
        from experiments.controller.reader_slot_credit import matched_reader_slots

        self.assertEqual(matched_reader_slots([], [9, 2]), [])
        self.assertIsNone(matched_reader_slots([0, 1], [7, 7, 7]))
        self.assertEqual(matched_reader_slots([2], [10, 9, 10, 8, 8]), [0])
        self.assertEqual(matched_reader_slots([2, 4], [7, 7, 7, 7, 7, 8]), [1, 3])
        self.assertEqual(set(matched_reader_slots([0, 3], [10, 3, 11, 2])), {1, 2})

    def test_single_rank_is_swapped_without_shifting_other_positions(self):
        from experiments.controller.reader_slot_witness import at_rank

        a, b = [(0, "keep"), (0, "del")], [(0, "keep"), (1, "new"), (1, "tail")]
        self.assertEqual(at_rank(a, b, 1), ["keep", "new"])
        self.assertEqual(at_rank(b, a, 2), ["keep", "new"])
        self.assertEqual(at_rank(a, b, 2), ["keep", "del", "tail"])
        with self.assertRaises(ValueError):
            at_rank(a, b, 3)

    def test_rank_crosses_keep_order_duplicates_and_missing_positions(self):
        left = [(0, "old"), (1, "A"), (0, "old"), (1, "common")]
        right = [(0, "other"), (0, "other"), (1, "B")]
        labels, (a, b) = slots(
            left, right, [{"old", "A", "common"}, {"other", "B", "common"}], 1
        )
        self.assertEqual(labels, ["other", "direct", "direct", "other"])
        self.assertEqual(a, ["old", "other", "B", "common"])
        self.assertEqual(b, ["other", "A", "old"])
        labels, (a, b) = slots([(0, "same")], [(1, "same")], [{"same"}, {"same"}], 1)
        self.assertEqual(labels, ["same"])
        self.assertEqual(a, b)

    def test_physical_credit_is_conserved_and_bounds_include_all_crosses(self):
        for low, high in [(0.0, 0.8), (0.8, 0.0)]:
            credit = (high - low) / 2
            r = {
                "low": {"f1": low, "prompt_sha256": "L"},
                "high": {"f1": high, "prompt_sha256": "H"},
                "hybrids": ["a", "b"],
                "credit": credit,
            }
            lower, upper = direct_range(r)
            for a, b in itertools.product([0.0, 0.3, 0.7, 1.0], repeat=2):
                direct, extra = components(low, high, [a, b], credit)
                self.assertAlmostEqual(direct, ((a - low) + (high - b)) / 2)
                self.assertGreaterEqual(extra * credit, 0.0)
                self.assertLessEqual(abs(extra), abs(credit))
                self.assertLessEqual(lower - 1e-12, direct + extra)
                self.assertGreaterEqual(upper + 1e-12, direct + extra)
            r["hybrids"] = ["a", "a"]
            self.assertAlmostEqual(*direct_range(r))
            r["hybrids"] = ["H", "L"]
            self.assertAlmostEqual(direct_range(r)[0], high - low + credit)
        self.assertEqual(components(0.1, 0.7, [0.7, 0.1], 0.0)[1], 0.0)

    def test_kernel_change_uses_only_source_for_direct_component(self):
        s = torch.tensor(
            [1.0, 0.0, 1.0, 0.0],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        states = torch.stack([s, s.new_tensor([1.0, 0.0, -1.0, 0.0])])
        w = s.new_ones(2)
        totals = s.new_tensor([[0.02, 0.02], [-0.02, -0.02]])
        bounds = s.new_tensor([[-0.04, 0.05], [-0.02, 0.06]])
        zero = s.new_zeros(2, 3)
        model = (states, w, totals, zero, bounds)
        self.assertTrue(
            torch.allclose(
                effects(s, model), s.new_full((4,), 0.01), atol=1e-13, rtol=0
            )
        )
        decision = {"state": s.tolist(), "base_logit": 0.25}
        lo, hi = possible_probability(decision, model)
        self.assertGreater(hi, lo)
        realized = []
        for a, b in itertools.product(range(2), repeat=2):
            ds = torch.stack((bounds[0, a], bounds[1, b]))
            p = effects(s, (states, w, totals, ds[:, None].expand(2, 3), bounds))[0]
            realized.append(float((0.25 + (p / 0.05).clamp(-2, 2)).sigmoid()))
        self.assertAlmostEqual(min(realized), lo)
        self.assertAlmostEqual(max(realized), hi)
