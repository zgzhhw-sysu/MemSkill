import tempfile
import unittest
from pathlib import Path

import torch

from experiments.controller.witness_passage_factorial import ready
from experiments.controller.witness_passage_labels import assignments
from experiments.controller.witness_passage_policy import (
    cases,
    contrast_scores,
    matched_probabilities,
    offsets,
    strength,
)
from experiments.controller.witness_passages import clean


class WitnessPassagesTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tensor(self, values):
        return torch.tensor(values, device=self.device, dtype=torch.float64)

    def test_only_exact_date_and_whole_names_are_removed(self):
        history = {"speaker_a": "Ann", "speaker_b": "Lee"}
        self.assertEqual(
            clean(
                "Monday | ANN visited Annette with Lee. INSERT block:",
                history,
                "Monday",
            ),
            "Speaker 1 visited Annette with Speaker 2.",
        )
        self.assertEqual(
            clean("Tuesday | Lee", history, "Monday"), "Tuesday | Speaker 2"
        )
        self.assertIsNone(clean(None, history, "Monday"))

    def test_margin_and_match_must_come_from_same_utterance(self):
        turns = self.tensor([[1, 0], [0, 1]])
        # The strongest positive match has no treatment contrast.
        scores, index = contrast_scores(
            turns, self.tensor([[0.8, 0.6]]), self.tensor([[1, 0]]), self.tensor([1])
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(float(scores[0]), 0.36)
        self.assertAlmostEqual(float(scores[1]), 0.8)

    def test_calibration_counts_ties_and_binds_reference_population(self):
        ref = self.tensor([0] * 32)
        self.assertEqual(float(strength(self.tensor(0), ref)), 0)
        self.assertAlmostEqual(float(strength(self.tensor(1), ref)), 1 - 2 / 33)
        with self.assertRaises(ValueError):
            strength(self.tensor(0), ref[:31])

    def test_donor_and_target_histories_are_excluded_from_reference(self):
        names = [f"conv-{h}_s{j}" for h in range(6) for j in range(5, 13)]
        names.append("conv-0_s13")
        turns = dict.fromkeys(names)
        target, donor = ("conv-0", 3), ("conv-1", 7)
        scores = self.tensor(
            [
                [1, 1] if n.split("_")[0] in (target[0], donor[0]) else [0, 0]
                for n in names
            ]
        )
        fit, used = offsets(
            {target: {"credit": -1}, donor: {"credit": 1}},
            {donor: {"scores": scores, "hits": [[2, 1]] * len(names)}},
            turns,
            target[0],
            len(names) - 1,
        )
        self.assertEqual([r["donor"] for r in used], [list(donor)])
        self.assertAlmostEqual(fit["credit"], 2 * (1 - 2 / 33), places=6)

    def test_repeated_donor_question_counts_once_with_confirmed_role_weights(self):
        rows = [
            {
                "eligible": True,
                "dialogue": "conv-1",
                "index": 7,
                "credit": -0.8,
                "job": i,
                "keys": [0, 1],
            }
            for i in range(3)
        ]
        labels = {
            i: {"sufficient_in_loser": i == 0, "necessary_in_winner": i != 2}
            for i in range(3)
        }
        fit = cases(rows, labels)
        self.assertEqual(list(fit), [("conv-1", 7)])
        self.assertEqual(fit["conv-1", 7]["ranks"], [(0, 0, 1, 1), (1, 0, 1, 0.5)])

    def test_rate_control_preserves_use_when_combined_shift_exceeds_two(self):
        z = self.tensor([-1, 0, 1])
        rates = (z + self.tensor([3, 4, 5])).sigmoid()
        w = self.tensor([3, 1, 2])
        matched = matched_probabilities(z, rates, w)
        self.assertAlmostEqual(float(w @ matched), float(w @ rates), places=10)
        self.assertTrue(bool(torch.allclose(matched.logit().diff(), z.diff())))

    def test_assignment_control_preserves_labels_in_each_donor_history(self):
        rows = [
            {"donor": [h, i], "credit": v}
            for h, i, v in [("b", 1, -4), ("a", 8, 2), ("a", 3, -1)]
        ]
        keys, variants, index = assignments({"x": rows, "y": rows})
        self.assertEqual(keys, [("a", 3), ("a", 8), ("b", 1)])
        self.assertEqual(len(variants), 2)
        self.assertEqual(variants[index], [-1, 2, -4])
        self.assertTrue(all(x[-1] == -4 for x in variants))

    def test_no_new_window_is_exposed_prior_to_witness_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trial = base / "trial"
            spec = {"dialogue": "conv-0", "start": 13}
            self.assertFalse(ready(base, trial, spec))
            self.assertFalse(ready(base, trial, {"dialogue": "conv-0", "start": 15}))
            folder = base / "late-pairs/outcomes/conv-0_s13"
            folder.mkdir(parents=True)
            (folder / "pair.json").touch()
            self.assertTrue(ready(base, trial, spec))
            (trial / "witness").mkdir(parents=True)
            (trial / "witness/forecast.json").touch()
            self.assertTrue(ready(base, trial, {"dialogue": "conv-0", "start": 15}))
