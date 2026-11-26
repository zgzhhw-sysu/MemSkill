import unittest
from itertools import product

import torch

from experiments.controller.conditional_credit import (
    coupled_paths,
    orientation_support,
    fit_windows,
    score_questions,
    check_coupling,
)
from experiments.controller.query_prototypes import mask_speakers, prototype_logits
from experiments.controller.audit_query_prototypes import compare


class ConditionalCreditTest(unittest.TestCase):
    def test_marginals_and_online_conditionals_even_at_boundaries(self):
        for p, q, rho in product(
            (0.0, 0.24, 0.5, 1.0), (0.11, 0.82, 1.0, 0.0), (-1.0, -0.38, 0.0, 0.96, 1.0)
        ):
            d = coupled_paths(p, q, rho)
            u, v = d["conditional_second"]
            for a, b in zip(d["mass"], ((1 - p) * (1 - u), (1 - p) * u, p * (1 - v), p * v), strict=True):
                self.assertAlmostEqual(a, b)
            self.assertAlmostEqual((1 - p) * u + p * v, q)
            self.assertGreaterEqual(min(d["mass"]), 0.0)

    def test_other_dialogues_only_and_registration_roundtrip(self):
        features, probs, rows = {}, {}, []
        for sid, n in product("ABCDEF", range(1, 9)):
            features[sid, n] = {"features": [n * 0.01, (-1) ** n, float(ord(sid) - 64)]}
            probs[sid, n] = 0.1 * n
            if n <= 4:
                rows.append({"dialogue": sid, "session": n, "signed": 0.1 * n, "prob": 0.5, "use": n == 1})
        fits, pred = fit_windows(rows, features, probs)
        altered = [
            dict(r, signed=-r["signed"], use=not r["use"]) if r["dialogue"] == "A" else r for r in rows
        ]
        new, new_pred = fit_windows(altered, features, probs)
        self.assertEqual(fits["A"], new["A"])
        self.assertEqual(
            [v for v in pred if v["dialogue"] == "A"], [v for v in new_pred if v["dialogue"] == "A"]
        )
        report = {"fits": fits, "forecasts": pred}
        check_coupling(report, features, probs)
        pred[0]["mass"][1] += 0.01
        with self.assertRaisesRegex(ValueError, "reproduce"):
            check_coupling(report, features, probs)

    def test_negative_only_is_identical_to_always_favor_different(self):
        windows, pred = [], []
        for sid, start in product("ABCDEF", (1, 3)):
            qs = {2: {"f1": {"00": 0.0, "01": 1.0, "10": 1.0, "11": 0.0}, "question": {"category": 4}}}
            windows.append({"dialogue": sid, "start": start, "questions": qs})
            pred.append(
                {
                    "dialogue": sid,
                    "start": start,
                    "p": 0.3,
                    "q": 0.7,
                    "strength": -0.4,
                    **coupled_paths(0.3, 0.7, -0.4),
                }
            )
        out = score_questions(windows, pred)
        m = out["aggregate"]
        self.assertGreater(m["conditional"], m["independent"])
        self.assertEqual(m["conditional"], m["favor_different"])
        self.assertEqual(m["conditional"], m["shuffled_sign"])
        self.assertFalse(out["placement_criterion"])
        self.assertEqual(out["questions"], 12)
        self.assertFalse(orientation_support(pred)["can_beat_both_fixed_signs"])
        self.assertEqual(orientation_support(pred)["max_dialogues_beating_shuffled_sign"], 0)


class PrototypeTest(unittest.TestCase):
    def test_name_mask_uses_complete_names(self):
        self.assertEqual(
            mask_speakers("Ann met Joanne and JO's friend", {"speaker_a": "Ann", "speaker_b": "Jo"}),
            "the speaker met Joanne and the speaker's friend",
        )

    def test_unsigned_and_unsupported_queries_have_no_vote(self):
        x = torch.tensor([[0.0, -2.0], [1.0, 1.0], [1.0, 0.0]])
        v = prototype_logits(x, torch.zeros(2), torch.ones(2), torch.tensor([0.5, -0.5]))
        self.assertEqual(v[:2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(v[2]), 2 / 3, places=6)

    def test_temporal_first_and_activity_second(self):
        keys = {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}
        row = {"dialogue": "a", "start": 1, "questions": {42: {"f1": keys, "question": {"category": 2}}}}
        model = {
            "forecasts": [
                {"dialogue": "a", "session": i, "p_activity": p} for i, p in [(1, 0.2), (2, 0.8)]
            ],
            "fits": {"a": {"training_mean": 0.5}},
        }
        r = compare([row], model)
        self.assertAlmostEqual(r["aggregate"]["credit"], 0.64)
        self.assertAlmostEqual(r["aggregate"]["shuffled"], 0.34)
        self.assertEqual(r["aggregate"]["greedy"], 1.0)
        self.assertFalse(r["passes"])


class NeighborReadTest(unittest.TestCase):
    def test_same_index_changes_read_slots_without_crossing_sessions(self):
        from experiments.controller.neighbor_read import retrieve_slots

        ranking = [2, 3, 0, 5, 1, 6, 4]
        self.assertEqual(retrieve_slots(ranking, [3, 4], (0, 0), budget=4), [2, 3, 0, 5])
        self.assertEqual(retrieve_slots(ranking, [3, 4], (1, 1), budget=4), [2, 3, 4, 0])
        self.assertEqual(retrieve_slots(ranking, [3, 4], (0, 1), budget=1), [2])
        with self.assertRaisesRegex(ValueError, "complete"):
            retrieve_slots([0, 1, 1], [1, 2], (0, 1))

    def test_oracle_control_permutates_within_window(self):
        from experiments.controller.neighbor_read import summarize

        cells = {
            k: {"recall": v, "complete": bool(v), "words": 10}
            for k, v in (("00", 0.0), ("01", 1.0), ("10", 0.0), ("11", 0.0))
        }
        rows = [{"dialogue": sid, "start": 1, "type": "4", "cells": cells} for sid in "ABCDEF"]
        result = summarize(rows)
        self.assertEqual(result["aggregate"]["oracle"], 1.0)
        self.assertEqual(result["aggregate"]["same_count"], 0.5)
        self.assertTrue(result["advance_to_reader"])
