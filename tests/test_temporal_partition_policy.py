import math
import unittest

import torch

from experiments.controller.temporal_partition_policy import (
    donor_example,
    features,
    panel,
    predict,
)


class TemporalPartitionPolicyTests(unittest.TestCase):
    def test_distinct_effects_use_distinct_state_features_and_exclude_target_history(
        self,
    ):
        windows = [{"name": n, "dialogue": n} for n in ("a", "b", "c")]
        states = {"a": [0.0] * 6, "b": [0.0, 0.0, 0.0, 9.0, 9.0, 9.0], "c": [9.0] * 6}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target = windows[0]
        sample = {
            "name": "b",
            "dialogue": "b",
            "old": 0.1,
            "new": -0.1,
            "total": 0.0,
            "bonus": 0.0,
        }
        y = predict(target, windows, states, [sample], device)
        self.assertEqual(y["pooled_return"], 0.5)
        self.assertGreater(y["credit"], 0.5)
        self.assertEqual(y["credit"], y["split_return"])
        outside = {
            "name": "a",
            "dialogue": "a",
            "old": 1.0,
            "new": 1.0,
            "total": 2.0,
            "bonus": 1.0,
        }
        self.assertEqual(y, predict(target, windows, states, [sample, outside], device))
        self.assertTrue(
            all(
                p == 0.5
                for p in predict(target, windows, states, [outside], device).values()
            )
        )
        aligned = dict(states, a=states["b"])
        self.assertGreater(
            predict(target, windows, aligned, [dict(sample, bonus=0.05)], device)[
                "credit"
            ],
            predict(target, windows, aligned, [sample], device)["credit"],
        )

    def test_source_features_and_structurally_zero_extra_credit(self):
        v = torch.eye(2)
        x = features(v, v, [{"text": "Last week?"}, {"text": "Thanks."}])
        self.assertEqual(x[:3], [1.0, 1.0, 0.5])
        self.assertAlmostEqual(x[3], math.log(4))
        self.assertEqual(x[4:], [0.0, 0.5])
        w = {"name": "small", "dialogue": "a", "questions": [3]}
        r = dict(
            w,
            status="matched",
            rows=[
                {
                    "index": 3,
                    "effects": {"old": 0.0, "new": 0.5, "total": 0.5},
                    "probe": {"signed_effect": 0.0, "status": "unavailable_repair"},
                }
            ],
        )
        self.assertEqual(donor_example(w, r)["bonus"], 0.0)
        r["rows"][0]["probe"]["signed_effect"] = 0.5
        with self.assertRaises(ValueError):
            donor_example(w, r)

    def test_rate_and_costs_use_same_windows_and_all_questions(self):
        def window(s, p, end, n):
            choices = {"credit": p, "native": 0.25, "split_return": p}
            row = {
                "temporal": {"f1": 0.0, "tokens": [8, 2]},
                "update": {"f1": end, "tokens": [8, 4]},
                "type": 1,
            }
            return {
                "p": choices,
                "writer_tokens": [10, 20],
                "dialogue": s,
                "rows": [row] * n,
            }

        out = panel([window("a", 1.0, 1.0, 2), window("b", 0.0, 0.0, 1)])
        self.assertEqual(out["questions"], 3)
        self.assertAlmostEqual(out["f1"]["credit"], 2 / 3)
        self.assertAlmostEqual(out["f1"]["rate"], 1 / 3)
        self.assertEqual(out["writer_tokens_per_window"]["rate"], 15)
        self.assertEqual(out["reader_tokens_per_question"]["rate"], 11)
        self.assertEqual(out["by_history"]["b"]["questions"], 1)
