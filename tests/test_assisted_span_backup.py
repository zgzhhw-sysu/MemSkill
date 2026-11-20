import unittest

import torch

from experiments.controller.assisted_span_backup import (
    AssistedTransport,
    adjustment,
    cell_table,
    estimates,
    outcome_baseline,
)
from experiments.controller.assisted_trial import promote
from experiments.controller.evidence_transport import ARMS, Transport, root_targets
from experiments.controller.masked_span_policy import mask


def full(sid, value=0.7):
    return {
        "sid": sid,
        "ids": [0, 1],
        "types": [1, 2],
        "links": [([0], []), ([], [0])],
        "base": [0.0, -0.1, 0.2],
        "pivot": None,
        "y": {
            "00": [0.1, value],
            "01": [0.7, 0.1],
            "10": [value, 0.4],
            "11": [0.2, 0.8],
        },
        "root": [[0.1, -0.2], [0.0, 0.1]],
        "last": [[0.0, 0.1], [-0.1, 0.1]],
    }


class AssistedBackupTests(unittest.TestCase):
    def setUp(self):
        self.w = full("held")
        self.p = torch.tensor(
            [[0.2, 0.8, 0.4, 0.7, 0.9], [0.7, 0.1, 0.6, 0.2, 0.5]], dtype=torch.float64
        )

    def test_arbitrary_predictions_preserve_the_full_target_in_expectation(self):
        m = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.7], [0.2, 0.3]]], dtype=torch.float64
        )
        measured = cell_table(self.w, "cpu")
        z = [estimates(mask(self.w, b), self.p, m, measured) for b in (0, 1)]
        expected = root_targets(self.w, self.p, "cpu")
        for arm in ("paired", "difficulty", "assisted", "resolved"):
            self.assertTrue(
                torch.allclose(
                    (z[0][arm] + z[1][arm]) / 2, expected, atol=1e-12, rtol=0
                )
            )
        raw = ARMS.index("raw")
        self.assertTrue(torch.allclose(z[0]["resolved"][:, raw], expected[:, raw]))
        self.assertTrue(torch.allclose(z[1]["resolved"][:, raw], expected[:, raw]))
        self.assertFalse(torch.allclose(z[0]["assisted"], z[1]["assisted"]))

    def test_constant_half_reproduces_old_backup_even_for_equal_continuation(self):
        table = torch.full((2, 2, 2), 0.5, dtype=torch.float64)
        record = mask(self.w, 1)
        self.assertTrue(
            torch.equal(adjustment(table, self.p, 1), torch.zeros(2, len(ARMS)))
        )
        observed = estimates(record, self.p, table)
        self.assertTrue(
            torch.equal(observed["paired"], root_targets(record, self.p, "cpu"))
        )
        self.assertTrue(torch.equal(observed["paired"], observed["assisted"]))

    def test_baseline_is_shrunk_and_excludes_both_histories(self):
        anchors = {sid: full(sid, i / 5) for i, sid in enumerate(("a", "b", "c", "d"))}
        excluded = {"a", "b"}
        m = outcome_baseline(anchors, excluded, anchors["b"], "cpu")
        pool = torch.cat([cell_table(anchors[s], "cpu") for s in ("c", "d")])
        self.assertTrue(
            torch.allclose(m[0], (pool[[0, 2]].sum(0) + 8 * pool.mean(0)) / 10)
        )
        anchors.update(a=full("a", 1.0), b=full("b", 0.0))
        self.assertTrue(
            torch.equal(m, outcome_baseline(anchors, excluded, anchors["b"], "cpu"))
        )
        target = dict(anchors["b"], types=[5, 5])
        self.assertTrue(
            torch.equal(
                outcome_baseline(anchors, excluded, target, "cpu")[0], pool.mean(0)
            )
        )
        with self.assertRaises(ValueError):
            outcome_baseline(anchors, {"a"}, target, "cpu")

    def test_paired_forecast_matches_base_and_target_outcomes_cannot_change_policy(
        self,
    ):
        anchors = {s + "_s5": full(s, (i + 1) / 5) for i, s in enumerate("abcd")}
        extra = {s + "_s7": mask(full(s, i / 5), i % 2) for i, s in enumerate("abcd")}
        turns = {
            s + "_s" + str(k): (["t"], torch.ones(1, 1), "src", "file")
            for s in "abcd"
            for k in (5, 6, 7, 8)
        }

        def run(physical=None):
            learner = AssistedTransport(turns, anchors, extra, "cpu", physical=physical)
            return learner.forecast("d_s7")

        result = run()
        base = Transport(turns, anchors | extra, "cpu").forecast("d_s7")
        for j, row in enumerate(result["offsets"]):
            self.assertAlmostEqual(
                row["paired"], base["offsets"][j]["credit"], places=12
            )
            self.assertEqual(row["assisted"], row["credit"])
            if j:
                self.assertEqual(row["difficulty"], row["credit"])
        anchors["d_s5"] = full("d", 0.0)
        extra["d_s7"] = mask(full("d", 1.0), 0)
        self.assertEqual(result, run())
        physical = {n: cell_table(full(w["sid"]), "cpu") for n, w in extra.items()}
        resolved = run(physical)
        for j, row in enumerate(resolved["offsets"]):
            self.assertEqual(row["assisted"], result["offsets"][j]["credit"])
            self.assertEqual(row["paired"], result["offsets"][j]["paired"])

    def test_gate_requires_improvement_over_paired_and_prospective_returns(self):
        f = {
            k: 0.4
            for k in ("paired", "native", "terminal", "raw", "sham", "blind", "rate")
        }
        f["credit"] = 0.45
        overall = {
            "questions": 378,
            "f1": f,
            "ht": {"credit": 0.45, "native": 0.4},
            "by_dialogue": {s: {"f1": f} for s in "abcdef"},
        }
        fresh = {"questions": 20, "f1": f}
        self.assertTrue(promote(overall, fresh))
        f["paired"] = 0.45
        self.assertFalse(promote(overall, fresh))
        f["paired"] = 0.4
        self.assertFalse(promote(overall, {"questions": 0}))
