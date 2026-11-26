import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.optimize import linprog

from experiments.controller.audit_cross_window import verified_decision
from experiments.controller.credit_rank_allocation import (
    allocate,
    apply_allocation,
    fit,
    permutation,
)
from experiments.controller.cross_window_prefix import trajectory_hashes


class CreditRankAllocationTests(unittest.TestCase):
    def tensor(self, x):
        return torch.tensor(x, dtype=torch.float64)

    def test_two_thresholds_attain_independent_linear_program_with_tied_scores(self):
        score, base, value, w = map(
            self.tensor,
            (
                [0, 0, 1, 2, 3, 4],
                [-0.9, -0.4, 0, -1, -0.6, -1.3],
                [0.3, -0.6, 0.2, -0.1, 0.8, 0.7],
                [1, 3, 2, 4, 2, 1],
            ),
        )
        rule = allocate(score, base, value, w)
        p = apply_allocation(score, base, rule)
        lo, hi = (base - 2).sigmoid(), (base + 2).sigmoid()
        native = base.sigmoid()
        weight = w / w.sum()
        keys, ix = score.unique(return_inverse=True)
        widths = np.bincount(ix, weights=(weight * (hi - lo)).numpy())
        rewards = np.bincount(ix, weights=(weight * (hi - lo) * value).numpy())
        n = len(keys)
        ordered = np.eye(n)[:-1] - np.eye(n)[1:]
        sol = linprog(
            -rewards,
            A_ub=ordered,
            b_ub=np.zeros(n - 1),
            A_eq=[widths],
            b_eq=[float(weight @ (native - lo))],
            bounds=(0, 1),
            method="highs",
        )
        self.assertTrue(sol.success)
        optimum = float(weight @ ((lo - native) * value)) - sol.fun
        self.assertGreater(optimum, 0.01)
        self.assertAlmostEqual(rule["training_gain"], optimum, places=10)
        self.assertAlmostEqual(
            float(weight @ ((p - native) * value)), optimum, places=10
        )
        self.assertAlmostEqual(float(weight @ (p - native)), 0.0, places=12)
        fraction = (p - lo) / (hi - lo)
        self.assertAlmostEqual(float(fraction[0]), float(fraction[1]), places=12)
        self.assertTrue(torch.all(torch.diff(fraction) >= -1e-12))

    def test_uninformative_or_inverted_order_preserves_native(self):
        base = self.tensor([-1, -1, -1])
        y = self.tensor([1, 0.3, -1])
        for score in (self.tensor([0, 0, 0]), self.tensor([0, 1, 2])):
            rule = allocate(score, base, y, torch.ones_like(y))
            self.assertTrue(rule["native"])
            self.assertTrue(
                torch.equal(apply_allocation(score, base, rule), base.sigmoid())
            )

    def test_nested_scores_exclude_both_complete_histories_and_permute_within_history(
        self,
    ):
        rows = [
            {
                "dialogue": h,
                "window": h + str(i),
                "stage": 0,
                "state": [1.0, i * 0.25, k * 0.1],
                "base": -1.3,
                "raw": 0.2 * (i - k),
                "credit": 0.1 * (i + 1),
                "weight": 0.5,
            }
            for k, h in enumerate("ABCD")
            for i in (0, 1)
        ]
        shuffled = permutation(rows)
        self.assertEqual([r["shuffled"] for r in shuffled], [0.2, 0.1] * 4)
        _, first = fit(shuffled, "A", 0, "cpu")
        self.assertEqual(
            first["inner_histories"],
            {"B": ["C", "D"], "C": ["B", "D"], "D": ["B", "C"]},
        )
        modified = permutation(
            [
                dict(r, raw=20.0, credit=-14.0) if r["dialogue"] in ("A", "B") else r
                for r in rows
            ]
        )
        _, second = fit(modified, "A", 0, "cpu")
        for a, b in zip(first["calibration"], second["calibration"], strict=True):
            self.assertNotEqual(a["history"], "A")
            if a["history"] == "B":
                self.assertEqual(a["scores"], b["scores"])

    def test_long_prefix_requires_the_executed_ordered_three_skills(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {"dialogue": "a", "end": 2, "held": True, "background": [1]}
            pair = dict(spec, trajectories={})
            versions = ["a", "b", "t", "x"]
            for action in (0, 1):
                cell = root / f"1{action}"
                (cell / "traces").mkdir(parents=True)
                choices = [
                    {
                        "use": i,
                        "plain": [0, 1, 2],
                        "grounded": [0, 1, 3],
                        "state": [0.0, 1.0],
                        "base_logit": -0.5,
                    }
                    for i in (1, action)
                ]
                spans = [
                    {
                        "kind": "span",
                        "source": "same",
                        "source_turns": [0],
                        "retrieved": [],
                        "refs": [],
                        "span_id": j,
                        "selected_skills": [
                            {"version": versions[k]}
                            for k in ([0, 1, 3] if d["use"] else [0, 1, 2])
                        ],
                    }
                    for j, d in enumerate(choices)
                ]
                (cell / "traces/log.jsonl").write_text(
                    "\n".join(map(json.dumps, spans))
                )
                (cell / "decisions.json").write_text(json.dumps(choices))
                (cell / "usage.jsonl").write_text("")
                (cell / "scores.json").write_text("{}")
                pair["trajectories"][cell.name] = trajectory_hashes(cell)
            dec = verified_decision(root, spec, pair, versions)
            self.assertEqual(dec["plain"], [0, 1, 2])
            with self.assertRaisesRegex(ValueError, "fixed three"):
                verified_decision(root, spec, pair, ["b", "a", "t", "x"])

    def test_adjacent_float_scores_keep_distinct_decisions(self):
        a = self.tensor(1.0)
        x = torch.stack((a, torch.nextafter(a, a + 1)))
        z = self.tensor([-1.0, -1.0])
        rule = allocate(x, z, self.tensor([-1.0, 1.0]), torch.ones_like(x))
        p = apply_allocation(x, z, rule)
        self.assertGreater(float(p[1] - p[0]), 0.1)
