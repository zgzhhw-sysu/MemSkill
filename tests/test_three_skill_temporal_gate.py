import unittest
from unittest.mock import patch

import torch

from experiments.controller.three_skill_margin import margin_fit
from experiments.controller.three_skill_temporal_gate import (
    attribution,
    gate_targets,
    simplex,
    train,
    utilities,
)
from experiments.controller.writer_format_policy import fit


class TemporalGateTests(unittest.TestCase):
    def test_registration_entry_does_not_require_a_forecast_or_gpu(self):
        from experiments.controller import three_skill_temporal_gate as gate

        with (
            patch.object(gate, "registration", return_value="frozen") as reg,
            patch.object(gate, "gpu", side_effect=AssertionError),
        ):
            self.assertEqual(gate.run("study", register_only=True), "frozen")
            reg.assert_called_once_with("study")

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tensor(self, v):
        return torch.tensor(v, dtype=torch.float64, device=self.device)

    def test_edge_offsets_preserve_sign_when_margin_is_already_met(self):
        x = self.tensor([[1.0]])
        y = self.tensor([0.0])
        one = self.tensor([1.0])
        for c, offset, expected in (
            (0.2, 0.5, 0.0),
            (0.2, -0.4, 0.2),
            (-0.2, -0.5, 0.0),
            (-0.2, 0.4, -0.2),
        ):
            raw, b = margin_fit(
                x,
                y,
                one,
                self.tensor([c]),
                one,
                penalty=one,
                constraint_offset=self.tensor([offset]),
            )
            self.assertAlmostEqual(raw.item(), 0.0)
            self.assertAlmostEqual(b.item(), expected)

    def test_shared_temporal_evidence_and_conditional_only_evidence(self):
        pts = [
            {"index": 2, "scores": [0.0, 1.0, 0.5], "edges": [1.0, 0.5, -0.5]},
            {"index": 4, "scores": [0.1, 0.0, 1.0], "edges": [0.0, 0.0, 1.0]},
        ]
        value, edges, qs = gate_targets(pts)
        self.assertAlmostEqual(value, (0.75 + 0.4) / 2)
        self.assertEqual(edges, [0.25, 0.125])
        self.assertEqual([q["id"] for q in qs], [2])
        pts[1]["edges"][0] = 0.1
        with self.assertRaises(ValueError):
            gate_targets(pts)

    def test_nested_holdout_and_changed_temporal_probability(self):
        donors = [
            {
                "history": str(h),
                "node": f"{h}_s{i}",
                "features": [0.2 * h, 0.5 * i - 0.1 * h],
                "raw": 0.02 * i,
                "component_credit": [0.08 * i, 0.0],
                "gate_raw": 0.015 * i,
                "temporal_edges": [0.15 * i, -0.05 * i],
            }
            for h in range(4)
            for i in (-1, 1)
        ]
        outer = "3"
        remaining = [r for r in donors if r["history"] != outer]
        inners = {h: fit(remaining, h, self.device) for h in ("0", "1", "2")}
        m = train(donors, outer, inners, self.device)
        donors[-1].update(gate_raw=1e5, temporal_edges=[10.0, 10.0])
        self.assertEqual(m, train(donors, outer, inners, self.device))
        raw = utilities(
            m, {"raw": 0.01, "credit": 0.15, "rotated": -0.2}, [0.4, 0.7], self.device
        )
        p = simplex(self.tensor([0.6, 0.3, 0.1]), raw)
        self.assertGreater(abs(float(p["credit"][0] - p["raw"][0])), 1e-6)
        for v in p.values():
            self.assertAlmostEqual(v.sum().item(), 1.0)
        inners["0"] = {**inners["0"], "histories": ["0"]}
        with self.assertRaises(ValueError):
            train(donors, outer, inners, self.device)

    def test_two_order_contributions_match_expected_f1(self):
        a = [0.4, 0.3, 0.3]
        b = [0.2, 0.2, 0.6]
        y = [0.1, 1.0, 0.6]
        g, q = attribution(a, b, y)
        self.assertAlmostEqual(
            g + q, sum((p - r) * v for p, r, v in zip(b, a, y, strict=True))
        )
        self.assertAlmostEqual(attribution(a, [0.4, 0.1, 0.5], y)[0], 0.0)
        self.assertAlmostEqual(attribution(a, [0.2, 0.4, 0.4], y)[1], 0.0)
