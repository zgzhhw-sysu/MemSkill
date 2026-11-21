import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from experiments.controller.capped_source_repair import writer
from experiments.controller.legacy_terminal_symmetry import make_writer, ops
from experiments.controller.semantic_writer import sha
from experiments.controller.three_skill_margin import margin_fit
from experiments.controller.writer_format_audit import (
    interval_map,
    parameters,
    reader_inputs,
)
from experiments.controller.writer_format_history import consistent_credit
from experiments.controller.writer_format_policy import fit, probabilities, scores_at
from experiments.controller.writer_format_trial import (
    CELLS,
    PAIRS,
    SINGLE,
    effects,
    reparse,
    shares,
)
from src.episode_trace import skill_spec


class WriterFormatCreditTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float64)

    def test_marginal_effects_and_opposing_mechanisms(self):
        for v in itertools.product((0, 0.4, 1), repeat=4):
            r = effects(dict(zip(CELLS, v, strict=True)))
            self.assertAlmostEqual(r["template"] + r["grouping"], v[3] - v[0])
            self.assertAlmostEqual(sum(shares(r, r["raw"])), v[3] - v[0])
            self.assertEqual(shares(r, 0), [0, 0])
        r = effects(dict(zip(CELLS, [0.6, 0.1, 1.0, 0.4], strict=True)))
        self.assertGreater(r["template"], 0)
        self.assertLess(r["grouping"], 0)
        self.assertEqual(shares(r, r["raw"]), [0.0, r["raw"]])
        self.assertEqual(consistent_credit(-1, -0.8, -0.65), -0.65)
        self.assertEqual(consistent_credit(1, 0.8, -0.65), 0.0)
        with self.assertRaises(ValueError):
            shares(r, 0.2)

    def test_prompt_equivalence_bounds_are_tight_even_with_interaction(self):
        for ids in (["a", "a", "c", "d"], ["a", "b", "b", "d"], ["a", "b", "c", "d"]):
            prompts = dict(zip(CELLS, ids, strict=True))
            fixed = {CELLS[0]: 0.35, CELLS[-1]: 0.65}
            group, known = parameters(prompts, fixed)
            bounds = interval_map(
                torch.tensor([group], device=self.device), self.tensor([known])
            )[0]
            unknown = set(ids) - {ids[0], ids[-1]}
            possible = []
            for vals in itertools.product((0.0, 1.0), repeat=len(unknown)):
                answer = {
                    **dict(zip(unknown, vals, strict=True)),
                    ids[0]: 0.35,
                    ids[-1]: 0.65,
                }
                r = effects({k: answer[s] for k, s in prompts.items()})
                possible.append(
                    [
                        r[k]
                        for k in (
                            "template",
                            "grouping",
                            "grouping_at_activity",
                            "grouping_at_relation",
                            "interaction",
                        )
                    ]
                )
            outcomes = self.tensor(possible)
            torch.testing.assert_close(
                bounds, torch.stack((outcomes.amin(0), outcomes.amax(0)), dim=1)
            )
        with self.assertRaises(ValueError):
            parameters(dict.fromkeys(CELLS, "repeated"), {CELLS[0]: 0.0, CELLS[3]: 1.0})

    def test_failed_repeats_widen_bounds_and_preserve_current_measurements(self):
        first, last = CELLS[0], CELLS[-1]
        diag = {first: 0.0, last: 1.0}
        packet = {
            "arms": {k: {"hash": k, "old": {"f1": diag.get(k)}} for k in CELLS},
            "mismatch": [first],
        }

        def fixed(status, packet, response):
            _, values = reader_inputs(diag, status, packet, response)
            return {
                k: v for k, v in zip(CELLS, values, strict=True) if math.isfinite(v)
            }

        self.assertEqual(fixed("different_pure_bank", {}, {}), {})
        self.assertEqual(fixed("prepared", packet, {}), {last: 1.0})
        packet["mismatch"] = []
        failed = {"status": "different_pure_answer", "reads": {first: {"f1": 0.4}}}
        self.assertEqual(fixed("prepared", packet, failed), {first: 0.4})
        self.assertEqual(diag, {first: 0.0, last: 1.0})

    def test_reparse_uses_same_writer_but_different_source_items(self):
        source = 'DATE: 1:00 pm on 9 March, 2023\nCONVERSATION:\nAnn said, "I walked home."\nLee said, "I slept."'
        spec = skill_spec(
            SimpleNamespace(
                name="activity",
                description="A",
                instruction_template="Record activity",
                source_rule=PAIRS,
                update_type="insert",
            )
        )
        ex = make_writer({"model": "local", "writer_tokens": 32, "retriever": "unused"})
        req = ex._build_executor_prompt(ops([spec]), source, [])
        r = {
            "prompt_sha256": sha(req),
            "response": "ACTION: INSERT\nMEMORY_ITEM: The two people spoke.",
            "input_tokens": 20,
            "output_tokens": 8,
        }
        event = {"source": source, "retrieved": [], "executor_responses": [r]}
        pure = writer(ex, ops([spec]), event, repair=False)
        changed = reparse(ex, event, [spec], SINGLE, [r], pure)
        self.assertEqual(changed["actions"][0], pure[0])
        self.assertEqual(
            [a["reasoning"] for a in changed["actions"][1:]], [SINGLE, SINGLE]
        )
        self.assertEqual([a["reasoning"] for a in pure[1:]], [PAIRS])
        self.assertNotEqual(changed["writer_skills"][0]["version"], spec["version"])
        with self.assertRaises(ValueError):
            reparse(ex, event, [spec], SINGLE, [dict(r, prompt_sha256="wrong")], pure)

    def test_matching_raw_held_history_and_historical_ablation(self):
        rows = [
            {
                "node": f"{h}_{i}",
                "history": h,
                "features": [0.5 * i, 0.2 * int(h) + i],
                "raw": 0.02 * i - 0.03 * int(h),
                "component_credit": [0.0, 0.06 * i],
                "component_raw": [0.02 * i, -0.03 * int(h)],
            }
            for h in ("1", "2", "3")
            for i in (0, 1)
        ]
        fit0 = fit(rows, "3", self.device)
        t = self.tensor
        x = t(
            [
                [
                    1.0,
                    r["features"][0] / 2,
                    (r["features"][1] - fit0["center"]) / fit0["sd"],
                ]
                for r in rows[:4]
            ]
        )
        raw, _ = margin_fit(
            x,
            t([r["raw"] for r in rows[:4]]),
            t([0.25] * 4),
            t([0.0] * 4),
            t([0.0] * 4),
        )
        target = [0.2, 0.1]
        self.assertAlmostEqual(
            scores_at(fit0, target, self.device)["raw"],
            float(raw @ t([1.0, 0.1, (0.1 - fit0["center"]) / fit0["sd"]])),
        )
        hist = [
            {**r, "node": "old" + r["node"], "raw": -0.5, "credit": -0.65} for r in rows
        ]
        m = fit(rows, "3", self.device, history=hist)
        for r in (rows[-1], hist[-1]):
            r["features"] = [3e5, -1e5]
            r["raw"] = 20
        self.assertEqual(m, fit(rows, "3", self.device, history=hist))
        ps = probabilities(
            t([0.4, 0.5, 0.1]), t(0.41), scores_at(m, target, self.device)
        )
        self.assertGreater(float((ps["credit"] - ps["zero_history"]).abs().sum()), 1e-8)
        for v in ps.values():
            self.assertAlmostEqual(float(v[0]), 0.41)
            self.assertAlmostEqual(float(v.sum()), 1.0)
        dense = fit(rows, "3", self.device, dense=True)
        self.assertNotEqual(dense["coef"]["raw"], fit0["coef"]["raw"])
