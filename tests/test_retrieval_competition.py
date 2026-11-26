import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.controller.audit_competition import diagnostics
from experiments.controller.retrieval_competition import (
    ARMS,
    learn,
    mass,
    probabilities,
    simulation,
)
from experiments.controller.root_write_repairs import check_root, replay


class CompetitionTests(unittest.TestCase):
    def test_two_stage_oracle_matches_root_and_both_conditional_usage_counts(self):
        from experiments.controller.two_stage_capacity import capacity

        native = torch.full((4, 3), 0.3, dtype=torch.float64)
        values = native.new_tensor(
            [
                [[0.9, 0.1], [0.2, 0.4]],
                [[0.1, 0.2], [0.9, 0.7]],
                [[0.4, 0.9], [0.1, 0.2]],
                [[0.2, 0.1], [0.4, 0.9]],
            ]
        )
        out = capacity(native, values, native.new_ones(4))
        self.assertTrue(
            torch.allclose(
                torch.tensor(out["native_expected_uses"]),
                torch.tensor(out["witness_expected_uses"]),
            )
        )
        self.assertGreater(out["feasible_both_stages"], out["native"] + 0.02)
        self.assertLessEqual(out["feasible_both_stages"], out["relaxed_upper"])

    def test_overlap_credit_is_zero_for_disjoint_continuations_and_is_not_renormalized(
        self,
    ):
        from experiments.controller.retrieval_competition import common_suffix_effect

        self.assertEqual(common_suffix_effect((0.0, 1.0), (1.0, 1.0)), 0.0)
        self.assertAlmostEqual(common_suffix_effect((0.2, 0.8), (1.0, 1.0)), 0.4)
        self.assertAlmostEqual(common_suffix_effect((0.7, 0.7), (-1.0, 1.0)), 0.4)

    def test_null_earlier_evidence_reproduces_original_two_stage_fit(self):
        states = [{"features": [0.0, 1.0, 0.0, 1.0], "base": -1.0}] * 3
        rows = [
            {
                "dialogue": d,
                "window": d,
                "states": states,
                "values": [0.2, 0.5, 0.6, 0.4],
                "signed": [0.05, 0.0],
            }
            for d in ("a", "b", "c")
        ]
        prior = learn(rows, True)
        self.assertEqual(
            learn(rows, True, root_effects={d: [0.0, 0.0] for d in ("a", "b", "c")}),
            prior,
        )
        changed = learn(
            rows, True, root_effects={d: [-0.4, -0.2] for d in ("a", "b", "c")}
        )
        self.assertEqual(changed["terminal"], prior["terminal"])
        self.assertLess(changed["root"]["intercept"], prior["root"]["intercept"])

    def test_changed_root_cannot_change_paired_focal_return_at_terminal_pivot(self):
        from experiments.controller.root_overlap_credit import ARMS as ROOT_ARMS
        from experiments.controller.root_overlap_credit import score

        ps = {
            0: dict.fromkeys(ROOT_ARMS, 0.5),
            1: dict.fromkeys(ROOT_ARMS, 0.5),
            2: dict.fromkeys(ROOT_ARMS, 0.5),
        }
        ps[0]["overlap"] = 0.8
        prior = {k: 0.5 for k in ("credit", "return_only", "terminal", "native")}
        original = {
            "dialogue": "a",
            "window": "a_s9",
            "rows": [{"index": 11, "type": 2, "f1": prior, "ht": prior}],
        }
        out = score(
            original,
            ps,
            {"10": [0.0], "11": [1.0]},
            ["10", "11"],
            {b: {"reader": 4, "executor": 9} for b in ("10", "11")},
            1,
            1,
        )
        self.assertAlmostEqual(out["rows"][0]["f1"]["overlap"], 0.5)
        self.assertAlmostEqual(out["rows"][0]["ht"]["overlap"], 0.8)

    def test_known_constant_quality_removes_random_mass_variation(self):
        from experiments.controller.paired_design_variance import estimates

        states = {
            j: {"credit": 0.85, "native": 0.4, "rate": 0.6, "return_only": 0.2}
            for j in range(3)
        }
        measured = {b: 0.65 for b in ("00", "01", "10", "11")}
        est = estimates(states, measured, 0.65)
        self.assertGreater(
            max(x for x, _ in est["credit"]) - min(x for x, _ in est["credit"]), 0.1
        )
        for _, adjusted in est["credit"]:
            self.assertAlmostEqual(adjusted, 0.65)
        for _, adjusted in est["credit-native"]:
            self.assertAlmostEqual(adjusted, 0.0)

    def test_top_five_can_displace_existing_memories_and_ignores_unpaired_padding(self):
        mem = ["old " + str(i) for i in range(5)]
        vectors = {s: torch.tensor([0.1, 0.1]) for s in mem}
        vectors.update({"new": torch.tensor([1.0, 1.0])})
        q = torch.ones((6, 2))
        x = simulation(
            q,
            vectors,
            mem,
            [("new", [(0, "one")])],
            torch.tensor([0]),
            torch.tensor([-1]),
        )
        self.assertTrue(torch.allclose(x, torch.tensor([1.0, 0.0, 0.8, 0.09])))

    def test_rule_two_zero_is_identical_to_return_only_and_zero_return_keeps_native(
        self,
    ):
        states = [{"features": [0.0, 1.0, 0.0, 1.0], "base": -1.0}] * 3
        rows = [
            {"dialogue": s, "states": states, "values": [0.6] * 4, "signed": [0.0, 0.0]}
            for s in ("a", "b", "c")
        ]
        full = learn(rows, True)
        raw = learn(rows, False)
        self.assertEqual(full, raw)
        p = probabilities(states[0], 0, {"credit": full, "return_only": raw}, 0.4)
        self.assertAlmostEqual(p["native"], p["credit"])

    def test_root_paths_use_the_observed_continuation(self):
        ps = {0: {"x": 0.3}, 1: {"x": 0.1}, 2: {"x": 0.9}}
        self.assertAlmostEqual(mass("01", ps, "x"), 0.07)
        self.assertAlmostEqual(mass("11", ps, "x"), 0.27)
        self.assertAlmostEqual(
            sum(mass(b, ps, "x") for b in ("00", "01", "10", "11")), 1.0
        )

    def test_costs_use_physical_inclusion_probability_only_in_randomized_windows(self):
        ps = {j: dict.fromkeys(ARMS, 0.5) for j in range(3)}
        costs = {"reader": 12, "executor": 6}
        row = dict.fromkeys(ARMS, 0.4)
        arm = {"rows": [{"f1": row, "ht": row}] * 2, "probabilities": ps}
        for partial in (False, True):
            bits = ("00", "01") if partial else ("00", "01", "10", "11")
            w = dict(arm, tokens=dict.fromkeys(bits, costs))
            if partial:
                w["focal_probabilities"] = ps[1]
            self.assertEqual(
                diagnostics([w])["costs"]["credit"],
                {"writer_tokens_per_window": 6.0, "reader_tokens_per_question": 6.0},
            )

    def test_first_span_repair_preserves_original_actions_and_reruns_suffix(self):
        events = [
            {
                "parsed_actions": [
                    {
                        "action_type": "INSERT",
                        "success": True,
                        "memory_content": "original",
                    }
                ]
            },
            {},
        ]
        with patch(
            "experiments.controller.root_write_repairs.rebuild_suffix",
            return_value=(None, 1),
        ) as rebuild:
            bank, record = replay(SimpleNamespace(), events, ["added"])
        self.assertIsNone(bank)
        self.assertEqual(record["failed"], 1)
        self.assertEqual(
            [a.memory_content for a in rebuild.call_args.args[-1]],
            ["original", "added"],
        )
        self.assertEqual(len(events[0]["parsed_actions"]), 1)
        self.assertIn("observe_suffix", rebuild.call_args.kwargs)

    def test_changed_suffix_skill_cannot_yield_earlier_credit(self):
        a = [{"selected_skills": ["s", "t", "temporal"]}, {"selected_skills": ["x"]}]
        b = [{"selected_skills": ["s", "t", "activity"]}, {"selected_skills": ["y"]}]
        out = check_root(
            None,
            {"qa": [{}]},
            {"qa_index": 0, "background": 0, "recovery_sign": 1},
            {(0, 0): {"spans": a}, (1, 0): {"spans": b}},
            None,
        )
        self.assertEqual(out["status"], "different_fixed_context")
        self.assertEqual(out["signed_effect"], 0)
