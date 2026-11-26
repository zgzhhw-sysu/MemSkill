import logging
import unittest

import torch

from experiments.controller.terminal_credit_envelope import (
    apply_bounds,
    continuation_bounds,
    ridge_weights,
)
from experiments.controller.writer_block_repair import (
    HEADER,
    bounded_credit,
    parsed_repair,
)
from src.executor import ExecutionResult, Executor


class WriterBlockRepairTests(unittest.TestCase):
    def test_only_standalone_heading_before_its_own_explicit_action_is_removed(self):
        source = (
            "ACTION: INSERT\nMEMORY_ITEM: They discussed the INSERT block: yesterday.\n"
            "INSERT block:\nACTION: UPDATE\nMEMORY_INDEX: 0\nUPDATED_MEMORY: Keep this text.\n"
            "\tdelete block:\r\n\r\nACTION: DELETE\r\nMEMORY_INDEX: 0\nDELETE block:"
        )
        self.assertEqual(
            HEADER.sub("", source), source.replace("\tdelete block:\r\n", "")
        )

    def test_reparse_preserves_all_native_source_inserts_and_other_fields(self):
        executor = Executor.__new__(Executor)
        executor.logger = logging.getLogger(__name__)
        raw = "INSERT block:\nACTION: INSERT\nMEMORY_ITEM: first\n\nINSERT block:\nACTION: INSERT\nMEMORY_ITEM: second"
        actions = [vars(a) for a in executor._parse_response(raw, 0)]
        source = vars(
            ExecutionResult(
                "INSERT",
                True,
                memory_content="DATE | original",
                reasoning="dated_single_turns",
            )
        )
        event = {
            "parsed_actions": [*actions, source],
            "refs": [],
            "executor_responses": [{"response": raw}],
        }
        fixed, n = parsed_repair(executor, event)
        self.assertEqual(n, 1)
        self.assertEqual(
            [a["memory_content"] for a in fixed], ["first", "second", "DATE | original"]
        )
        self.assertEqual(fixed[-1], source)
        self.assertIn("INSERT block:", event["parsed_actions"][0]["memory_content"])
        with self.assertRaises(ValueError):
            parsed_repair(
                executor, dict(event, parsed_actions=event["parsed_actions"][1:])
            )

    def test_weak_recovery_vetoes_credit_in_either_direction(self):
        no = {"answer": "unspecified", "ok": True, "f1": 0.0}
        yes = {"answer": "red bicycle", "ok": True, "f1": 1.0}
        partial = {"answer": "red", "ok": True, "f1": 0.5}
        for sign in (-1, 1):

            def order(lo, hi, direction=sign):
                return [lo, hi] if direction == 1 else [hi, lo]

            self.assertEqual(
                bounded_credit(sign, order(no, yes), order(yes, yes), "red bicycle"),
                (0.0, "same_skill_recovery"),
            )
            self.assertEqual(
                bounded_credit(sign, order(no, yes), order(no, no), "red bicycle"),
                (0.0, "recovery_lost"),
            )
            self.assertEqual(
                bounded_credit(
                    sign, order(no, yes), order(partial, yes), "red bicycle"
                ),
                (0.5 * sign, "retained"),
            )


class CreditRangeTests(unittest.TestCase):
    def test_root_bonus_extremum_on_diagonal_inside_probability_box(self):
        p = torch.tensor([[[0.2, 0.8]], [[0.4, 0.6]]], dtype=torch.float64)
        bounds = continuation_bounds([0.0, 0.0, 0.0, 0.0], [1.0, 1.0], p)
        torch.testing.assert_close(bounds, p.new_tensor([[0.6, 1.0]]))

    def test_affine_interval_reverses_ends_for_negative_influence(self):
        bounds = torch.tensor([[[2.0, 4.0]], [[0.25, 1.0]]], dtype=torch.float64)
        result = apply_bounds(bounds.new_tensor([[-2.0, 1.0]]), bounds)
        torch.testing.assert_close(result, bounds.new_tensor([[[-7.75, -3.0]]]))

    def test_effective_ridge_weights_reproduce_centered_point_predictions(self):
        x = torch.tensor([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]], dtype=torch.float64)
        y = x.new_tensor([1.0, -1.0, 2.0])
        weights = ridge_weights(x, x)
        z = (x[:, 0] - 2.0) / x[:, 0].std(unbiased=False)
        fitted = y.mean() + z * (z @ (y - y.mean())) / (z @ z + len(y))
        torch.testing.assert_close(weights @ y, fitted)
