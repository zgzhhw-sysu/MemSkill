import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.controller import factorial_repairs
from experiments.controller.temporal_update_audit import (
    contribution,
    mixed_banks,
    summarize,
)
from src.memory_bank import MemoryBank


class TemporalUpdateAuditTests(unittest.TestCase):
    def test_cross_handles_deleted_prefix_items_and_overlapping_suffix_ids(self):
        parts = [[(4, 1, "old"), (6, 1, "left")], [(3, 2, "changed"), (6, 1, "right")]]
        banks = []
        for part in parts:
            bank = MemoryBank(top_k=1)
            for n, (_, _, t) in enumerate(part):
                bank.add_memory(t, np.eye(2)[n])
            banks.append(bank)
        crossed, refs = mixed_banks(banks, parts, {3, 4}, 1, 0)
        self.assertEqual(crossed.get_all_contents(), ["changed", "left"])
        self.assertEqual(refs, [["update", 3, 2], ["temporal", 6, 1]])
        self.assertEqual(crossed.retrieve(np.array([0, 1])), (["left"], [1]))
        self.assertEqual(banks[0].get_memory_at(1).access_count, 0)

    def test_joint_gain_and_probability_shift_are_counted_once(self):
        effects = contribution(0, 0.25, 0.5, 1)
        self.assertEqual(
            effects, {"old": 0.375, "new": 0.625, "interaction": 0.25, "total": 1}
        )
        report = {
            "status": "matched",
            "questions": [8],
            "rows": [{"effects": effects, "probe": None}],
            "touched_old": [3],
            "choice": {"source": 0.75, "native": 0.25},
        }
        summary = summarize(
            [report, {"status": "ineligible_fixed_slots", "questions": [3]}]
        )
        self.assertEqual(
            summary["source_minus_native"], {"old": 0.1875, "new": 0.3125, "total": 0.5}
        )
        self.assertEqual(summary["registered_questions_arrived"], 2)
        self.assertEqual(summary["matched_questions"], 1)
        self.assertEqual(summary["positive"], 0)

    def test_old_evidence_allows_retrieval_veto_when_focal_write_has_no_quote(self):
        q = {"question": "Where?", "evidence": ["D1:1"]}
        convo = {
            "qa": [q],
            "conversation": {
                "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "place"}]
            },
        }
        step = {
            "skills": [{"update_type": "insert"}],
            "parsed_actions": [{"action_type": "INSERT", "success": True}],
            "applied": True,
            "source_turns": ["D2:2"],
        }
        row = {"qa_index": 0, "background": 0, "recovery_sign": 1, "delta": 1}
        trainer = SimpleNamespace(
            config=SimpleNamespace(mem_top_k_eval=1),
            evaluator=SimpleNamespace(get_ground_truth=lambda q: "Paris"),
        )
        cells, banks = {}, {}
        for i, text in enumerate(["Rome", "Paris"]):
            bank = MemoryBank(top_k=1)
            for j, t in enumerate([text, "Paris is the city"]):
                bank.add_memory(t, np.eye(2)[j])
            a = {"answer": text, "tokens": [4, 1], "ok": True, "f1": float(i)}
            cells[0, i] = {"spans": [step, step], "scores": {0: dict(a, indices=[0])}}
            banks[0, i] = bank

        def read(trainer, q, ctx):
            return {
                "answer": ctx[0],
                "tokens": [4, 1],
                "ok": True,
                "f1": float("Paris" in ctx[0]),
            }

        with patch.object(factorial_repairs, "_answer", read):
            result = factorial_repairs.final_probe(
                trainer,
                convo,
                row,
                cells,
                banks,
                np.eye(2)[0],
                probe_if_write_unavailable=True,
            )
            old = factorial_repairs.final_probe(
                trainer, convo, row, cells, banks, np.eye(2)[0]
            )
        self.assertEqual(old["status"], "unavailable_repair")
        self.assertEqual(
            (result["status"], result["competing"], result["signed_effect"]),
            ("competing_recovery", ["retrieval"], 0),
        )
