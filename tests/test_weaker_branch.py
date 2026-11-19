import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.controller.probe_weaker import check_weaker_span, matches_recorded_swap
from src.credit_assignment import _digest


class WeakerBranchTest(unittest.TestCase):
    def test_repair_of_losing_alternative_vetoes_positive_credit(self):
        before = {"ok": True, "answer": "unknown", "f1": 0., "tokens": [12, 2],
                  "indices": [0], "context_hash": _digest(["old"])}
        weak = {"actions": ["same writer"], "suffix": [], "bank_hash": _digest(["old"]),
                "contract_fault": None, "qa": {0: before}, "contents": ["old"]}
        stored = dict(weak, qa={"0": before})
        improved = dict(before, answer="the violin is his instrument", f1=.4, context_hash=_digest(["quote"]))
        repair = dict(weak, qa={0: improved}, bank_hash=_digest(["old", "quote"]))
        diagnosis = {"span_id": 0, "alternative": [{"update_type": "INSERT"}], "swap": stored, "swap_repeat": stored,
                     "questions": [{"qa_index": 0, "original_index": 18, "controller_delta": 1.,
                                    "control_matched": True, "original_f1": 1.}]}
        convo = {"qa": [{"question": "Which instrument?", "evidence": ["D1:1"], "original_index": 18}], "conversation": {}}
        trainer = SimpleNamespace(args=SimpleNamespace(retriever="fixed"), evaluator=SimpleNamespace(get_ground_truth=lambda _: "violin"))
        turns = [{"dia_id": "D1:1", "text": "violin", "session_date": "Monday", "speaker": "Jon"}]
        def probe(*branches):
            with patch("experiments.controller.probe_weaker._write_branch", side_effect=branches), \
                    patch("experiments.controller.probe_weaker.get_embeddings", return_value=[[0]]), \
                    patch("experiments.controller.probe_weaker.evidence_turns", return_value=turns), \
                    patch("experiments.controller.probe_weaker.check_responses"):
                return check_weaker_span(trainer, convo, [{"source_turns": ["D1:1"]}], diagnosis)["questions"][0]
        self.assertEqual(probe(weak.copy(), repair, repair)["competing"], ["execution"])
        self.assertEqual(probe(weak.copy(), weak.copy())["status"], "survived_probes")
        self.assertEqual(probe(weak.copy(), repair, dict(repair, bank_hash="different"))["status"], "unresolved")
        self.assertEqual(probe(dict(weak, bank_hash="different"))["status"], "unreproduced")
        self.assertFalse(matches_recorded_swap(dict(weak, contract_fault="not applied"), stored))
