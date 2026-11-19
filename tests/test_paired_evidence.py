import hashlib
import json
import unittest
from types import SimpleNamespace

from experiments.controller.audit_paired import answer_change, development_decision, reader_inputs
from src.eval import get_evaluator


class ReaderEvidenceTest(unittest.TestCase):
    def test_shortening_an_already_present_reference_is_separate_from_recovering_it(self):
        answer = {"answer": "Python and C++.", "f1": 1., "ok": True}
        long = {"answer": "James has worked with Python and C++.", "f1": .5, "ok": True}
        missing = {"answer": "There is no mention of the languages.", "f1": 0., "ok": True}
        self.assertEqual(answer_change(long, answer, ["Python and C++"]), "credit_shorter_with_shared_reference")
        self.assertEqual(answer_change(answer, long, "Python and C++"), "control_shorter_with_shared_reference")
        self.assertEqual(answer_change(missing, answer, "Python and C++"), "credit_recovered_reference")
        self.assertEqual(answer_change(answer, missing, "Python and C++"), "control_recovered_reference")

    def test_selection_distinguishes_local_gains_from_a_global_rate_shift(self):
        gains = {"f1_change": 0.03, "control_to_credit_on_same_states": {"greedy_switches": 2},
                 "changed_question_evidence": [
                     {"f1_delta": 0.4, "reader_prompt_changed": True, "actions_changed_in_history": True},
                     {"f1_delta": -0.1, "reader_prompt_changed": True, "actions_changed_in_history": True}]}
        controls = {name: gains.copy() for name in ("native", "terminal", "always", "rate")}
        report = {"phase": "validation", "credit_minus": controls}
        self.assertTrue(development_decision(report)["advance"])
        controls["rate"]["control_to_credit_on_same_states"] = {"greedy_switches": 0}
        self.assertFalse(development_decision(report)["advance"])
        controls["rate"] = dict(gains, changed_question_evidence=[
            {"f1_delta": 0.5, "reader_prompt_changed": False, "actions_changed_in_history": True}])
        self.assertFalse(development_decision(report)["advance"])
        with self.assertRaises(ValueError):
            development_decision(dict(report, phase="test"))

    def test_ordered_versioned_context_and_question_identity(self):
        evaluator = get_evaluator("longmemeval", SimpleNamespace())
        qa = {"question": "Which team?", "question_date": "2023-05-20"}
        write = {"kind": "span", "span_id": 0, "refs": [], "retrieved": [], "changes": [
            {"memory_id": 0, "version": 1, "action": "INSERT", "before": None, "after": "First."},
            {"memory_id": 1, "version": 1, "action": "INSERT", "before": None, "after": "Second."}]}
        rewrite = {"kind": "span", "span_id": 1, "refs": [[0, 1]], "retrieved": ["First."], "changes": [
            {"memory_id": 0, "version": 2, "action": "UPDATE", "before": "First.", "after": "First."}]}
        snapshot = {"kind": "snapshot", "memory_refs": [[0, 2], [1, 1]],
                    "text_sha256": hashlib.sha256(json.dumps([[0, "First."], [1, "Second."]]).encode()).hexdigest()}
        row = {"qa_index": 0, "retrieved_indices": [1, 0], "tokens": [40, 2], "prediction": "Team A", "f1": 1.0}
        query = {"kind": "qa", "question": qa["question"], "success": True, "query_id": 0,
                 "refs": [[1, 1], [0, 2]], "tokens": [40, 2], "prediction": "Team A", "f1": 1.0}
        events = [write, rewrite, snapshot, query]
        prompt = evaluator.build_prompt(qa["question"], ["Second.", "First."], qa)
        self.assertEqual(reader_inputs(events, [row], {0: qa}, evaluator)[0],
                         (hashlib.sha256(prompt.encode()).hexdigest(), {0, 1}))
        for invalid in [dict(query, refs=[[0, 2], [1, 1]]), dict(query, question="When?")]:
            with self.assertRaises(ValueError):
                reader_inputs([write, rewrite, snapshot, invalid], [row], {0: qa}, evaluator)
        with self.assertRaises(ValueError):
            reader_inputs([write, rewrite, dict(snapshot, memory_refs=[[0, 1], [1, 1]]), query],
                          [row], {0: qa}, evaluator)
