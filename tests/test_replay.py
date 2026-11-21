import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from src.episode_trace import EpisodeTrace, calendar_proposal, confirmed_proposal, reference_match, reference_recovered, skill_spec
from src.designer import Designer, DesignerCase
from src.executor import Executor
from src.operation_bank import Operation
from src.memory_bank import MemoryBank
from src.replay import anchor_relative_time, restore_text
from src.revision_validation import check_exposure, compare_revisions


class ReplayTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"), "requires CUDA and Triton")
    def test_skylakex_cuda_search_preserves_duplicate_memory_versions(self):
        rng = np.random.default_rng(31)
        query = rng.standard_normal(1024, dtype=np.float32)
        vector = rng.standard_normal(1024, dtype=np.float32)
        bank = MemoryBank()
        for i in range(7):
            bank.add_memory(str(i), vector, state_encoder_embedding=vector)
        for use_state in (False, True):
            # Measured on OpenBLAS 0.3.29 SkylakeX. Rows 4 and 5 score higher
            # than the identical first four. Normalization also lifts the final row.
            groups = np.array([0, 0, 0, 0, 1, 1, int(use_state)], dtype=np.float32)
            expected = np.argsort(groups)[::-1][:3].tolist()
            with self.subTest(use_state=use_state), patch.dict(os.environ, MEMSKILL_MEMORY_SEARCH="cuda-skylakex"):
                with patch.object(np, "dot", side_effect=AssertionError("unexpected CPU dot product")):
                    contents, ids = bank.retrieve(query, top_k=3, use_state_encoder=use_state)
                self.assertEqual(ids, expected)
                self.assertEqual(contents, [str(i) for i in expected])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA memory ranking requires a GPU")
    def test_cuda_search_preserves_native_ids_for_tied_memories(self):
        bank = MemoryBank()
        for index in range(40):
            vector = np.array([1, index % 3], dtype=np.float32)
            bank.add_memory(str(index), vector, state_encoder_embedding=vector)
        for use_state in (False, True):
            for k in (0, 1, 20, 40):
                with self.subTest(use_state=use_state, k=k):
                    options = dict(top_k=k, use_state_encoder=use_state, return_embeddings=True)
                    with patch.dict(os.environ, MEMSKILL_MEMORY_SEARCH="numpy"):
                        expected = bank.retrieve([1, 1], **options)
                    with patch.dict(os.environ, MEMSKILL_MEMORY_SEARCH="cuda"):
                        actual = bank.retrieve([1, 1], **options)
                    self.assertEqual(actual[:2], expected[:2])
                    np.testing.assert_array_equal(actual[2], expected[2])

    def test_executor_records_the_response_used_for_a_write(self):
        args = SimpleNamespace(retriever="unused", model="local", api_base="local", api_key="test",
                               max_new_tokens=32, temperature=0)
        op = SimpleNamespace(name="insert", description="d", update_type="insert", instruction_template="store")
        answer = "ACTION: INSERT\nMEMORY_ITEM: John went to work."
        audit = []
        with patch("src.executor.get_llm_response_via_api", return_value=(answer, 23, 9)) as endpoint:
            results = Executor(args).execute_operation([op], "source", [], audit=audit)
        self.assertEqual(endpoint.call_count, 1)
        self.assertEqual(audit[0]["response"], answer)
        self.assertEqual((audit[0]["input_tokens"], audit[0]["output_tokens"]), (23, 9))
        self.assertEqual(results[0].memory_content, "John went to work.")

    def test_dated_source_rule_preserves_unambiguous_turn_only_for_selected_insert(self):
        op = Operation("capture_time", "d", "capture", "insert")
        baseline = skill_spec(op)["version"]
        args = SimpleNamespace(retriever="unused", model="local", api_base="local", api_key="test",
                               max_new_tokens=32, temperature=0)
        source = 'DATE: 1:00 pm on 1 January, 2023\nAli said, "I arrived yesterday."\nBo said, "I travel a lot."'
        executor = Executor(args)
        with patch("src.executor.get_llm_response_via_api", return_value=("ACTION: NOOP", 11, 5)):
            self.assertEqual(len(executor.execute_operation([op], source, [])), 1)
            op.source_rule = "dated_relative_turns"
            restored = Operation.from_dict(op.to_dict())
            self.assertNotEqual(baseline, skill_spec(restored)["version"])
            actions = executor.execute_operation([restored], source, [])
            self.assertEqual(len(actions), 2)
            self.assertEqual(actions[1].memory_content, "1:00 pm on 1 January, 2023 | Ali: I arrived yesterday (31 December 2022).")
            self.assertEqual(len(executor.execute_operation([restored], source, [actions[1].memory_content])), 1)

    def test_versioned_write_and_shifted_index(self):
        emb = np.array([1.0, 0.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            bank = MemoryBank()
            trace = bank.trace = EpisodeTrace(Path(d), "sample")
            trace.begin(0, "source", [], [], [])
            for content in ("first", "second"):
                bank.add_memory(content, emb)
            trace.finish([], True)
            trace.begin(1, "next", [], ["second", "first"], [1, 0])
            bank.update_memory(1, "revised", emb)
            bank.delete_memory(0)
            trace.finish([], True)
            trace.begin(2, "last", [], ["revised"], [0])
            bank.add_memory("new", emb)
            trace.finish([], True)
            trace.verify(bank)
            self.assertEqual(trace.ref(0), [1, 2])
            self.assertEqual(restore_text(trace.events), {1: "revised", 2: "new"})
            corrupted = copy.deepcopy(trace.events)
            corrupted[2]["refs"][0][1] = 1
            with self.assertRaises(ValueError):
                restore_text(corrupted)

    def test_replay_preserves_date_precision(self):
        turn = {"session_date": "6:18 pm on 1 January, 2023", "text": "Yesterday, last year and last spring."}
        self.assertEqual(anchor_relative_time(turn), "Yesterday (31 December 2022), last year (2022) and last spring.")

    def test_revision_is_bound_to_probed_version(self):
        op = SimpleNamespace(name="temporal", description="d", update_type="insert", instruction_template="old")
        bank = SimpleNamespace(get_all_operations=lambda: [op])
        record = {"selected_skills": [(op.name, skill_spec(op)["version"])],
                  "reader_probes": {"control": {"ok": True, "f1": 0}, "probes": [
                      {"type": "dated_speaker_source", "ok": True, "f1": 0},
                      {"type": "calendar_expansion", "ok": True, "answer": "Yesterday (19 January 2023)", "f1": 1}]}}
        case = SimpleNamespace(query_id="sample_1", ground_truth="19 January, 2023", replay=record)
        self.assertEqual(calendar_proposal(bank, [case])["action"], "refine_existing")
        op.instruction_template = "another version"
        self.assertEqual(calendar_proposal(bank, [case])["action"], "no_change")

    def test_designer_withholds_stale_replay(self):
        op = SimpleNamespace(name="insert", description="d", update_type="insert", instruction_template="old")
        bank = SimpleNamespace(operations={op.name: op})
        case = DesignerCase("q", "question", "answer", replay={"skill": op.name,
                            "skill_version": skill_spec(op)["version"], "finding": "suffix_rule_edit_helped"})
        designer = object.__new__(Designer)
        self.assertIn("suffix_rule_edit_helped", designer._format_failure_cases_details([case], bank))
        op.instruction_template = "revised"
        prompt = designer._format_failure_cases_details([case], bank)
        self.assertIn("older skill version", prompt)
        self.assertNotIn("suffix_rule_edit_helped", prompt)

    def test_confirmed_proposal_uses_exact_intervention_and_downstream_effect(self):
        op = SimpleNamespace(name="insert", description="d", update_type="insert", instruction_template="old")
        bank = SimpleNamespace(operations={op.name: op})
        record = {"scope_kind": "local", "unchanged_write_reproduced": True, "skill": op.name,
                  "skill_version": skill_spec(op)["version"], "intervention": " tested words", "f1_delta": .5,
                  "paired_answers": {"unchanged": "unknown", "edited": "museum"}}
        case = SimpleNamespace(query_id="q1", ground_truth="museum", replay=record)
        self.assertEqual(confirmed_proposal(bank, [case])["action"], "no_change")
        record["scope_kind"] = "suffix"
        proposal = confirmed_proposal(bank, [case])
        changes = proposal["changes"][0]["refined_operation"]["changes"]
        self.assertEqual(changes, {"instruction_template": "old tested words"})
        regression = SimpleNamespace(query_id="q2", ground_truth="bridge", replay=dict(record, f1_delta=-1))
        self.assertEqual(confirmed_proposal(bank, [case, regression])["action"], "no_change")

    def test_wrong_day_and_partial_abstention_do_not_confirm_a_revision(self):
        self.assertTrue(reference_match("May 7, 2023.", "7 May 2023"))
        self.assertFalse(reference_match("May 8, 2023, at 1:56 pm.", "7 May 2023"))
        self.assertFalse(reference_match("There is no mention of self-care.", "self-care is important"))
        self.assertFalse(reference_recovered("John is a shooting guard for the Wolves.", "Shooting guard.", "shooting guard"))
        self.assertFalse(reference_recovered('The bird is the "eagle".', "Eagles.", "Eagles"))

    def test_reject_regression_and_check_question_identity(self):
        def report(values):
            return {"answers": [{"sample_id": s, "qa_index": 0, "f1": np.float64(v)} for s, v in values]}
        control = report([("a", 0), ("b", .5)])
        result = compare_revisions(control, report([("a", 1), ("b", .25)]))
        self.assertFalse(result["accept"])
        self.assertEqual(json.loads(json.dumps(result))["worse"], 1)
        with self.assertRaises(ValueError):
            compare_revisions(control, report([("a", 1), ("a", .5)]))

    def test_positive_score_cannot_promote_an_unexecuted_version(self):
        old = Operation("time", "d", "old", "insert")
        revised = Operation("time", "d", "revision", "insert")
        original_bank = SimpleNamespace(get_all_operations=lambda: [old])
        revised_bank = SimpleNamespace(get_all_operations=lambda: [revised])
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d) / "episode.jsonl"
            event = {"kind": "span", "sample": "s", "span_id": 0, "skills": [skill_spec(old)]}
            journal.write_text(json.dumps(event))
            self.assertFalse(check_exposure(original_bank, revised_bank, d, {"accept": True})["accept"])
            event["skills"] = [skill_spec(revised)]
            journal.write_text(json.dumps(event))
            self.assertTrue(check_exposure(original_bank, revised_bank, d, {"accept": True})["accept"])


if __name__ == "__main__":
    unittest.main()
