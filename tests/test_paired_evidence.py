import hashlib
import json
import os
import unittest
import numpy as np
import torch
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from statistics import mean
from types import SimpleNamespace

from experiments.controller.audit_paired import answer_change, development_decision, reader_inputs
from experiments.controller.audit_duration_credit import duration_alias
from experiments.controller.audit_budget_policy import crossfit as crossfit_budget
from experiments.controller.audit_granularity_transfer import placement_ceiling
from experiments.controller.skill_source_budget import dialogue_recall, retained_turns
from experiments.controller.replay_validation_choices import verified_schedule
from experiments.controller.probe_omitted import check_responses, omitted_final_indices, probe_training_windows
from experiments.controller.audit_training_replays import probability_placement
from experiments.controller.granularity_placement import compare as compare_granularity, measured_return
from experiments.controller.transfer_grounding import external_histories
from experiments.controller.writer_granularity import missing_multiline_turns, source_variants
from src.data_processing.locomo import LoCoMoProcessor
from src.eval import get_evaluator


class LinkedQuestionTests(unittest.TestCase):
    def test_selection_requires_count_control_gain_when_present(self):
        good = {"f1_change": .02, "control_to_credit_on_same_states": {"greedy_switches": 1},
                "changed_question_evidence": [{"f1_delta": .2, "actions_changed_in_history": True,
                                               "reader_prompt_changed": True}]}
        report = {"phase": "validation", "credit_minus": dict.fromkeys(("native", "terminal", "always", "rate"), good)}
        self.assertTrue(development_decision(report)["advance"])
        report["credit_minus"]["count"] = dict(good, f1_change=0.)
        self.assertFalse(development_decision(report)["advance"])

    def test_all_question_coverage(self):
        from experiments.controller.paired_grounding import verify_linked_questions
        questions = [{"evidence": ["a", "b"], "original_index": 12}, {"evidence": ["b"], "original_index": 31}]
        events = [{"kind": "qa", "query_id": f"q_{i}", "success": True} for i in range(2)]
        for i, (turn, ids) in enumerate((("a", [0]), ("b", [0, 1]))):
            events.append({"kind": "span", "span_id": i, "source_turns": [turn]})
            rows = [{"qa_index": j, "original_index": questions[j]["original_index"]} for j in ids]
            events.append({"kind": "span_credit", "diagnosis": {"span_id": i, "evidence_rule_version": 2, "questions": rows}})
        verify_linked_questions(events, questions)
        events[-1]["diagnosis"]["questions"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            verify_linked_questions(events, questions)
        with self.assertRaisesRegex(ValueError, "factual"):
            verify_linked_questions(events[1:], questions)

    def test_evidence_forecasts_exclude_held_rewards(self):
        from experiments.controller.evidence_preference import dialogue_fits
        entries = [{"dialogue": sid, "session": k, "activity": 0., "temporal": gain,
                    "contrast": torch.tensor([feature], dtype=torch.float64)}
                   for sid in ("a", "b", "c") for k, gain, feature in ((1, .1, 1.), (2, 0., 0.))]
        before, fitted = dialogue_fits(entries)
        changed = [dict(r, temporal=-r["temporal"]) if r["dialogue"] == "a" else r for r in entries]
        after, _ = dialogue_fits(changed)
        self.assertEqual([r for r in before if r["dialogue"] == "a"], [r for r in after if r["dialogue"] == "a"])
        self.assertEqual(fitted["a"]["training_dialogues"], ["b", "c"])
        self.assertTrue(all(r["p"] == .5 for r in before if r["session"] == 2))
        self.assertTrue(all(r["p"] > .5 for r in before if r["session"] == 1))

    def test_serial_encoder_scope(self):
        from src import rag_utils as rag
        with TemporaryDirectory() as work, patch.dict(os.environ, {"MEMSKILL_QWEN_EMBEDDING_PATH": work}):
            model = SimpleNamespace(device=torch.device("cuda"))
            base = SimpleNamespace(model_name=work, model=model, _use_qwen_embedding=True,
                                   use_flash_attn=rag._RETRIEVER_USE_FLASH_ATTN)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with rag.reuse_qwen_retriever(base, "qwen3-embedding-0.6b"):
                    for load in (rag.init_context_model, rag.init_query_model):
                        self.assertIs(load("qwen3-embedding-0.6b")[1], model)
                    raise RuntimeError("simulated")
            self.assertIsNone(rag._serial_qwen.get())
            with self.assertRaises(ValueError):
                with rag.reuse_qwen_retriever(base, "contriever"):
                    pass


class ReaderEvidenceTest(unittest.TestCase):
    def test_held_reader_answers_do_not_change_that_dialogues_policy_probabilities(self):
        windows = {str(i): [{"start": 1, "state": [1., 0.], "n": i+1, "temporal": .8, "activity": .1},
                            {"start": 3, "state": [0., 1.], "n": i+2, "temporal": .1, "activity": .8}]
                   for i in range(6)}
        baseline = crossfit_budget(windows, "cpu")
        self.assertTrue(baseline["advance_to_full_controller"])
        for row in windows["0"]:
            row["temporal"], row["activity"] = row["activity"], row["temporal"]
        changed = crossfit_budget(windows, "cpu")
        def probabilities(report):
            return [r["temporal_probability"] for r in report["rows"] if r["dialogue"] == "0"]
        self.assertEqual(probabilities(changed), probabilities(baseline))
        self.assertLess(changed["folds"]["0"]["greedy"], baseline["folds"]["0"]["greedy"])

    def test_fixed_turn_budget_preserves_source_order_and_allows_only_distinct_turns(self):
        turns = [{"dia_id": str(i), "speaker": "A", "text": f"utterance {i}"} for i in range(3)]
        processor = LoCoMoProcessor(chunk_mode="full-session")
        picks = {"activity": ["2", "0"], "temporal": ["1", "2"]}
        result = retained_turns(turns, "May 4", picks, processor, 2)
        self.assertEqual(result["activity"], [f'May 4 | A said, "utterance {i}"' for i in (0, 2)])
        with self.assertRaisesRegex(ValueError, "turn budget"):
            retained_turns(turns, "May 4", dict(picks, temporal=["2", "2"]), processor, 2)
        entries = [{"covered_question_mass": {"activity": 1., "temporal": 0.}},
                   {"covered_question_mass": {"activity": 0., "temporal": 1.}}]
        recall = dialogue_recall(entries, 2)["recall"]
        self.assertEqual(recall["best_per_span"], 1.)
        self.assertEqual(recall["same_count_within_dialogue"], .5)

    def test_grouping_ceiling_chooses_once_for_all_questions_in_a_window(self):
        def row(pair, single, source):
            return {"dialogue": "one", "source_delta": source, "writer_delta": pair-single,
                    "writer_f1": {"pair": pair, "single": single}}
        scores = placement_ceiling({"first": [row(1., 0., 1.), row(0., .5, 1.)],
                                    "second": [row(0., 1., -1.)]})
        self.assertAlmostEqual(scores["f1"]["best_per_window"], 2/3)
        self.assertAlmostEqual(scores["f1"]["source_count_within_dialogue"], 1.25/3)
        self.assertEqual(scores["windows_with_exchangeable_choices"], 2)

    def test_live_probe_waits_for_publication_and_checks_original_backend(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, MEMSKILL_BACKEND_ID="gpu"):
            study = Path(tmp)
            episode = study / "train" / "example"
            trainer = SimpleNamespace(args=SimpleNamespace(api_base="unused"))
            windows = [({"sample_id": "example"}, episode)]
            probe_training_windows(trainer, study, windows, completed_only=True)
            progress, = study.rglob("progress.json")
            self.assertEqual(json.loads(progress.read_bytes())["pending_windows"], ["example"])
            self.assertFalse(progress.with_name("summary.json").exists())
            episode.mkdir(parents=True)
            (episode / "backend.json").write_text('{"inference": "different"}')
            with self.assertRaisesRegex(ValueError, "different inference backends"):
                probe_training_windows(trainer, study, windows, completed_only=True)

    def test_multiline_repair_preserves_the_omitted_speaker_date_and_image(self):
        with patch("src.data_processing.locomo.tiktoken.get_encoding"):
            processor = LoCoMoProcessor(chunk_mode="full-session")
        date = "10:12 am on 8 May, 2023"
        turns = [{"dia_id": "D1:1", "speaker": "Bo", "text": "A complete line."},
                 {"dia_id": "D1:2", "speaker": "Ada", "text": "We played trivia.\n",
                  "blip_caption": "a photo of a prize"}]
        source, = processor._extract_session_chunks(turns, date)
        self.assertEqual(missing_multiline_turns(processor, turns, date, source),
                         {"D1:2": date + " | Ada: We played trivia. [shared a photo of a prize.]"})
        with self.assertRaises(ValueError):
            missing_multiline_turns(processor, turns, date, source + "different")

    def test_split_only_the_source_pairs_actually_written(self):
        date = "8 May, 2023"
        pair = date + " | Bo: Your plan? Ada: Travel."
        written = "The writer summary."
        source = ('DATE: ' + date + '\nBo said, "Hi."\nAda said, "Hi."'
                  '\nBo said, "Your plan?"\nAda said, "Travel."')
        last = {"source": source, "parsed_actions": [{"success": True, "reasoning": "attributed_turn_pairs",
                "memory_content": pair}], "changes": [
                {"action": "INSERT", "version": 1, "memory_id": i, "before": None, "after": text}
                for i, text in enumerate([written, pair])], "refs": [], "retrieved": []}
        live, fixed, tails = source_variants([last])
        self.assertEqual(list(live.values()), [written, pair])
        self.assertEqual(fixed, [written])
        self.assertEqual(tails, {"pair": [pair], "single": [date + " | Bo: Your plan?", date + " | Ada: Travel."]})
        self.assertIsNone(source_variants([dict(last, parsed_actions=[])]))
        with self.assertRaises(ValueError):
            source_variants([dict(last, changes=last["changes"][::-1])])

    def test_equivalent_numeral_and_spelled_duration_is_separate_from_fact_recovery(self):
        correct = "Three years."
        self.assertTrue(duration_alias("Nate has had his first two turtles for 3 years.", correct, "three years"))
        self.assertTrue(duration_alias(correct, "3 years", "3 years"))
        for old in ("For 13 years.", "For 3.5 years.", "For 3 months.", "Not for 3 years.",
                    "For twenty-three years.", "For almost 3 years.", "He might keep them for 3 years."):
            self.assertFalse(duration_alias(old, correct, "three years"))
        self.assertFalse(duration_alias("For 3 years.", correct, "three years in Boston"))

    def test_conditional_placement_removes_the_probability_of_using_two_or_zero_pairs(self):
        logits = torch.tensor([1.5, -.3], dtype=torch.float64)
        scores = {"00": .2, "10": .8, "01": .1, "11": .4}
        result = measured_return(logits, scores)
        p, q = result["probabilities"]
        one = p*(1-q) + (1-p)*q
        self.assertAlmostEqual(result["given_one_f1"], (.8*p*(1-q) + .1*(1-p)*q) / one)
        self.assertAlmostEqual(result["unconditional_f1"], one*result["given_one_f1"] + .2*(1-p)*(1-q) + .4*p*q)
        windows = [{"dialogue": str(i), "start": j, "n": i+2, "state": [[v], [-v]],
                    "f1": dict(scores, **{"10": .8 if v > 0 else .1, "01": .1 if v > 0 else .8})}
                   for i in range(6) for j, v in ((1, 1.), (3, -1.))]
        for window in windows:
            window["f1"] = {key: np.float64(value) for key, value in window["f1"].items()}
        study = json.loads(json.dumps(compare_granularity(windows)))
        self.assertTrue(study["advance_to_full_writer_probes"])
        self.assertEqual(study["positive_conditional_folds_vs_uniform"], 6)
        self.assertAlmostEqual(study["aggregate"]["slot"]["given_one_f1"], .45)

    def test_external_validation_requires_separate_question_families_and_complete_histories(self):
        def sample(sid):
            return {"question_id": sid, "question_type": "temporal", "haystack_sessions": [[], []]}
        data = [sample(sid) for sid in ("a", "b_abs", "c", "a_abs", "c_abs")]
        splits = {"test": [2], "val": [0, 1, 3, 4]}
        formal = {"ids": ["c"], "history_sessions": 2}
        val = {"ids": ["a", "b_abs"], "abstentions": 1, "answerable_types": {"temporal": 1}, "history_sessions": 4}
        self.assertEqual(external_histories(data, splits, formal), [data[2]])
        self.assertEqual(external_histories(data, splits, formal, val), data[:2])
        self.assertEqual(external_histories(data, splits, formal, val, part="even"), [data[0]])
        self.assertEqual(external_histories(data, splits, formal, val, part="odd"), [data[1]])
        with self.assertRaisesRegex(ValueError, "external validation"):
            external_histories(data, splits, formal, part="even")
        for bad in (dict(val, ids=["a", "a_abs"]), dict(val, ids=["a", "c_abs"]),
                    dict(val, answerable_types={"multi": 1}), dict(val, history_sessions=3)):
            with self.assertRaises(ValueError):
                external_histories(data, splits, formal, bad)

    def test_probability_placement_preserves_dialogue_rates_and_weights_spans_equally(self):
        def link(dialogue, span, p, plain, grounded):
            return {"dialogue": dialogue, "span": span, "window": 1, "probabilities": {"actor": p},
                    "scores": {"plain": plain, "always": grounded}}
        a = link("a", 0, .9, .1, .8)
        b = link("a", 1, .2, .7, .2)
        c = link("a", 2, .5, .4, .4)
        d = link("b", 0, .3, 0., 1.)
        report = probability_placement([a, a, b, c, d])
        metrics = report["arms"]["actor"]
        def expected(ps):
            return mean(r["scores"]["plain"] + p * (r["scores"]["always"] - r["scores"]["plain"])
                        for r, p in zip((a, b, c, d), (*ps, .3), strict=True))
        self.assertEqual(report["spans"], 4)
        bounds = report["measured_action_upper_bound"]
        self.assertAlmostEqual(bounds["best_measured_action_f1"], (.8 + .7 + .4 + 1.) / 4)
        self.assertAlmostEqual(bounds["always_grounded_f1"], (.8 + .2 + .4 + 1.) / 4)
        self.assertEqual((bounds["grounded_better_spans"], bounds["plain_better_spans"], bounds["tied_spans"]), (2, 1, 1))
        self.assertAlmostEqual(metrics["expected_f1"], expected((.9, .2, .5)))
        self.assertAlmostEqual(metrics["permuted_expected_f1"], mean(expected(ps) for ps in permutations((.9, .2, .5))))
        self.assertAlmostEqual(report["dialogues"]["b"]["arms"]["actor"]["placement_delta"], 0.)
        with self.assertRaises(ValueError):
            probability_placement([a, dict(a, probabilities={"actor": .3})])

    def test_final_training_probe_requires_observed_unexamined_questions_linked_to_the_last_span(self):
        spans = [{"source_turns": ["D1:1"]}, {"source_turns": ["D2:1", "D2:2"]}]
        questions = [{"evidence": ev} for ev in (["D1:1"], ["D2:1"], ["D1:1", "D2:2"], ["D2:2"], [])]
        self.assertEqual(omitted_final_indices(spans, questions, {0, 1, 2, 4}, {"questions": [{"qa_index": 1}]}), [2])
        record = {"questions": [], "control": {"qa": {0: {"ok": True}}, "actions": [{"reasoning": "INSERT"}]}}
        check_responses(record)
        for invalid in (dict(record, swap={"qa": {0: {"ok": False}}}),
                        dict(record, execution_repair={"actions": [{"reasoning": "API call failed: timeout"}]}),
                        dict(record, questions=[{"retrieval_probe":{"reply":{"ok":True}, "repeat":{"ok":False}}}])):
            with self.assertRaises(RuntimeError):
                check_responses(invalid)

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
        controls["plain"] = dict(gains, f1_change=0.)
        self.assertFalse(development_decision(report)["advance"])
        controls["plain"] = gains.copy()
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

    def test_external_schedule_must_match_its_recorded_policy(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "decisions.json"
            row = {"use": True, "prob": .5, "base_logit": 0., "state": [1.]}
            path.write_text(json.dumps([row]))
            self.assertEqual(verified_schedule(directory, [0., 0.], 1), [True])
            for weights, steps in (([1., 0.], 1), ([0., 0.], 2)):
                with self.assertRaises(ValueError):
                    verified_schedule(directory, weights, steps)
            path.write_text(json.dumps([dict(row, use=False)]))
            with self.assertRaises(ValueError):
                verified_schedule(directory, [0., 0.], 1)
