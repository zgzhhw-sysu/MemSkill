import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.controller.paper_f1 import PROTOCOL, load_cohort, load_split_cohort, run_baseline, score_answers


class PaperF1Test(unittest.TestCase):
    def test_single_swap_changes_one_forward_and_rejects_prefix_drift(self):
        import torch
        from src.replay import swap_boundary_logits, verify_native_swap

        class Controller(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor_head = torch.nn.Identity()

            def forward(self, state, candidates):
                return self.actor_head(candidates.reshape(1, -1, 1)).flatten()

        controller = Controller()
        names = ["noop", "insert", "delete"]
        values = torch.tensor([3., 2., 1.])
        def events():
            rows = []
            for i in range(3):
                state = torch.tensor([float(i)])
                logits = controller(state, values)
                skills = [{"name": names[j], "update_type": names[j]} for j in logits.topk(2).indices.tolist()]
                rows.append({"sample": "train", "span_id": i, "source": str(i), "source_turns": [], "changes": [],
                             "retrieved": [], "refs": [], "selected_skills": skills, "skills": skills[1:],
                             "policy": {"state_sha256": hashlib.sha256(state.numpy().tobytes()).hexdigest(),
                                        "logits": logits.tolist(), "first_draw_probabilities": logits.softmax(0).tolist(),
                                        "candidates": names},
                             "executor_responses": [{"prompt_sha256": skills[-1]["name"]}]})
            return rows
        original = events()
        with swap_boundary_logits(controller, original, 1):
            changed = events()
        report = verify_native_swap(original, changed, 1)
        self.assertEqual((report["outgoing"], report["incoming"], report["suffix_action_changes"]), ("insert", "delete", 0))
        self.assertTrue(report["writer_prompt_changed"])
        self.assertEqual(events(), original)
        for index, key in [(0, "executor_responses"), (1, "refs"), (1, "skills")]:
            corrupt = copy.deepcopy(changed)
            corrupt[index][key].append("drift")
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_native_swap(original, corrupt, 1)
        wrong = copy.deepcopy(original)
        wrong[0]["policy"]["state_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "state differs"), swap_boundary_logits(controller, wrong, 1):
            events()
        self.assertEqual(events(), original)

    def test_independent_repeat_seeds_only_preceding_writes_and_checks_them_on_resume(self):
        from experiments.designer.paper_date_pair import seed_writer_prefix
        from src.evaluation_cache import publish_response, read_response

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "original-cache"
            isolated = base / "repeat" / "native-cache"
            keys = [str(i) * 64 for i in (1, 2, 3)]
            spans, usage = [], []
            for i, (key, kind) in enumerate(zip(keys, ("executor", "executor", "reader"), strict=True)):
                row = publish_response(source, key, f"prompt-{i}", f"response-{i}", 10, 5)
                usage.append({"kind": kind, "cache_key": key, "response_sha256": row["sha256"], "input": 10, "output": 5})
                if kind == "executor":
                    spans.append({"executor_responses": [{"prompt_sha256": f"prompt-{i}", "response": f"response-{i}",
                                                          "input_tokens": 10, "output_tokens": 5}]})
            (base / "usage.jsonl").write_text("\n".join(map(json.dumps, usage)))
            seed_writer_prefix(base, spans, 1, source, isolated)
            self.assertEqual(read_response(isolated, keys[0]), read_response(source, keys[0]))
            self.assertIsNone(read_response(isolated, keys[1]))
            self.assertIsNone(read_response(isolated, keys[2]))
            independent = publish_response(isolated, keys[1], "prompt-1", "different writer result", 10, 6)
            seed_writer_prefix(base, spans, 1, source, isolated)
            self.assertEqual(read_response(isolated, keys[1]), independent)
            with self.assertRaisesRegex(ValueError, "separate cache"):
                seed_writer_prefix(base, spans, 1, source, source)
            with self.assertRaisesRegex(ValueError, "recorded prefix"):
                seed_writer_prefix(base, spans, 0, source, isolated)
            spans[0]["executor_responses"][0]["response"] = "modified"
            with self.assertRaisesRegex(ValueError, "native response receipt"):
                seed_writer_prefix(base, spans, 1, source, isolated)

    def test_development_ids_are_separate_from_paper_test_ids(self):
        def key_hash(sid):
            return hashlib.sha256(json.dumps([[sid, 1]], separators=(",", ":")).encode()).hexdigest()

        data = [{"sample_id": sid, "qa": [{"category": 5}, {"category": 2}]} for sid in ("a", "b", "c")]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(data))
            spec = {"data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "excluded_category": 5,
                    "test_indices": [2], "scored_questions": 1, "scored_key_sha256": key_hash("c"),
                    "development_splits": {s: {"indices": [i], "scored_questions": 1, "scored_key_sha256": key_hash(sid)}
                                           for i, (s, sid) in enumerate((("train", "a"), ("validation", "b")))}}
            for split, sid in (("train", "a"), ("validation", "b"), ("test", "c")):
                samples, questions = load_split_cohort("locomo", path, spec, split)
                self.assertEqual([s["sample_id"] for s in samples], [sid])
                self.assertEqual(list(questions), [(sid, 1)])
            self.assertEqual(list(load_cohort("locomo", path, spec)), [("c", 1)])
            spec["development_splits"]["train"]["indices"] = [2]
            with self.assertRaisesRegex(ValueError, "locked cohort"):
                load_split_cohort("locomo", path, spec, "train")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_split_cohort("longmemeval", path, {}, "validation")

    def test_complete_history_shards_rescore_native_baseline(self):
        from experiments.designer.paper_date_pair import summarize_if_complete

        questions = {("a", 0): {"answer": "red, blue", "category": 1},
                     ("b", 0): {"answer": "run", "category": 4}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "baseline.json"
            for sid, prediction in (("a", "blue"), ("b", "running")):
                folder = root / "native" / sid
                folder.mkdir(parents=True)
                (folder / "scores.json").write_text(json.dumps({"answers": [
                    {"sample_id": sid, "qa_index": 0, "f1": 0, "prediction": prediction}]}))
                summarize_if_complete("locomo", out, questions, {}, ["native"], 20.59)
                self.assertEqual(out.exists(), sid == "b")
            result = json.loads(out.read_text())
            self.assertEqual(result["native_f1"], 75)
            self.assertEqual(result["f1_by_type"]["1"], {"questions": 1, "native": 50})
            self.assertAlmostEqual(result["difference_from_paper_points"], 54.41)
            summarize_if_complete("locomo", out, questions, {}, ["native"], None, split="train")
            training = json.loads(out.read_bytes())
            self.assertEqual(training["split"], "train")
            self.assertEqual(training["native_f1"], 75)
            self.assertNotIn("paper_f1", training)

    def test_native_scores_use_original_keys_and_references(self):
        questions = {("a", 2): {"answer": "red, blue", "category": 1},
                     ("a", 5): {"answer": "running; other", "category": 3}}
        rows = [{"sample_id": "a", "qa_index": 5, "prediction": "run", "f1": 0},
                {"sample_id": "a", "qa_index": 2, "prediction": "red", "f1": 0}]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.json"
            path.write_text(json.dumps({"answers": rows}))
            self.assertEqual(score_answers("locomo", questions, [path]), 75)
            for bad, message in [(rows[:1], "Missing"), (rows * 2, "Duplicate"),
                                 ([dict(rows[0], qa_index=3)], "Unexpected")]:
                with self.subTest(message=message):
                    path.write_text(json.dumps({"answers": bad}))
                    with self.assertRaisesRegex(ValueError, message):
                        score_answers("locomo", questions, [path])

    def test_longmemeval_zero_and_numeric_answers(self):
        questions = {("x", 0): {"answer": 42}, ("y", 0): {"answer": "July"}}
        with TemporaryDirectory() as tmp:
            paths = []
            for sample, prediction in [("x", "42"), ("y", "")]:
                path = Path(tmp) / sample
                path.write_text(json.dumps({"answers": [
                    {"sample_id": sample, "qa_index": 0, "prediction": prediction}]}))
                paths.append(path)
            self.assertEqual(score_answers("longmemeval", questions, paths), 50)

    @patch("experiments.controller.paper_f1.subprocess.run")
    def test_baseline_passes_paper_defaults_to_native_entry_point(self, run):
        defaults = json.loads(PROTOCOL.read_text())["inference_defaults"]
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "baseline"
            output = run_baseline("longmemeval", Path(tmp) / "data.json", dest,
                                  ["--model", "test-model"], defaults)
            argv = run.call_args.args[0]
            self.assertEqual(argv[1], "main.py")
            self.assertIn("--eval-only", argv)
            self.assertIn("--eval-f1-only", argv)
            for key, value in {"--action-top-k": "7", "--chunk-size": "512",
                               "--chunk-overlap": "64", "--mem-top-k-eval": "20",
                               "--out-file": str(output), "--model": "test-model"}.items():
                self.assertEqual(argv[argv.index(key) + 1], value)
