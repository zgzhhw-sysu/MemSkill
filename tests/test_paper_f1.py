import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.controller.paper_f1 import PROTOCOL, run_baseline, score_answers


class PaperF1Test(unittest.TestCase):
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
