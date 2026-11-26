import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.controller.resume_status import check_equal_source_choice, load_episode


class ResumeEpisodeTest(unittest.TestCase):
    def test_both_equal_source_branches_refer_to_the_frozen_versions(self):
        bank = [{"name": "shared1"}, {"name": "shared2"},
                {"name": "capture_activity_details", "version": "pair", "source_rule": "dated_adjacent_turns"},
                {"name": "capture_temporal_context", "version": "single", "source_rule": "dated_single_turns"}]
        row = {"grounded": [0, 1, 2], "plain": [0, 1, 3], "use": False}
        selected = [bank[i] for i in row["plain"]]
        swapped = [bank[i] for i in row["grounded"]]
        check_equal_source_choice(row, bank, selected, swapped)
        with self.assertRaisesRegex(ValueError, "frozen bank"):
            check_equal_source_choice(row, bank, selected, swapped[:2] + [dict(bank[2], version="other")])
        with self.assertRaisesRegex(ValueError, "third slot"):
            check_equal_source_choice(dict(row, plain=[1, 0, 3]), bank, selected, swapped)

    def test_resume_requires_matching_ids_diagnosis_and_sampled_actions(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "conv-26_s1"
            (directory / "traces").mkdir(parents=True)
            (directory / "questions.json").write_text("[7]")
            (directory / "usage.jsonl").touch()
            row = {"episode": directory.name, "span": 0, "use": False, "plain": [2, 1, 0],
                   "grounded": [2, 1, 3], "delta": -0.25, "prob": 0.5, "state": [0.3],
                   "old_log_probability": math.log(0.5), "terminal": 0.5, "base_logit": -1.0}
            events = [{"kind": "span", "span_id": 0, "candidate_versions": ["a", "b", "c", "d"]},
                      {"kind": "snapshot"},
                      {"kind": "span_credit", "diagnosis": {"span_id": 0, "mean_controller_delta": -0.25,
                       "sampled_skills": [{"version": v} for v in ("c", "b", "a")]}}]
            (directory / "traces" / "episode.jsonl").write_text(
                "".join(json.dumps(dict(e, sample="conv-26")) + "\n" for e in events))
            path = directory / "rows.json"
            path.write_text(json.dumps([row]))
            self.assertEqual(load_episode(directory, "conv-26", 1, [7]), [row])
            with self.assertRaisesRegex(ValueError, "Writer protocol differs"):
                load_episode(directory, "conv-26", 1, [7], deployment_writer=True)
            with self.assertRaises(ValueError):
                load_episode(directory, "conv-26", 1, [3])
            selected = [dict(version=v, name=n, update_type=t) for v,n,t in
                        (("c","insert","INSERT"), ("b","noop","NOOP"), ("a","capture","INSERT"))]
            effective = [selected[0], selected[2]]
            events[0].update(selected_skills=selected, skills=effective)
            diagnosis = events[2]["diagnosis"]
            diagnosis.update(sampled_skills=selected, control={"writer_skills": effective})
            trace = directory / "traces" / "episode.jsonl"
            trace.write_text("".join(json.dumps(dict(e, sample="conv-26")) + "\n" for e in events))
            self.assertEqual(load_episode(directory, "conv-26", 1, [7], deployment_writer=True), [row])
            with self.assertRaisesRegex(ValueError, "Writer protocol differs"):
                load_episode(directory, "conv-26", 1, [7], deployment_writer=False)
            diagnosis["control"]["writer_skills"] = selected
            trace.write_text("".join(json.dumps(dict(e, sample="conv-26")) + "\n" for e in events))
            with self.assertRaisesRegex(ValueError, "in control"):
                load_episode(directory, "conv-26", 1, [7], deployment_writer=True)
            diagnosis["control"]["writer_skills"] = effective
            trace.write_text("".join(json.dumps(dict(e, sample="conv-26")) + "\n" for e in events))
            for wrong in [[], [dict(row, use=True)], [dict(row, delta=0.25)], [dict(row, episode="conv-26_s3")]]:
                with self.subTest(wrong=wrong):
                    path.write_text(json.dumps(wrong))
                    with self.assertRaises(ValueError):
                        load_episode(directory, "conv-26", 1)
