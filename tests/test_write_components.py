import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from experiments.controller.component_policy import relabel
from experiments.controller.factorial_rollout import record_scored_paths
from experiments.controller.write_components import (
    contributions,
    crossed_banks,
    retain_credit,
    trace_exposure,
)
from src.credit_assignment import _digest


class WriteComponentsTest(unittest.TestCase):
    def test_composition_keeps_prefix_and_duplicate_contents(self):
        def branch(source, tag):
            parts = [("No reasoning provided", "same"), (tag, source)]
            return {
                "actions": [
                    {"action_type": "INSERT", "success": True, "reasoning": r, "memory_content": t}
                    for r, t in parts
                ],
                "bank_hash": _digest(["same", "same", source]),
            }

        branches = [branch("T", "dated_single_turns"), branch("A", "dated_adjacent_turns")]
        banks = crossed_banks(["same"], branches)
        self.assertEqual(banks["01"], ["same", "same", "A"])
        self.assertEqual(banks["10"], ["same", "same", "T"])
        branches[1]["bank_hash"] = "changed"
        with self.assertRaises(ValueError):
            crossed_banks(["same"], branches)

    def test_overwrite_delete_and_never_read_are_distinct(self):
        def write(mid, v, before, after, action):
            return dict(memory_id=mid, version=v, before=before, after=after, action=action)

        def span(i, changes):
            return dict(kind="span", span_id=i, refs=[], retrieved=[], changes=changes)

        events = [
            span(0, [write(i, 1, None, t, "INSERT") for i, t in enumerate(["a", "b", "c"])]),
            span(1, [write(0, 2, "a", "z", "UPDATE"), write(1, 2, "b", None, "DELETE")]),
            dict(kind="snapshot", memory_refs=[[0, 2], [2, 1]], text_sha256=_digest([(0, "z"), (2, "c")])),
            dict(
                kind="qa", query_id="x_0", question="q", success=True, prediction="x", f1=1.0, refs=[[0, 2]]
            ),
        ]
        qa = {7: dict(question="q", answer="x")}
        ev = SimpleNamespace(
            compute_f1=lambda a, b, q: float(a == b), get_ground_truth=lambda q: q["answer"]
        )
        queries, stats = trace_exposure(events, [7], qa, ev)
        self.assertEqual(queries[0]["context_hash"], _digest(["z"]))
        self.assertEqual(
            (
                stats[0]["overwritten_or_deleted"],
                stats[0]["survived_unread"],
                stats[1]["read_by_any_question"],
            ),
            (2, 1, 1),
        )
        events[-1]["refs"] = [[0, 1]]
        with self.assertRaises(ValueError):
            trace_exposure(events, [7], qa, ev)

    def test_additive_decomposition_and_credit_bound(self):
        c = contributions({"00": 0.1, "01": 0.0, "10": 0.6, "11": 0.9})
        self.assertAlmostEqual(c["source"], 0.1)
        self.assertAlmostEqual(c["generated"], 0.7)
        self.assertAlmostEqual(retain_credit(0.8, c["source"], True), 0.1)
        self.assertEqual(retain_credit(-0.5, c["source"], True), 0.0)
        self.assertEqual(retain_credit(0.8, c["source"], False), 0.0)
        self.assertEqual(retain_credit(-0.1, -0.9, True), -0.1)

    def test_only_final_eligible_credit_changes_and_all_linked_questions_count(self):
        first = {"span": 0, "episode": "demo", "dialogue": "D", "signed": -0.3}
        last = {
            "span": 1,
            "episode": "demo",
            "dialogue": "D",
            "credits": {1: 1.0, 2: 0.0, 3: 0.0},
            "signed": 1 / 3,
        }
        f1 = {"00": 0.0, "01": 0.6, "10": 0.4, "11": 1.0}
        row = dict(
            episode="demo",
            dialogue="D",
            original_index=1,
            f1=f1,
            strict_credit=1.0,
            retained=0.6,
            endpoints_reproduced=True,
            **contributions(f1),
        )
        result = {"manifest": {"backend": "test"}, "rows": [row]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "answers").mkdir()
            (path / "answers/demo-q1.json").write_text(
                json.dumps(
                    {"manifest": result["manifest"], "answers": {k: {"f1": v} for k, v in f1.items()}}
                )
            )
            filtered, first_only, support = relabel([first, last], result, path)
            self.assertEqual(support, ["D"])
            self.assertIs(filtered[0], first)
            self.assertEqual(first_only[1]["signed"], 0.0)
            self.assertAlmostEqual(filtered[1]["signed"], 0.2)
            row["strict_credit"] = 0.5
            with self.assertRaises(ValueError):
                relabel([first, last], result, path)

    def test_even_one_scored_path_excludes_a_window_and_is_bound_to_its_contents(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            locked = root / "frozen.json"
            locked.write_text("{}")
            cutoff = locked.stat().st_mtime_ns
            for window, seconds in (("early", -1), ("late", 1)):
                p = root / "cells" / window / "01" / "scores.json"
                p.parent.mkdir(parents=True)
                p.write_text("[]")
                os.utime(p, ns=(cutoff, cutoff + seconds * 10**9))
            self.assertEqual(record_scored_paths(root / "cells", locked), {"early"})
            (root / "cells/early/01/scores.json").write_text("[0]")
            with self.assertRaises(ValueError):
                record_scored_paths(root / "cells", locked)
