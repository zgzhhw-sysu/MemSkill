import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from experiments.controller import legacy_root_replay as root
from experiments.controller import matched_root_credit
from experiments.controller.relation_prefix_interaction import combine, metrics
from experiments.controller.relation_skill_repairs import candidates
from experiments.controller.source_compression import file_hash
from src.credit_assignment import _digest


def scores(temporal, relation):
    return {"temporal": temporal, "relation": relation}


class PrefixContrastTests(unittest.TestCase):
    def test_ceiling_chooses_once_per_state_and_weights_each_question_once(self):
        # The third question would prefer the opposite action to the first two.
        a = [scores(1, 0), scores(0, 1)]
        b = [scores(0, 1), scores(1, 0)]
        first = metrics([a, a, b])
        self.assertTrue(first["strict_flip"])
        self.assertAlmostEqual(first["conditional_oracle"], 2 / 3)
        self.assertAlmostEqual(first["shared_oracle"], 0.5)
        second = metrics([[scores(1, 1), scores(1, 1)]])
        self.assertFalse(second["strict_flip"])
        numeric_tie = metrics([[scores(0, 1), scores(0.5, 0.5 - 1e-16)]])
        self.assertFalse(numeric_tie["strict_flip"])
        total = combine(
            [
                dict(first, matched=True, same_prefix=False),
                dict(second, matched=True, same_prefix=True),
                {"matched": False},
            ]
        )
        self.assertEqual(
            (total["attempted"], total["matched"], total["questions"]), (3, 2, 4)
        )
        self.assertAlmostEqual(total["oracle_gain"], 0.125)
        self.assertAlmostEqual(total["interaction"], 0.5)

    def test_only_wrong_to_complete_answers_linked_to_focal_span_qualify(self):
        qa = [
            {
                "question": "Where?",
                "answer": "New York",
                "category": 1,
                "original_index": i,
                "evidence": [turn],
            }
            for i, turn in enumerate(("s2", "s1", "s2"))
        ]
        low = {"answer": "Boston", "ok": True, "f1": 0.0}
        high = {"answer": "New York", "ok": True, "f1": 1.0}
        rows = [
            {"original_index": i, "category": 1, "relation": l, "temporal": h}
            for i, (l, h) in enumerate(
                ((low, high), (high, low), (low, dict(high, answer="York", f1=0.5)))
            )
        ]
        linked, jobs = candidates(qa, rows, ["s2"])
        self.assertEqual(linked, 2)
        self.assertEqual(
            [(j["question_index"], j["weak"], j["signed_credit"]) for j in jobs],
            [(0, "swap", -1.0)],
        )

    def test_changed_suffix_receipt_excludes_root_credit_before_repair(self):
        inp = {"retrieved": ["old"], "prompt": root.sha("old")}
        packet = {"branches": {a: inp for a in root.ARMS}}
        spans = [{}, {"source": "source", "skills": [], "selected_skills": []}]
        prior = {
            "control": {
                "suffix": [{"actions_hash": _digest([{"action": "different"}])}]
            }
        }
        ex = Mock()
        ex._build_executor_prompt.return_value = "old"
        with (
            patch.object(root, "make_writer", return_value=ex),
            patch.object(root, "ops", return_value=[]),
            patch.object(
                root, "call", return_value={"text": "response", "tokens": [2, 1]}
            ) as api,
            patch.object(root, "writer", return_value=[]),
        ):
            result = root.reissue(None, spans, prior, {"writer_tokens": 32}, packet)
        self.assertEqual(
            (result["status"], result["arm"]), ("suffix_actions_changed", "control")
        )
        api.assert_called_once()


class EarlierRootCreditTests(unittest.TestCase):
    def test_full_window_count_is_retained_and_a_competing_recovery_has_zero_weight(
        self,
    ):
        with TemporaryDirectory() as td:
            src = Path(td)
            early = src / "legacy-terminal-symmetry/registration.json"
            early.parent.mkdir()
            early.write_text("{}")
            audit = src / "legacy-root-symmetry"
            (audit / "outcomes").mkdir(parents=True)
            output = src / "matched-span-rank"
            (output / "root-augmented").mkdir(parents=True)
            (output / "expanded").mkdir()
            (output / "expanded/forecast.json").write_text("{}")
            jobs = [
                {"id": key, "window": window, "pivot": 0, "signed_credit": delta}
                for key, window, delta in (("p", "a_s1", -0.5), ("q", "b_s3", 1.0))
            ]
            reg = audit / "registration.json"
            reg.write_text(
                json.dumps({"jobs": jobs, "terminal_registration": file_hash(early)})
            )
            hashes = {}
            for k, status, c in (("p", "confirmed", -0.5), ("q", "veto", 0.0)):
                f = audit / "outcomes" / (k + ".json")
                f.write_text(json.dumps({"status": status, "credit": c}))
                hashes[k] = file_hash(f)
            (audit / "outcomes-complete.json").write_text(json.dumps(hashes))
            (audit / "summary.json").write_text(
                json.dumps(
                    {
                        "registration": file_hash(reg),
                        "outcomes": hashes,
                        "cases": {k: {} for k in hashes},
                    }
                )
            )
            nodes = {
                "a_s1": (None, None, None, "a_s1", 1, 5),
                "b_s3": (None, None, None, "b_s3", 0, 8),
            }
            with patch.object(
                matched_root_credit, "early_sources", return_value=(early, nodes)
            ):
                rows = matched_root_credit.root_units(None, None, None)
                self.assertEqual(
                    [(r["node"], r["background"], r["credit"]) for r in rows],
                    [("a_s1", 1, -0.1), ("b_s3", 0, 0.0)],
                )
                (audit / "outcomes/p.json").write_text("{}")
                with self.assertRaises(ValueError):
                    matched_root_credit.root_units(None, None, None)
