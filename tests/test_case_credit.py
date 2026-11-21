import tempfile
import unittest
from pathlib import Path

from experiments.controller.audit_case_credit import interval, score
from experiments.controller.case_credit import (
    corrections,
    examples,
    excerpt,
    possible_offsets,
    prompt,
    vote,
)
from experiments.controller.source_compression import register
from src.evaluation_cache import digest


class CaseCreditTests(unittest.TestCase):
    def test_balancing_orders_and_strict_parse(self):
        self.assertEqual([vote("A", False), vote("A", True)], [-1, 1])
        self.assertEqual([vote("B.", False), vote(" a! ", True)], [1, 1])
        self.assertEqual(vote("A or B", True), 0)
        self.assertEqual(
            corrections({"true": [1, 1], "rotated": [-1, -1], "empty": [1, 1]})[
                "credit"
            ],
            2,
        )
        self.assertEqual(
            corrections({"true": [1, -1], "rotated": [-1, 1], "empty": [0, 0]})[
                "credit"
            ],
            0,
        )

    def test_label_permutation_and_donor_separation(self):
        history = {
            "speaker_a": "Tim",
            "speaker_b": "Jo",
            "session_2": [
                {"dia_id": "D2:1", "speaker": "Jo", "text": "Tim arrived today."}
            ],
        }
        qa = {"question": "When did Tim arrive?", "evidence": ["D2:1"]}
        data = {s: {"conversation": history, "qa": [qa]} for s in ("A", "B", "C")}
        cases = [
            {
                "dialogue": s,
                "session": 2,
                "original_index": 0,
                "query": "When did the speaker arrive?",
                "effect": v,
            }
            for s, v in (("A", 1), ("B", -1), ("C", 1))
        ]
        ds = examples(data, cases, "C")
        self.assertEqual([d["donor"] for d in ds], ["A", "B"])
        self.assertTrue(all("Tim" not in d["clue"] for d in ds))
        target = excerpt(history, 2)
        self.assertEqual(target, "Speaker 2: Speaker 1 arrived today.")
        a, b = [prompt(target, ds, k, False) for k in ("true", "rotated")]
        self.assertEqual(
            [v for v in a.splitlines() if v.startswith("Better:")],
            ["Better: B", "Better: A"],
        )
        self.assertEqual(
            [v for v in b.splitlines() if v.startswith("Better:")],
            ["Better: A", "Better: B"],
        )
        self.assertNotIn("arrive?", prompt(target, ds, "empty", True))

    def test_bound_uses_one_decision_across_questions_and_prefixes(self):
        nodes = [
            {"native": 0.5, "weight": w, "cases": [[0.0, 1.0], [1.0, 0.0]]}
            for w in (0.25, 0.75)
        ]
        result = interval(nodes, [-2, 0, 2], "cpu")
        self.assertEqual(result["affected"], 2)
        self.assertAlmostEqual(result["max"], 0.5)
        self.assertGreater(result["question_oracle"], 0.85)

    def test_exact_repeat_prompts_constrain_unobserved_votes(self):
        t = {"prompts": {k: ["first", "second"] for k in ("true", "rotated", "empty")}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "responses").mkdir()
            self.assertEqual(possible_offsets(t, p)[0], [-1, -0.5, 0, 0.5, 1])
            for s in ("first", "second"):
                key = digest(
                    {
                        "model": "llama3-8b",
                        "temperature": 0,
                        "max_tokens": 3,
                        "messages": [{"role": "user", "content": s}],
                    }
                )
                register(
                    p / "responses" / (key + ".json"), {"request": key, "reply": "B"}
                )
            self.assertEqual(possible_offsets(t, p)[0], [0])

    def test_f1_uses_native_first_action_and_matches_rate_without_labels(self):
        vals = [[0.0, 1.0], [1.0, 1.0]]
        nodes = [
            {
                "prefix": b,
                "native": 0.5,
                "raw": 0.6,
                "terminal": 0.4,
                "cases": [[vals[0][b]], [vals[1][b]]],
                "weight": w,
                "costs": [{"reader": 1, "executor": 5}, {"reader": 3, "executor": 5}],
            }
            for b, w in enumerate((0.3, 0.7))
        ]
        w = {
            "window": "s_s5",
            "dialogue": "s",
            "indices": [1],
            "types": [4],
            "offsets": {
                "credit": 1.0,
                "permuted": 0.0,
                "semantic": 0.0,
                "verified": 1.0,
            },
            "nodes": nodes,
            "native_f1": [0.85],
            "votes": {},
            "inputs": {},
            "strict": False,
            "decision_tokens": {"prompt_tokens": 12, "completion_tokens": 2},
        }
        summary, rows = score([w], "cpu")
        self.assertGreater(summary["f1"]["credit"], 0.85)
        self.assertAlmostEqual(summary["f1"]["credit"], summary["f1"]["rate"])
        self.assertAlmostEqual(rows[0]["expected_tokens"]["native"]["executor"], 5)


class JointCaseCreditTests(unittest.TestCase):
    def test_joint_bound_keeps_conditional_return_and_chooses_one_offset_per_source(
        self,
    ):
        import torch

        from experiments.controller.case_credit_joint import value_table

        logits = torch.zeros(3, dtype=torch.float64)
        y = logits.new_tensor([[[0.0], [1.0]], [[1.0], [0.0]]])
        table = value_table(logits, y, [0, 2], [0, -2])
        self.assertAlmostEqual(float(table[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(table[1, 0, 0]), 0.5)
        self.assertGreater(float(table[1, 1, 0]), 0.78)

    def test_joint_two_path_score_separates_focal_and_design_values(self):
        from experiments.controller.audit_case_credit_joint import evaluate

        ref = {
            "window": "x_s13",
            "dialogue": "x",
            "probabilities": {
                str(j): {"native": 0.5, "return_only": 0.5, "terminal": 0.5}
                for j in range(3)
            },
            "rows": [
                {
                    "index": 10,
                    "type": 3,
                    "f1": dict.fromkeys(("native", "return_only", "terminal"), 0.5),
                    "ht": dict.fromkeys(("native", "return_only", "terminal"), 0.5),
                }
            ],
        }
        c = {
            "offsets": {
                "credit": 1.0,
                "verified": 1.0,
                "permuted": 0.0,
                "semantic": 0.0,
            },
            "usage": {"prompt_tokens": 2},
        }
        d = {
            "ref": ref,
            "nodes": [0, 1, 2],
            "cases": {"00": [0.0], "10": [1.0]},
            "costs": {k: {"reader": 4, "executor": 8} for k in ("00", "10")},
            "pivot": 0,
            "background": 0,
            "choice": [c, c],
            "start": 13,
            "strict": True,
            "inputs": {},
            "states": {},
        }
        s, rows = evaluate([d], "cpu")
        self.assertGreater(s["f1"]["credit"], 0.72)
        self.assertLess(s["ht"]["credit"], 0.40)
        self.assertAlmostEqual(s["ht"]["credit"], s["ht"]["rate"])
        self.assertTrue(rows[0]["strict"])

    def test_linkage_audit_distinguishes_opposite_effects_across_prefixes(self):
        import torch

        from experiments.controller.case_credit_joint import question_changes

        y = torch.tensor([[[0.0, 0.5], [1.0, 0.5]], [[1.0, 0.1], [0.0, 0.1]]])
        changes = {k: v.tolist() for k, v in question_changes(y).items()}
        self.assertEqual(changes["affected_first"], [True, True])
        self.assertEqual(changes["affected_last"], [True, False])
        self.assertEqual(changes["opposite_last_by_prefix"], [True, False])
