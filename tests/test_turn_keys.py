import unittest

import torch

from experiments.controller.turn_key_credit import (
    fit,
    keyed_scores,
    offset,
    read_metrics,
)
from experiments.controller.turn_key_reader import live_keys, reorder, summarize
from src.replay import dated_group_parts, dated_turn_groups


class TurnKeyTests(unittest.TestCase):
    def test_clipped_parent_keys_cannot_expose_words_from_the_following_chunk(self):
        words = " ".join(f"alpha{i}" for i in range(100))
        src = f"Session Date: 2025-01-01\n\nuser: {words}\n\nassistant: A reply with two lines\nand a continuation"
        pairs = dated_group_parts(src, 2)
        self.assertEqual([s for s, _ in pairs], dated_turn_groups(src, 2))
        self.assertEqual([i for i, _ in pairs[0][1]], [0])
        self.assertEqual([i for i, _ in pairs[1][1]], [0, 1])
        self.assertEqual(pairs[0][0].split(" | ")[1].split()[-1], "alpha94")
        self.assertNotIn("alpha94", pairs[1][1][0][1])
        for parent, parts in pairs:
            self.assertEqual(
                parent.split(" | ", 1)[1],
                " ".join(s.split(" | ", 1)[1] for _, s in parts),
            )
        self.assertIn("and a continuation", pairs[-1][0])

    def test_only_focal_source_inserts_get_keys_and_repeated_text_stays_distinct(self):
        src = 'DATE: 6 July, 2023\nCONVERSATION:\nA said, "one"\nB said, "two"'
        text = dated_turn_groups(src, 2)[0]

        def event(mid, reason):
            return {
                "source": src,
                "retrieved": [],
                "refs": [],
                "parsed_actions": [
                    {
                        "action_type": "INSERT",
                        "success": True,
                        "memory_content": text,
                        "reasoning": reason,
                    }
                ],
                "changes": [
                    {
                        "action": "INSERT",
                        "memory_id": mid,
                        "version": 1,
                        "before": None,
                        "after": text,
                    }
                ],
            }

        first = event(9, "dated_adjacent_turns")
        self.assertEqual(
            set(live_keys([first, event(12, "dated_adjacent_turns")])), {1}
        )
        self.assertEqual(live_keys([first, event(12, "model")]), {})

    def test_key_rotation_retains_storage_and_leaves_all_other_scores_unchanged(self):
        base = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        child = torch.tensor([[[0.8, 0.6], [0.3, 0.2]]])
        new, rot = keyed_scores(base, child, [1, 3])
        torch.testing.assert_close(new, torch.tensor([[0.1, 0.5, 0.3, 0.35]]))
        torch.testing.assert_close(rot, torch.tensor([[0.1, 0.25, 0.3, 0.6]]))
        self.assertEqual(reorder(torch.tensor([1.0, 1.0, 0.0]), [1, 0, 2], 2), [1, 0])

    def test_multiple_fragments_of_one_evidence_turn_only_count_once(self):
        ids = torch.tensor([[0, 2], [0, 1], [1, 2]])
        gold = torch.tensor([[True, True, False]])
        recall, complete, words = read_metrics(
            torch.tensor([[0, 1]]), ids, torch.tensor([4, 5, 7]), gold
        )
        self.assertEqual((recall.item(), complete.item(), words.item()), (1.0, True, 9))

    def test_neutral_auxiliary_credit_keeps_the_logit_unchanged(self):
        model = fit(
            [
                {"features": [1.0, 2.0, 0.0], "target": 0.0},
                {"features": [1.0, 3.0, 1.0], "target": 0.0},
            ]
        )
        self.assertEqual(offset(model, [4.0, 5.0, 1.0]), 0.0)

    def test_f1_separates_index_effect_from_changing_skill_probability(self):
        arms = {
            k: {"f1": f, "words": 5}
            for k, f in {
                "temporal": 0.2,
                "activity": 0.4,
                "keyed": 0.8,
                "rotated": 0.3,
            }.items()
        }
        w = {
            "native": 0.25,
            "terminal": 0.3,
            "p_keyed_activity": 0.75,
            "dialogue": "a",
            "rows": [{"arms": arms, "category": 1}, {"exclusion": "unreproduced"}],
        }
        s = summarize([w])
        self.assertEqual((s["matched_questions"], s["intended_questions"]), (1, 2))
        self.assertAlmostEqual(s["f1"]["old_index_policy"], 0.35)
        self.assertAlmostEqual(s["f1"]["new_native"] - s["f1"]["native"], 0.1)
        self.assertAlmostEqual(s["f1"]["policy"] - s["f1"]["new_native"], 0.3)
        self.assertEqual(s["f1"]["policy"], s["f1"]["matched_rate"])
