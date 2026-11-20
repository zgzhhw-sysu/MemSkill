import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

from experiments.controller import factorial_repairs as audit


class Bank:
    def __init__(self, texts):
        self.texts = list(texts)

    def get_all_contents(self):
        return self.texts.copy()

    def retrieve(self, embedding, top_k):
        # Inserting an annotated quote makes it compete for the single read slot.
        i = len(self.texts) - 1 if len(self.texts) > 2 else 0
        return [self.texts[i]], [i]


def reply(text):
    return {"answer": text, "ok": True, "f1": float(text == "Paris"), "tokens": [12, 1]}


def fixture():
    source = {"source": "source", "refs": [], "retrieved": [], "source_turns": ["D2:1"]}
    q = {
        "original_index": 41,
        "question": "Where?",
        "category": 4,
        "evidence": ["D2:1"],
    }
    cells = {}
    for a, b in product((0, 1), repeat=2):

        def event(bit, linked):
            return dict(
                source,
                source_turns=["D2:1"] if linked else ["D1:1"],
                selected_skills=["fixed", str(bit)],
                skills=[{"update_type": "insert"}],
                parsed_actions=[{"action_type": "INSERT", "success": True}],
                applied=True,
            )

        cells[a, b] = {
            "spans": [event(a, False), event(b, True)],
            "scores": {0: dict(reply("Paris" if b else "Rome"), indices=[0])},
        }
    return q, cells


class FactorialRepairsTests(unittest.TestCase):
    def test_backgrounds_share_an_original_question_and_unchanged_prefix(self):
        q, cells = fixture()
        evaluator = SimpleNamespace(get_ground_truth=lambda qa: "Paris")
        rows = audit.contrasts(cells, [q], evaluator)
        s = audit.summarize(
            [{"dialogue": "d", "start": 1, "questions": 1, "rows": rows}]
        )
        self.assertEqual(s["positions"]["1"]["linked"], 2)
        self.assertEqual(s["positions"]["1"]["unique_linked_questions"], 1)
        self.assertEqual(s["positions"]["1"]["both_backgrounds_agree"], 1)
        cells[0, 1]["spans"][0]["source"] = "different"
        with self.assertRaises(ValueError):
            audit.contrasts(cells, [q], evaluator)

    def test_weak_bank_repair_and_hidden_read_veto_are_independent(self):
        q, cells = fixture()
        trainer = SimpleNamespace(
            config=SimpleNamespace(mem_top_k_eval=1),
            evaluator=SimpleNamespace(get_ground_truth=lambda qa: "Paris"),
        )

        def insert(actions, bank, indices):
            bank.texts.extend(a.memory_content for a in actions)
            return True

        trainer.executor = SimpleNamespace(apply_to_memory_bank=insert)
        convo = {
            "qa": [q],
            "conversation": {
                "session_2": [{"dia_id": "D2:1", "speaker": "A", "text": "quote"}],
                "session_2_date_time": "date",
            },
        }
        row = {"qa_index": 0, "background": 0, "recovery_sign": 1, "delta": 1.0}
        for write, hidden, status in [
            (True, False, "competing_recovery"),
            (False, True, "competing_recovery"),
            (False, False, "survived_probes"),
        ]:
            with self.subTest(write=write, hidden=hidden):
                banks = {(0, 0): Bank(["weak", "hidden"]), (0, 1): Bank(["strong"])}

                def read(_, qa, ctx, hidden=hidden, write=write):
                    solved = (
                        ctx[0] == "strong"
                        or (ctx[0] == "hidden" and hidden)
                        or ("quote" in ctx[0] and write)
                    )
                    return reply("Paris" if solved else "Rome")

                with (
                    patch.object(audit, "_answer", read),
                    patch.object(
                        audit, "replay_writes", return_value=Bank(banks[0, 0].texts)
                    ),
                ):
                    result = audit.final_probe(trainer, convo, row, cells, banks, None)
                self.assertEqual(result["status"], status)
                self.assertEqual(
                    result["signed_effect"], float(not write and not hidden)
                )
                self.assertEqual(
                    result["competing"],
                    (["execution"] if write else [])
                    + (["retrieval"] if hidden else []),
                )
                self.assertEqual(banks[0, 0].texts, ["weak", "hidden"])
