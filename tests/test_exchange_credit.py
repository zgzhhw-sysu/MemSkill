import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.controller.exchange_paths import clean_marker, measure
from experiments.controller.exchange_policy import collect, predict
from experiments.controller.exchange_questions import (
    parse,
    quote,
    screen,
    teacher_prompt,
    verify_question,
)
from experiments.controller.exchange_status import resolved, stage_jobs, state
from experiments.controller.source_compression import file_hash, register
from src.eval.locomo import LoCoMoEvaluator
from src.evaluation_cache import digest, publish_response


class ExchangeTest(unittest.TestCase):
    def setUp(self):
        self.job = {
            "asker": "Bob",
            "responder": "Emma",
            "ask_text": "Which hobby?",
            "source_question": "Which hobby?",
            "reply": "I practice martial arts at the party.",
            "ask_id": "D5:1",
            "reply_id": "D5:2",
            "date": "1 May",
            "other_replies": [{"id": "D5:4", "text": "We met a friend."}],
        }
        self.item = {
            "question": "Which hobby does Emma practice?",
            "answer": "martial arts",
        }

    def test_format_changes_only_framing(self):
        raw = json.dumps(self.item)
        self.assertEqual(parse(raw, self.job)[1], "candidate")
        self.assertEqual(
            parse("Here is the item:\n" + raw, self.job)[1], "malformed_json"
        )
        self.assertEqual(
            parse("Here is the item:\n" + raw, self.job, True), parse(raw, self.job)
        )
        for text in (raw + "\nExtra.", raw + raw, "My question?:\n" + raw):
            self.assertEqual(parse(text, self.job, True)[1], "malformed_json")
        self.assertEqual(
            parse(json.dumps(dict(self.item, answer="art")), self.job)[1],
            "reference_not_in_reply",
        )
        self.assertEqual(
            parse(
                json.dumps(
                    dict(self.item, question="Does Emma practice martial arts?")
                ),
                self.job,
            )[1],
            "answer_in_question",
        )

    def test_decoy_and_reader_exclusions(self):
        ev = LoCoMoEvaluator(SimpleNamespace())

        def judge(answers, job):
            with patch(
                "experiments.controller.exchange_questions.call",
                side_effect=[{"text": a} for a in [json.dumps(self.item), *answers]],
            ):
                return screen(Path("strict"), job, ev)["status"]

        self.assertEqual(
            judge([], dict(self.job, other_replies=[])), "no_matched_decoy"
        )
        self.assertEqual(judge(["unknown"], self.job), "source_not_recovered")
        self.assertEqual(
            judge(["martial arts", "perhaps martial arts"], self.job), "decoy_recovered"
        )
        self.assertEqual(judge(["martial arts", "reading"], self.job), "accepted")

    def test_legacy_reader_receipt_binds_real_and_decoy_sources(self):
        job = self.job
        qa, _ = parse(json.dumps(self.item), job)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            reg = out / "registration.json"
            register(reg, {"jobs": [job]})
            ev = LoCoMoEvaluator(SimpleNamespace())

            def prompt(reply):
                return ev.build_prompt(
                    qa["question"],
                    [
                        quote(job, job["asker"], job["ask_text"]),
                        quote(job, job["responder"], reply),
                    ],
                    qa,
                )

            packet = {
                "population": file_hash(reg),
                "qa": qa,
                "decoy_id": job["other_replies"][0]["id"],
            }
            for kind, text, ans in (
                (
                    "teacher",
                    teacher_prompt(
                        job["asker"],
                        job["source_question"],
                        job["responder"],
                        job["reply"],
                    ),
                    json.dumps(self.item),
                ),
                ("source", prompt(job["reply"]), qa["answer"]),
                ("decoy", prompt(job["other_replies"][0]["text"]), "unknown"),
            ):
                key = digest(text)
                rec = publish_response(
                    out / "responses",
                    key,
                    hashlib.sha256(text.encode()).hexdigest(),
                    ans,
                    10,
                    5,
                )
                packet[kind] = {"key": key, "text": ans, "sha256": rec["sha256"]}
            verify_question(reg, job, packet)
            for other in (
                dict(job, reply="Changed source"),
                dict(
                    job,
                    other_replies=[
                        {"id": packet["decoy_id"], "text": "Different decoy"}
                    ],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Different source"):
                    verify_question(reg, other, packet)

    def test_causal_credit_requires_persistence_and_no_competition(self):
        self.assertEqual(clean_marker("one INSERT block:"), "one INSERT block:")
        self.assertEqual(clean_marker("one\nINSERT block:\n"), "one")
        qa = {
            "question": "Which hobby does Emma practice?",
            "answer": "martial arts",
            "category": 4,
        }
        ev = SimpleNamespace(
            build_prompt=lambda _, ctx, qa: ctx[0],
            compute_f1=LoCoMoEvaluator(SimpleNamespace()).compute_f1,
        )

        def judge(
            low="hiking", high="martial arts", later=None, repair=None, capped=None
        ):
            rows = []
            for answer in (low, high):
                rows.append(
                    {
                        "native": [answer],
                        "delayed": [later or answer],
                        "execution": [repair or answer],
                        "retrieval": None,
                        "format": [answer],
                    }
                )
            obj = {
                "status": "prepared",
                "paths": [{"faults": [None, None], "capped": [capped, None]}] * 2,
                "arms": rows,
            }
            with patch(
                "experiments.controller.exchange_paths.call",
                side_effect=lambda _, prompt, *args: {"text": prompt},
            ):
                return measure(None, obj, qa, ev, 0)

        self.assertEqual(judge()["credit"], 1.0)
        self.assertEqual(judge("martial arts", "hiking")["credit"], -1.0)
        for result, cause in (
            (judge(later="reading"), "delayed_disagrees"),
            (judge(repair="perhaps martial arts"), "execution"),
            (judge(capped="cap"), "writer_defect"),
        ):
            self.assertEqual((result["raw"], result["credit"]), (1.0, 0.0))
            self.assertIn(cause, result["veto"])

    def test_held_history_and_equal_dialogue_mass(self):
        vec = torch.tensor([[1.0, 1.0]])
        names = ["a_s5", "b_s5", "b_s6", "c_s5"]
        donors = {
            n: {"history": n[0], "pivot": 0, "credit": v, "raw": v}
            for n, v in zip(names, [1.0, 0.03, 0.03, -0.01], strict=True)
        }
        v = dict.fromkeys(names, vec)
        result = predict("a_s5", 0, donors, v)
        self.assertAlmostEqual(result["credit"], 0.2, places=5)
        self.assertEqual(predict("a_s5", 1, donors, v)["credit"], 0.0)
        donors["a_s5"]["credit"] = -1.0
        self.assertEqual(predict("a_s5", 0, donors, v), result)

    def test_failed_attempts_in_denominator_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for k in ("questions", "prepared", "outcomes"):
                (out / k).mkdir()
            jobs = [
                {"id": str(i), "dialogue": "a", "pivot": 0, "session": 5, "slot": i}
                for i in range(2)
            ]
            reg = out / "registration.json"
            register(reg, {"jobs": jobs})
            for i, j in enumerate(jobs):
                name = str(i) + ".json"
                q = out / "questions" / name
                register(
                    q,
                    {
                        "status": ("accepted" if i == 0 else "malformed_json"),
                        "source": digest(j),
                        "population": file_hash(reg),
                    },
                )
                if i == 0:
                    self.assertFalse(resolved(out, j))
                    p = out / "prepared" / name
                    register(p, {"inputs": file_hash(q)})
                    register(
                        out / "outcomes" / name,
                        {
                            "inputs": file_hash(p),
                            "question": file_hash(q),
                            "status": "measured",
                            "raw": 1,
                            "credit": 1,
                        },
                    )
            self.assertTrue(all(resolved(out, j) for j in jobs))
            fit = collect(reg, jobs)
            self.assertEqual(fit["sessions"]["a_s5"]["credit"], 0.5)
            self.assertEqual(len(stage_jobs(jobs)["breadth"]), 1)
            self.assertEqual(state(out, "policy"), "work")
