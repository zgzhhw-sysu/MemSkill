import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.controller.audit_paired import audit_writer_ranges
from src.executor import Executor
from src.writer_context import WriterContext


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt and messages[0]["role"] == "user"
        return list(messages[0]["content"])


class WriterContextTest(unittest.TestCase):
    def setUp(self):
        self.ctx = object.__new__(WriterContext)
        self.ctx.limit = 115
        self.ctx.tokenizer = CharacterTokenizer()
        self.source = "Session Date: Friday\n\nuser: " + "雪🌻abcd " * 30 + "\n\nassistant: Yes.\nMore."

    def test_short_and_split_inputs_preserve_original_text(self):
        build = lambda s: "<" + s + ">"
        short = self.ctx.prepare("短", build, 16)
        self.assertEqual(short, [(0, 1, "<短>", 3, 16)])
        plan = self.ctx.prepare(self.source, build, 16)
        self.assertGreater(len(plan), 1)
        self.assertEqual("".join(self.source[a:b] for a,b,*_ in plan), self.source)
        self.assertEqual(sum(part[-1] for part in plan), 16)
        for a,b,prompt,tokens,cap in plan:
            self.assertIn(self.source[a:b], prompt)
            self.assertTrue(prompt.startswith("<Session Date: Friday\n\n"))
            self.assertTrue(0 < cap <= self.ctx.limit - tokens)
        with self.assertRaises(ValueError):
            self.ctx.prepare(self.source, lambda s: "x" * self.ctx.limit + s, 16)

    def test_executor_audits_every_request_and_applies_the_rule_to_the_whole_source(self):
        args = SimpleNamespace(retriever="test", model="test", api_base="local", api_key="local",
                               max_new_tokens=16, temperature=0)
        executor = Executor(args)
        executor.writer_context = self.ctx
        op = SimpleNamespace(name="detail", update_type="insert", source_rule="attributed_turn_pairs")
        def reply(**kwargs):
            return "ACTION: INSERT\nMEMORY_ITEM: item", len(kwargs["prompt"]), 1
        history = []
        with patch.object(executor, "_build_executor_prompt", side_effect=lambda ops, text, mems: text), \
                patch("src.executor.get_llm_response_via_api", side_effect=reply), \
                patch("src.replay.attributed_turn_pairs", return_value=["source item"]) as extract:
            results = executor.execute_operation([op], self.source, [], audit=history)
        self.assertTrue(all(r.success for r in results))
        extract.assert_called_once_with(self.source)
        self.assertEqual(results[-1].memory_content, "source item")
        inputs = {"writer_tokens":16, "writer_context":{"context_tokens":115}}
        event = {"source": self.source, "executor_responses": history}
        self.assertGreater(audit_writer_ranges(event, inputs), 1)
        for invalid in (dict(history[-1], source_range=[0, len(self.source)]),
                        dict(history[-1], input_tokens=115), dict(history[-1], output_tokens=0),
                        dict(history[-1], output_cap=0)):
            with self.assertRaises(ValueError):
                audit_writer_ranges(dict(event, executor_responses=[*history[:-1], invalid]), inputs)
        with self.assertRaises(ValueError):
            audit_writer_ranges(dict(event, executor_responses=history[:-1]), inputs)
