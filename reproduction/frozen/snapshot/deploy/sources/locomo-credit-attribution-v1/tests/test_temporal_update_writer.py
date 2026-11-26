import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash
from experiments.controller.temporal_update_forecast import UPDATE, continuity
from experiments.controller.temporal_update_writer import (
    choices,
    passed_screen,
    replay_provider,
    score_prepared,
    summarize,
)
from src.episode_trace import skill_spec
from src.executor import Executor, deployment_operations


def spec(name, kind, **kw):
    return skill_spec(
        SimpleNamespace(
            name=name,
            description=name,
            update_type=kind,
            instruction_template=name,
            **kw,
        )
    )


class TemporalUpdateTests(unittest.TestCase):
    def test_common_insert_preserves_temporal_prompt_and_all_dated_records(self):
        specs = [
            spec("insert", "insert"),
            spec("noop", "noop"),
            spec(
                "capture_temporal_context", "insert", source_rule="dated_single_turns"
            ),
        ]
        old = [SimpleNamespace(**s) for s in specs]
        low, high = choices(specs, spec(UPDATE, "update"))
        self.assertEqual(
            [skill_spec(op) for op in low[:2]], [skill_spec(op) for op in high[:2]]
        )
        self.assertEqual(high[-1].update_type, "update")
        args = SimpleNamespace(
            retriever="unused",
            model="local",
            api_key="test",
            api_base="local",
            max_new_tokens=32,
            temperature=0,
        )
        executor = Executor(args)
        source = 'DATE: 1:00 pm on 9 March, 2023\nCONVERSATION:\nAnn said, "I walked home."\nLee said, "I slept."'
        ctx = ["Ann's former plan"]
        reply = "ACTION: INSERT\nMEMORY_ITEM: Ann walked home."
        archive = []
        check = []
        with patch(
            "src.executor.get_llm_response_via_api", return_value=(reply, 20, 8)
        ) as endpoint:
            old_actions = executor.execute_operation(
                deployment_operations(old), source, ctx, audit=archive
            )
            control = executor.execute_operation(
                deployment_operations(low),
                source,
                ctx,
                audit=check,
                response_provider=replay_provider(archive),
            )
            self.assertEqual(endpoint.call_count, 1)
        self.assertEqual(archive, check)
        self.assertEqual([vars(a) for a in old_actions], [vars(a) for a in control])
        self.assertGreater(len(control), 1)
        updated = executor.execute_operation(
            deployment_operations(high),
            source,
            ctx,
            response_provider=lambda **kw: (
                "ACTION: UPDATE\nMEMORY_INDEX: 0\nUPDATED_MEMORY: Ann walked home on Thursday.",
                28,
                12,
            ),
        )
        self.assertEqual(updated[0].action_type, "UPDATE")
        self.assertEqual(
            [a.memory_content for a in updated[1:]],
            [a.memory_content for a in control[1:]],
        )
        with self.assertRaises(KeyError):
            replay_provider(archive)(prompt="mismatched prompt", MAX_TOKENS=32)

    def test_ineligible_fixed_slots_are_identified(self):
        original = [
            spec("noop", "noop"),
            spec(UPDATE, "update"),
            spec(
                "capture_temporal_context", "insert", source_rule="dated_single_turns"
            ),
        ]
        self.assertIsNone(choices(original, spec(UPDATE, "update")))

    def test_mask_uses_visible_temporal_turns_and_allows_no_cue(self):
        previous = torch.tensor([[1.0, 0.0]])
        current = torch.eye(2)
        self.assertEqual(
            continuity(
                previous, current, [{"text": "Thanks."}, {"text": "Last week."}]
            ),
            0.0,
        )
        self.assertIsNone(
            continuity(previous, current, [{"text": "We may go."}, {"text": "Hello!"}])
        )
        self.assertEqual(
            continuity(
                previous, current, [{"text": "In May 2017."}, {"text": "Great!"}]
            ),
            1.0,
        )

    def test_coverage_and_rate_match_use_windows_consistently(self):
        def report(n, prob, delta):
            p = {
                "source": prob,
                "sham": 0.5,
                "native": 0.5,
                "proposed": prob,
                "fixed_temporal": 0.0,
                "fixed_update": 1.0,
            }
            return {
                "status": "matched",
                "dialogue": str(n),
                "questions": list(range(n)),
                "choice": p,
                "edited": 1,
                "strict": True,
                "writer_tokens": {"control": 24, "update": 25},
                "update_fault": None,
                "rows": [
                    {
                        "type": 1,
                        "first_only": True,
                        "temporal": {"f1": 0.5, "tokens": [10, 3]},
                        "update": {"f1": 0.5 + delta, "tokens": [11, 4]},
                    }
                ]
                * n,
            }

        r = summarize(
            [
                report(1, 0.2, -0.5),
                report(3, 0.8, 0.5),
                {"status": "unmatched_temporal_write", "questions": [1, 2]},
                {"status": "ineligible_fixed_slots", "questions": [3]},
            ]
        )
        self.assertEqual(
            (r["arrived_questions"], r["eligible_questions"], r["questions"]), (7, 6, 4)
        )
        self.assertAlmostEqual(r["f1"]["rate"], 0.625)
        self.assertAlmostEqual(r["f1"]["source"], 0.775)
        self.assertFalse(passed_screen(dict(r, total_windows=4)))

    def test_prepared_reader_scores_only_changed_prompts_and_checks_writer(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"MEMSKILL_BACKEND_ID": "frozen"}),
        ):
            root = Path(tmp)
            writes = root / "writes.json"
            writes.write_text("{}")
            path = root / "reader-inputs.json"
            low = {
                "answer": "old",
                "tokens": [4, 2],
                "f1": 0.2,
                "ok": True,
                "context_hash": "a",
                "refs": [[8, 1]],
            }
            rows = [
                {
                    "index": 0,
                    "temporal": low,
                    "prompt": None,
                    "prompt_hash": None,
                    "update": {"refs": [[1, 2]], "context_hash": "a", "indices": [0]},
                },
                {
                    "index": 1,
                    "temporal": low,
                    "prompt": "changed",
                    "prompt_hash": sha("changed"),
                    "update": {"refs": [[2, 2]], "context_hash": "b", "indices": [1]},
                },
            ]
            packet = {
                "backend": "frozen",
                "model": "llama3-8b",
                "dialogue": "d",
                "writes": file_hash(writes),
                "rows": rows,
            }
            path.write_text(json.dumps(packet))
            corpus = {"d": {"qa": [{}, {}]}}
            with patch(
                "experiments.controller.temporal_update_writer.measured_answer",
                return_value={"f1": 0.8, "answer": " resolved ", "tokens": [5, 2]},
            ) as call:
                score_prepared(path, root, corpus, None)
                self.assertEqual(call.call_count, 1)
                scored = json.loads((root / "pair.json").read_bytes())
                self.assertEqual(scored["rows"][0]["update"]["answer"], "old")
                self.assertEqual(scored["rows"][0]["update"]["refs"], [[1, 2]])
                self.assertEqual(scored["rows"][1]["update"]["answer"], "resolved")
                writes.write_text('{"altered":true}')
                with self.assertRaises(ValueError):
                    score_prepared(path, root, corpus, None)
