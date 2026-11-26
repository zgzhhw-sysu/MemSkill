import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from experiments.controller.first_evidence_credit import (
    background,
    cohort,
    pair_value,
    paths,
    verify_switch,
)
from experiments.controller.first_evidence_prefill import request_for
from experiments.controller.paired_policy import PairedPolicy
from experiments.controller.prefix_interaction import capacity
from experiments.controller.semantic_writer import sha
from experiments.controller.source_compression import file_hash
from src.controller import PPOController
from src.episode_trace import skill_spec
from src.executor import Executor, deployment_operations


class FirstEvidenceTests(unittest.TestCase):
    def test_group_by_earliest_evidence_and_keep_each_question_once(self):
        specs = []
        for sid, questions in (("A", [[1, 2, 4], [2, 4], [1, 4]]), ("B", [[1, 4]])):
            specs.append(
                {
                    "dialogue": sid,
                    "end": 4,
                    "background": [background(sid, i) for i in (1, 2, 3)],
                    "questions": [
                        {"original_index": i, "type": 1, "sessions": ns}
                        for i, ns in enumerate(questions)
                    ],
                }
            )
        groups = cohort(specs)
        self.assertEqual(
            [(g["dialogue"], g["first"]) for g in groups],
            [("A", 1), ("B", 1), ("A", 2)],
        )
        self.assertEqual([v["original_index"] for v in groups[0]["questions"]], [0, 2])
        with self.assertRaises(ValueError):
            cohort([*specs, specs[0]])
        a, b = paths("A", 2, 4)
        self.assertEqual([j for j in range(4) if a[j] != b[j]], [1])
        self.assertEqual(a[-1], b[-1])
        self.assertEqual(
            pair_value(0.1, 0.9, {"off": 0.0, "on": 1.0, "mix": 0.25}),
            {"off": 0.1, "on": 0.9, "mix": 0.1 + 0.8 * 0.25},
        )

    def test_precomputed_first_write_matches_the_physical_factual_prompt(self):
        specs = [
            skill_spec(
                SimpleNamespace(
                    name=n, description=n, instruction_template=n, update_type=t
                )
            )
            for n, t in (
                ("write", "insert"),
                ("pass", "noop"),
                ("time", "insert"),
                ("activity", "insert"),
            )
        ]
        group = {"dialogue": "A", "first": 1, "end": 3}
        bits, alt = paths("A", 1, 3)
        low, high = [0, 1, 2], [0, 1, 3]
        selected = [specs[k] for k in (high if bits[0] else low)]
        ex = Executor(SimpleNamespace(retriever="unused"))
        source = "Ann said hello."
        prompt = ex._build_executor_prompt(
            deployment_operations([SimpleNamespace(**v) for v in selected]), source, []
        )
        span = {
            "kind": "span",
            "source": source,
            "retrieved": [],
            "selected_skills": selected,
            "executor_responses": [{"prompt_sha256": sha(prompt)}],
            "changes": [],
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "".join(map(str, bits))
            d = root / name
            (d / "traces").mkdir(parents=True)
            trace = d / "traces/log.jsonl"
            trace.write_text(json.dumps(span))
            dec = d / "decisions.json"
            dec.write_text(
                json.dumps([{"plain": low, "grounded": high, "use": bits[0]}])
            )
            pair = {
                "trajectories": {
                    name: {
                        "traces/log.jsonl": file_hash(trace),
                        "decisions.json": file_hash(dec),
                    }
                }
            }
            r = request_for(group, root, pair, specs, "model", 256)
            self.assertEqual(
                r["selected"], [specs[i]["version"] for i in (high if alt[0] else low)]
            )
            self.assertEqual(r["factual_prompt"], sha(prompt))
            self.assertNotEqual(r["prompt_hash"], r["factual_prompt"])
            span["retrieved"] = ["wrong context"]
            trace.write_text(json.dumps(span))
            pair["trajectories"][name]["traces/log.jsonl"] = file_hash(trace)
            with self.assertRaisesRegex(ValueError, "reproduce"):
                request_for(group, root, pair, specs, "model", 256)

    def test_fixed_common_actions_override_state_order_and_reject_duplicate(self):
        torch.manual_seed(22)
        native = PPOController(
            state_dim=5, op_dim=4, hidden_dim=8, action_top_k=3, device="cpu"
        )
        actor = PairedPolicy(native, 5, 4)
        actor.mode = "count"
        actor.schedule = [False, True, False]
        actor.common_schedule = [[2, 0], [0, 2], [2, 2]]
        s, ops = torch.randn(5), torch.randn(6, 4)
        self.assertEqual(actor(s, ops, deterministic=True)[0], [2, 0, 4])
        self.assertEqual(actor(s + 1, ops, deterministic=True)[0], [0, 2, 5])
        with self.assertRaisesRegex(ValueError, "ordered background"):
            actor(s, ops, deterministic=True)

    def test_intervention_allows_later_retrieval_but_nothing_else_in_schedule_to_change(
        self,
    ):
        versions = list("abcd")
        old_dec = [
            {
                "state": [i],
                "base_logit": 0.0,
                "plain": [0, 1, 2],
                "grounded": [0, 1, 3],
                "use": False,
            }
            for i in range(4)
        ]
        old = [
            {
                "source": str(i),
                "source_turns": [i],
                "span_id": i,
                "refs": [],
                "retrieved": [],
                "selected_skills": [{"version": v} for v in "abc"],
            }
            for i in range(4)
        ]
        new, changed = copy.deepcopy(old), copy.deepcopy(old_dec)
        new[1]["selected_skills"][-1]["version"] = "d"
        changed[1]["use"] = True
        new[2]["retrieved"] = ["different early write"]
        changed[2]["state"] = [12]
        verify_switch(old, new, old_dec, changed, 2, versions)
        for kind in ("prefix", "focal", "continuation"):
            with self.subTest(kind=kind):
                x, y = copy.deepcopy(new), copy.deepcopy(changed)
                if kind == "prefix":
                    x[0]["refs"] = [[123, 1]]
                elif kind == "focal":
                    x[1]["retrieved"] = ["changed too early"]
                else:
                    x[-1]["selected_skills"][0]["version"] = "b"
                with self.assertRaises(ValueError):
                    verify_switch(old, x, old_dec, y, 2, versions)

    def test_prefix_capacity_respects_each_window_budget_and_oracle_corner(self):
        # The terminal skill should be opposite to the first skill.
        y = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64
        )
        s = y.new_tensor([[1, 0, 1, 0], [1, 0, 0, 1]])
        r = capacity(y, torch.zeros(3, dtype=y.dtype), s)
        self.assertAlmostEqual(r["native_f1"], 0.5)
        self.assertAlmostEqual(r["unrestricted_gain"], 0.5)
        self.assertAlmostEqual(sum(r["bounded_probabilities"]) / 2, 0.5)
        self.assertAlmostEqual(
            r["bounded_gain"], torch.sigmoid(y.new_tensor(2.0)).item() - 0.5
        )
        self.assertTrue(r["opposite_preferences"])
        self.assertEqual(r["root_only_gain"], 0.0)
        s[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "different source"):
            capacity(y, torch.zeros(3, dtype=y.dtype), s)
