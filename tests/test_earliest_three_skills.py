import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.controller.checkpoint_odds_credit import distribute, fit
from experiments.controller.earliest_credit_update import points
from experiments.controller.earliest_skill_evaluation import contrast_bounds
from experiments.controller.earliest_three_skills import pair_rows
from experiments.controller.exchange_questions import BACKEND, MODEL, call
from experiments.controller.semantic_writer import sha
from src.evaluation_cache import publish_response, read_response, request_cache


class OriginalStateTests(unittest.TestCase):
    def test_state_moderation_is_pre_action_and_target_history_is_excluded(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        donors = [
            {
                "history": str(h),
                "window": str((h, i)),
                "features": [0.1 * i, 0.2 + h / 100],
                "native_pair": [0.4, 0.3 + 0.4 * i],
                "points": [
                    {
                        "scores": [0.5, 0.4, 0.2 + 0.4 * i],
                        "edges": [0, 0, 0.2 if i else 0],
                    }
                ],
            }
            for h in range(3)
            for i in range(2)
        ]
        model = fit(donors, "2", device)
        self.assertEqual(model["histories"], ["0", "1"])
        donors[-1]["points"][0]["scores"] = [1000, 1000, 1000]
        self.assertEqual(model, fit(donors, "2", device))

        def tensor(x):
            return torch.tensor(x, dtype=torch.float64, device=device)

        # When the actual odds equal the donor center, removing this feature has no effect.
        z = tensor(model["centers"][1]).sigmoid()
        native = torch.stack((tensor(0.3), 0.7 * (1 - z), 0.7 * z))
        p = distribute(native, tensor(0.2), [0.1, 0.2], model)
        torch.testing.assert_close(p["credit"], p["masked_state"])
        moved = distribute(tensor([0.3, 0.05, 0.65]), tensor(0.2), [0.1, 0.2], model)
        self.assertFalse(torch.allclose(moved["credit"], moved["masked_state"]))
        for prob in moved.values():
            self.assertAlmostEqual(float(prob[0]), 0.2)
            self.assertAlmostEqual(float(prob.sum()), 1.0)

    def test_bounds_are_attained_with_partly_observed_branches(self):
        d = torch.tensor(
            [-0.2, 0.3, -0.1],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        x = d.new_tensor([[0.4, 0.6, float("nan")], [float("nan"), float("nan"), 0.7]])
        observed = []
        for e in itertools.product((0, 1), repeat=3):
            y = x.clone()
            y[y.isnan()] = d.new_tensor(e)
            observed.append(float((y @ d).sum()))
        torch.testing.assert_close(
            contrast_bounds(d, x), d.new_tensor([min(observed), max(observed)])
        )
        with self.assertRaises(ValueError):
            contrast_bounds(d, d.new_tensor([[0, 1.1, 0.1]]))

    def test_actual_action_order_and_focal_credit(self):
        for native_activity in (False, True):
            qs = [
                {
                    "original_index": i,
                    "question": "On what date?",
                    "answer": "1 January 2023",
                    "category": 2,
                    "evidence": ["D2:5" if i == 0 else "D1:2"],
                }
                for i in range(2)
            ]
            prep = {"questions": []}
            rel = {"rows": []}
            binary = {"rows": []}
            for q in qs:
                i = q["original_index"]
                weak = {"answer": "Tokyo", "f1": 0, "receipt": {}}
                high = {"answer": "1 January 2023", "f1": 1, "receipt": {}}
                # The relation skill loses to both other skills on the very same physical answer.
                prep["questions"].append(
                    {
                        "qa": q,
                        "linked": i == 0,
                        "native": {"indices": [1], "hash": "A"},
                        "alternative": {"indices": [2], "hash": "B"},
                    }
                )
                rel["rows"].append(dict(weak, index=i, ok=True))
                activity = dict(high, f1=2 / 3, answer="1 Jan 2023")
                arms = {
                    "native": activity if native_activity else high,
                    "alternative": high if native_activity else activity,
                }
                binary["rows"].append(
                    {
                        "original_index": i,
                        "category": 2,
                        "linked": i == 0,
                        "arms": arms,
                        "delta": 2 / 3 - 1,
                    }
                )
            pair, auth = pair_rows(
                {"native_activity": native_activity},
                qs,
                [{"source_turns": ["D2:5"]}],
                prep,
                rel,
                binary,
            )
            self.assertEqual(pair["f1"], [[1, 2 / 3, 0]] * 2)
            self.assertEqual(len(auth), 4)
            self.assertEqual(pair["linked"], 1)
            for edge, d in (("activity", -2 / 3), ("temporal", -1)):
                (job,) = pair["jobs"][edge]
                self.assertEqual(
                    (job["question_index"], job["weak"], job["signed_credit"]),
                    (0, "swap", d),
                )

    def test_complete_update_uses_each_direct_credit_once_and_retains_veto(self):
        ids = [15, 19]
        pair = {
            "rows": [{"original_index": i, "category": 4} for i in ids],
            "f1": [[0, 0.5, 1], [0, 0, 1]],
            "jobs": {"activity": [{"qa": {"original_index": i}} for i in ids]},
        }
        ok = {
            "baseline": "activity",
            "index": 15,
            "raw": 0.5,
            "credit": 0.5,
            "status": "confirmed",
        }
        veto = {
            "baseline": "activity",
            "index": 19,
            "raw": 1.0,
            "credit": 0.0,
            "status": "veto",
        }
        cases = [ok, veto, dict(ok, baseline="temporal", raw=1, credit=1)]
        self.assertEqual(
            [q["edges"][2] for q in points(pair, {"cases": cases})], [0.5, 0]
        )
        with self.assertRaises(ValueError):
            points(pair, {"cases": cases + [ok]})
        with self.assertRaises(ValueError):
            points(pair, {"cases": [ok]})

    def test_exact_gpu_response_can_be_reused_only_with_same_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "old", root / "trial"
            url = "http://127.0.0.1:18773/v1"
            env = {
                "MEMSKILL_API_BASE": url,
                "MEMSKILL_BACKEND_ID": BACKEND,
                "MEMSKILL_GREEDY_EVAL_CACHE": str(old),
            }
            with patch.dict("os.environ", env):
                params = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "prompt"}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 42,
                    "max_tokens": 32,
                }
                _, key = request_cache(url, params, "reader")
                previous = publish_response(old, key, sha("prompt"), "answer", 20, 2)
                with patch(
                    "experiments.controller.exchange_questions.subprocess.run",
                    side_effect=RuntimeError("cache miss"),
                ):
                    receipt = call(new, "prompt", "reader", 32, reuse_from=old)
                    self.assertEqual(previous["sha256"], receipt["sha256"])
                    self.assertEqual(read_response(new / "responses", key), previous)
                    with self.assertRaisesRegex(RuntimeError, "cache miss"):
                        call(new, "prompt", "reader", 31, reuse_from=old)
