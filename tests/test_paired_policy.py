import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from experiments.controller.paired_grounding import reuse_identical_choices
from experiments.controller.paired_policy import PairedPolicy, fit_policy
from src.controller import PPOController
from src.replay import attributed_turn_pairs


class PairedPolicyTest(unittest.TestCase):
    def test_same_noise_replays_and_native_greedy_is_preserved(self):
        torch.manual_seed(41)
        native = PPOController(state_dim=5, op_dim=4, hidden_dim=8, action_top_k=3, device="cpu")
        state, ops = torch.randn(5), torch.randn(6, 4)
        expected, _, _ = native(state, ops, deterministic=True)
        for target in range(len(ops)):
            with self.subTest(target=target):
                p = PairedPolicy(native, target)
                noise = {}
                original = p(state, ops, record_noise=noise)
                self.assertEqual(original, p(state, ops, replay_noise=noise))
                self.assertEqual(len(p.decisions), 1)
                alternative = noise["paired_alternative"]
                self.assertEqual(sum(a != b for a,b in zip(original[0], alternative)), 1)
                self.assertEqual(len(set(alternative)), 3)
                self.assertAlmostEqual(original[1], math.log(0.5))
                p.mode = "native"
                self.assertEqual(p(state, ops, deterministic=True)[0], expected)

    def test_signed_gains_and_abstentions(self):
        def example(i, use, delta):
            return {"episode": str(i), "terminal": 0.3, "use": use, "delta": delta,
                    "state": [1., -0.5], "base_logit": -1.}
        labeled = [example(0, True, 0.4), example(1, False, -0.4)]
        unlabeled = [dict(row, delta=0.0) for row in labeled]
        self.assertTrue(torch.equal(fit_policy(unlabeled, 1), fit_policy(labeled, 0)))
        weights = fit_policy(labeled, 1)
        x = torch.tensor([1., 1./math.sqrt(1.25), -0.5/math.sqrt(1.25)])
        self.assertGreater(float(weights @ x) - 1, 0)
        self.assertLess(float(fit_policy([dict(r, delta=-r["delta"]) for r in labeled], 1) @ x), 0)

    def test_grounded_skill_uses_only_dated_source_and_handles_odd_turns(self):
        text = ('DATE: 7:00 pm on 21 May, 2023\nCONVERSATION:\nA said, "I joined yesterday."\n'
                'B said, "Congratulations!"\nA said, "Thanks!" and shared a team photo.')
        rows = attributed_turn_pairs(text)
        self.assertEqual(len(rows), 2)
        self.assertIn('A: I joined yesterday (20 May 2023). B: Congratulations!', rows[0])
        self.assertIn('A: Thanks! [shared a team photo.]', rows[1])
        self.assertFalse(attributed_turn_pairs("No date header"))

    def test_long_history_role_and_date_are_retained_without_dropping_words(self):
        body = " ".join(f"word{i}" for i in range(225))
        src = f"Session Date: 2023/04/02 (Sun) 00:00\n\nuser: {body}\n\nassistant: Details\ncontinued."
        chunks = attributed_turn_pairs(src)
        self.assertEqual(len(chunks), 3)
        words = " ".join(c.split(" | ", 1)[1] for c in chunks)
        self.assertEqual(words, f"user: {body} assistant: Details continued.")
        self.assertTrue(all(c.startswith("2023/04/02 (Sun) 00:00 | ") for c in chunks))

    def test_reuse_requires_identical_reachable_choices_and_model_inputs(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            native = base / "native" / "history"
            (native / "traces").mkdir(parents=True)
            for name in ("scores.json", "question_ids.json"):
                (native/name).write_text("[]")
            (native/"inputs.json").write_text(json.dumps({"arm":"native", "backend":"fixed"}))
            (native/"decisions.json").write_text(json.dumps([{"use":False,"prob":0.2,"base_logit":-1,"state":[1.]}]))
            dst = base / "credit" / "history"
            dst.parent.mkdir()
            self.assertFalse(reuse_identical_choices(dst, "credit", torch.tensor([3.,0.]), {"arm":"credit","backend":"fixed"}))
            self.assertFalse(reuse_identical_choices(dst, "credit", None, {"arm":"credit","backend":"different"}))
            self.assertTrue(reuse_identical_choices(dst, "credit", torch.tensor([0.2,0.]), {"arm":"credit","backend":"fixed"}))
            self.assertEqual(json.loads((dst/"reuse.json").read_text())["checked_decisions"], 1)
            self.assertFalse(json.loads((dst/"decisions.json").read_text())[0]["use"])


if __name__ == "__main__":
    unittest.main()
