import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

import torch

from experiments.controller.audit_paired import audit_rate_control
from experiments.controller.audit_trajectory_return import policy_value
from experiments.controller.evaluate_last_switches import final_switches
from experiments.controller.paired_grounding import load_arm_weights, reuse_identical_choices
from experiments.controller.paired_policy import (PairedPolicy, count_schedule, fit_policy, fit_anchored_credit,
    fit_constrained_placement, fit_memory_novelty, fit_source_count_correction, match_training_probability, memory_novelty, novelty_logit_shift)
from scripts.experiments.serve_low_duty import DutyCycle
from src.controller import PPOController
from src.replay import attributed_turn_pairs, dated_turn_groups


class PairedPolicyTest(unittest.TestCase):
    def test_empty_final_longmem_turn_keeps_its_source_position(self):
        text = "Session Date: 2023-06-01\n\nassistant: Hello\n\nuser: Again\n\nassistant: "
        self.assertEqual(dated_turn_groups(text, 1), ["2023-06-01 | assistant: Hello",
                         "2023-06-01 | user: Again", "2023-06-01 | assistant:"])
        self.assertEqual(dated_turn_groups(text, 2), ["2023-06-01 | assistant: Hello user: Again",
                         "2023-06-01 | assistant:"])

    def test_equal_source_arms_keep_identical_words_on_multiline_locomo_and_longmem(self):
        src = ('DATE: Thursday on 1 June, 2023\nCONVERSATION:\nTim said, "First line\nsecond yesterday."\n'
               'John said, "I have a picture" and shared a photo.\nTim said, "Thanks!"')
        singles = dated_turn_groups(src, 1)
        self.assertEqual(len(singles), 3)
        self.assertIn("second yesterday (31 May 2023).", singles[0])
        self.assertIn("[shared a photo.]", singles[1])
        long = "Session Date: 2023-01-01\n\nuser: " + "long " * 145 + "\n\nassistant: First line\nsecond line."
        for source in (src, long):
            texts = [[" ".join(x.partition(" | ")[2].split()) for x in dated_turn_groups(source, k)] for k in (1, 2)]
            self.assertEqual(" ".join(texts[0]), " ".join(texts[1]))
            self.assertTrue(all(len(part.split()) <= 96 for group in texts for part in group))
        self.assertEqual(len(dated_turn_groups(src, 2)), 2)

    def test_fixed_pair_retains_native_common_skills_and_exactly_one_source_rule(self):
        torch.manual_seed(73)
        native = PPOController(state_dim=5, op_dim=4, hidden_dim=8, action_top_k=3, device="cpu")
        state, ops = torch.randn(5), torch.randn(6, 4)
        logits = native.get_action_logits(native.encode_state(state[None]), native.encode_ops(ops[None]))[0]
        ordered = logits.argsort(descending=True).tolist()
        for target, other in ((ordered[0], ordered[1]), (ordered[-1], ordered[-2])):
            p = PairedPolicy(native, target, other)
            common = [i for i in ordered if i not in (target, other)][:2]
            for mode, chosen, prob in (("plain", other, 0.), ("always", target, 1.)):
                p.mode = mode
                self.assertEqual(p(state, ops, deterministic=True)[0], common + [chosen])
                self.assertEqual(p.decisions[-1]["prob"], prob)
            p.mode = "explore"
            noise = {}
            actions, _, _ = p(state, ops, record_noise=noise)
            self.assertEqual(actions[:2], noise["paired_alternative"][:2])
            self.assertEqual({actions[-1], noise["paired_alternative"][-1]}, {target, other})

    def test_constrained_credit_redirects_a_fixed_training_rate_and_kl(self):
        rows = [dict(state=[x, 0.], base_logit=0., use=True, delta=.4*x) for x in (-1., 1.)]
        rate, anchor = torch.zeros(3), torch.tensor([0., -.8, 0.])
        w, fit = fit_constrained_placement(rows, rate, anchor)
        self.assertGreater(w[1], 0.)
        self.assertAlmostEqual(fit["achieved_rate"], .5, places=6)
        self.assertAlmostEqual(fit["achieved_kl"], fit["target_kl"], places=7)
        with self.assertRaises(ValueError):
            fit_constrained_placement(rows, torch.tensor([1., 0., 0.]), anchor)
        fallback, info = fit_constrained_placement([dict(r, delta=0.) for r in rows], rate, anchor)
        self.assertTrue(torch.equal(fallback, anchor))
        self.assertEqual(info["fit"], "anchor")

    def test_final_switch_marks_whether_both_policies_reached_the_observed_prefix(self):
        arms = {"credit": [.8, .6], "credit_rate": [.1, .55], "anchor": [.6, .49],
                "anchor_rate": [.7, .5], "novelty": [.2, .7]}
        episode = {"uses": [True, False], "probabilities": arms}
        switched = final_switches(episode)
        self.assertEqual(set(switched), {"anchor", "novelty"})
        self.assertEqual(switched["anchor"]["probabilities"], [.49, .5])
        self.assertTrue(switched["anchor"]["both_follow_recorded_prefix"])
        self.assertFalse(switched["novelty"]["both_follow_recorded_prefix"])
        episode["uses"][0] = False
        self.assertFalse(final_switches(episode)["anchor"]["both_follow_recorded_prefix"])

    def test_trajectory_estimate_recovers_exact_adaptive_two_step_return(self):
        episodes = []
        for first in (False, True):
            for second in (False, True):
                # Only the real history with first=1 and second=0 pays a reward.
                episodes.append({"uses": [first, second], "probabilities": {"target": [.8, .25 if first else .9]},
                                 "questions": 5, "return": float(first and not second), "baseline": .4})
        estimates = policy_value(episodes, "target")
        for key in ("baseline_adjusted_f1", "ordinary_is_f1", "self_normalized_f1"):
            self.assertAlmostEqual(estimates[key], .8 * .75)
        short = [dict(episodes[0], probabilities={"target": [1., 1.]})]
        self.assertIsNone(policy_value(short, "target")["self_normalized_f1"])

    def test_novelty_uses_only_pre_action_memory_and_preserves_mean_probability(self):
        states = [[1., 0., 0., 0.], [1., 0., 2., 0.], [1., 0., 0., 2.], [1., 0., -2., 0.]]
        rows = [dict(state=x, base_logit=-.2, use=True, delta=d) for x, d in zip(states, [0., -.4, 0., .4])]
        fit = fit_memory_novelty(rows, torch.zeros(5))
        self.assertIsNone(memory_novelty(states[0]))
        self.assertEqual([memory_novelty(s) for s in states[1:]], [0., 1., 2.])
        shifts = [novelty_logit_shift(s, fit) for s in states]
        self.assertEqual(shifts[0], fit["intercept"])
        self.assertLess(shifts[1], shifts[3])
        base = torch.full((4,), -.2, dtype=torch.float64)
        self.assertAlmostEqual(float((base + torch.tensor(shifts)).sigmoid().mean()), float(base.sigmoid().mean()), 7)
        self.assertEqual(fit_memory_novelty([rows[0]], torch.zeros(5))["slope"], 0.)

    def test_source_count_can_change_order_while_preserving_training_propensity(self):
        anchor = torch.tensor([-.35, 0.])
        rows = [dict(state=[1.], base_logit=i/10, use=True, delta=delta, source_item_count=n)
                for i, (n, delta) in enumerate([(0, -.5), (1, 0.), (2, 0.), (9, .4)])]
        intercept, slope = fit_source_count_correction(rows, anchor)
        base = [r["base_logit"] + float(anchor[0]) for r in rows]
        logits = [x + intercept + slope * math.log1p(r["source_item_count"]) for x, r in zip(base, rows)]
        self.assertGreater(slope, 0.)
        self.assertLess(logits[0], base[0])
        self.assertGreater(logits[-1], base[-1])
        self.assertAlmostEqual(torch.tensor(base).sigmoid().mean().item(), torch.tensor(logits).sigmoid().mean().item(), 6)
        self.assertEqual(fit_source_count_correction([dict(r, source_item_count=2) for r in rows], anchor), (0., 0.))

    def test_count_control_uses_only_the_count_and_executes_the_entire_schedule(self):
        choices = [True, True, True, False, False, True]
        schedule = count_schedule(42, "conv-47", choices)
        self.assertEqual(schedule, count_schedule(42, "conv-47", list(reversed(choices))))
        self.assertEqual(sum(schedule), sum(choices))
        self.assertEqual(count_schedule(42, "conv-47", [False] * 3), [False] * 3)
        self.assertEqual(count_schedule(42, "conv-47", [True] * 3), [True] * 3)
        with self.assertRaises(ValueError):
            count_schedule(42, "conv-47", [])
        native = PPOController(state_dim=5, op_dim=4, hidden_dim=8, action_top_k=3, device="cpu")
        gate = PairedPolicy(native, 0)
        gate.mode, gate.schedule = "count", schedule
        ops = torch.randn(6, 4)
        for use in schedule:
            selected, log_prob, _ = gate(torch.randn(5), ops, deterministic=True)
            self.assertEqual(0 in selected, use)
            self.assertEqual(len(set(selected)), 3)
            self.assertEqual(log_prob, 0.)
        self.assertEqual([r["prob"] for r in gate.decisions], list(map(float, schedule)))

    def test_split_writer_accounts_for_both_gpus_before_rest(self):
        with patch("torch.cuda.synchronize") as sync, patch("time.sleep") as sleep:
            with patch("time.monotonic", side_effect=[10, 12, 18]):
                clock = DutyCycle(0.25, 2)
                clock.rest()
            self.assertEqual(sync.call_args_list, [call(0), call(1)])
            sleep.assert_called_once_with(6.0)
            self.assertEqual(clock.started, 18)

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

    def test_rate_control_preserves_terminal_slopes_and_matches_mean_probability(self):
        rows = [{"state": [float(x)], "base_logit": -0.5} for x in (-1, 1, 1)]
        terminal, credit = torch.tensor([-0.4, 0.2]), torch.tensor([1.0, -1.5])
        rate = match_training_probability(rows, terminal, credit)
        states = torch.tensor([-1., 1., 1.])
        def probs(w):
            return (-0.5 + w[0] + states*w[1]).sigmoid()
        self.assertTrue(torch.equal(rate[1:], terminal[1:]))
        self.assertAlmostEqual(probs(rate).mean().item(), probs(credit).mean().item(), places=6)
        self.assertFalse(torch.allclose(probs(rate), probs(credit)))
        self.assertTrue(torch.equal(match_training_probability(rows, terminal, terminal), terminal))

    def test_legacy_rate_fit_uses_only_training_then_both_benchmarks_load_frozen_weights(self):
        with TemporaryDirectory() as tmp:
            study = Path(tmp)
            (study / "protocol.json").write_text(json.dumps({"train_sessions": [[1, 2]], "train": ["sample"]}))
            terminal, credit = torch.tensor([-0.4, 0.2]), torch.tensor([1., -1.5])
            for name, weight in (("terminal", terminal), ("credit", credit)):
                (study / (name + ".json")).write_text(json.dumps(weight.tolist()))
            rows = [{"state": [float(x)], "base_logit": -0.5} for x in (-1, 1, 1)]
            with patch("experiments.controller.paired_grounding.load_episode", return_value=rows) as read_rows:
                rate = load_arm_weights(study, "rate")
                self.assertTrue(torch.equal(rate, load_arm_weights(study, "rate")))
                read_rows.assert_called_once_with(study / "train/sample_s1", "sample", 2, deployment_writer=False)
            weights = {name: w.numpy() for name, w in (("terminal", terminal), ("credit", credit), ("rate", rate))}
            with patch("experiments.controller.audit_paired.load_episode", return_value=rows):
                self.assertEqual(audit_rate_control(study, weights)["training_spans"], 3)
                for bad in (rate + torch.tensor([1., 0.]), credit):
                    with self.assertRaises(ValueError):
                        audit_rate_control(study, dict(weights, rate=bad.numpy()))

    def test_credit_anchor_preserves_the_terminal_actor_without_local_evidence(self):
        terminal = torch.tensor([-0.5, 0.1])
        rows = [{"use": False, "delta": 0.0, "base_logit": -0.1, "state": [x]} for x in (1., -1.)]
        self.assertTrue(torch.equal(fit_anchored_credit(rows, terminal), terminal))
        rows[0]["delta"] = -0.4  # Swapping the sampled plain action for the grounded action helped.
        learned = fit_anchored_credit(rows, terminal)
        rate = match_training_probability(rows, terminal, learned)
        def logits(w):
            return -0.1 + w[0] + torch.tensor([1., -1.]) * w[1]
        shift = logits(learned) - logits(terminal)
        self.assertGreater(shift[0].item(), 0.0)
        self.assertLess(abs(shift[1].item()), abs((logits(rate) - logits(terminal))[1].item()))
        self.assertAlmostEqual(logits(learned).sigmoid().mean().item(), logits(rate).sigmoid().mean().item(), places=6)

    def test_centered_credit_separates_states_at_the_reference_training_rate(self):
        terminal = torch.tensor([-.4, .1, .2])
        rows = [{"use": True, "delta": d, "base_logit": -.2, "state": [1., y]}
                for d, y in [(.4, .2), (0., 0.), (0., -.2)]]
        anchor = fit_anchored_credit(rows, terminal)
        centered = match_training_probability(rows, fit_anchored_credit(rows, terminal, center_states=True), anchor)
        x = torch.nn.functional.normalize(torch.tensor([r["state"] for r in rows]), dim=1)
        def probabilities(w):
            return (-.2 + w[0] + x @ w[1:]).sigmoid()
        self.assertAlmostEqual(probabilities(anchor).mean().item(), probabilities(centered).mean().item(), places=6)
        self.assertGreater(probabilities(centered)[0].item(), probabilities(centered)[-1].item())
        self.assertFalse(torch.allclose(probabilities(anchor), probabilities(centered)))
        identical = [dict(r, state=[1., 0.]) for r in rows]
        self.assertTrue(torch.equal(fit_anchored_credit(identical, terminal, center_states=True),
                                    fit_anchored_credit(identical, terminal)))

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
            context = {"algorithm": "source-ranges-v1", "context_tokens": 8192}
            self.assertFalse(reuse_identical_choices(dst, "credit", None, {"arm":"credit","backend":"fixed","writer_context":context}))
            (native/"inputs.json").write_text(json.dumps({"arm":"native","backend":"fixed","writer_context":context}))
            self.assertFalse(reuse_identical_choices(dst, "credit", None, {"arm":"credit","backend":"fixed"}))
            (native/"inputs.json").write_text(json.dumps({"arm":"native","backend":"fixed"}))
            self.assertTrue(reuse_identical_choices(dst, "credit", torch.tensor([0.2,0.]), {"arm":"credit","backend":"fixed"}))
            self.assertEqual(json.loads((dst/"reuse.json").read_text())["checked_decisions"], 1)
            self.assertFalse(json.loads((dst/"decisions.json").read_text())[0]["use"])
            control = base / "count" / "history"
            control.parent.mkdir()
            def signature(schedule):
                return {"arm": "count", "backend": "fixed", "count_control": {"schedule": schedule}}
            self.assertFalse(reuse_identical_choices(control, "count", None, signature([True])))
            self.assertFalse(reuse_identical_choices(control, "count", None, signature([False, True])))
            self.assertTrue(reuse_identical_choices(control, "count", None, signature([False])))
            self.assertEqual(json.loads((control/"decisions.json").read_text())[0]["prob"], 0.)


if __name__ == "__main__":
    unittest.main()
