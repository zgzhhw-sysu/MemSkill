import math
import unittest

from experiments.controller.audit_late_pairs import check_design
from experiments.controller.late_pair_policy import (
    STEPS,
    actor,
    calibrated_models,
    correction,
    logistic,
    select_step,
)
from experiments.controller.late_pair_trial import paths, summarize_pair


class LatePairTest(unittest.TestCase):
    def test_rate_matching_and_zero_step(self):
        rows = [
            dict(features=[i, i % 2], base_logit=-0.9 + i / 20, signed=(-1) ** i * 0.3) for i in range(8)
        ]
        f = actor(rows)
        self.assertAlmostEqual(correction(f, [100.0, 80.0], 0.0), 0.0)
        for step in STEPS:
            before = sum(logistic(r["base_logit"]) for r in rows)
            after = sum(logistic(r["base_logit"] + correction(f, r["features"], step)) for r in rows)
            self.assertAlmostEqual(before, after)

    def test_uncertain_or_one_dialogue_gain_does_not_move_policy(self):
        gains = {s: [0.0] * 5 for s in STEPS}
        gains[1.0] = [1.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(select_step(gains)[0], 0.0)
        gains[0.5] = gains[2.0] = [0.03, 0.01, 0.02, 0.04, 0.02]
        self.assertEqual(select_step(gains)[0], 0.5)

    def test_nested_holdout_does_not_use_test_dialogue_scores_or_credit(self):
        rows = [
            dict(
                dialogue=d,
                features=[i + j],
                base_logit=-1.0,
                signed=0.2 if i == 1 else -0.1,
                scores=[{"activity": 0.6 if i == 1 else 0.2, "temporal": 0.4}],
            )
            for i in range(2)
            for j, d in enumerate("ABCDEF")
        ]
        a = calibrated_models(rows)["A"]
        altered = [
            dict(r, signed=-r["signed"], scores=[{"activity": 0.0, "temporal": 1.0}])
            if r["dialogue"] == "A"
            else r
            for r in rows
        ]
        self.assertEqual(a, calibrated_models(altered)["A"])

    def test_two_path_estimator_includes_policy_reaction_to_first_action(self):
        cells = {}
        for a in (0, 1):
            for b in (0, 1):

                def dec(logit):
                    return dict(state=[1.0], base_logit=logit, plain=[0], grounded=[1])

                cells[a, b] = ([dec(0.0), dec(2 * a - 1)], {"first": a}, {0: 0.1 + 0.1 * a + 0.4 * b})
        pred = [
            dict(residual=0.4, unit_residual=0.6, source=0.25, component=0.51),
            dict(residual=-0.1, unit_residual=0.2, source=0.7, component=0.2),
        ]
        w = dict(dialogue="D", start=9, questions=[123], types=[4])
        out = check_design(w, cells, pred, [0.0, 0.0])
        g = lambda x: 1 / (1 + math.exp(-x))
        target = (1 - g(0.4)) * (0.1 + 0.4 * g(-1.1)) + g(0.4) * (0.2 + 0.4 * g(0.9))
        self.assertAlmostEqual(out["questions"][0]["returns"]["calibrated"], target)
        focus = dict(w, pivot=1, background=0)
        cells[0, 1][0][1]["base_logit"] += 0.2
        with self.assertRaises(ValueError):
            summarize_pair(focus, {k: cells[k] for k in paths(1, 0)}, pred, [0.0, 0.0])

    def test_resampling_counts_and_probability_mixture(self):
        from experiments.controller.step_mixture import propensity, resampling_weights

        calibration = {str(s): {"dialogue_gains": [0.0] * 5} for s in STEPS}
        self.assertEqual(resampling_weights(calibration), [1.0, 0.0, 0.0, 0.0, 0.0])
        # Only one dialogue supports the unit step. All resamples that include it vote for that step.
        calibration["1.0"]["dialogue_gains"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        weights = resampling_weights(calibration)
        self.assertAlmostEqual(weights[3], 1 - (4 / 5) ** 5)
        self.assertAlmostEqual(sum(weights), 1.0)
        offsets = [0.0, 0.3, 0.7, 1.3, 1.5]
        self.assertAlmostEqual(
            propensity(-1.0, weights, offsets), weights[0] * logistic(-1.0) + weights[3] * logistic(0.3)
        )

    def test_complete_blocks_cover_every_pivot_and_background(self):
        from experiments.controller.late_pair_policy import assignments

        convo = {}
        for i in range(1, 18):
            convo[f"session_{i}"] = [{"dia_id": f"D{i}:1"}]
            convo[f"session_{i}_date_time"] = "unused"
        data = {
            "held": {
                "conversation": convo,
                "qa": [dict(category=4, evidence=[f"D{i}:1"]) for i in (9, 11, 13, 15)]
                + [dict(category=5, evidence=["D9:1"]), dict(category=1, evidence=["D9:1", "D16:1"])],
            }
        }
        rows = assignments(data)
        self.assertEqual([r["questions"] for r in rows], [[0], [1], [2], [3]])
        self.assertEqual({(r["pivot"], r["background"]) for r in rows}, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertEqual(rows, assignments(data))

    def test_six_dialogue_resampling_uses_all_46656_draws(self):
        from experiments.controller.step_mixture import resampling_weights

        gains = {str(s): {"dialogue_gains": [0.0] * 6} for s in STEPS}
        gains["0.25"]["dialogue_gains"][-1] = 0.7
        p = resampling_weights(gains)
        self.assertAlmostEqual(p[1], 1 - (5 / 6) ** 6, places=12)
        self.assertAlmostEqual(p[0], (5 / 6) ** 6, places=12)

    def test_no_development_decision_on_a_partial_six_history_panel(self):
        from experiments.controller.late_pair_trial import aggregate

        arms = (
            "calibrated",
            "calibrated_greedy",
            "unit",
            "source",
            "terminal",
            "native",
            "permuted",
            "temporal",
            "activity",
        )
        scores = dict.fromkeys(arms, 0.4)
        reports = [
            {
                "dialogue": d,
                "pivot": j % 2,
                "sampling_mass": dict.fromkeys(arms, 1.0),
                "questions": [{"type": 1, "focal": scores, "ht": scores}],
            }
            for j, d in enumerate("ABCDEF")
        ]
        partial = aggregate(reports)
        self.assertFalse(partial["complete"])
        self.assertNotIn("eligible_for_development", partial)
        completed = aggregate(reports, complete=True)
        self.assertFalse(completed["eligible_for_development"])
        self.assertAlmostEqual(completed["all"]["arms"]["native"]["self_normalized"], 0.4)
