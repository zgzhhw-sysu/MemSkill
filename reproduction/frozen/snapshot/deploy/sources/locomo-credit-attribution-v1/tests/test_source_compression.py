import unittest

import torch

from experiments.controller.audit_source_compression import check_forecasts, forecasts, summarize
from experiments.controller.factorial_rollout import CELLS, summarize_window, trajectory_masses
from experiments.controller.source_compression import compression_features, fit_actor


class SourceCompressionTests(unittest.TestCase):
    def test_singleton_and_pair_merge(self):
        eyes = torch.eye(2)
        features = compression_features(eyes, eyes)
        self.assertTrue(torch.allclose(features, torch.zeros_like(features)))
        a, b, crowding = compression_features(eyes, eyes[0:1]).tolist()
        self.assertAlmostEqual(a, .5)
        self.assertAlmostEqual(b, 1.)
        self.assertAlmostEqual(crowding, 0.)

    def test_neutral_actor_when_no_verified_effects(self):
        x = torch.tensor([[1., 2., 3.], [1., 3., 4.]], dtype=torch.float64)
        _, scale, actor = fit_actor(x, x.new_zeros(2))
        self.assertEqual(float(scale[0]), 1.)
        self.assertFalse(bool(actor.any()))

    def test_forecasts_exclude_held_dialogue_and_future_outcomes(self):
        rows = []
        for sid in ("A","B","C"):
            for i in range(1,4):
                rows.append({"dialogue": sid, "session": i, "episode": f"{sid}-{i}", "terminal": .1*i, "state": [1.,2.],
                             "base_logit": -.3, "use": i == 1, "delta": 0., "signed": .1*(i-2), "features": [i,0.,1.]})
        future = [{"dialogue": r["dialogue"], "session": r["session"]+4, "features": r["features"]} for r in rows]
        fits, a = forecasts(rows, future)
        fits2, b = forecasts([dict(r, signed=-r["signed"], terminal=1.) if r["dialogue"] == "A" else r for r in rows], future)
        self.assertEqual(fits["A"], fits2["A"])
        self.assertEqual([r for r in a if r["dialogue"]=="A"], [r for r in b if r["dialogue"]=="A"])
        self.assertLess(a[0]["p_activity"], a[2]["p_activity"])
        self.assertEqual(fits["A"]["training_dialogues"], ["B","C"])
        inputs = {(r["dialogue"], r["session"]): r for r in future}
        report = {"fits": fits, "forecasts": a}
        check_forecasts(inputs, report)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_forecasts(inputs, dict(report, forecasts=a+[a[0]]))
        a[0]["p_activity"] = 1. - a[0]["p_activity"]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            check_forecasts(inputs, report)

    def test_matched_placement_and_question_weight(self):
        fit = {"A":{"terminal":[0.,0.],"matched_logit_shift":0.}}
        rows = [{"dialogue":"A", "session":i, "state":[1.], "base_logit": 0., "links":i,
                 "scores":[{"activity": float(i==1),"temporal": float(i==2)}]*i} for i in (1,2)]
        p = [{"dialogue":"A","session":i,"p_activity":v} for i,v in ((1,.8),(2,.4))]
        result = summarize(rows,fit,p)
        self.assertAlmostEqual(result["by_span"]["mean"]["credit"],.7)
        self.assertAlmostEqual(result["by_link"]["mean"]["credit"],2/3)
        self.assertAlmostEqual(result["by_span"]["mean"]["shuffled"],.5)
        self.assertAlmostEqual(result["by_link"]["mean"]["shuffled"],(0.6+2*.4)/3)
        self.assertEqual(result["by_span"]["mean"]["greedy"],1.)
        self.assertFalse(result["passes"])

class FactorialRolloutTests(unittest.TestCase):
    def test_conditional_path_mass_and_exact_count_placement(self):
        masses = trajectory_masses([.9,.2], [0.,0.,0.], [0.,-1.,1.], 0.)
        self.assertAlmostEqual(masses[1,0]["credit"],.72)
        self.assertEqual(masses[1,0]["greedy"],1.)
        self.assertAlmostEqual(masses[1,0]["shuffled"],(.72+.02)/2)
        self.assertEqual(masses[1,0]["greedy_shuffled"],.5)
        self.assertEqual(masses[1,1]["greedy_shuffled"],0.)
        self.assertEqual(masses[1,1]["terminal_greedy"],1.)
        self.assertGreater(masses[1,1]["terminal"],masses[0,1]["terminal"])
        for arm in masses[0,0]:
            self.assertAlmostEqual(sum(masses[k][arm] for k in CELLS),1.)

    def test_full_return_uses_scores_from_each_end_state(self):
        def state(v):
            return {"state":[float(v),1.], "base_logit":.1, "plain":[0,1,2], "grounded":[0,1,3]}
        cells = {(a,b):([state(0),state(a)],{"first":a},{0:float(a!=b)}) for a,b in CELLS}
        fit = {"terminal":[0.,0.,0.],"matched_logit_shift":.3}
        r = summarize_window(cells,[.9,.2],fit,[88])["per_question"][0]
        self.assertEqual(r["original_index"],88)
        self.assertAlmostEqual(r["f1"]["credit"],.74)
        self.assertEqual(r["f1"]["greedy"],1.)
        cells[(1,1)][0][1]["state"] = [7.,8.]
        with self.assertRaisesRegex(ValueError, "prefix"):
            summarize_window(cells,[.9,.2],fit,[88])
