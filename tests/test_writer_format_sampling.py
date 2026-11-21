import unittest
from itertools import combinations
from pathlib import Path
from statistics import mean, pvariance
from tempfile import TemporaryDirectory

from experiments.controller.factorial_rollout import read
from experiments.controller.source_compression import file_hash, register
from experiments.controller.writer_format_policy import surveyed_samples
from experiments.controller.writer_format_sampling import (
    auxiliary,
    estimate,
    measured,
    report,
)
from experiments.controller.writer_format_trial import CELLS, effects


class SampledCreditTests(unittest.TestCase):
    def test_estimated_split_is_the_measured_shapley_split(self):
        v = dict(zip(CELLS, (0.76, 0.29, 0.62, 0.52), strict=True))
        total = v[CELLS[-1]] - v[CELLS[0]]
        diff = measured({"reads": {k: {"f1": z} for k, z in v.items()}})
        s = {"windows": {"a": {"n": 1, "corrected": diff}, "b": {"excluded": "failed"}}}
        (a,) = surveyed_samples(
            [{"node": "a", "raw": total}, {"node": "b", "raw": 0}], s, "corrected"
        )
        true = effects(v)
        for z, k in zip(a["component_raw"], ("template", "grouping"), strict=True):
            self.assertAlmostEqual(z, true[k])

    def test_residual_estimator_and_its_variance_across_every_sample(self):
        y = [1.0, -0.3, 0.4, 0.0, -0.15]
        rows = {
            0: {"exact": y[0]},
            **{k: {"proxy": v} for k, v in enumerate([0.6, 0.5, -0.7, 0.3], 1)},
        }
        for k in (1, 2, 3, 4):
            values = dict(enumerate(y))
            outcomes = [estimate(rows, s, values) for s in combinations(range(1, 5), k)]
            for method in ("corrected", "sample_only"):
                self.assertAlmostEqual(mean(d[method] for d in outcomes), mean(y))
            if k == 1:
                self.assertIsNone(outcomes[0]["variance"])
            else:
                self.assertAlmostEqual(
                    mean(d["variance"] for d in outcomes),
                    pvariance(d["corrected"] for d in outcomes),
                )
        self.assertIsNone(estimate(rows, [], {})["corrected"])
        self.assertEqual(estimate({0: {"exact": 0.2}}, [], {})["variance"], 0)
        with self.assertRaises(ValueError):
            estimate(rows, [2, 2], dict(enumerate(y)))

    def test_auxiliary_is_blind_to_new_answers_and_ties(self):
        arms = {k: {"prompt": "x y"} for k in CELLS}
        arms[CELLS[0]].update(prompt="first old", old={"f1": 1.0})
        arms[CELLS[-1]].update(prompt="second old", old={"f1": 0.0})
        self.assertEqual(auxiliary(arms), 0.0)
        arms["relation_pairs"]["prompt"] = "first old"
        self.assertEqual(auxiliary(arms), 0.5)

    def test_failed_draw_is_not_replaced_by_another_available_question(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp)
            out = src / "sampling"
            out.mkdir()
            f = out / "frame.json"
            p = src / "questions/w"
            p.mkdir(parents=True)
            register(
                f,
                {
                    "inputs": {"registration": "r"},
                    "windows": {
                        "w": {
                            "n": 3,
                            "sample": ["1", "2"],
                            "prepared": "p",
                            "rows": {i: {"proxy": 0.0} for i in ("1", "2", "3")},
                        }
                    },
                },
            )
            receipt = {"inputs": {"registration": "r", "parent": "p"}}
            register(p / "1.json", dict(receipt, status="different_pure_answer"))
            register(p / "3.json", dict(receipt, status="matched"))
            self.assertIsNone(report(src, f, 2))
            j = read(report(src, f, 1))
            self.assertEqual(j["frame"], file_hash(f))
            self.assertEqual(
                j["windows"]["w"]["excluded"], {"1": "different_pure_answer"}
            )
