import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.controller.audit_cross_window import PLAN, held_models, register_models
from experiments.controller.source_compression import file_hash

from experiments.controller.cross_window_prefix import (
    background,
    compare_prefixes,
    endpoint_cohort,
    focused_conversation,
    probabilities,
    read_origins,
    summarize,
)
from experiments.controller.late_pair_policy import STEPS


class CrossWindowPrefixTests(unittest.TestCase):
    def test_each_cross_block_question_once_and_all_intervening_sessions_retained(self):
        sessions = {}
        for i in range(1, 7):
            sessions[f"session_{i}"] = [{"dia_id": f"D{i}:1", "text": "turn"}]
            sessions[f"session_{i}_date_time"] = str(i)
        qa = [
            dict(category=1, evidence=["D1:1", "D6:1"]),
            dict(category=2, evidence=["D2:1", "D3:1"]),
            dict(category=1, evidence=["D1:1", "D2:1"]),
            dict(category=5, evidence=["D1:1", "D5:1"]),
            dict(category=3, evidence=["missing", "D5:1"]),
            dict(category=3, evidence=[]),
        ]
        convo = dict(conversation=sessions, qa=qa)
        rows = endpoint_cohort({"held": convo, "in-fit": convo}, {"in-fit"})
        self.assertEqual(
            [(r["dialogue"], r["end"]) for r in rows],
            [("held", 3), ("in-fit", 3), ("held", 6), ("in-fit", 6)],
        )
        self.assertEqual(
            [r["questions"][0]["original_index"] for r in rows], [1, 1, 0, 0]
        )
        c = focused_conversation(convo, rows[2])
        self.assertEqual(len(c["conversation"]), 12)
        self.assertEqual(c["qa"][0]["original_index"], 0)
        self.assertEqual(c["conversation"]["session_4"], sessions["session_4"])
        self.assertEqual(rows[0]["background"], rows[2]["background"][:2])
        self.assertEqual(rows[2]["background"][3], background("held", 4))

    def test_read_origins_use_the_retrieved_version_of_an_updated_memory(self):
        spans = [
            dict(
                span_id=0,
                changes=[
                    dict(memory_id=7, version=1, after="v1"),
                    dict(memory_id=8, version=1, after="x"),
                ],
            ),
            dict(span_id=2, changes=[dict(memory_id=7, version=2, after="v2")]),
        ]
        self.assertEqual(
            read_origins(spans, [dict(query_id="conv_0", refs=[[7, 2], [8, 1]])]),
            {0: [3, 1]},
        )

    def test_two_main_fits_lock_preexisting_scores_once(self):
        def write(p, data):
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_text(json.dumps(data))

        with TemporaryDirectory() as folder:
            dest = Path(folder)
            root = dest / "cross-window-prefix"
            out = root / "main-crossfit"
            out.mkdir(parents=True)
            write(dest / "frozen.json", {"frozen": True})
            baseline = file_hash(dest / "frozen.json")
            pilot_hash = {}
            for key, family in (("strength", "prefix-return"), ("rank", "rank-return")):
                path = dest / family / "pilot/forecast.json"
                write(path, {"inputs": {"history": baseline}, "fits": {"pilot": {}}})
                pilot_hash[key] = file_hash(path)
                inputs = {"history": baseline}
                if key == "rank":
                    inputs["strength_forecast"] = file_hash(
                        dest / "prefix-return/main/forecast.json"
                    )
                write(
                    dest / family / "main/forecast.json", {"inputs": inputs, "fits": {}}
                )
            write(root / "manifest.json", {"baseline": baseline, "pilots": pilot_hash})
            write(root / "main-plan.json", json.loads(PLAN.read_bytes()))
            older = root / "conv-A_to03" / "100" / "scores.json"
            write(older, {"scores": [0.5]})
            receipt = register_models(dest, root, out)[2]
            self.assertEqual(
                list(receipt["published_branches"]), ["conv-A_to03/100/scores.json"]
            )
            write(root / "conv-B_to03" / "100" / "scores.json", {"scores": [0.2]})
            self.assertEqual(register_models(dest, root, out)[2], receipt)
            write(older, {"scores": [0.8]})
            with self.assertRaises(ValueError):
                register_models(dest, root, out)

    def test_crossfit_rejects_target_history_in_either_parameter_set(self):
        ids = set("ABCDEF")

        def fit():
            folds = {d: dict(gate=dict(dialogues=sorted(ids - {d}))) for d in ids}
            folds["deployment"] = dict(gate=dict(dialogues=sorted(ids)))
            return dict(stage="main", fits=folds)

        records = {"strength": fit(), "rank": fit()}
        self.assertEqual(
            held_models(records, "A", ids)["rank"], records["rank"]["fits"]["A"]
        )
        records["rank"]["fits"]["A"]["gate"]["dialogues"].append("A")
        with self.assertRaises(ValueError):
            held_models(records, "A", ids)

    def test_inference_records_use_numeric_reader_query_ids(self):
        spans = [dict(span_id=1, changes=[dict(memory_id=1, version=1, after="item")])]
        self.assertEqual(
            read_origins(spans, [dict(query_id=0, refs=[[1, 1]])]), {0: [2]}
        )

    def test_requires_identical_all_earlier_writes_and_final_state(self):
        def branch(bit):
            return (
                [
                    dict(use=a, base_logit=0.1, state=[1.0], plain=[0], grounded=[1])
                    for a in (1, 0, bit)
                ],
                {},
                {},
            )

        cells = [branch(0), branch(1)]

        def trace():
            return [
                dict(
                    source=str(i),
                    source_turns=[i],
                    retrieved=[],
                    refs=[],
                    span_id=i,
                    changes=[],
                )
                for i in range(3)
            ]

        a, b = trace(), trace()
        b[-1]["changes"] = ["different memory"]
        self.assertEqual(compare_prefixes(cells, [a, b], [1, 0]), cells[0][0][-1])
        b[1]["changes"] = ["earlier memory"]
        with self.assertRaises(ValueError):
            compare_prefixes(cells, [a, b], [1, 0])
        b[1]["changes"] = []
        cells[1][0][-1]["base_logit"] += 0.1
        with self.assertRaises(ValueError):
            compare_prefixes(cells, [a, b], [1, 0])

    def test_probabilities_use_previous_observed_skill_and_focal_scores_are_not_pooled_per_endpoint(
        self,
    ):
        direction = dict(
            center=[0.0], scale=[1.0], slope=[1.0], shifts={str(s): 0.0 for s in STEPS}
        )
        strength = dict(steps=[0.0, 0.0, 2.0], shared=0.0, rate_shifts=[0.0] * 3)
        ranks = dict(
            rules=[0, 0, 1],
            shared_rule=0,
            context_rules=[0] * 3,
            cuts=[[0.0] * 3] * 3,
            rate_shifts=[0.0] * 3,
        )
        pilots = {
            "strength": dict(direction=direction, gate=strength),
            "rank": dict(direction=direction, gate=ranks, strength=strength),
        }
        get = lambda last: probabilities(
            dict(state=[1.0], base_logit=0.0), last, [1.0], [0.0, 0.0], pilots
        )
        self.assertGreater(get(1)["strength"], get(0)["strength"])
        self.assertLess(get(1)["rank"], get(0)["rank"])
        probs = get(0)

        def q(y):
            return dict(
                type=1,
                sessions=[1, 4],
                temporal=y,
                activity=y,
                focal={k: y for k in probs},
                last_writer_sessions={"activity": [1, 4], "temporal": [4]},
            )

        r = summarize(
            [
                dict(dialogue="A", end=4, probabilities=probs, questions=[q(1)]),
                dict(dialogue="B", end=4, probabilities=probs, questions=[q(0), q(0)]),
            ],
            complete=True,
        )
        self.assertEqual(
            r["all"]["last_writes_cover_all_evidence_sessions"],
            {"activity": 1.0, "temporal": 0.0},
        )
        self.assertEqual(r["paired_dialogue_bootstrap"]["rank-native"], [0.0, 0.0])
        self.assertAlmostEqual(r["all"]["f1"]["native"], 1 / 3)
        self.assertEqual(r["evidence_gap"]["3"]["questions"], 3)
