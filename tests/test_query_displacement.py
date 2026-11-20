import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller.legacy_terminal_symmetry import measure, report
from experiments.controller.policy_capacity import window_capacity
from experiments.controller.query_displacement_audit import score
from experiments.controller.query_displacement_data import encode_questions
from experiments.controller.query_displacement_policy import (
    compile_fold,
    contrast,
    index_mask,
    projected_queries,
    reciprocal_ranks,
    turn_mass,
    vote,
)
from experiments.controller.source_compression import file_hash


class QueryDisplacementTest(unittest.TestCase):
    def test_terminal_summary_checks_receipts_and_keeps_empty_histories(self):
        packet = {
            "jobs": [{"id": "a1", "history": "a", "weak": "swap"}],
            "root_separate": [{"id": "b0"}],
            "windows": {"a_s1": {}, "b_s3": {}},
            "questions": 2,
            "linked": 3,
        }
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "registration.json").write_text(json.dumps(packet))
            (out / "outcomes").mkdir()
            result = out / "outcomes/a1.json"
            result.write_text(
                json.dumps({"status": "veto", "veto": ["retrieval"], "credit": 0})
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                report(out, packet)
            (out / "outcomes-complete.json").write_text(
                json.dumps({"a1": file_hash(result)})
            )
            report(out, packet)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["cases"][0]["prior_repair_on_weaker"])
            self.assertEqual(summary["by_history"]["a"]["veto"], 1)
            self.assertEqual(summary["by_history"]["b"]["candidates"], 0)
            self.assertEqual(summary["root_cases_outside_terminal_audit"], ["b0"])
            result.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Changed terminal result"):
                report(out, packet)

    def test_embedding_batches_cannot_cross_held_histories(self):
        q = {"b/10": "long future question", "a/2": "short", "a/1": "extra"}
        calls = []

        def encode(strings):
            calls.append(strings)
            return np.full((len(strings), 2), sum(map(len, strings)))

        keys, values = encode_questions(q, encode)
        self.assertEqual(keys, ["a/1", "a/2", "b/10"])
        self.assertEqual(calls, [("extra", "short"), ("long future question",)])
        q["b/10"] = "changed"
        np.testing.assert_array_equal(encode_questions(q, encode)[1][:2], values[:2])

    def test_top_five_spans_and_stable_ties(self):
        query = torch.tensor([[1.0, 0.0]])
        vec = torch.tensor([[1.0, 0.0]] * 6)
        memberships = torch.eye(6, dtype=torch.bool)
        rr = reciprocal_ranks(query, (vec, memberships))[0]
        self.assertEqual((rr[0], rr[-1]), (1.0, 0.0))
        second = memberships.clone()
        second[0, 5] = True
        idx, mask = index_mask([[5]], query.device)
        d = contrast(query, [(vec, memberships), (vec, second)], idx, mask)
        self.assertEqual(float(d), 1.0)

    def test_mass_uses_the_same_turn_population_for_different_chunk_counts(self):
        q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        single = (
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            torch.eye(3, dtype=torch.bool),
        )
        paired = (
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool),
        )
        split = (
            single[0],
            torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.bool),
        )
        for group in (single, paired, split):
            torch.testing.assert_close(turn_mass(q, group), q.new_full((1, 3), 1 / 3))
        first = (q.new_tensor([[1.0, 0.0], [0.8, 0.6]]), torch.eye(2, dtype=torch.bool))
        second = (q.new_tensor([[1.0, 0.0], [0.96, 0.28]]), first[1])
        i, m = index_mask([[0]], q.device)
        self.assertEqual(float(contrast(q, [first, second], i, m)), 0.0)
        self.assertLess(float(contrast(q, [first, second], i, m, metric="mass")), 0.0)

    def test_displacement_identity_on_exact_evidence(self):
        v = torch.eye(2, dtype=torch.float64)
        f = {
            "evidence": v[None, :],
            "mask": torch.ones(1, 2, dtype=torch.bool),
            "average": v.mean(0, keepdim=True),
            "queries": v[:1],
        }
        groups = [
            (v, torch.eye(2, dtype=torch.bool)),
            (v.flip(0), torch.eye(2, dtype=torch.bool)),
        ]
        mapped, sim, vals = projected_queries(f, (v, groups))
        self.assertEqual(mapped.tolist(), [[0, 1]])
        self.assertEqual(float(sim), 1.0)
        torch.testing.assert_close(vals["credit"], vals["untranslated"])

    def test_vote_requires_donor_support_and_reverses_on_other_rankings(self):
        labels = torch.tensor([0.1, 0.1, -0.1], dtype=torch.float64)
        donor = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        target = -donor
        sim = donor
        histories = ["a", "b", "c", "d", "e"]
        shift, n, rev = vote(labels, donor, target, sim, ["a", "a", "b"], histories)
        self.assertEqual((n, rev), (2, 2))
        shift2, _, _ = vote(
            labels[:1], donor[:1], target[:1], sim[:1], ["a"], histories
        )
        self.assertAlmostEqual(shift, shift2)
        self.assertLess(shift, 0)
        self.assertAlmostEqual(
            shift, -vote(labels, donor, donor, sim, ["a", "a", "b"], histories)[0]
        )
        self.assertEqual(
            vote(-labels[:1], donor[:1], target[:1], sim[:1], ["a"], histories),
            (0.0, 0, 0),
        )

    def test_fold_removes_the_held_history_before_credit_rotation(self):
        histories = list("abcdef")
        vec = F.normalize(torch.tensor([[1.0, 2.0]]))
        rows = [
            {
                "history": h,
                "stage": 0,
                "linked": [0],
                "window": h + "_s1",
                "index": 1,
                "bits": ["00", "10"],
                "key": h,
                "node": h,
                "raw": i / 10,
                "credit": i / 10,
            }
            for i, h in enumerate(histories, 1)
        ]
        keys = dict.fromkeys(histories, vec[0])
        banks = {h: (vec, None) for h in histories}

        def fold(rs):
            return compile_fold(
                rs, "a", 0, histories, keys, banks, dict.fromkeys(range(6), 0.2)
            )

        all_six = compile_fold(
            rows, "new-history", 0, histories, keys, banks, dict.fromkeys(range(6), 0.2)
        )
        self.assertEqual(all_six["histories"], histories)
        clean = fold(rows)
        changed = copy.deepcopy(rows)
        changed[0].update(
            credit=100.0, raw=-200.0, key="unavailable", node="unavailable"
        )
        new = fold(changed)
        self.assertNotIn("a", new["owners"])
        for k in ("credit", "reassigned", "raw", "queries"):
            torch.testing.assert_close(clean[k], new[k])

    def test_earlier_repairs_follow_the_weaker_action_in_both_directions(self):
        qa = {
            "question": "What was the destination?",
            "answer": "London",
            "category": 4,
        }

        def response(_, prompt, *rest):
            return {
                "text": "London"
                if "signal" in prompt.split("Question:")[0]
                else "unknown"
            }

        for weak, high in [("control", "swap"), ("swap", "control")]:
            job = {"qa": qa, "weak": weak, "signed_credit": 0.7}
            branches = {
                weak: {
                    "native": ["unknown"],
                    "original": {"answer": "unknown", "f1": 0.0},
                    "execution": ["signal"],
                    "retrieval": None,
                    "format": ["unknown"],
                },
                high: {
                    "native": ["signal"],
                    "original": {"answer": "London", "f1": 1.0},
                },
            }
            with patch(
                "experiments.controller.legacy_terminal_symmetry.call",
                side_effect=response,
            ):
                scored = measure(None, job, {"branches": branches})
                self.assertEqual(
                    (scored["status"], scored["veto"], scored["credit"]),
                    ("veto", ["execution"], 0.0),
                )
                branches[weak]["execution"] = ["unknown"]
                self.assertEqual(
                    measure(None, job, {"branches": branches})["credit"], 0.7
                )
                branches[high]["original"]["answer"] = "different archived answer"
                self.assertEqual(
                    measure(None, job, {"branches": branches})["status"],
                    "native_answers_changed",
                )

    def test_direction_envelope_preserves_abstentions(self):
        probs = {"raw": 0.5, "native": 0.5, "credit": 0.25}
        w = {
            "window": "test",
            "dialogue": "external",
            "rows": [{"f1": probs}],
            "endpoints": [[0.0, 1.0]],
            "focal_probabilities": probs,
        }
        v = window_capacity(w, directional=True)
        self.assertAlmostEqual(v["same_direction"], 0.5)
        self.assertGreater(v["bounded"], v["same_direction"])
        probs["credit"] = 0.5
        self.assertAlmostEqual(
            window_capacity(w, directional=True)["same_direction"], 0.5
        )
        probs["credit"] = 0.6
        v = window_capacity(w, directional=True)
        self.assertAlmostEqual(v["shrink_only"], 0.6)
        self.assertAlmostEqual(v["same_direction"], v["bounded"])

    def test_window_oracle_cannot_select_actions_per_question(self):
        endpoints = [(1.0, 0.0), (0.0, 1.0)]
        p = {"native": 0.6, "raw": 0.5, "credit": 0.2}
        rows = [
            {"f1": {k: (1 - z) * lo + z * hi for k, z in p.items()}}
            for lo, hi in endpoints
        ]
        w = {
            "window": "w",
            "dialogue": "h",
            "rows": rows,
            "endpoints": endpoints,
            "focal_probabilities": p,
        }
        v = window_capacity(w)
        self.assertEqual((v["bounded"], v["unbounded"]), (0.5, 0.5))
        w.pop("focal_probabilities")
        w.update(
            cells={
                "00": [1.0, 0.0],
                "01": [0.0, 1.0],
                "10": [0.0, 1.0],
                "11": [1.0, 0.0],
            },
            probabilities={str(i): p for i in range(3)},
        )
        self.assertEqual(window_capacity(w)["unbounded"], 0.5)
        self.assertEqual(
            score(
                [
                    {
                        "actual": 1.0,
                        "translated": 0.0,
                        "untranslated": 1.0,
                        "source_only": -1.0,
                        "zero": 0.0,
                        "histories": 0,
                    }
                ]
            )["sign_matches"]["translated"],
            0,
        )
