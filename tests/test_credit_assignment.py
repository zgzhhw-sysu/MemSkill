import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.controller import PPOBuffer, PPOController
from src.credit_assignment import allocate, classify, diagnose_episode, execution_fault, single_swap, _digest
from src.episode_trace import skill_spec
from src.trainer import BaseTrainer


class SpanCreditTest(unittest.TestCase):
    def test_disabled_native_episode_has_identical_transitions_and_no_probes(self):
        def run(weight):
            args = SimpleNamespace(trace_dir=None)
            if weight is not None:
                args.span_credit_weight = weight
            cfg = SimpleNamespace(outer_epochs=1, inner_epochs=1, redistribute_reward=True,
                                  reward_redistribution_decay=.7, final_reward_last_ratio=.6)
            trainer = SimpleNamespace(args=args, config=cfg, _evaluate_qa=lambda *a,**kw:(.6,.6))
            def sample(**kwargs):
                i=kwargs["session_idx"]
                kwargs["ppo_buffer"].push(np.array([i]),np.ones((2,2)),[1,0],-.5,.15,.01)
                return dict(session_idx=i, action_idx=[1,0], log_prob=-.5, value=.15, process_reward=.01)
            trainer._process_session=sample
            b=PPOBuffer()
            BaseTrainer.train_episode(trainer,{"sample_id":"test"},SimpleNamespace(trace=None),["a","b"],b,
                                      episode_length=2,precompute_embeddings=False)
            return b
        with patch("src.credit_assignment.diagnose_episode", side_effect=AssertionError("Unexpected probe")):
            old,disabled=run(None),run(0.0)
        for field in vars(old):
            np.testing.assert_array_equal(getattr(old,field),getattr(disabled,field))

    def test_recorded_policy_noise_preserves_sampling_and_replay_consumes_no_rng(self):
        for k in (1, 3):
            controller = PPOController(state_dim=4, op_dim=3, hidden_dim=8, device="cpu", action_top_k=k)
            state, ops = torch.randn(4), torch.randn(5,3)
            for seed in range(12):
                torch.manual_seed(seed)
                untouched = controller(state, ops)
                untouched_rng = torch.get_rng_state()
                torch.manual_seed(seed)
                noise = {}
                recorded = controller(state, ops, record_noise=noise)
                self.assertEqual(untouched, recorded)
                self.assertTrue(torch.equal(untouched_rng, torch.get_rng_state()))
                self.assertEqual(recorded, controller(state, ops, replay_noise=noise))
                self.assertTrue(torch.equal(untouched_rng, torch.get_rng_state()))

    def test_swap_is_fixed_k_distinct_and_does_not_modify_the_sample(self):
        ops = [dict(name=n, update_type=t) for n,t in (("a","DELETE"), ("b","UPDATE"), ("c","INSERT"))]
        selected = ops[:2]
        self.assertEqual(single_swap(ops, selected), [ops[0], ops[2]])
        self.assertEqual(selected, ops[:2])
        self.assertIsNone(single_swap(ops, ops))
        self.assertEqual(single_swap(ops, [ops[2]]), [ops[0]])

    def test_signed_reference_recovery_and_competing_explanations(self):
        correct = dict(ok=True, answer="Paris", f1=1.)
        wrong = dict(ok=True, answer="London", f1=0.)
        def label(a, b, **kwargs):
            return classify(a, b, "Paris", **dict({"retrieval": False, "execution": False}, **kwargs))
        self.assertEqual(label(correct, wrong)["controller_delta"], 1.)
        self.assertEqual(label(wrong, correct)["controller_delta"], -1.)
        for args, cause in [({"execution": True}, "execution"), ({"retrieval": True},"retrieval"),
                            ({"execution": True, "retrieval": True},"mixed"), ({}, "unresolved")]:
            self.assertEqual(label(wrong, None, **args)["cause"], cause)
        mixed = label(wrong, correct, retrieval=True)
        self.assertEqual((mixed["cause"],mixed["controller_delta"]), ("mixed",0.))
        self.assertEqual(label(dict(ok=True,answer="It is Paris.",f1=.5), correct)["controller_delta"],0.)

    def test_components_clip_and_do_not_cross_episode_or_replace_samples(self):
        buf = PPOBuffer()
        steps = []
        for i in range(3):
            buf.push(np.array([i]), np.array([[2]]), 0, -.4, .1, .05)
            if i == 0:
                buf.finish_episode(.7)
            else:
                steps.append(dict(session_idx=i-1, action_idx=0, log_prob=-.4, value=.1, process_reward=.05))
        buf.finish_episode(.8, True, .9, .6)
        diagnoses = [dict(span_id=0, mean_controller_delta=-.9), dict(span_id=1, mean_controller_delta=.1)]
        before = list(buf.rewards)
        actions, probs, values = map(list,(buf.actions,buf.log_probs,buf.values))
        allocate(buf,1,steps,diagnoses,0.,.5,.8,.6,True)
        self.assertEqual(before,buf.rewards)
        bad = [diagnoses[0],dict(diagnoses[1],span_id=4)]
        with self.assertRaises(ValueError):
            allocate(buf,1,steps,bad,1.,.5,.8,.6,True)
        self.assertEqual(before,buf.rewards)
        rows = allocate(buf,1,steps,diagnoses,.2,.5,.8,.6,True)
        np.testing.assert_allclose(buf.rewards,np.array(before)+[0,-.1,.02])
        self.assertEqual([r["terminal"] for r in rows],[0,.48])
        self.assertAlmostEqual(sum(r["redistributed"] for r in rows),.32)
        self.assertEqual((buf.actions,buf.log_probs,buf.values),(actions,probs,values))

    def test_parser_contract_and_failed_apply_are_executor_faults(self):
        allowed = [dict(update_type="insert")]
        self.assertIsNone(execution_fault([dict(success=True,action_type="INSERT")],True,allowed))
        self.assertEqual(execution_fault([dict(success=True,action_type="DELETE")],True,allowed),
                         "action_outside_selected_types")
        self.assertEqual(execution_fault([dict(success=False)],False,allowed),"parse_or_apply_failure")

    def test_full_diagnosis_localizes_known_interventions_and_abstains_on_unstable_swap(self):
        """Real trace restoration and routing, with controlled responses for each isolated intervention."""
        op = lambda n: SimpleNamespace(name=n,update_type="INSERT",description=n,instruction_template=n)
        ops = [op("a"),op("b")]
        specs = list(map(skill_spec,ops))
        trainer = SimpleNamespace(args=SimpleNamespace(retriever="dummy",span_credit_spans=1,span_credit_questions=4),
                                  evaluator=SimpleNamespace(get_ground_truth=lambda q:q["answer"]),
                                  operation_bank=SimpleNamespace(get_candidate_operations=lambda:ops))
        qa = dict(question="What city?",answer="Paris",evidence=["D1:1"])
        conv = dict(qa=[qa],conversation=dict(session_1=[dict(dia_id="D1:1",text="Paris",speaker="Ali")]))
        action = dict(action_type="INSERT", success=True)
        event = dict(span_id=0,skills=specs[:1],candidate_versions=[s["version"] for s in specs],
                     source="Ali said Paris",source_turns=["D1:1"],retrieved=[],refs=[],parsed_actions=[action],
                     applied=True,old_log_probability=-.3,changes=[
                         dict(action="INSERT",before=None,after=text,memory_id=i,version=1)
                         for i,text in enumerate(["input A","Paris"])])
        step = dict(action_idx=0,log_prob=-.3)
        original = dict(kind="qa",query_id="conv_0",success=True,refs=[[0,1]],
                        prediction="London",f1=0.,question=qa["question"])
        def prediction(text):
            score = {"Paris":1., "They met in Paris.":.4}.get(text, 0.)
            return dict(ok=True,answer=text,f1=score,tokens=[8,1])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"episode.jsonl"
            path.write_text(json.dumps(original)+"\n")
            trace=SimpleNamespace(path=path,events=[event])
            with self.assertRaisesRegex(ValueError, "candidate bank changed"):
                diagnose_episode(trainer,conv,trace,[dict(step,action_idx=1)])
            for cause in ("selection","execution","retrieval","mixed","unstable","unchanged_context","no_effect",
                          "plausible_execution", "plausible_retrieval", "only_plausible"):
                calls=[]
                def writer(t, events, start, selected, queries, quotes=()):
                    alt=selected[0]["name"]=="b"
                    calls.append(alt)
                    pred="London"
                    if (quotes and cause in ("execution","mixed")) or (alt and cause in ("selection","mixed","unchanged_context")):
                        pred="Paris"
                    if alt and cause in ("plausible_execution", "plausible_retrieval"):
                        pred = "Paris"
                    if quotes and cause in ("plausible_execution", "only_plausible"):
                        pred = "They met in Paris."
                    if alt and cause=="unstable" and calls.count(True)==1:
                        pred="Paris"
                    return dict(actions=[action],failed_span=None,contract_fault=None,
                                bank_hash=_digest(["input A","Paris"]) if not alt and not quotes else "other",
                                qa={0:dict(prediction(pred),indices=[0],
                                           context_hash="edited" if (alt or quotes) and cause!="unchanged_context" else "original")})
                retrieval_answer = {"retrieval":"Paris", "plausible_retrieval":"They met in Paris."}.get(cause, "London")
                with self.subTest(cause=cause), patch("rag_utils.get_embeddings",return_value=[[1]]), \
                     patch("src.credit_assignment._write_branch",side_effect=writer), \
                     patch("src.credit_assignment._answer",return_value=prediction(retrieval_answer)):
                    result,=diagnose_episode(trainer,conv,trace,[step])
                    expected="unresolved" if cause in ("unstable","unchanged_context","no_effect","only_plausible") else cause
                    if cause.startswith("plausible_"):
                        expected = "ambiguous"
                        self.assertEqual(result["questions"][0]["supported"], ["selection"])
                        self.assertEqual(result["questions"][0]["possible_causes"], [cause.removeprefix("plausible_")])
                        with patch.object(trainer.args, "span_credit_rule_version", 1, create=True):
                            original_rule, = diagnose_episode(trainer, conv, trace, [step])
                        self.assertEqual(original_rule["mean_controller_delta"], -1.)
                    self.assertEqual(result["questions"][0]["cause"],expected)
                    self.assertEqual(result["mean_controller_delta"],-1. if cause=="selection" else 0.)
                    if cause=="retrieval":
                        self.assertEqual(result["questions"][0]["retrieval_probe"]["inserted_ref"],[1,1])
