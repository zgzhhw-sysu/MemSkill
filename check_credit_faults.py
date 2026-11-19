"""Blindly route known single-stage faults. The writer/ranking are controlled; LoCoMo Reader is real."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.credit_assignment import _answer, diagnose_episode
from src.episode_trace import EpisodeTrace, evidence_turns, skill_spec
from src.eval import get_evaluator
from src.executor import ExecutionResult, Executor
from src.memory_bank import MemoryBank


def case(args, evaluator, conversation, index, fault, directory):
    qa = conversation["qa"][index]
    turn, = evidence_turns(conversation["conversation"], qa["evidence"])
    source = f'DATE: {turn["session_date"]}\n{turn["speaker"]} said, "{turn["text"]}"'
    quote = f'{turn["session_date"]} | {turn["speaker"]}: {turn["text"]}'
    # Unrelated items come from the same real dialogue; no reference answer is written by the harness.
    turns = [t for k, seq in conversation["conversation"].items() if isinstance(seq,list)
             for t in seq if t["dia_id"]!=turn["dia_id"]]
    decoy = next(t["text"] for t in turns if t["text"] and t["speaker"] != turn["speaker"])
    def embeddings(retriever, inputs, mode):
        if mode == "query":
            return np.array([[1.,0.] for _ in inputs], dtype="float32")
        # Make exactly one known ranking error by putting the decoy ahead of the supporting memory.
        return np.array([[1.,0.] if (text==quote and fault!="retrieval") else
                         ([0.,1.] if text==quote else [.7,.7]) for text in inputs],dtype="float32")
    def op(name, kind, instruction):
        return SimpleNamespace(name=name,description=instruction,instruction_template=instruction,
                               update_type=kind,embedding=np.zeros(2,dtype="float32"))
    ops = [op("retain","INSERT","Record the supplied speaker and the entire quoted turn."),
           op("skip","NOOP","Leave memory as it is.")]
    def writer(selected, text, memories):
        if selected[0].name=="skip":
            return [ExecutionResult("NOOP",True)]
        items = [decoy] if fault=="execution" else [quote,decoy]
        return [ExecutionResult("INSERT",True,memory_content=t) for t in items]
    executor = Executor(args)
    executor.execute_operation = writer
    trainer = SimpleNamespace(args=args, evaluator=evaluator, device="cpu", executor=executor,
                              operation_bank=SimpleNamespace(get_candidate_operations=lambda:ops),
                              _initialize_memory_bank=lambda:MemoryBank("fixture",top_k=1))
    sample = dict(conversation,qa=[dict(qa,original_index=index)])
    chosen = 1 if fault=="selection" else 0
    record = EpisodeTrace(directory,conversation["sample_id"],conversation["conversation"])
    bank = trainer._initialize_memory_bank()
    bank.trace = record
    with patch("rag_utils.get_embeddings",side_effect=embeddings), patch("src.executor.get_embeddings",side_effect=embeddings):
        record.begin(0,source,[ops[chosen]],[],[])
        record.current.update(candidate_versions=[skill_spec(o)["version"] for o in ops],old_log_probability=0.)
        actions = writer([ops[chosen]],source,[])
        record.finish(actions,executor.apply_to_memory_bank(actions,bank,[]))
        record.verify(bank)
        context,indices = bank.retrieve(embeddings(None,["question"],"query")[0])
        answer = _answer(trainer,qa,context)
        record.write("qa",dict(query_id=conversation["sample_id"]+"_0",question=qa["question"],
                               success=answer["ok"],prediction=answer["answer"],f1=answer["f1"],
                               refs=[record.ref(i) for i in indices],tokens=answer["tokens"]))
        diagnosis, = diagnose_episode(trainer,sample,record,[dict(action_idx=chosen,log_prob=0.)])
        q, = diagnosis["questions"]
        record.write("injection_review",{"known":fault,"diagnosis":diagnosis})
    return {"sample":conversation["sample_id"],"qa_index":index,"known":fault,"cause":q["cause"],
            "supported":q["supported"],"controller_delta":q["controller_delta"],
            "control_matched":q["control_matched"],"observed":answer,
            "correct_source_answer":_answer(trainer,qa,[quote]),"reference":evaluator.get_ground_truth(qa),
            "trace":record.path.name}


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--port",type=int,required=True)
    options=parser.parse_args()
    output=options.output
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    data={c["sample_id"]:c for c in json.loads((root/"runs/llama3_8b/assets/locomo10.json").read_text())}
    args=SimpleNamespace(api=True,api_base=f"http://127.0.0.1:{options.port}/v1",api_key="local-only",model="llama3-8b",
                         retriever="fixture",span_credit_spans=1,span_credit_questions=1,temperature=0.,max_new_tokens=32)
    evaluator=get_evaluator("locomo",args)
    os.environ["MEMSKILL_USAGE_LOG"]=str(output/"usage.jsonl")
    cases=[("conv-26",3),("conv-30",39),("conv-41",68)]
    (output/"protocol.json").write_text(json.dumps({"cases":cases,"stages":["selection","execution","retrieval"],
                 "writer":"controlled retain/skip operations", "ranking":"fixed injection", "reader":"real Llama3 8B",
                 "optimizer":"unused; all injected trajectories excluded from training"},indent=2))
    results=[]
    for sid,i in cases:
        for fault in ("selection","execution","retrieval"):
            row=case(args,evaluator,data[sid],i,fault,output)
            results.append(row)
            (output/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False))
            print(sid,i,fault,row["cause"],row["controller_delta"],flush=True)


if __name__=="__main__":
    run()
