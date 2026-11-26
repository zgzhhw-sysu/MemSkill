"""Frozen three-arm official-test comparison; no training or policy fitting."""
from pathlib import Path
import hashlib,json,os
from contextlib import contextmanager
from unittest.mock import patch
R=Path(__file__).resolve().parent
F=R/'fair-test-three-v1'
BASE=R.parents[1]/'runs/llama31-paper-v1-locomo-batch64'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
textsha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def verify_registration():
 reg=read(F/'registration.json')
 for n,h in reg['files_sha256'].items():assert sha(R/n)==h,n
 return reg
def unique_actions(actions):
 seen=set();kept=[];suppressed=[]
 for i,a in enumerate(actions):
  if a.success and a.action_type=='INSERT':
   if a.memory_content in seen:suppressed.append(i);continue
   seen.add(a.memory_content)
  kept.append(a)
 return kept,suppressed
@contextmanager
def test_reader(target,sid,arm):
 from transformers import AutoTokenizer
 from src.eval.locomo import LoCoMoEvaluator
 plan=read(F/'plans'/sid/'plan.json');qkeys={}
 for k,c in plan['rows'].items():qkeys.setdefault(c['question'],[]).append(k)
 tok=AutoTokenizer.from_pretrained(R/'snapshot/models/Llama-3.1-8B-Instruct',local_files_only=True)
 enc=lambda s:tok.encode(s,add_special_tokens=False,verbose=False)
 old=LoCoMoEvaluator.build_prompt;observed={};oldcache=os.environ['MEMSKILL_GREEDY_EVAL_CACHE']
 def build(self,question,retrieved_memories,qa_item=None):
  keys=qkeys[question];cfg=plan['rows'][keys[0]]
  assert all(plan['rows'][k]==cfg for k in keys[1:])
  top20=list(retrieved_memories);assert len(top20)==20
  if arm=='native':
   assert top20==cfg['native_top20'],'Identity GPU retrieval differs'
   shown=top20;prompt=old(self,question,shown,qa_item);cropped=0
   assert textsha(prompt)==cfg['native_prompt_sha256']
  else:
   shown=top20[:10];before=shown.copy();raw=cfg['excerpt'];limit=cfg['limit']
   render=lambda:old(self,question,shown+[raw],qa_item)
   prompt=render()
   for i in range(9,-1,-1):
    tokens=enc(shown[i])
    while len(enc(prompt))>limit and tokens:
     tokens=tokens[:max(0,len(tokens)-max(1,len(enc(prompt))-limit))]
     shown[i]=tok.decode(tokens).strip();prompt=render()
    if len(enc(prompt))<=limit:break
   cropped=sum(a!=b for a,b in zip(before,shown))
   assert len(enc(prompt))<=limit
  rec={'arm':arm,'prompt_sha256':textsha(prompt),'prompt_tokens':len(enc(prompt)),
       'limit':cfg['limit'],'retrieved_top20':top20,'shown_memories':shown,
       'excerpt':cfg['excerpt'] if arm!='native' else None,'cropped_slots':cropped}
  assert rec['prompt_tokens']<=cfg['limit']
  for k in keys:assert observed.setdefault(k,rec)==rec
  os.environ['MEMSKILL_GREEDY_EVAL_CACHE']=str(F/'reader-cache')
  return prompt
 try:
  with patch.object(LoCoMoEvaluator,'build_prompt',build):yield
  assert set(observed)==set(plan['rows'])
  save(target/'reader-receipts.json',{'arm':arm,'history':sid,'plan_sha256':sha(F/'plans'/sid/'plan.json'),'rows':observed})
 finally:os.environ['MEMSKILL_GREEDY_EVAL_CACHE']=oldcache
