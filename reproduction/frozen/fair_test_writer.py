"""Original K7 control versus frozen best-training K1/source-only/unique package."""
import sys,os,json,contextlib,collections
from pathlib import Path
from unittest.mock import patch
from fair_test_common import R,F,BASE,read,save,sha,textsha,unique_actions,test_reader,verify_registration
sys.path.insert(0,str(R/'snapshot/run-linked-questions-v1'))
import torch
from experiments.designer import paper_date_pair as paper
reg=verify_registration();arm=os.environ['FAIR_TEST_ARM'];sid=os.environ['FAIR_TEST_HISTORY']
assert arm in ['native','best'] and sid in reg['histories']
assert os.environ.get('GPU_GUARD_LEASE_FDS') and torch.cuda.is_available()
if arm=='best':assert read(F/'writers/native'/sid/'verified.json')['exact_writer_and_reader_context_parity']
args=paper.get_agentic_memory_args();assert args.device=='cuda'
root=F/'writers'/arm;engine=root/'engines'/sid;engine.mkdir(parents=True,exist_ok=True)
os.environ['MEMSKILL_GREEDY_EVAL_CACHE']=str(root/'greedy-cache');assert (root/'greedy-cache').exists()
samples,cohort=paper.load_split_cohort('locomo',R/'assets/locomo10.json',read(paper.PROTOCOL)['datasets']['locomo'],'test')
sample,=(s for s in samples if s['sample_id']==sid);author_run=paper.run_traced_history
class Done(Exception):pass
def execute(trainer,unused_sample,local,unused_directory):
 protocol=read(engine/'protocol.json');reference=read(BASE/'locomo/protocol.json')
 assert protocol==reference,'Full original test protocol mismatch'
 assert trainer.controller.action_top_k==7
 target=root/sid;assert not target.exists();target.mkdir()
 original=paper.history_spans(BASE/'locomo/native'/sid);chunks,n,_=trainer._prepare_sessions(sample)
 assert len(chunks)==len(original)==n==reg['histories'][sid]['blocks']
 before_execute=trainer.executor.execute_operation;records=[];decisions=[]
 names=[o.name for o in trainer.operation_bank.get_candidate_operations()]
 def observe(module,inputs,values):
  assert values.is_cuda and trainer.controller.action_top_k==(7 if arm=='native' else 1)
  decisions.append([names[i] for i in torch.topk(values[0,:,0],trainer.controller.action_top_k).indices.tolist()])
 def filtered(operation,session_text,retrieved_memories,**kw):
  i=len(records);assert chunks[i]==session_text==original[i]['source']
  if arm=='native':
   from src.executor import deployment_operations
   deployed=deployment_operations(operation if isinstance(operation,(list,tuple)) else [operation])
   prompt=trainer.executor._build_executor_prompt(deployed,session_text,retrieved_memories)
   assert textsha(prompt)==original[i]['executor_responses'][0]['prompt_sha256'],'Identity writer differs before request'
  from src.executor import deployment_operations
  deployed=deployment_operations(operation if isinstance(operation,(list,tuple)) else [operation])
  isolated=arm=='best' and [o.name for o in deployed]==['insert']
  actual_retrieved=[] if isolated else list(retrieved_memories)
  receipts=kw.get('audit')
  if receipts is None:receipts=[];kw['audit']=receipts
  start=len(receipts);actions=before_execute(operation,session_text,actual_retrieved,**kw)
  raw=[vars(a).copy() for a in actions];kept,removed=unique_actions(actions) if arm=='best' else (actions,[])
  row={'span':i,'source_sha256':textsha(session_text),'controller_retrieved':list(retrieved_memories),'executor_retrieved':actual_retrieved,'insert_source_only':isolated,
       'before':raw,'after':[vars(a).copy() for a in kept],'suppressed_indices':removed,'requests':receipts[start:]}
  records.append(row)
  with (target/'filter.live.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  print('FAIR_TEST_BLOCK',arm,sid,i,'suppressed',len(removed),flush=True)
  return kept
 with patch.object(trainer.controller,'action_top_k',7 if arm=='native' else 1),patch.object(trainer.executor,'execute_operation',filtered),trainer.controller.actor_head.register_forward_hook(observe):
  with test_reader(target,sid,arm):author_run(trainer,sample,local,target)
 new=paper.history_spans(target);assert len(new)==len(records)==len(decisions)==n
 for a,b,c,decision in zip(original,new,records,decisions,strict=True):
  assert a['source']==b['source'] and b['parsed_actions']==c['after'] and b['executor_responses']==c['requests']
  assert b['retrieved']==c['controller_retrieved']
  assert [x['name'] for x in b['selected_skills']]==decision
  if arm=='native':
   for k in ['changes','parsed_actions','selected_skills','executor_responses','retrieved','refs','policy']:assert a[k]==b[k],(b['span_id'],k)
  else:
   kept,removed=unique_actions([type('Action',(),x)() for x in c['before']])
   assert [x for i,x in enumerate(c['before']) if i not in removed]==c['after'] and removed==c['suppressed_indices']
 for k in ['state_sha256','logits']:assert new[0]['policy'][k]==original[0]['policy'][k],'Same initial actual GPU Controller state required'
 usage=[json.loads(l) for l in (target/'usage.jsonl').read_text().splitlines()]
 wu=[u for u in usage if u['kind']=='executor']
 assert len(wu)==sum(len(t['executor_responses']) for t in new)
 if arm=='native':assert all(u['cache_hit'] for u in wu),'Identity writer must use exact archived responses'
 save(target/'filter.json',records)
 save(target/'verified.json',{'complete':True,'arm':arm,'history':sid,'blocks':n,'questions':len(read(target/'scores.json')['answers']),
   'exact_writer_and_reader_context_parity':arm=='native','controller_k':7 if arm=='native' else 1,'source_only_for_insert':arm=='best','actual_choice_counts':dict(collections.Counter('/'.join(v) for v in decisions)),
   'suppressed_inserts':sum(len(r['suppressed_indices']) for r in records),'writer_calls':len(wu),'new_physical_writer_calls':sum(not u['cache_hit'] for u in wu),
   'registration_sha256':sha(F/'registration.json'),'scores_sha256':sha(target/'scores.json'),'device':torch.cuda.get_device_name(),
   'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']})
 print('FAIR_TEST_WRITER_COMPLETE',arm,sid,flush=True);raise Done
paper.run_traced_history=execute
try:paper.run(args,engine,0,1)
except Done:pass
