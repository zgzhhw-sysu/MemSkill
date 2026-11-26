"""CPU-only receipt/prompt preparation from already verified GPU artifacts."""
from pathlib import Path
import sys,os,json,hashlib,collections
from fair_test_common import R,F,BASE,read,save,sha,textsha,unique_actions
sys.path.insert(0,str(R/'snapshot/deploy/sources/locomo-credit-attribution-v1'))
import numpy as np
from types import SimpleNamespace
from experiments.controller.paper_f1 import load_split_cohort,PROTOCOL
from experiments.controller.passage_prefix_sources import raw_turn_passages,fit_raw_passages,readback_tokenizer
from experiments.controller.reader_slot_credit import archive_digests,measured_contexts
from src.evaluation_cache import digest
from src.eval.locomo import LoCoMoEvaluator
meta=read(BASE/'locomo/protocol.json');samples,cohort=load_split_cohort('locomo',R/'assets/locomo10.json',read(PROTOCOL)['datasets']['locomo'],'test')
assert {s['sample_id'] for s in samples}=={'conv-49','conv-50'} and len(cohort)==314
enc,dec=readback_tokenizer(meta);ev=LoCoMoEvaluator(SimpleNamespace());proof=[]
for sample in sorted(samples,key=lambda s:s['sample_id']):
 sid=sample['sample_id'];qa={i:q for (s,i),q in cohort.items() if s==sid};old=F/'imported'/sid
 rank=read(old/'ranks.json');source=read(old/'source-plan.json');direct=read(old/'direct-plan.json')
 passages=raw_turn_passages(sample['conversation']);inputs=rank['inputs']
 assert inputs['corpus_sha256']==sha(R/'assets/locomo10.json') and inputs['passage_sha256']==digest(passages)
 assert inputs['question_sha256']==digest([(i,q['question']) for i,q in qa.items()])
 assert rank['dimension']==1024 and '3090' in rank['device']
 assert rank['cosines_sha256']==sha(old/'cosines.npy')
 co=np.load(old/'cosines.npy',allow_pickle=False);assert co.shape==(len(qa),len(passages)) and co.dtype==np.float32
 orders=np.argsort(-co,axis=1,kind='stable')[:,:20]
 for j,i in enumerate(qa):
  assert rank['ranks'][str(i)]['ids']==orders[j].tolist()
  assert rank['ranks'][str(i)]['scores']==co[j,orders[j]].tolist()
 cell=BASE/'locomo/native'/sid;arch=archive_digests(cell);assert arch==source['archive']
 os.environ['MEMSKILL_GREEDY_EVAL_CACHE']=str(BASE/'greedy-cache')
 spans,contexts,answers=measured_contexts(cell,BASE/'greedy-cache',arch,qa,ev,meta['model']['model'])
 rows={}
 for (i,q),ctx,answer in zip(qa.items(),contexts,answers,strict=True):
  key=str(i);texts=[t for _,t in ctx];assert len(texts)==20
  native=ev.build_prompt(q['question'],texts,q);assert textsha(native)==answer['prompt_sha256']==source['rows'][key]['hashes']['native']
  captured={}
  def build(v):captured['texts']=list(v);return ev.build_prompt(q['question'],v,q)
  ranked=[(j,passages[j]) for j in rank['ranks'][key]['ids']]
  prompt,used=fit_raw_passages(texts[:10],ranked,build,enc,dec,source['rows'][key]['prompt_tokens']['source'],True)
  assert textsha(prompt)==direct['rows'][key]['hashes']['direct']
  assert len(captured['texts'])==11 and len(used)==1,'Exactly one raw passage required'
  excerpt=captured['texts'][-1];limit=len(enc(native))
  assert len(enc(prompt))<=limit and len(enc(ev.build_prompt(q['question'],['']*10+[excerpt],q)))<=limit
  rows[key]={'question':q['question'],'category':q['category'],'limit':limit,'historical_direct_limit':direct['rows'][key]['limit'],
    'native_top20':texts,'native_prompt':native,'native_prompt_sha256':textsha(native),'mixed_prompt':prompt,'mixed_prompt_sha256':textsha(prompt),
    'excerpt':excerpt,'used':used,'raw_rank_ids':rank['ranks'][key]['ids'],'native_token_overhead':answer['tokens'][0]-len(enc(native))}
 save(F/'plans'/sid/'plan.json',{'history':sid,'questions':len(qa),'rows':rows,'original_archive':arch,'raw_rank_sha256':sha(old/'ranks.json')})
 suppressed=0
 for t in spans:
  original=[SimpleNamespace(**a) for a in t['parsed_actions']];kept,removed=unique_actions(original)
  assert [vars(a) for a in kept if a.action_type!='INSERT']==[vars(a) for a in original if a.action_type!='INSERT']
  assert {a.memory_content for a in kept if a.success and a.action_type=='INSERT'}=={a.memory_content for a in original if a.success and a.action_type=='INSERT'}
  assert all(original[i].success and original[i].action_type=='INSERT' for i in removed)
  suppressed+=len(removed)
 proof.append({'history':sid,'questions':len(qa),'blocks':len(spans),'original_same_block_duplicate_inserts':suppressed,'all_original_prompt_and_physical_receipts_verified':True,'historical_mixed_prompt_exact':True,'new_model_calls':0})
save(F/'data-preflight.json',{'complete':True,'questions':314,'histories':proof,'gpu_raw_rank_receipts_verified':True,'filter_actual_archived_actions_checked':True,'new_model_calls':0})
print(json.dumps(proof),flush=True)
