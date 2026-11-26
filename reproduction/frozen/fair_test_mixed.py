"""Second arm: verified native top10 plus the same frozen question passage."""
import sys,os,json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from fair_test_common import R,F,read,save,verify_registration
sys.path.insert(0,str(R/'snapshot/deploy/sources/locomo-credit-attribution-v1'))
from experiments.controller.reader_slot_credit import measured_prompt
from src.eval.locomo import LoCoMoEvaluator
reg=verify_registration();sid=sys.argv[1];assert sid in reg['histories']
assert read(F/'writers/native'/sid/'verified.json')['exact_writer_and_reader_context_parity']
qa={i:q for s in read(R/'assets/locomo10.json') if s['sample_id']==sid for i,q in enumerate(s['qa']) if int(q.get('category',1))!=5}
plan=read(F/'plans'/sid/'plan.json');target=F/'readers/mixed'/sid;assert not target.exists();target.mkdir(parents=True)
os.environ['MEMSKILL_GREEDY_EVAL_CACHE']=str(F/'reader-cache');os.environ['MEMSKILL_USAGE_LOG']=str(target/'usage.jsonl')
ev=LoCoMoEvaluator(SimpleNamespace())
def run(item):
 key,cfg=item;q=qa[int(key)]
 assert ev.build_prompt(q['question'],cfg['native_top20'][:10]+[cfg['excerpt']],q)==cfg['mixed_prompt']
 result=measured_prompt(cfg['mixed_prompt'],q,ev,reg['model'],F/'reader-cache',True)
 save(target/'per-question'/(key+'.json'),result);print('FAIR_TEST_MIXED_QA',sid,key,flush=True)
 return key,result
with ThreadPoolExecutor(max_workers=2) as pool:answers=dict(pool.map(run,plan['rows'].items()))
save(target/'answers.json',answers);print('FAIR_TEST_MIXED_COMPLETE',sid,len(answers),flush=True)
