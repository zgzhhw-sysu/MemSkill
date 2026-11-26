"""Independent pinned-author F1 plus actual-cache, budget and three-arm receipts."""
from pathlib import Path
import sys,json,hashlib,statistics,math,collections,datetime,fcntl
from fair_test_common import R,F,read,save,sha,textsha,verify_registration
reg=verify_registration();sid=sys.argv[1];assert sid in reg['histories']
A=R/'audit-author';sys.path.insert(0,str(A))
from src.eval import get_evaluator
ev=get_evaluator('locomo',None);assert Path(sys.modules[type(ev).__module__].__file__).resolve().is_relative_to(A)
manifest=read(A/'manifest.json')
for n,h in manifest['files'].items():assert sha(A/n)==h,n
sample,=(s for s in read(R/'assets/locomo10.json') if s['sample_id']==sid)
qa={str(i):q for i,q in enumerate(sample['qa']) if int(q.get('category',1))!=5}
plan=read(F/'plans'/sid/'plan.json')['rows'];assert set(qa)==set(plan)
dig=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
answers={};metrics={}
for arm in ['native','mixed']:
 target=F/('readers/mixed' if arm=='mixed' else 'writers/'+arm)/sid
 if arm=='mixed':rows=read(target/'answers.json');receipts=None
 else:
  v=read(target/'verified.json');assert v['complete'] and v['scores_sha256']==sha(target/'scores.json')
  raw=read(target/'scores.json')['answers'];rows={str(x['qa_index']):{'answer':x['prediction'],'tokens':x['tokens'],'f1':x['f1']} for x in raw};assert len(raw)==len(rows)
  receipts=read(target/'reader-receipts.json')['rows'];assert set(receipts)==set(qa)
 usage=[u for l in (target/'usage.jsonl').read_text().splitlines() if (u:=json.loads(l))['kind']=='reader']
 cache_by_prompt={}
 for u in usage:
  k=u['cache_key'];c=read(F/'reader-cache'/k[:2]/(k+'.json'));p=c['payload']
  assert c['sha256']==dig(p)==u['response_sha256'] and p['key']==k
  cache_by_prompt.setdefault(p['prompt_sha256'],[]).append((u,c))
 assert set(rows)==set(qa) and len(usage)==len(qa)
 report={};vals=[];cost=[]
 for key,q in qa.items():
  cfg=plan[key];row=rows[key]
  if arm=='mixed':ph=cfg['mixed_prompt_sha256'];prompt_tokens=None
  else:
   rc=receipts[key];ph=rc['prompt_sha256'];prompt_tokens=rc['prompt_tokens'];assert rc['limit']==cfg['limit']
   if arm=='native':assert ph==cfg['native_prompt_sha256'] and rc['retrieved_top20']==cfg['native_top20']
   else:assert rc['excerpt']==cfg['excerpt'] and len(rc['shown_memories'])==10
  assert cache_by_prompt.get(ph),(arm,key,ph)
  u,c=cache_by_prompt[ph].pop();p=c['payload']
  assert p['response'].strip()==row['answer'].strip() and p['tokens']==row['tokens']
  if arm=='mixed':assert row['prompt_sha256']==ph and row['key']==u['cache_key'] and row['response_sha256']==c['sha256']
  else:assert row['tokens'][0]-prompt_tokens==cfg['native_token_overhead']
  assert row['tokens'][0]<=cfg['limit']+cfg['native_token_overhead'] and 0<row['tokens'][1]<=32
  score=float(ev.compute_f1(row['answer'],ev.get_ground_truth(q),q));assert math.isclose(score,row['f1'],abs_tol=1e-12)
  vals.append(score);cost.append(row['tokens'])
  report[key]={'f1':score,'answer':row['answer'],'tokens':row['tokens'],'prompt_sha256':ph,'cache_key':u['cache_key'],'cache_hit':u['cache_hit'],'response_sha256':c['sha256']}
 assert not any(cache_by_prompt.values())
 answers[arm]=report;metrics[arm]={'questions':len(vals),'f1':100*statistics.mean(vals),'input_tokens':sum(t[0] for t in cost),'output_tokens':sum(t[1] for t in cost),'new_physical_qa_calls':sum(not u['cache_hit'] for u in usage)}
pairs={}
for a,b in [('native','mixed')]:
 ds=[answers[b][k]['f1']-answers[a][k]['f1'] for k in qa]
 pairs[b+'_vs_'+a]={'delta_f1':100*statistics.mean(ds),'higher_equal_lower':[sum(d>0 for d in ds),sum(d==0 for d in ds),sum(d<0 for d in ds)],'changed_answers':sum(answers[b][k]['answer']!=answers[a][k]['answer'] for k in qa)}
out={'complete':True,'history':sid,'questions':len(qa),'arms':metrics,'paired':pairs,'registration_sha256':sha(F/'registration.json'),'scorer_commit':manifest['upstream_commit'],'scorer_manifest_sha256':sha(A/'manifest.json'),'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save(F/'audits-controls'/sid/'per-question.json',answers);save(F/'audits-controls'/sid/'summary.json',out)
with (F/'aggregate-controls.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 paths=[F/'audits-controls'/h/'summary.json' for h in reg['histories']]
 if all(p.exists() for p in paths):
  histories=[read(p) for p in paths];assert sum(h['questions'] for h in histories)==314
  combined={a:{'questions':314,'f1':sum(h['arms'][a]['f1']*h['questions'] for h in histories)/314,'input_tokens':sum(h['arms'][a]['input_tokens'] for h in histories),'output_tokens':sum(h['arms'][a]['output_tokens'] for h in histories)} for a in metrics}
  combined_pairs={k:{'delta_f1':sum(h['paired'][k]['delta_f1']*h['questions'] for h in histories)/314,'higher_equal_lower':[sum(h['paired'][k]['higher_equal_lower'][i] for h in histories) for i in range(3)]} for k in pairs}
  save(F/'controls-summary.json',{'complete':True,'questions':314,'histories':histories,'arms':combined,'paired':combined_pairs,'registration_sha256':sha(F/'registration.json'),'scope':'Completed native and mixed controls only, official historically exposed314 test questions. Best package is not included in this summary.'})
print('FAIR_TEST_AUDIT_COMPLETE',json.dumps(out),flush=True)
