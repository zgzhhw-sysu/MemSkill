"""One full history, native parity gate then fixed best package; never auto-restart."""
from pathlib import Path
import os,sys,subprocess,time,json,datetime
from fair_test_common import R,F,read,save,verify_registration
reg=verify_registration();sid=sys.argv[1];gpu=int(sys.argv[2]);assert reg['histories'][sid]['gpu']==gpu
py='/home/lixian/Documents/AgentMemory/.venv-judge24b/bin/python';guard=R.parents[1]/'tools/gpu_guard.py'
def foreign():
 rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 return [int(x.strip()) for x in rows.splitlines() if x.strip().isdigit() and (Path('/proc')/x.strip()).exists() and (Path('/proc')/x.strip()).stat().st_uid!=os.getuid()]
for arm in ['native','mixed','best','score']:
 print('FAIR_TEST_STAGE',sid,arm,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 if arm in ['native','best']:
  while (pids:=foreign()):print('WAIT_FOREIGN_GPU',gpu,pids,flush=True);time.sleep(30)
  command=[py,str(guard),'run','--gpus',str(gpu),'--','/bin/bash',str(R/'run_fair_test.sh'),'writer',arm,sid]
 else:command=['/bin/bash',str(R/'run_fair_test.sh'),arm,arm,sid]
 subprocess.run(command,check=True)
save(F/('queue-complete-'+sid+'.json'),{'complete':True,'history':sid,'gpu':gpu,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
