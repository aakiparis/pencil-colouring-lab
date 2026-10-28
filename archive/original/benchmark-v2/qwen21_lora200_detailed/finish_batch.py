"""Finite output synchronisation for the user-authorised full LoRA batch; no recurring automation."""
import json,sys,time
from pathlib import Path
F=Path(__file__).resolve().parent;ROOT=F.parent;sys.path.insert(0,str(ROOT))
from launch import call
from compare import sync,OUT
last=None
while True:
 job=call(['ai','job','get',json.loads((F/'batch-job.json').read_text())['metadata']['id']]);state=job['status']['state'];progress=sync()['qwen21_lora200_detailed'];record=dict(model='Qwen-Image-2.1 LoRA · update 200 · strength 1 · detailed background v4',stage='full dataset inference',state=state,job_id=job['metadata']['id'],**progress)
 (F/'progress.json').write_text(json.dumps(record,indent=2)+'\n');(OUT/'pipeline-state.json').write_text(json.dumps(record,indent=2)+'\n')
 if record!=last:print(json.dumps(record),flush=True);last=record
 if state in ['COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED']:
  if state=='COMPLETED' and progress['completed']!=progress['total']:raise RuntimeError('Completed job has missing or failed outputs')
  break
 time.sleep(30)
