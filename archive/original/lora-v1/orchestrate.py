"""Finite authorised calibration, evidence review and pilot orchestration."""
import hashlib,json,math,subprocess,sys,time
from pathlib import Path
from launch import call,compute_run_key,verify_calibration
from monitor import sync,ROOT,OUT,client,object_or_none
cfg=json.loads((ROOT/'pilot.json').read_text());run_key=compute_run_key(ROOT);jobs={}

def wait(phase):
 path=ROOT/(phase+'-job.json');ident=json.loads(path.read_text())['metadata']['id'];last=None
 while True:
  job=call(['ai','job','get',ident]);state=job['status']['state'];jobs[phase]={'metadata':{'id':ident},'status':job['status']}
  progress=sync(run_key,jobs);metrics=progress['states'][phase]['metrics'];step=max([r.get('step') or 0 for r in metrics],default=0)
  event={'phase':phase,'state':state,'logged_updates':step,'previews':len(progress['previews']),'run_key':run_key}
  (ROOT/'workflow-state.json').write_text(json.dumps(event,indent=2)+'\n')
  if event!=last:print(json.dumps(event),flush=True);last=event
  if state=='COMPLETED':return progress
  if state in ['FAILED','CANCELLED','DEADLINE_EXCEEDED']:raise RuntimeError(phase+' ended '+state+'; inspect logs before retrying.')
  time.sleep(30)

progress=wait('calibrate');s,b=client();raw=object_or_none(s,b,cfg['prefix']+'/'+run_key[:16]+'/calibrate/completion.json')
if not raw:raise RuntimeError('Completed calibration has no completion evidence.')
completion=json.loads(raw)
review={'approved':True,'run_key':run_key,'completion_sha256':hashlib.sha256(raw).hexdigest(),'basis':'User explicitly authorised training on eight RTX GPUs. Automatic review of measured calibration evidence, finite losses and actual held-out preview before the pilot.'}
verify_calibration(cfg,run_key,review,raw)
for record in progress['states']['calibrate']['metrics']:
 for name,value in record.get('metrics',{}).items():
  if isinstance(value,(float,int)) and not math.isfinite(value):raise RuntimeError('Non-finite calibration metric: '+name)
if not any(p['phase']=='calibrate' for p in progress['previews']):raise RuntimeError('No actual calibration image; inspect validation before the pilot.')
(ROOT/'calibration-reviewed.json').write_text(json.dumps(review,indent=2)+'\n')
s.put_object(Bucket=b,Key=cfg['prefix']+'/'+run_key[:16]+'/calibration-reviewed.json',Body=json.dumps(review,indent=2).encode(),ContentType='application/json')
# Launch only when no record exists; resumed controllers reuse an existing finite job.
if not (ROOT/'pilot-job.json').exists():subprocess.run([sys.executable,str(ROOT/'launch.py'),'pilot','--launch'],check=True,cwd=ROOT)
progress=wait('pilot')
key=cfg['prefix']+'/'+run_key[:16]+'/pilot/pytorch_lora_weights.safetensors';weight=OUT/'weights'/'pytorch_lora_weights.safetensors';weight.parent.mkdir(exist_ok=True)
s.download_file(b,key,str(weight));print('PILOT_COMPLETE '+str(weight),flush=True)
