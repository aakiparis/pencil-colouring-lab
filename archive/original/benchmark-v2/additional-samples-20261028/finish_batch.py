"""Collect the additional sample outputs and keep the main benchmark section current."""
import hashlib,json,sys,time
from pathlib import Path
F=Path(__file__).resolve().parent;ROOT=F.parent;sys.path.insert(0,str(ROOT));from launch import call
from compare import build,OUT,MODELS,DATA
sys.path.insert(0,str(ROOT.parent));from storage_ops import client
from botocore.exceptions import ClientError
s,b=client();cfg=json.loads((F/'run.json').read_text());runhash=hashlib.sha256((F/'run.json').read_bytes()).hexdigest();last=None
while True:
 job=call(['ai','job','get',json.loads((F/'batch-job.json').read_text())['metadata']['id']]);state=job['status']['state'];results={}
 for t in cfg['tasks']:
  try:raw=s.get_object(Bucket=b,Key=cfg['prefix']+'/results/'+t['id']+'.json')['Body'].read()
  except ClientError as ex:
   if ex.response['Error']['Code'] in ['NoSuchKey','404']:continue
   raise
  m=json.loads(raw);assert m['run_sha256']==runhash and m['prompt']==t['prompt'] and m['seed']==t['seed'];assert m['additional_loras']==cfg['additional_loras'];assert m['input_sha256']=={n:cfg['input_sha256'][n] for n in t['inputs']}
  dest=DATA/cfg['prefix']/'outputs'/(t['id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
  if m['status']=='completed':
   if not dest.exists():s.download_file(b,cfg['prefix']+'/results/'+t['id']+'.png',str(dest))
   assert hashlib.sha256(dest.read_bytes()).hexdigest()==m['output_sha256']
  dest.with_suffix('.json').write_bytes(raw);results[t['id']]=m
 manifest=json.loads((ROOT/'additional-samples.json').read_text());count=sum(r['status']=='completed' for r in results.values());failed=sum(r['status']=='failed' for r in results.values())
 for row in manifest['rows']:
  for v in row['variants'].values():
   if v.get('task_id') in results:
    r=results[v['task_id']];v.update(status=r['status'],output_sha256=r.get('output_sha256'))
    if r.get('error'):v['error']=r['error']
 manifest.update(generation_status=f'{state} · {count}/10 new images received · 2 existing seed-123 results retained',job_id=job['metadata']['id']);(ROOT/'additional-samples.json').write_text(json.dumps(manifest,indent=2)+'\n')
 progress={'state':state,'completed':count,'failed':failed,'total':10,'job_id':job['metadata']['id']};(F/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
 configs={m:json.loads((ROOT/m/'run.json').read_text()) for m in MODELS};historical=json.loads((OUT/'progress.json').read_text());build(configs,historical)
 if progress!=last:print(json.dumps(progress),flush=True);last=progress
 if state in ['COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED']:
  s.upload_file(str(ROOT/'additional-samples.json'),b,cfg['prefix']+'/comparison-manifest.json')
  if state!='COMPLETED' or count!=10:raise RuntimeError('Additional sample batch incomplete: '+str(progress))
  print('COMPLETE: all 12 result cells available in the additional-samples section.',flush=True);break
 time.sleep(20)
