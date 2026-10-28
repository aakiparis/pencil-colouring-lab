"""Prepare or launch one finite preemptible training job, after dataset selection."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
CLI='/Users/kiparis/.nebius/bin/nebius'
from run_identity import compute_run_key
def call(args):
 result=subprocess.run([CLI,*args,'--format','json','--no-check-update'],capture_output=True,text=True)
 if result.returncode:raise RuntimeError(result.stderr)
 return json.loads(result.stdout)
def verify_calibration(cfg,run_key,review,raw):
 if review.get('approved') is not True or review.get('run_key')!=run_key:raise SystemExit('Calibration review does not match the current model/configuration/dataset/trainer.')
 completion=json.loads(raw)
 if hashlib.sha256(raw).hexdigest()!=review.get('completion_sha256'):raise SystemExit('Reviewed calibration completion checksum does not match private S3.')
 if completion.get('state')!='complete' or completion.get('mode')!='calibrate' or completion.get('run_key')!=run_key or completion.get('last_logged_update',completion.get('unique_logged_update_steps',0))<cfg['calibration_updates'] or completion.get('world_size')!=cfg['infrastructure'].get('gpu_count',8) or not (completion.get('median_recent_update_seconds') or 0)>0 or not (completion.get('peak_allocated_bytes') or 0)>0:raise SystemExit('Calibration is not complete or lacks measured update timing and GPU memory evidence.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['calibrate','pilot']);ap.add_argument('--launch',action='store_true');ap.add_argument('--retry',action='store_true');a=ap.parse_args()
 cfg=json.loads((ROOT/'pilot.json').read_text());dataset=ROOT/'frozen-dataset'
 if not a.launch:
  print(json.dumps({'action':'plan_only_no_upload_or_compute','mode':a.mode,'infrastructure':cfg['infrastructure'],'dataset_frozen':(dataset/'manifest.json').exists(),'training_updates':cfg['calibration_updates'] if a.mode=='calibrate' else cfg['training']['max_train_steps']},indent=2));return
 if not (dataset/'manifest.json').exists():raise SystemExit('Freeze reviewed targets and split before allocating compute.')
 attempt=1
 record=ROOT/(a.mode+'-job.json')
 if record.exists():
  old=json.loads(record.read_text());state=call(['ai','job','get',old['metadata']['id']])['status']['state']
  if not a.retry or state not in ['FAILED','CANCELLED','DEADLINE_EXCEEDED']:raise SystemExit('Existing job '+old['metadata']['id']+' '+state+'; no duplicate job created.')
  (ROOT/(a.mode+'-previous-job-'+old['metadata']['id']+'.json')).write_text(json.dumps(old,indent=2)+'\n')
  attempt=1+len(list(ROOT.glob(a.mode+'-previous-job-*.json')))
 run_key=compute_run_key(ROOT)
 from storage_ops import client
 s,bucket=client()
 if a.mode=='pilot':
  review_path=ROOT/'calibration-reviewed.json'
  if not review_path.exists():raise SystemExit('Review the completed calibration before the pilot.')
  review=json.loads(review_path.read_text())
  if review.get('approved') is not True or review.get('run_key')!=run_key:raise SystemExit('Calibration review does not match the current model/configuration/dataset/trainer.')
  raw=s.get_object(Bucket=bucket,Key=cfg['prefix']+'/'+run_key[:16]+'/calibrate/completion.json')['Body'].read()
  verify_calibration(cfg,run_key,review,raw)
 archive=ROOT/'training-package.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for base in [ROOT/'vendor',dataset]:
   for p in sorted(base.rglob('*')):
    if p.is_file():z.write(p,str(p.relative_to(ROOT)))
  for name in ['pilot.json','selection.json','MODEL-LICENCE.txt','source-provenance.json','infer_adapter.py','requirements.txt','train_worker.py','bootstrap.sh','distributed_train.py','run_identity.py','trainer_ddp.py']:z.write(ROOT/name,name)
 digest=hashlib.sha256(archive.read_bytes()).hexdigest();key='lora-v1/packages/'+digest+'.zip'
 s.upload_file(str(archive),bucket,key)
 project=json.loads((PARENT/'project.json').read_text())['metadata']['id'];fs=json.loads((PARENT/'filesystem.json').read_text())['metadata']['id'];secret=json.loads((PARENT/'runtime-secret.json').read_text())['metadata']['id'];subnet=json.loads((PARENT/'subnets.json').read_text())['items'][0]['metadata']['id'];container=json.loads((PARENT/'container.json').read_text());infra=cfg['infrastructure']
 args=['ai','job','create','--parent-id',project,'--name','pencil-lora-v1-'+a.mode+'-'+run_key[:8]+'-a'+str(attempt),'--image','pytorch/pytorch@'+container['digest'],'--container-command','/bin/bash','--args','/app/bootstrap.sh '+a.mode,'--volume',fs+':/cache:rw','--subnet-id',subnet,'--env','S3_ENDPOINT=https://storage.uk-south2.nebius.cloud','--env','S3_BUCKET='+bucket,'--env','PACKAGE_KEY='+key,'--env','PACKAGE_SHA256='+digest,'--env','HF_HOME=/cache/huggingface','--env','OMP_NUM_THREADS=8','--env-secret','AWS_ACCESS_KEY_ID='+secret,'--env-secret','AWS_SECRET_ACCESS_KEY='+secret,'--platform',infra['platform'],'--preset',infra['preset'],'--preemptible','--follows-spot-price','--restart-policy','never','--disk-size','100Gi','--timeout',infra['timeout'],'--async']
 for name in ['train_worker.py','bootstrap.sh','requirements.txt']:args+=['--inject-file',str(ROOT/name)+':/app/'+name]
 result=call(args)
 if 'resource_id' in result:result={'metadata':{'id':result['resource_id']},'status':{'state':'CREATING'},'operation':result}
 record.write_text(json.dumps(result,indent=2)+'\n');print(result['metadata']['id'])
if __name__=='__main__':main()
