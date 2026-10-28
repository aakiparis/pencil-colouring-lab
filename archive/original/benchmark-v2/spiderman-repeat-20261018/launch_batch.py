"""Launch the selected LoRA over the reviewed dataset, without changing historical runs."""
import hashlib,json,sys
from pathlib import Path
F=Path(__file__).resolve().parent;ROOT=F.parent;PARENT=ROOT.parent
sys.path.insert(0,str(ROOT));from launch import call
sys.path.insert(0,str(PARENT));from storage_ops import client
if (F/'batch-job.json').exists():raise SystemExit('Existing batch job recorded; refusing a duplicate launch.')
cfg=json.loads((F/'run.json').read_text());s,b=client()
for name in ['run.json','worker.py','bootstrap.sh']:s.upload_file(str(F/name),b,cfg['prefix']+'/'+name)
s.upload_file(str(PARENT/'lora-v1/candidate-selection.json'),b,cfg['prefix']+'/candidate-selection.json')
secret=json.loads((PARENT/'runtime-secret.json').read_text())['metadata']['id'];fs=json.loads((PARENT/'filesystem.json').read_text())['metadata']['id'];subnet=json.loads((PARENT/'subnets.json').read_text())['items'][0]['metadata']['id'];project=json.loads((PARENT/'project.json').read_text())['metadata']['id'];image='pytorch/pytorch@'+json.loads((PARENT/'container.json').read_text())['digest']
args=['ai','job','create','--parent-id',project,'--name','pencil-spiderman-repeat-'+hashlib.sha256((F/'run.json').read_bytes()).hexdigest()[:8],'--image',image,'--container-command','/bin/bash','--args','/app/benchmark_bootstrap.sh batch','--volume',fs+':/cache:rw','--subnet-id',subnet,'--env','S3_ENDPOINT=https://storage.uk-south2.nebius.cloud','--env','S3_BUCKET='+b,'--env','HF_HOME=/cache/huggingface','--env','OMP_NUM_THREADS=8','--env','RUN_SHA256='+hashlib.sha256((F/'run.json').read_bytes()).hexdigest(),'--env','RUN_PREFIX='+cfg['prefix'],'--env-secret','AWS_ACCESS_KEY_ID='+secret,'--env-secret','AWS_SECRET_ACCESS_KEY='+secret,'--restart-policy','never','--disk-size','100Gi','--timeout','2h','--preemptible','--follows-spot-price','--platform','gpu-rtx6000-a','--preset','1gpu-24vcpu-218gb','--async']
for source,dest in [(F/'worker.py','benchmark_worker.py'),(F/'bootstrap.sh','benchmark_bootstrap.sh'),(PARENT/'lora-v1/requirements.txt','requirements.txt')]:args+=['--inject-file',str(source)+':/app/'+dest]
r=call(args);r={'metadata':{'id':r['resource_id']},'status':{'state':'CREATING'},'operation':r} if 'resource_id' in r else r;(F/'batch-job.json').write_text(json.dumps(r,indent=2)+'\n');print(r['metadata']['id'])
