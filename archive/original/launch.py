"""Launch finite Nebius Jobs using an existing public PyTorch container."""
import json,sys,shlex,subprocess
from pathlib import Path
from provision import cli,ROOT,PROJECT

def command(mode,gpus=1,dry_run=False):
 fs=json.loads((ROOT/'filesystem.json').read_text())['metadata']['id']
 subnet=json.loads((ROOT/'subnets.json').read_text())['items'][0]['metadata']['id']
 bucket=json.loads((ROOT/'bucket.json').read_text())
 secret=json.loads((ROOT/'runtime-secret.json').read_text())['metadata']['id']
 container=json.loads((ROOT/'container.json').read_text())
 image='pytorch/pytorch@'+container['digest'] if container.get('digest') else container['image']
 args=['ai','job','create','--parent-id',PROJECT,'--name','pencil-baseline-'+mode+'-v3','--image',image,'--container-command','/bin/bash','--args','/app/bootstrap.sh '+mode,'--inject-file',str(ROOT/'worker.py')+':/app/worker.py','--inject-file',str(ROOT/'bootstrap.sh')+':/app/bootstrap.sh','--volume',fs+':/cache:rw','--subnet-id',subnet,'--env','S3_ENDPOINT=https://'+bucket['status']['domain_name'],'--env','S3_BUCKET='+bucket['metadata']['name'],'--env','HF_HOME=/cache/huggingface','--env','RUN_SHA256='+__import__('hashlib').sha256((ROOT/'run.json').read_bytes()).hexdigest(),'--env','OMP_NUM_THREADS=8','--env-secret','AWS_ACCESS_KEY_ID='+secret,'--env-secret','AWS_SECRET_ACCESS_KEY='+secret,'--restart-policy','never','--disk-size','100Gi','--timeout','2h' if mode=='warm' else '4h']
 if mode=='warm':args+=['--platform','cpu-d3','--preset','8vcpu-32gb']
 else:args+=['--preemptible','--follows-spot-price','--platform','gpu-rtx6000-a','--preset','8gpu-192vcpu-1744gb' if gpus==8 else '1gpu-24vcpu-218gb']
 if dry_run:args+=['--dry-run']
 return args
if __name__=='__main__':
 mode=sys.argv[1];gpus=int(sys.argv[2]) if len(sys.argv)>2 else 1
 dry='--dry-run' in sys.argv
 if mode in ('smoke','batch') and not dry:
  approval=ROOT/'prompt-confirmed.json'
  if not approval.exists():raise SystemExit('Inference requires a confirmed prompt selection.')
  if json.loads(approval.read_text())['run_sha256']!=__import__('hashlib').sha256((ROOT/'run.json').read_bytes()).hexdigest():raise SystemExit('Run configuration differs from the confirmed selection.')
 if dry:
  result=subprocess.run(['nebius',*command(mode,gpus,True),'--no-check-update'],capture_output=True,text=True)
  print(result.stdout or result.stderr)
  raise SystemExit(result.returncode)
 response=cli(command(mode,gpus,False))
 dest=ROOT/(mode+'-job.json');dest.write_text(json.dumps(response,indent=2));print(mode,response.get('metadata',{}).get('id'),response.get('status',{}).get('state'))
