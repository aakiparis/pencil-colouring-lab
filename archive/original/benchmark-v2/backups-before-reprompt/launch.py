"""Launch the authorised benchmark as finite preemptible Nebius Jobs."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
from storage_ops import client
CLI = '/Users/kiparis/.nebius/bin/nebius'

def call(args):
    p = subprocess.run([CLI, *args, '--format','json','--no-check-update'], capture_output=True, text=True)
    if p.returncode: raise RuntimeError(p.stderr)
    if '--dry-run' in args: return {'dry_run': p.stdout}
    return json.loads(p.stdout)

def launch(slug, mode, gpus=1, dry=False, retry=False):
    folder = ROOT / slug
    run = folder / 'run.json'; cfg = json.loads(run.read_text())
    output = folder / (mode + '-job.json')
    attempt = 1
    if output.exists() and not dry:
        old = json.loads(output.read_text())
        if not retry:
            print('Existing job', old['metadata']['id']); return
        state = call(['ai','job','get',old['metadata']['id']])['status']['state']
        if state not in ['FAILED','CANCELLED','DEADLINE_EXCEEDED']:
            raise RuntimeError('Retry requires a failed or interrupted job: '+state)
        attempt = len(list(folder.glob(mode+'-job-attempt-*.json'))) + 2
        (folder/(mode+f'-job-attempt-{attempt-1}.json')).write_text(json.dumps(old,indent=2)+'\n')
    if slug == 'flux3':
        from huggingface_hub import get_hf_file_metadata, hf_hub_url, get_token
        # Fail before allocating a VM when model access has not been granted.
        get_hf_file_metadata(hf_hub_url(cfg['model']['id'], 'modular_model_index.json', revision=cfg['model']['revision']), token=get_token())
    s, bucket = client()
    for file in [run, ROOT/'worker.py', ROOT/'bootstrap.sh', ROOT/'prompt-policy.json']:
        s.upload_file(str(file), bucket, cfg['prefix'] + '/' + file.name)
    secret = json.loads((PARENT/'runtime-secret.json').read_text())['metadata']['id']
    fs = json.loads((PARENT/'filesystem.json').read_text())['metadata']['id']
    subnet = json.loads((PARENT/'subnets.json').read_text())['items'][0]['metadata']['id']
    project = json.loads((PARENT/'project.json').read_text())['metadata']['id']
    container = json.loads((PARENT/'container.json').read_text())
    image = 'pytorch/pytorch@' + container['digest']
    name = f'pencil-v2-{slug}-{mode}' + (f'-a{attempt}' if attempt>1 else '')
    args = ['ai','job','create','--parent-id',project,'--name',name,
            '--image',image,'--container-command','/bin/bash','--args','/app/benchmark_bootstrap.sh '+mode,
            '--inject-file',str(ROOT/'worker.py')+':/app/benchmark_worker.py',
            '--inject-file',str(ROOT/'bootstrap.sh')+':/app/benchmark_bootstrap.sh',
            '--volume',fs+':/cache:rw','--subnet-id',subnet,
            '--env','S3_ENDPOINT=https://storage.uk-south2.nebius.cloud','--env','S3_BUCKET='+bucket,
            '--env','HF_HOME=/cache/huggingface','--env','OMP_NUM_THREADS=8',
            '--env','RUN_SHA256='+hashlib.sha256(run.read_bytes()).hexdigest(),
            '--env','RUN_PREFIX='+cfg['prefix'],
            '--env-secret','AWS_ACCESS_KEY_ID='+secret,'--env-secret','AWS_SECRET_ACCESS_KEY='+secret,
            '--restart-policy','never','--disk-size','100Gi','--timeout','2h' if mode=='warm' else '4h','--async']
    if slug == 'flux3':
        if mode == 'warm': args += ['--env','AWAIT_SHARED_ENVIRONMENT=1']
        from huggingface_hub import get_token
        token = get_token()
        if token:
            hf_secret = ROOT/'hf-secret.json'
            if not hf_secret.exists():
                value = call(['mysterybox','secret','create','--parent-id',project,'--name','pencil-v2-huggingface',
                              '--secret-version-payload',json.dumps([{'key':'HF_TOKEN','string_value':token}]),
                              '--secret-version-set-primary'])
                hf_secret.write_text(json.dumps(value,indent=2)); hf_secret.chmod(0o600)
            args += ['--env-secret','HF_TOKEN='+json.loads(hf_secret.read_text())['metadata']['id']]
    if mode == 'warm': args += ['--platform','cpu-d3','--preset','8vcpu-32gb']
    else: args += ['--preemptible','--follows-spot-price','--platform','gpu-rtx6000-a',
                   '--preset','8gpu-192vcpu-1744gb' if gpus==8 else '1gpu-24vcpu-218gb']
    if slug == 'flux3' and mode == 'warm' and attempt > 1:
        args[args.index('--args')+1] = '/app/repair_http.sh warm'
        args += ['--inject-file',str(ROOT/'repair_http.sh')+':/app/repair_http.sh']
    if dry: args += ['--dry-run']
    result = call(args)
    if dry: print(json.dumps(result,indent=2)); return
    if 'resource_id' in result:
        (folder/(mode+'-operation.json')).write_text(json.dumps(result,indent=2)+'\n')
        result={'metadata':{'id':result['resource_id']},'status':{'state':'CREATING'}}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(slug,mode,result['metadata']['id'],result.get('status',{}).get('state'),flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['qwen21','flux3']);ap.add_argument('mode',choices=['warm','smoke','batch'])
    ap.add_argument('--gpus',type=int,choices=[1,8],default=1);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--retry',action='store_true');a=ap.parse_args()
    launch(a.model,a.mode,a.gpus,a.dry_run,a.retry)
