"""Launch the authorised validation comparison once on eight preemptible GPUs."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from launch import call
PARENT=ROOT.parent;E=ROOT/'evaluation'
if (E/'job.json').exists():raise SystemExit('An evaluation job is already recorded; inspect it before creating another.')
cfg=json.loads((ROOT/'pilot.json').read_text());status=json.loads(Path('/Users/kiparis/Downloads/Pencil colouring dataset/lora-v1/status.json').read_text());prov=status['states']['pilot']['provenance'];run_key=status['run_key']
variants=[{'id':'base-short','title':'Qwen-Image-2.1 · no LoRA','checkpoint':None,'scale':0,'prompt':cfg['instruction']}]
for step in [50,75,100]:
 for scale in [.75,1.0]:variants.append({'id':f'step-{step}-scale-{scale:g}','title':f'Qwen-Image-2.1 LoRA · update {step} · strength {scale:g}','checkpoint':step,'scale':scale,'prompt':cfg['instruction']})
variants += [{'id':'step-200-scale-1','title':'Qwen-Image-2.1 LoRA · update 200 · strength 1','checkpoint':200,'scale':1,'prompt':cfg['instruction']},{'id':'step-100-background','title':'Qwen-Image-2.1 LoRA · update 100 · background instruction','checkpoint':100,'scale':1,'prompt':cfg['instruction']+' Preserve a very light, simplified version of the original background.'}]
manifest=json.loads((ROOT/'frozen-dataset/manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='validation']
eval_cfg={'model':cfg['model'],'training_run_key':run_key,'training_prefix':cfg['prefix']+'/'+run_key[:16]+'/pilot','package_sha256':prov['package_sha256'],'package_key':'lora-v1/packages/'+prov['package_sha256']+'.zip','checkpoints':[50,75,100,200],'variants':variants,'rows':rows,'tasks':[dict(id=r['id'],variant=v['id'],checkpoint=v['checkpoint'],scale=v['scale'],prompt=v['prompt']) for v in variants for r in rows]}
id=hashlib.sha256(json.dumps(eval_cfg,sort_keys=True).encode()+(E/'worker.py').read_bytes()+(E/'bootstrap.sh').read_bytes()).hexdigest();eval_cfg.update(id=id,prefix='lora-v1/evaluation/'+id[:16]);(E/'evaluation.json').write_text(json.dumps(eval_cfg,indent=2)+'\n')
project=json.loads((PARENT/'project.json').read_text())['metadata']['id'];fs=json.loads((PARENT/'filesystem.json').read_text())['metadata']['id'];secret=json.loads((PARENT/'runtime-secret.json').read_text())['metadata']['id'];subnet=json.loads((PARENT/'subnets.json').read_text())['items'][0]['metadata']['id'];container=json.loads((PARENT/'container.json').read_text())
args=['ai','job','create','--parent-id',project,'--name','pencil-lora-eval-'+id[:12],'--image','pytorch/pytorch@'+container['digest'],'--container-command','/bin/bash','--args','/app/eval_bootstrap.sh','--volume',fs+':/cache:rw','--subnet-id',subnet,'--env','S3_ENDPOINT=https://storage.uk-south2.nebius.cloud','--env','S3_BUCKET=kiparis-pencil-baseline-uks2-20261017','--env','HF_HOME=/cache/huggingface','--env','OMP_NUM_THREADS=8','--env-secret','AWS_ACCESS_KEY_ID='+secret,'--env-secret','AWS_SECRET_ACCESS_KEY='+secret,'--platform','gpu-rtx6000-a','--preset','8gpu-192vcpu-1744gb','--preemptible','--follows-spot-price','--restart-policy','never','--disk-size','100Gi','--timeout','2h','--async']
for p,dest in [(E/'worker.py','eval_worker.py'),(E/'bootstrap.sh','eval_bootstrap.sh'),(E/'evaluation.json','evaluation.json'),(ROOT/'requirements.txt','requirements.txt')]:args+=['--inject-file',str(p)+':/app/'+dest]
r=call(args);r={'metadata':{'id':r['resource_id']},'status':{'state':'CREATING'},'operation':r} if 'resource_id' in r else r;(E/'job.json').write_text(json.dumps(r,indent=2)+'\n');print(r['metadata']['id'])
