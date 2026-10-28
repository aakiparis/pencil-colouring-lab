"""Finite eight-GPU DDP orchestration with immutable inputs and complete-checkpoint S3 recovery."""
import hashlib,json,os,runpy,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import boto3
from botocore.config import Config

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
S3=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],region_name='uk-south2',config=Config(signature_version='s3v4',request_checksum_calculation='when_required',response_checksum_validation='when_required'))
BUCKET=os.environ['S3_BUCKET']
disk_checks=[]
def require_space(stage,required_bytes):
 usage=shutil.disk_usage('/cache');record={'stage':stage,'filesystem':'/cache','capacity_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free,'required_free_bytes':required_bytes}
 disk_checks.append(record);print(json.dumps({'filesystem_preflight':record}),flush=True)
 if usage.free<required_bytes:raise RuntimeError('Shared cache capacity is insufficient at '+stage+'; no caches have been deleted and no extra volume has been allocated.')
package_sha=os.environ['PACKAGE_SHA256'];base=Path('/cache/pencil-lora-v1')/package_sha;base.mkdir(parents=True,exist_ok=True)
z=base/'package.zip'
package_bytes=S3.head_object(Bucket=BUCKET,Key=os.environ['PACKAGE_KEY'])['ContentLength']
require_space('package_download',package_bytes+8*1024**3)
S3.download_file(BUCKET,os.environ['PACKAGE_KEY'],str(z))
assert sha(z)==package_sha,'Training package checksum mismatch'
with zipfile.ZipFile(z) as archive:
 require_space('package_extraction',sum(info.file_size for info in archive.infolist())+8*1024**3)
 for name in archive.namelist():
  if not (base/name).resolve().is_relative_to(base.resolve()):raise ValueError('Unsafe archive member')
 archive.extractall(base)
cfg=json.loads((base/'pilot.json').read_text());manifest=json.loads((base/'frozen-dataset/manifest.json').read_text())
for rel,digest in manifest['files'].items():assert sha(base/'frozen-dataset'/rel)==digest,rel
mode=sys.argv[1] if len(sys.argv)>1 else 'calibrate'
assert mode in ['calibrate','pilot']
sys.path.insert(0,str(base))
from run_identity import compute_run_key
run_key=compute_run_key(base)
for injected in ['train_worker.py','bootstrap.sh','requirements.txt']:
 assert sha(base/injected)==sha(Path('/app')/injected),'Injected file differs from immutable package: '+injected
prefix=cfg['prefix']+'/'+run_key[:16]+'/'+mode
output=Path('/cache/pencil-lora-v1/runs')/run_key/mode;output.mkdir(parents=True,exist_ok=True)
def upload(path,key):S3.upload_file(str(path),BUCKET,prefix+'/'+key)
# Restore only a checkpoint whose commit manifest was uploaded after every constituent file.
try:
 latest=json.loads(S3.get_object(Bucket=BUCKET,Key=prefix+'/latest-complete.json')['Body'].read())
 cp_manifest=json.loads(S3.get_object(Bucket=BUCKET,Key=prefix+'/'+latest['manifest_key'])['Body'].read())
 for rel,digest in cp_manifest['files'].items():
  dest=output/latest['checkpoint']/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists() or sha(dest)!=digest:S3.download_file(BUCKET,prefix+'/checkpoints/'+latest['checkpoint']+'/'+rel,str(dest))
  assert sha(dest)==digest
 write(output/latest['checkpoint']/'COMPLETE.json',cp_manifest)
except S3.exceptions.NoSuchKey:pass
except Exception as exc:
 # Missing object is expected on the first attempt; network, permission and checksum errors are fatal.
 from botocore.exceptions import ClientError
 if not (isinstance(exc,ClientError) and exc.response['Error']['Code'] in ['NoSuchKey','404']):raise
for cp in output.glob('checkpoint-*'):
 if cp.is_dir() and not (cp/'COMPLETE.json').exists():cp.rename(output/('incomplete-'+cp.name+'-'+str(int(time.time()))))
from huggingface_hub import snapshot_download
model_plan=snapshot_download(cfg['model']['id'],revision=cfg['model']['revision'],dry_run=True)
missing_model_bytes=sum(item.file_size for item in model_plan if item.will_download)
reserve=cfg['infrastructure'].get('minimum_output_reserve_bytes',8*1024**3)
require_space('model_download_and_output_reserve',missing_model_bytes+reserve)
model=snapshot_download(cfg['model']['id'],revision=cfg['model']['revision'])
# The orchestrator must not initialise CUDA: all eight devices belong to training children.
from importlib.metadata import version
provenance={'run_key':run_key,'mode':mode,'model':cfg['model'],'dataset_manifest_sha256':sha(base/'frozen-dataset/manifest.json'),'package_sha256':package_sha,'packages':{name:version(name) for name in ['torch','accelerate','diffusers','transformers','peft','datasets']},'configuration':cfg,'filesystem_preflight':disk_checks,'missing_model_download_bytes_at_preflight':missing_model_bytes,'world_size':8,'effective_batch_size':8,'strategy':'One adapter with eight DDP replicas'}
write(output/'provenance.json',provenance);upload(output/'provenance.json','provenance.json')
(output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True));upload(output/'environment.txt','environment.txt')
complete=sorted([p for p in output.glob('checkpoint-*') if (p/'COMPLETE.json').exists()],key=lambda p:int(p.name.split('-')[-1]))
env=dict(os.environ,TRAIN_PACKAGE_ROOT=str(base),TRAIN_MODE=mode,TRAIN_RUN_KEY=run_key,TRAIN_OUTPUT=str(output),TRAIN_S3_PREFIX=prefix,TRAIN_MODEL_SNAPSHOT=model,TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
if complete:
 marker=json.loads((complete[-1]/'COMPLETE.json').read_text())
 assert marker['run_key']==run_key and marker['world_size']==8,'Resume checkpoint belongs to another run or world size'
 env['TRAIN_RESUME_CHECKPOINT']=complete[-1].name
command=[sys.executable,'-m','accelerate.commands.launch','--multi_gpu','--num_processes','8','--num_machines','1','--mixed_precision','bf16','--dynamo_backend','no','--num_cpu_threads_per_process','8','--main_process_port','29500',str(base/'distributed_train.py')]
write(output/'distributed-command.json',command);upload(output/'distributed-command.json','distributed-command.json')
started=time.monotonic()
try:
 with (output/'training.log').open('a') as log:
  process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in process.stdout:
   log.write(line);log.flush();print(line,end='',flush=True)
  returncode=process.wait()
 if returncode:raise RuntimeError('Eight-GPU Accelerate process exited with code '+str(returncode))
 ranks=[json.loads((output/('rank-'+str(i)+'-completion.json')).read_text()) for i in range(8)]
 assert all(r['state']=='complete' and r['world_size']==8 and r['run_key']==run_key for r in ranks)
 adapter=output/'pytorch_lora_weights.safetensors'
 assert adapter.exists(),'Final shared adapter is missing'
 rank0=ranks[0]
 report={'state':'complete','run_key':run_key,'mode':mode,'world_size':8,'effective_batch_size':8,'elapsed_seconds_including_precomputation':time.monotonic()-started,'unique_logged_update_steps':rank0['unique_logged_update_steps'],'last_logged_update':rank0['last_logged_update'],'median_recent_update_seconds':rank0['median_recent_update_seconds'],'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in ranks),'per_rank':ranks,'adapter_sha256':sha(adapter),'timing_note':'Eight-GPU DDP updates, effective batch eight. Pilot update computation is approximately ten times a fresh twenty-update calibration; add preparation, checkpoint transfers and separate inference.'}
 for p in output.rglob('*'):
  if p.is_file() and not any(part.startswith('checkpoint-') or part.startswith('incomplete-') for part in p.relative_to(output).parts):upload(p,str(p.relative_to(output)))
 write(output/'completion.json',report);upload(output/'completion.json','completion.json');print(json.dumps(report),flush=True)
except BaseException as exc:
 write(output/'failure.json',{'state':'failed','exception_type':type(exc).__name__,'message':str(exc),'run_key':run_key,'world_size':8})
 for name in ['failure.json','training.log','metrics.jsonl']:
  p=output/name
  if p.exists():upload(p,name)
 raise
