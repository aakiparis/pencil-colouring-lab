"""Finite independent eight-GPU validation inference; never modifies training artefacts."""
import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
import boto3
from botocore.config import Config
S=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],region_name='uk-south2',config=Config(signature_version='s3v4',request_checksum_calculation='when_required',response_checksum_validation='when_required'))
B=os.environ['S3_BUCKET'];cfg=json.loads(Path('/app/evaluation.json').read_text());prefix=cfg['prefix']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def upload(p,key):S.upload_file(str(p),B,prefix+'/'+key)
def parent():
 base=Path('/cache/pencil-lora-v1')/cfg['package_sha256'];base.mkdir(parents=True,exist_ok=True)
 if not (base/'frozen-dataset/manifest.json').exists():
  z=base/'package.zip';S.download_file(B,cfg['package_key'],str(z));assert sha(z)==cfg['package_sha256']
  with zipfile.ZipFile(z) as archive:
   assert all((base/n).resolve().is_relative_to(base.resolve()) for n in archive.namelist());archive.extractall(base)
 manifest=json.loads((base/'frozen-dataset/manifest.json').read_text())
 for rel,digest in manifest['files'].items():assert sha(base/'frozen-dataset'/rel)==digest
 adapters=Path('/cache/pencil-lora-evaluation')/cfg['id'];adapters.mkdir(parents=True,exist_ok=True)
 for step in cfg['checkpoints']:
  m=json.loads(S.get_object(Bucket=B,Key=cfg['training_prefix']+f'/checkpoints/checkpoint-{step}/manifest.json')['Body'].read());assert m['run_key']==cfg['training_run_key']
  p=adapters/str(step)/'pytorch_lora_weights.safetensors';p.parent.mkdir(exist_ok=True)
  S.download_file(B,cfg['training_prefix']+f'/checkpoints/checkpoint-{step}/pytorch_lora_weights.safetensors',str(p));assert sha(p)==m['files'][p.name]
 env=dict(os.environ,EVAL_BASE=str(base),EVAL_ADAPTERS=str(adapters))
 processes=[subprocess.Popen([sys.executable,__file__,'child',str(rank)],env=dict(env,CUDA_VISIBLE_DEVICES=str(rank))) for rank in range(8)]
 codes=[p.wait() for p in processes]
 if any(codes):raise RuntimeError('Evaluation child failures: '+str(codes))
 report={'state':'complete','evaluation_id':cfg['id'],'task_count':len(cfg['tasks']),'world_size':8}
 S.put_object(Bucket=B,Key=prefix+'/completion.json',Body=json.dumps(report).encode(),ContentType='application/json');print(json.dumps(report),flush=True)
def child(rank):
 import torch
 from PIL import Image
 from diffusers import QwenImage21Pipeline
 from huggingface_hub import snapshot_download
 base=Path(os.environ['EVAL_BASE']);manifest=json.loads((base/'frozen-dataset/manifest.json').read_text());rows={r['id']:r for r in manifest['rows']}
 out=Path('/tmp/pencil-eval')/str(rank);out.mkdir(parents=True,exist_ok=True)
 model=snapshot_download(cfg['model']['id'],revision=cfg['model']['revision'])
 pipe=QwenImage21Pipeline.from_pretrained(model,torch_dtype=torch.bfloat16).to('cuda');pipe.vae.disable_tiling();pipe.vae.disable_slicing()
 loaded=None
 for task in cfg['tasks'][rank::8]:
  stem=task['variant']+'/'+task['id'];key=prefix+'/outputs/'+stem+'.json'
  try:
   meta=json.loads(S.get_object(Bucket=B,Key=key)['Body'].read());assert meta['evaluation_id']==cfg['id'];continue
  except S.exceptions.NoSuchKey:pass
  step=task['checkpoint']
  if loaded!=step:
   if loaded is not None:pipe.unload_lora_weights()
   if step is not None:pipe.load_lora_weights(str(Path(os.environ['EVAL_ADAPTERS'])/str(step)),adapter_name='pencil_style')
   loaded=step
  if step is not None:pipe.set_adapters('pencil_style',adapter_weights=task['scale'])
  row=rows[task['id']];assert row['split']=='validation';source=base/'frozen-dataset'/row['cond_image'];image=Image.open(source).convert('RGB')
  width=round((1024*1024*image.width/image.height)**0.5/32)*32;height=round((1024*1024*image.height/image.width)**0.5/32)*32
  result=pipe(image=[image],prompt=task['prompt'],width=width,height=height,num_inference_steps=40,true_cfg_scale=1.0,generator=torch.Generator('cuda').manual_seed(42)).images[0]
  if result.mode=='RGBA':
   bg=Image.new('RGBA',result.size,'white');bg.alpha_composite(result);result=bg.convert('RGB')
  p=out/(task['variant']+'-'+task['id']+'.png');result.save(p)
  meta=dict(task,evaluation_id=cfg['id'],training_run_key=cfg['training_run_key'],model=cfg['model'],seed=42,num_inference_steps=40,true_cfg_scale=1.0,vae_tiling=False,input_sha256=sha(source),output_sha256=sha(p),width=result.width,height=result.height,adapter_sha256=sha(Path(os.environ['EVAL_ADAPTERS'])/str(step)/'pytorch_lora_weights.safetensors') if step is not None else None)
  upload(p,'outputs/'+stem+'.png');S.put_object(Bucket=B,Key=key,Body=json.dumps(meta,indent=2).encode(),ContentType='application/json');print(json.dumps({'rank':rank,'completed':stem}),flush=True)
if __name__=='__main__':
 try:child(int(sys.argv[2])) if len(sys.argv)>1 else parent()
 except BaseException as exc:
  S.put_object(Bucket=B,Key=prefix+'/failure-'+(sys.argv[2] if len(sys.argv)>2 else 'parent')+'.json',Body=json.dumps({'error':str(exc),'type':type(exc).__name__}).encode());raise
