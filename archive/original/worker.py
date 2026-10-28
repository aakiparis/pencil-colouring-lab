"""Frozen Qwen image-edit baseline: one independent model per GPU, no added LoRA."""
import os,json,time,hashlib,tarfile,traceback,sys,argparse,math
from pathlib import Path

def storage():
 import boto3
 from botocore.config import Config
 return boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],region_name='uk-south2',config=Config(signature_version='s3v4',request_checksum_calculation='when_required',response_checksum_validation='when_required'))

def load_config():
 path=Path('/cache/run.json')
 expected=os.environ.get('RUN_SHA256')
 if not path.exists() or (expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected):storage().download_file(os.environ['S3_BUCKET'],'baseline-v1/run.json',str(path))
 if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
  raise RuntimeError('Cached run configuration differs from the pinned experiment. Refresh the run configuration before starting.')
 return json.loads(path.read_text())

def upload_json(s,key,data):
 s.put_object(Bucket=os.environ['S3_BUCKET'],Key='baseline-v1/'+key,Body=json.dumps(data,indent=2).encode(),ContentType='application/json')

def warm():
 from huggingface_hub import snapshot_download
 config=load_config();s=storage();cache=Path('/cache');dataset=cache/'dataset';dataset.mkdir(exist_ok=True)
 s.download_file(os.environ['S3_BUCKET'],'baseline-v1/dataset.tar.gz',str(cache/'dataset.tar.gz'))
 with tarfile.open(cache/'dataset.tar.gz','r:gz') as t:t.extractall(dataset,filter='data')
 s.download_file(os.environ['S3_BUCKET'],'baseline-v1/dataset-checksums.json',str(cache/'dataset-checksums.json'))
 for row in json.loads((cache/'dataset-checksums.json').read_text()):
  assert hashlib.sha256((dataset/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('Dataset verified',flush=True)
 local=snapshot_download(config['model']['id'],revision=config['model']['revision'],cache_dir='/cache/huggingface/hub',max_workers=8)
 (cache/'model-path.txt').write_text(local)
 (cache/'warm-ready.json').write_text(json.dumps({'model':config['model'],'verified_dataset':True,'local_path':local}))
 print('Model and dataset ready',local,flush=True)

def dimensions(im,area):
 ratio=im.width/im.height
 return max(256,round(math.sqrt(area*ratio)/16)*16),max(256,round(math.sqrt(area/ratio)/16)*16)

def gpu_worker(rank,world,mode,limit):
 os.environ['CUDA_VISIBLE_DEVICES']=str(rank)
 import torch
 from PIL import Image,ImageOps
 from diffusers import QwenImageEditPlusPipeline
 config=load_config();tasks=config['tasks']
 if mode=='smoke':tasks=tasks[:1]
 if limit:tasks=tasks[:limit]
 tasks=tasks[rank::world]
 if not tasks:return
 if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
 torch.set_num_threads(8)
 torch.cuda.set_device(0)
 info={'rank':rank,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'capability':list(torch.cuda.get_device_capability(0)),'torch':torch.__version__,'cuda':torch.version.cuda,'model':config['model'],'mode':mode,'tasks':len(tasks)}
 from importlib.metadata import version
 info['packages']={name:version(name) for name in ['diffusers','transformers','accelerate','huggingface-hub','safetensors','Pillow','boto3']}
 s=storage();upload_json(s,f'{mode}/worker-{rank}-environment.json',info);print(json.dumps(info),flush=True)
 path=Path('/cache/model-path.txt').read_text().strip()
 start=time.time()
 pipe=QwenImageEditPlusPipeline.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
 assert pipe.transformer.config.zero_cond_t is True, 'Qwen-Image-Edit-2511 requires zero_cond_t=True'
 pipe.vae.enable_tiling();pipe.set_progress_bar_config(disable=True)
 print(f'Worker {rank} model loaded in {time.time()-start:.1f}s',flush=True)
 output=Path('/cache/results')/mode;output.mkdir(parents=True,exist_ok=True)
 failures=0
 for task in tasks:
  key=f'{mode}/{task["id"]}'
  # Skip only fully published outputs, allowing safe retries after interruption.
  try:
   old=json.loads(s.get_object(Bucket=os.environ['S3_BUCKET'],Key='baseline-v1/'+key+'.json')['Body'].read())
   if old.get('status')=='completed' and old.get('model')==config['model'] and old.get('run_sha256')==os.environ.get('RUN_SHA256'):
    print('Already completed',task['id'],flush=True);continue
  except s.exceptions.NoSuchKey:pass
  except Exception as e:
   if 'NoSuchKey' not in str(e) and '404' not in str(e):raise
  images=[]
  for name in task['inputs']:
   with Image.open(Path('/cache/dataset')/name) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
  width,height=dimensions(images[0],config['target_pixels'])
  result={'id':task['id'],'mode':mode,'rank':rank,'model':config['model'],'additional_loras':[],'zero_cond_t':True,'run_sha256':os.environ.get('RUN_SHA256'),'prompt':task['prompt'],'seed':task['seed'],'width':width,'height':height,'steps':config['num_inference_steps'],'true_cfg_scale':config['true_cfg_scale'],'guidance_scale':config['guidance_scale'],'input_sha256':{name:hashlib.sha256((Path('/cache/dataset')/name).read_bytes()).hexdigest() for name in task['inputs']}}
  start=time.time()
  try:
   torch.cuda.reset_peak_memory_stats()
   with torch.inference_mode():
    image=pipe(image=images,prompt=task['prompt'],negative_prompt=config['negative_prompt'],width=width,height=height,generator=torch.Generator(device='cuda').manual_seed(task['seed']),num_inference_steps=config['num_inference_steps'],true_cfg_scale=config['true_cfg_scale'],guidance_scale=config['guidance_scale']).images[0]
   dest=output/(task['id']+'.png');image.save(dest)
   s.upload_file(str(dest),os.environ['S3_BUCKET'],'baseline-v1/'+key+'.png',ExtraArgs={'ContentType':'image/png'})
   result.update(status='completed',seconds=round(time.time()-start,3),peak_gpu_bytes=torch.cuda.max_memory_allocated(),output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
  except Exception as e:
   result.update(status='failed',seconds=round(time.time()-start,3),error=repr(e),traceback=traceback.format_exc());failures+=1;torch.cuda.empty_cache()
  upload_json(s,key+'.json',result)
  print(json.dumps({k:result.get(k) for k in ['id','status','seconds','peak_gpu_bytes','error']}),flush=True)
 if failures:raise RuntimeError(f'{failures} images failed on worker {rank}')

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['warm','smoke','batch']);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
 if args.mode=='warm':warm();return
 assert Path('/cache/warm-ready.json').exists(),'Warm-up job must complete first'
 if args.mode=='smoke':gpu_worker(0,1,args.mode,args.limit);return
 import multiprocessing as mp
 import torch
 world=min(torch.cuda.device_count(),len(load_config()['tasks']))
 if world<1:raise RuntimeError('No GPUs available')
 ctx=mp.get_context('spawn');procs=[ctx.Process(target=gpu_worker,args=(i,world,args.mode,args.limit)) for i in range(world)]
 for proc in procs:proc.start()
 for proc in procs:proc.join()
 codes=[proc.exitcode for proc in procs]
 if any(codes):raise RuntimeError(f'Worker exit codes: {codes}')
 upload_json(storage(),'batch/completed.json',{'status':'completed','tasks':len(load_config()['tasks']),'workers':world,'model':load_config()['model']})
 print('All workers completed',flush=True)
if __name__=='__main__':main()
