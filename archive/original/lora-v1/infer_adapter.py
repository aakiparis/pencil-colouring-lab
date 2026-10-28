"""Apply a saved LoRA without merging; render held-out photos with a short instruction."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--scale',type=float,default=1.0);p.add_argument('--seeds',type=int,nargs='+',default=[42,123]);p.add_argument('--prompt-run',type=Path);a=p.parse_args()
 import torch
 from PIL import Image
 from diffusers import QwenImage21Pipeline
 from huggingface_hub import snapshot_download
 cfg=json.loads((Path(__file__).parent/'pilot.json').read_text());manifest=json.loads((a.dataset/'manifest.json').read_text())
 model=snapshot_download(cfg['model']['id'],revision=cfg['model']['revision'])
 pipe=QwenImage21Pipeline.from_pretrained(model,torch_dtype=torch.bfloat16).to('cuda')
 # Tiled decoding caused border smudges in this benchmark; retain the accepted full-frame decode.
 pipe.vae.disable_tiling();pipe.vae.disable_slicing()
 adapter_sha=None
 if a.adapter:
  weight=a.adapter/'pytorch_lora_weights.safetensors'
  if not weight.exists():raise ValueError('Adapter file missing: '+str(weight))
  adapter_sha=hashlib.sha256(weight.read_bytes()).hexdigest()
  pipe.load_lora_weights(str(a.adapter),adapter_name='pencil_style');pipe.set_adapters('pencil_style',adapter_weights=a.scale)
 prompts={r['id']:r['prompt'] for r in json.loads(a.prompt_run.read_text())['tasks']} if a.prompt_run else {}
 a.output.mkdir(parents=True,exist_ok=False)
 for row in manifest['rows']:
  if row['split']!=a.split:continue
  image=Image.open(a.dataset/row['cond_image']).convert('RGB');width=round((1024*1024*image.width/image.height)**0.5/32)*32;height=round((1024*1024*image.height/image.width)**0.5/32)*32
  for seed in a.seeds:
   path=a.output/(row['id']+'-seed-'+str(seed)+'.png')
   prompt=prompts[row['id']] if a.prompt_run else cfg['instruction']
   result=pipe(image=[image],prompt=prompt,width=width,height=height,num_inference_steps=40,true_cfg_scale=1.0,generator=torch.Generator('cuda').manual_seed(seed)).images[0]
   if result.mode=='RGBA':
    bg=Image.new('RGBA',result.size,'white');bg.alpha_composite(result);result=bg.convert('RGB')
   result.save(path)
   meta={'id':row['id'],'split':a.split,'model':cfg['model'],'instruction':prompt,'prompt_run_sha256':hashlib.sha256(a.prompt_run.read_bytes()).hexdigest() if a.prompt_run else None,'adapter':str(a.adapter) if a.adapter else None,'adapter_sha256':adapter_sha,'adapter_scale':a.scale if a.adapter else 0,'seed':seed,'width':result.width,'height':result.height,'vae_tiling':False,'vae_slicing':False,'input_sha256':hashlib.sha256((a.dataset/row['cond_image']).read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
   path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
