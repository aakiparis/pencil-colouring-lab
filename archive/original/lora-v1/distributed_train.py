"""Accelerate child: one adapter, all-rank checkpoint state and rank-zero S3 commits."""
import hashlib,json,os,runpy,shutil,statistics,sys,time
from pathlib import Path
import boto3,torch
from accelerate import Accelerator
from botocore.config import Config
ROOT=Path(os.environ['TRAIN_PACKAGE_ROOT']);sys.path.insert(0,str(ROOT))
from run_identity import compute_run_key
cfg=json.loads((ROOT/'pilot.json').read_text());mode=os.environ['TRAIN_MODE'];run_key=compute_run_key(ROOT)
assert run_key==os.environ['TRAIN_RUN_KEY']
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local_rank=int(os.environ['LOCAL_RANK'])
assert world==cfg['infrastructure']['gpu_count']==8
# Each child is explicitly bound before any trainer model is constructed.
torch.cuda.set_device(local_rank)
output=Path(os.environ['TRAIN_OUTPUT']);prefix=os.environ['TRAIN_S3_PREFIX'];bucket=os.environ['S3_BUCKET']
s3=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],region_name='uk-south2',config=Config(signature_version='s3v4',request_checksum_calculation='when_required',response_checksum_validation='when_required'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp-'+str(rank));temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def upload(path,key):s3.upload_file(str(path),bucket,prefix+'/'+key)
original_save=Accelerator.save_state;original_log=Accelerator.log;original_end=Accelerator.end_training
started=time.monotonic();timings={};last_step=0

def save(self,output_dir,*args,**kwargs):
 # Every worker reaches this callback, unlike the upstream main-only DDP save gate.
 self.wait_for_everyone()
 if self.is_main_process:
  usage=shutil.disk_usage('/cache');minimum=cfg['infrastructure']['minimum_output_reserve_bytes']
  print(json.dumps({'checkpoint_free_bytes':usage.free,'capacity_bytes':usage.total,'required_free_bytes':minimum}),flush=True)
  if usage.free<minimum:raise RuntimeError('Shared cache has insufficient checkpoint reserve; nothing deleted.')
 self.wait_for_everyone()
 result=original_save(self,output_dir,*args,**kwargs)
 # original_save writes rank-specific RNG files; never publish a checkpoint before all are durable.
 self.wait_for_everyone()
 if self.is_main_process:
  folder=Path(output_dir)
  expected=[folder/('random_states_'+str(i)+'.pkl') for i in range(world)]
  if not all(p.exists() for p in expected):raise RuntimeError('Incomplete distributed RNG state: '+str([p.name for p in expected if not p.exists()]))
  if not (folder/'pytorch_lora_weights.safetensors').exists():raise RuntimeError('Checkpoint adapter weights missing')
  files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
  marker={'run_key':run_key,'world_size':world,'checkpoint':folder.name,'files':files,'rank_rng_states':[p.name for p in expected]}
  for rel in files:upload(folder/rel,'checkpoints/'+folder.name+'/'+rel)
  write(folder/'COMPLETE.json',marker);upload(folder/'COMPLETE.json','checkpoints/'+folder.name+'/manifest.json')
  write(output/'latest-complete.json',{'run_key':run_key,'world_size':world,'checkpoint':folder.name,'manifest_key':'checkpoints/'+folder.name+'/manifest.json'})
  upload(output/'latest-complete.json','latest-complete.json')
 # Synchronise after the S3 commit as well, so no rank trains beyond an unpublished checkpoint.
 self.wait_for_everyone()
 return result

def log(self,values,*args,**kwargs):
 global last_step
 step=kwargs.get('step',args[0] if args else None)
 # Each rank logs the same update; reduce loss rather than reporting only rank zero's example.
 reported=dict(values)
 if 'loss' in reported:
  loss=torch.tensor(float(reported['loss']),device=self.device)
  reported['loss']=float(self.reduce(loss,reduction='mean').item())
 if self.is_main_process:
  epoch=sys._getframe(1).f_locals.get('epoch')
  record={'step':step,'epoch':epoch,'elapsed_seconds':time.monotonic()-started,'metrics':reported,'rank0_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'world_size':world,'effective_batch_size':8}
  with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
  if step is not None:timings[step]=record['elapsed_seconds'];last_step=step
  if step is not None and step%10==0:upload(output/'metrics.jsonl','metrics.jsonl')
 return original_log(self,reported,*args,**kwargs)

def end(self,*args,**kwargs):
 ordered=sorted(timings.items());deltas=[b[1]-a[1] for a,b in zip(ordered,ordered[1:]) if b[0]==a[0]+1]
 record={'state':'complete','rank':rank,'world_size':world,'local_rank':local_rank,'run_key':run_key,'mode':mode,'gpu':torch.cuda.get_device_name(local_rank),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'device_total_memory_bytes':torch.cuda.get_device_properties(local_rank).total_memory,'elapsed_seconds_including_precomputation':time.monotonic()-started,'unique_logged_update_steps':len(timings),'last_logged_update':last_step,'median_recent_update_seconds':statistics.median(deltas[-10:]) if deltas else None}
 write(output/('rank-'+str(rank)+'-completion.json'),record)
 self.wait_for_everyone()
 return original_end(self,*args,**kwargs)
Accelerator.save_state=save;Accelerator.log=log;Accelerator.end_training=end
def publish_training_previews(images,args,epoch,global_step):
 assert rank==0, 'Only rank zero publishes native validation previews'
 from PIL import Image
 validation_id=os.environ['TRAIN_VALIDATION_ID'];input_path=Path(args.validation_image)
 index_path=output/'preview-index.json'
 index=json.loads(index_path.read_text()) if index_path.exists() else {'run_key':run_key,'world_size':world,'previews':[]}
 for number,image in enumerate(images):
  name='step-'+str(global_step).zfill(6)+'-epoch-'+str(epoch).zfill(4)+'-'+str(number)
  folder=output/'previews';folder.mkdir(exist_ok=True)
  png=folder/(name+'.png')
  if image.mode=='RGBA':
   bg=Image.new('RGBA',image.size,'white');bg.alpha_composite(image);image=bg.convert('RGB')
  image.save(png)
  metadata={'run_key':run_key,'world_size':world,'id':validation_id,'global_step':global_step,'epoch':epoch,'prompt':args.validation_prompt or args.final_validation_prompt,'seed':args.seed,'num_inference_steps':args.validation_num_inference_steps,'true_cfg_scale':1.0,'vae_tiling':False,'width':image.width,'height':image.height,'input_sha256':sha(input_path),'output_sha256':sha(png),'image_key':prefix+'/previews/'+png.name,'metadata_key':prefix+'/previews/'+name+'.json'}
  meta=folder/(name+'.json');write(meta,metadata)
  upload(png,'previews/'+png.name);upload(meta,'previews/'+meta.name)
  index['previews']=[p for p in index['previews'] if p['image_key']!=metadata['image_key']]+[metadata];index['latest']=metadata
 write(index_path,index);upload(index_path,'preview-index.json')

train=cfg['training'];steps=cfg['calibration_updates'] if mode=='calibrate' else train['max_train_steps']
argv=['train','--pretrained_model_name_or_path',os.environ['TRAIN_MODEL_SNAPSHOT'],'--dataset_name',str(ROOT/'frozen-dataset'),'--cond_image_column','cond_image','--image_column','image','--caption_column','caption','--instance_prompt',cfg['instruction'],'--output_dir',str(output),'--mixed_precision','bf16','--gradient_checkpointing','--cache_latents','--offload','--use_aspect_ratio_buckets','--center_crop','--lr_scheduler','constant','--lr_warmup_steps','0','--report_to','tensorboard','--skip_final_inference','--dataloader_num_workers','0']
for key,value in train.items():
 if key not in ['max_train_steps','checkpointing_steps']:argv+=['--'+key,str(value)]
argv+=['--max_train_steps',str(steps),'--checkpointing_steps',str(10 if mode=='calibrate' else train['checkpointing_steps'])]
manifest=json.loads((ROOT/'frozen-dataset/manifest.json').read_text())
validation=[r for r in manifest['rows'] if r['split']=='validation']
if not validation:raise RuntimeError('Native validation needs a held-out image')
heldout=next((r for r in validation if r['id'].startswith('006')),validation[0])
os.environ['TRAIN_VALIDATION_ID']=heldout['id']
argv+=['--validation_prompt',cfg['instruction'],'--validation_image',str(ROOT/'frozen-dataset'/heldout['cond_image']),'--validation_epochs','5','--num_validation_images','1','--validation_num_inference_steps','40']
resume=os.environ.get('TRAIN_RESUME_CHECKPOINT')
if resume:argv+=['--resume_from_checkpoint',resume]
if rank==0:write(output/'training-command.json',argv);upload(output/'training-command.json','training-command.json')
sys.argv=argv
runpy.run_path(str(ROOT/'trainer_ddp.py'),run_name='__main__',init_globals={'publish_training_previews':publish_training_previews})
