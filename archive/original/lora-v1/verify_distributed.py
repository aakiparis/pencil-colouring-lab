"""Local DDP structural/identity checks; never imports CUDA or allocates cloud compute."""
import ast,hashlib,json,tempfile
from pathlib import Path
from run_identity import RUN_FILES,compute_run_key
from launch import verify_calibration
ROOT=Path(__file__).parent;checks={}
for name in ['trainer_ddp.py','distributed_train.py','train_worker.py','launch.py','run_identity.py']:ast.parse((ROOT/name).read_text())
checks['distributed_python_syntax']=True
s=(ROOT/'trainer_ddp.py').read_text();hook=(ROOT/'distributed_train.py').read_text();worker=(ROOT/'train_worker.py').read_text();cfg=json.loads((ROOT/'pilot.json').read_text())
assert cfg['infrastructure']['gpu_count']==8 and cfg['infrastructure']['preset']=='8gpu-192vcpu-1744gb' and cfg['training']['train_batch_size']==1 and cfg['training']['gradient_accumulation_steps']==1
checks['one_adapter_ddp_eight_effective_batch']=True
assert 'if True:  # Local DDP change: every rank must save' in s and 'weights.clear()' in s
assert hook.count('self.wait_for_everyone()')>=5 and 'random_states_' in hook and 'for i in range(world)' in hook
checks['all_rank_rng_checkpoint_commit_contract']=True
assert 'accelerator.skip_first_batches' in s and 'batch_sampler.generator.seed' in s
checks['resume_epoch_and_batch_offset_patch']=True
assert 'publish_training_previews(images, args, epoch, global_step)' in s and s.count('epoch % args.validation_epochs == 0')==3 and 'timeout=timedelta(minutes=20)' in s
assert 'pipeline.vae.disable_tiling()' in s and "'preview-index.json'" in hook and "'--validation_epochs','5'" in hook
checks['native_live_transformer_preview_contract']=True
assert '--multi_gpu' in worker and "'--num_processes','8'" in worker and 'import torch' not in worker
checks['parent_does_not_initialise_cuda']=True
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)
 for name in RUN_FILES:
  dest=p/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes() if (ROOT/name).exists() else b'{}')
 before=compute_run_key(p);(p/'distributed_train.py').write_bytes((p/'distributed_train.py').read_bytes()+b'\n')
 assert before!=compute_run_key(p)
checks['distributed_helper_change_invalidates_run_identity']=True
completion={'state':'complete','mode':'calibrate','run_key':'test-run','world_size':8,'last_logged_update':20,'median_recent_update_seconds':1,'peak_allocated_bytes':1}
raw=json.dumps(completion).encode();review={'approved':True,'run_key':'test-run','completion_sha256':hashlib.sha256(raw).hexdigest()}
verify_calibration(cfg,'test-run',review,raw)
for test in ['wrong_run','wrong_world','wrong_checksum']:
 changed=dict(completion);key='test-run';candidate_review=dict(review);candidate_raw=raw
 if test=='wrong_run':key='another-run'
 if test=='wrong_world':changed['world_size']=1;candidate_raw=json.dumps(changed).encode();candidate_review['completion_sha256']=hashlib.sha256(candidate_raw).hexdigest()
 if test=='wrong_checksum':candidate_raw+=b' '
 try:verify_calibration(cfg,key,candidate_review,candidate_raw);raise AssertionError('Invalid calibration accepted: '+test)
 except SystemExit:pass
checks['calibration_matches_code_dataset_world_and_s3_completion']=True
report={'state':'local_contract_checks_passed_gpu_calibration_required','checks':checks,'gpu_ddp_executed':False,'gpu_memory_fit_verified':False,'native_preview_executed':False,'cloud_compute_allocated_by_this_check':False}
(ROOT/'distributed-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
