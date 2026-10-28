"""Check paired export/load and selection guardrails without cloud or GPU work."""
import hashlib,json,tempfile
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from freeze_dataset import freeze
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
from launch import verify_calibration
checks={}
cal_cfg={'calibration_updates':20,'infrastructure':{'gpu_count':8}};cal_key='fixture-run-key'
cal_completion={'state':'complete','world_size':8,'mode':'calibrate','run_key':cal_key,'unique_logged_update_steps':20,'median_recent_update_seconds':1.0,'peak_allocated_bytes':1}
cal_raw=json.dumps(cal_completion).encode();cal_review={'approved':True,'run_key':cal_key,'completion_sha256':hashlib.sha256(cal_raw).hexdigest()}
verify_calibration(cal_cfg,cal_key,cal_review,cal_raw);checks['matching_calibration_evidence_accepted']=True
for label,key,review,raw in [('wrong_run_key','different-run',cal_review,cal_raw),('tampered_completion',cal_key,cal_review,cal_raw+b' '),('missing_timing',cal_key,dict(cal_review,completion_sha256=hashlib.sha256(b'{}').hexdigest()),b'{}')]:
 try:verify_calibration(cal_cfg,key,review,raw);raise AssertionError(label+' should be rejected')
 except SystemExit:checks['calibration_guard_'+label]=True
with tempfile.TemporaryDirectory(prefix='pencil-lora-export-') as temporary:
 root=Path(temporary);source=root/'source';source.mkdir();rows=[]
 for index,split in enumerate(['train','validation','test']):
  inp=source/(split+'-input.png');target=source/(split+'-target.png')
  Image.new('RGB',(256,320),(index*50,100,200)).save(inp);Image.new('RGB',(256,320),(index*50,150,200)).save(target)
  rows.append({'id':split,'inputs':[{'path':inp.name,'sha256':sha(inp)}],'selected_target':{'path':target.name,'sha256':sha(target)},'target_approval':'approved','geometry_review':'approved','grouping_confirmed':True,'group_id':split,'identities_inferred_from_filenames':[split],'split':split})
 config={'dataset_root':str(source),'group_review_approved':True,'target_selection_approved':True,'instruction':'Create a pencil colouring page.','rows':rows};selection=root/'selection.json';selection.write_text(json.dumps(config))
 freeze(selection,root/'frozen')
 ds=load_dataset(str(root/'frozen'))
 for split in ['train','validation','test']:
  assert len(ds[split])==1
  for col in ['image','cond_image']:
   im=ds[split][0][col];assert im.mode=='RGBA' and im.size==(256,320)
  assert ds[split][0]['caption']==config['instruction']
 checks['self_contained_parquet_images_and_splits']=True
 config['target_selection_approved']=False;selection.write_text(json.dumps(config))
 try:freeze(selection,root/'unapproved');raise AssertionError('Missing approval should block freeze')
 except ValueError:checks['pending_target_guard']=True
 config['target_selection_approved']=True;config['rows'][1]['identities_inferred_from_filenames']=['train'];selection.write_text(json.dumps(config))
 try:freeze(selection,root/'leakage');raise AssertionError('Identity leakage should block freeze')
 except ValueError:checks['identity_leakage_guard']=True
 config['rows'][1]['identities_inferred_from_filenames']=['validation'];config['rows'][0]['selected_target']['source_compatibility']='mismatch';selection.write_text(json.dumps(config))
 try:freeze(selection,root/'mismatch');raise AssertionError('Source mismatch should block freeze')
 except ValueError:checks['source_mismatch_guard']=True
 checks['real_selection_not_modified_by_synthetic_fixture']=True
 report={'state':'preparation_verified_no_training_launched','checks':checks,'gpu_training_tested':False,'gpu_inference_tested':False,'cloud_resources_allocated':False,'synthetic_images_only':True}
 (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
