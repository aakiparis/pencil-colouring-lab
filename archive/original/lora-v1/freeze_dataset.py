"""Validate reviewed choices and freeze a portable, checksummed paired dataset."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(selection,destination):
 cfg=json.loads(selection.read_text());src=Path(cfg['dataset_root'])
 if not cfg['group_review_approved'] or not cfg['target_selection_approved']:
  raise ValueError('Target selection and grouping are still pending. No training dataset has been frozen.')
 included=[]; group_splits={};identity_splits={};hash_splits={}
 for row in cfg['rows']:
  if row.get('include_in_pilot', True) is False:continue
  if row.get('target_approval')!='approved':continue
  if len(row['inputs'])!=1:raise ValueError('Single-input pilot only: '+row['id'])
  if not row.get('grouping_confirmed') or row.get('geometry_review')!='approved':raise ValueError('Grouping or geometry not approved: '+row['id'])
  if row['split'] not in ['train','validation','test']:raise ValueError('Invalid split')
  target=row['selected_target']
  if not isinstance(target,dict) or not target.get('path'):raise ValueError('No selected target: '+row['id'])
  if target.get('source_compatibility') in ['mismatch','output_checksum_mismatch']:raise ValueError('Selected target uses a different original: '+row['id'])
  for key,seen in [(row['group_id'],group_splits)]+[(x,identity_splits) for x in row['identities_inferred_from_filenames']]+[(x['sha256'],hash_splits) for x in row['inputs']]:
   if key in seen and seen[key]!=row['split']:raise ValueError('Split leakage: '+str(key))
   seen[key]=row['split']
  for item in [*row['inputs'],target]:
   path=(src/item['path']).resolve()
   if not path.is_relative_to(src.resolve()) or sha(path)!=item['sha256']:raise ValueError('Source checksum or path mismatch: '+row['id'])
  included.append(row)
 if not all(any(r['split']==s for r in included) for s in ['train','validation','test']):raise ValueError('Need separate non-empty train, validation and test splits.')
 if destination.exists():raise ValueError('Refusing to overwrite frozen dataset')
 destination.mkdir(parents=True)
 from PIL import Image,ImageOps
 from datasets import Dataset,Features,Image as HFImage,Value
 files={}; manifest=[]
 for split in ['train','validation','test']:
  records=[]
  for row in included:
   if row['split']!=split:continue
   with Image.open(src/row['inputs'][0]['path']) as im: source_size=ImageOps.exif_transpose(im).size
   with Image.open(src/row['selected_target']['path']) as im: target_size=ImageOps.exif_transpose(im).size
   aspect_change=abs((source_size[0]/source_size[1])/(target_size[0]/target_size[1])-1)
   if aspect_change>0.05 and row.get('geometry_normalisation')!='white_letterbox_to_source_aspect' and row.get('allow_aspect_ratio_warp') is not True:raise ValueError('Source/target aspect differs by more than 5%; review geometry explicitly before allowing the trainer to stretch the condition: '+row['id'])
   item={'id':row['id'],'split':split,'group_id':row['group_id'],'instruction':cfg['instruction'],'source':row,'geometry':{'input_size':source_size,'target_size':target_size,'relative_aspect_ratio_change':aspect_change,'shared_crop':True,'random_flip':False}}
   for field,original in [('cond_image',row['inputs'][0]),('image',row['selected_target'])]:
    p=destination/'images'/row['id']/(field+'.png');p.parent.mkdir(parents=True,exist_ok=True)
    # Preserve complete images and opaque alpha; the trainer applies shared geometry. No cropping here.
    with Image.open(src/original['path']) as im:
     im=ImageOps.exif_transpose(im).convert('RGBA')
     if row.get('geometry_normalisation')=='white_letterbox_to_source_aspect':
      import math
      ratio=source_size[0]/source_size[1];pixels=target_size[0]*target_size[1]
      canvas_size=(max(32,round(math.sqrt(pixels*ratio)/32)*32),max(32,round(math.sqrt(pixels/ratio)/32)*32))
      fitted=ImageOps.contain(im,canvas_size,Image.Resampling.LANCZOS)
      canvas=Image.new('RGBA',canvas_size,'white');canvas.alpha_composite(fitted,((canvas.width-fitted.width)//2,(canvas.height-fitted.height)//2));im=canvas
      item['geometry'].update(normalisation='white_letterbox_to_source_aspect',canvas_size=canvas_size,preserves_image_aspect=True)
     im.save(p)
    rel=str(p.relative_to(destination));files[rel]=sha(p);item[field]=rel
   records.append({'id':row['id'],'caption':cfg['instruction'],'cond_image':{'bytes':(destination/item['cond_image']).read_bytes(),'path':None},'image':{'bytes':(destination/item['image']).read_bytes(),'path':None}})
   manifest.append(item)
  features=Features({'id':Value('string'),'caption':Value('string'),'cond_image':HFImage(),'image':HFImage()})
  p=destination/'data'/(split+'-00000-of-00001.parquet');p.parent.mkdir(exist_ok=True)
  Dataset.from_list(records,features=features).to_parquet(str(p));files[str(p.relative_to(destination))]=sha(p)
 frozen={'schema_version':1,'selection_sha256':sha(selection),'instruction':cfg['instruction'],'rows':manifest,'files':files}
 (destination/'manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
 print('Frozen dataset:',destination,'rows:',len(manifest))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,default=ROOT/'selection.json');ap.add_argument('--output',type=Path,default=ROOT/'frozen-dataset');a=ap.parse_args();freeze(a.selection,a.output)
