"""Build a conservative, explicitly unapproved paired-image training selection."""
import hashlib,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).parent
DATA=Path('/Users/kiparis/Downloads/Pencil colouring dataset')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def normal(s): return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()
def identities(s):
 s=normal(s); found=[]
 for name,aliases in {'ronaldo':['ronaldo','cr7','cristiano'],'yamal':['yamal'],'raphinha':['raphinha'],'vinicius':['vini','vinicius'],'doue':['doue'],'mbappe':['mbappe'],'camavinga':['camavinga','cama'],'neymar':['neymar'],'bellingham':['bellingham','jude'],'messi':['messi'],'diogo_costa':['diogo costa'],'garnacho':['garnacho'],'nico_williams':['nico williams']}.items():
  if any(a in s for a in aliases):found.append(name)
 return found
rows=[]
for row in json.loads((DATA/'manifest.json').read_text()):
 pid=row['id'];n=int(pid[:3]);inp=row.get('inputs',[row['input']]);labels=' '.join(row.get('input_filenames',[]))+ ' '+row.get('output_filename','')
 ids=identities(labels); reasons=[]
 if len(inp)!=1:reasons.append('Multiple input images: excluded from the single-condition pilot.')
 if n in [26,40]:reasons.append('Pair was previously rejected.')
 if n==45:reasons.append('Corrected input: historical ChatGPT, Qwen 2.1 v2 and FLUX outputs use the previous input. Qwen reprompt v3 and background v4 use the corrected input; verify result metadata when selecting.')
 if not ids:
  if n in [1,2]:ids=['tabby_cat_identity_unconfirmed']
  elif n in [44,45,55]:ids=['football_boots_near_duplicates_unconfirmed']
  elif n==41:ids=['minecraft_scene']
  else:reasons.append('Human identity is not established by filenames; resolve before training or evaluation.')
 targets=[]
 for slug,path in [('chatgpt',row.get('output')),('qwen21_reprompt',f'benchmark-v2/qwen21_reprompt/outputs/{pid}.png'),('qwen21_background',f'benchmark-v2/qwen21_background/outputs/{pid}.png')]:
  p=DATA/path if path else None
  if p:
   candidate={'variant':slug,'path':path,'sha256':digest(p) if p.exists() else None,'availability':'available' if p.exists() else 'pending_generation','approval':'not_selected_for_training','source_compatibility':'mismatch' if n==45 and slug=='chatgpt' else 'verify_metadata'}
   meta=p.with_suffix('.json')
   if meta.exists():
    candidate['generation_metadata']=str(meta.relative_to(DATA))
    generated=json.loads(meta.read_text())
    expected={x:digest(DATA/x) for x in inp}
    candidate['source_compatibility']='matched_input_hashes' if generated.get('input_sha256')==expected else 'mismatch'
    if p.exists() and generated.get('output_sha256')!=candidate['sha256']:candidate['source_compatibility']='output_checksum_mismatch'
   targets.append(candidate)
 rows.append({'id':pid,'inputs':[{'path':x,'sha256':digest(DATA/x)} for x in inp], 'source_group':row.get('group'), 'identities_inferred_from_filenames':ids,'grouping_confirmed':False,'group_id':None,'split':None,'target_candidates':targets,'selected_target':None,'target_approval':'pending','geometry_review':'pending','eligibility_notes':reasons,'instruction':'Create a pencil colouring page.'})
# Connected identity and exact-source components prevent the same named person or identical input crossing splits.
parent=list(range(len(rows)))
def find(x):
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):parent[find(a)]=find(b)
seen={}
for i,r in enumerate(rows):
 if len(r['inputs']) != 1: continue
 for key in [('identity',x) for x in r['identities_inferred_from_filenames']]+[('hash',x['sha256']) for x in r['inputs']]+[('source',r['source_group'])]:
  if key in seen:union(i,seen[key])
  else:seen[key]=i
for i,r in enumerate(rows):
 component=[x for j,x in enumerate(rows) if find(j)==find(i)]
 names=sorted({n for x in component for n in x['identities_inferred_from_filenames']})
 r['group_id']='group-'+hashlib.sha256('|'.join(x['id'] for x in component).encode()).hexdigest()[:12]
 r['split']='test' if set(names)&{'vinicius','doue'} else 'validation' if set(names)&{'raphinha','bellingham'} else 'train'
 r['split_status']='provisional_filename_grouping_requires_review'
plan={'schema_version':1,'status':'pending_target_and_group_review','dataset_root':str(DATA),'instruction':'Create a pencil colouring page.','group_review_approved':False,'target_selection_approved':False,'notes':['No generated output is automatically approved as a training target.','Identity labels come from filenames, not face recognition. Confirm all included identities and near duplicates before freezing.','Keep connected co-occurring identities and near duplicates in a single split.','The first pilot is single-input only. Multi-input composition needs a separate trainer extension and evaluation.','Held-out photos must not be used for training, prompt changes or target tuning once the evaluation split is frozen.'],'rows':rows}
p=ROOT/'selection.json'
if p.exists():raise SystemExit('Selection already exists; refusing to overwrite review decisions.')
p.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
print('Created unapproved selection:',len(rows),'rows')
