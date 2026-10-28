"""Build a reproducible, reviewed image-edit evaluation package."""
from pathlib import Path
import json,tarfile,hashlib
ROOT=Path(__file__).parent
DATA=Path('/Users/kiparis/Downloads/Pencil colouring dataset')
PROMPT=('Transform the supplied image into a realistic graphite pencil drawing suitable for a child to colour in. '
'Preserve the identity, facial expression, pose, proportions, clothing, accessories, visible text and important object details. '
'Use precise natural pencil contours, fine but restrained graphite texture and light, soft mid-tones. '
'Leave broad areas of white paper for coloured pencils. Keep dark clothing, hair and fur light enough to colour; avoid dense black fills, heavy cross-hatching and deep shadows. '
'Use a clean white paper background and omit photographic backgrounds. Keep the existing framing; do not invent limbs, extra people, objects or text. '
'The result should remain recognisable and realistic, with a hand-drawn pencil finish rather than a photograph, cartoon, vector outline or fully shaded finished artwork. Monochrome graphite only.')
rows=json.loads((DATA/'manifest.json').read_text());tasks=[]
for r in rows:
 if r['review_status']!='approved':continue
 prompt=PROMPT
 if r['kind']=='multi_input':prompt=('Create one coherent graphite pencil colouring illustration combining the subjects in all supplied reference images. Preserve the recognisable identity, clothing and distinguishing details of each subject. Do not duplicate subjects. '+PROMPT.replace('Transform the supplied image into','Render the composition as').replace('Keep the existing framing;','Use a balanced composition;'))
 if r['id']=='029_pair':prompt+=' Show the entire car, completing its cropped edges, and retain the person beside the car. This full-car request is intentional.'
 tasks.append({'id':r['id'],'inputs':r['inputs'],'reference':r['output'],'kind':r['kind'],'source_group':r.get('group',r['id']),'prompt':prompt,'review_notes':r.get('review_notes',''),'seed':42})
config={'run_id':'qwen-edit-2511-baseline-v1','model':json.loads((ROOT/'model.json').read_text()),'additional_loras':[],'dtype':'bfloat16','num_inference_steps':40,'true_cfg_scale':4.0,'guidance_scale':1.0,'negative_prompt':' ','target_pixels':1024*1024,'input_preprocessing':'EXIF orientation and RGB conversion only. No cropping or reference-based conditioning.','tasks':tasks}
(ROOT/'run.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
files=[]
with tarfile.open(ROOT/'dataset.tar.gz','w:gz') as t:
 for p in sorted(DATA.rglob('*')):
  if not p.is_file() or 'thumbnails' in p.parts or p.suffix=='.html':continue
  rel=p.relative_to(DATA);t.add(p,arcname=str(rel));files.append({'path':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'dataset-checksums.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'tasks':len(tasks),'single_input':sum(r['kind']=='single_input' for r in tasks),'multi_input':sum(r['kind']=='multi_input' for r in tasks),'archive_bytes':(ROOT/'dataset.tar.gz').stat().st_size,'copied_files':len(files)}))
