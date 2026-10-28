"""Collect identical-seed repeats, compare actual decoded pixels and update the existing review page."""
import hashlib,html,itertools,json,sys,time
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
F=Path(__file__).resolve().parent;ROOT=F.parent;sys.path.insert(0,str(ROOT))
from launch import call
sys.path.insert(0,str(ROOT.parent));from storage_ops import client
from botocore.exceptions import ClientError
s,b=client();cfg=json.loads((F/'run.json').read_text());runhash=hashlib.sha256((F/'run.json').read_bytes()).hexdigest();out=Path('/Users/kiparis/Downloads/Pencil colouring dataset/inference-tests/spiderman-20261018');(out/'repeats').mkdir(exist_ok=True);last=None
if not (out/'review-before-repeats.html').exists():(out/'review-before-repeats.html').write_bytes((out/'review.html').read_bytes())
def compare(a,b):
 ia=Image.open(a).convert('RGB');ib=Image.open(b).convert('RGB');result={'same_dimensions':ia.size==ib.size,'same_png_sha256':hashlib.sha256(a.read_bytes()).hexdigest()==hashlib.sha256(b.read_bytes()).hexdigest()}
 if ia.size==ib.size:
  delta=ImageChops.difference(ia,ib);channels=delta.split();maximum=ImageChops.lighter(ImageChops.lighter(channels[0],channels[1]),channels[2]);hist=maximum.histogram();result.update(identical_pixels=delta.getbbox() is None,mean_absolute_channel_difference=sum(ImageStat.Stat(delta).mean)/3,maximum_channel_difference=max(x[1] for x in delta.getextrema()),changed_pixel_fraction=1-hist[0]/(ia.width*ia.height))
 else:result['identical_pixels']=False
 return result
while True:
 job=call(['ai','job','get',json.loads((F/'batch-job.json').read_text())['metadata']['id']]);state=job['status']['state'];results={}
 for t in cfg['tasks']:
  try:raw=s.get_object(Bucket=b,Key=cfg['prefix']+'/results/'+t['id']+'.json')['Body'].read()
  except ClientError as ex:
   if ex.response['Error']['Code'] in ['NoSuchKey','404']:continue
   raise
  m=json.loads(raw);assert m['run_sha256']==runhash and m['prompt']==t['prompt'] and m['seed']==42;assert m['additional_loras']==cfg['additional_loras'];assert m['input_sha256']=={n:cfg['input_sha256'][n] for n in t['inputs']}
  if m['status']=='completed':
   p=out/'repeats'/(t['id']+'.png')
   if not p.exists():s.download_file(b,cfg['prefix']+'/results/'+t['id']+'.png',str(p))
   assert hashlib.sha256(p.read_bytes()).hexdigest()==m['output_sha256'];p.with_suffix('.json').write_bytes(raw)
  results[t['id']]=m
 report={'job_id':job['metadata']['id'],'state':state,'completed':sum(m['status']=='completed' for m in results.values()),'total':6,'seed':42,'comparison_basis':cfg['comparison_basis'],'comparisons':{}}
 for original in ['spiderman-poster-01','spiderman-poster-02']:
  existing={'previous_run':out/'outputs'/(original+'.png')}
  for n in [1,2,3]:
   path=out/'repeats'/(original+f'-repeat-{n}.png')
   if path.exists():existing['repeat_'+str(n)]=path
  report['comparisons'][original]={a+'__'+b:compare(existing[a],existing[b]) for a,b in itertools.combinations(existing,2)}
 (F/'progress.json').write_text(json.dumps(report,indent=2)+'\n');(out/'repeat-status.json').write_text(json.dumps(report,indent=2)+'\n');(out/'repeat-run.json').write_bytes((F/'run.json').read_bytes())
 page='<html lang="en-GB"><meta charset="utf-8"><title>Pencil colouring · Spider-Man repeat test</title><style>body{font:15px system-ui;margin:24px;background:#f4f5f6}.pair{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:16px;overflow-x:auto}img{width:100%;height:500px;object-fit:contain;background:white}section{margin-bottom:36px}pre{white-space:pre-wrap}a{color:#14516f}</style><h1>Pencil colouring · Spider-Man repeat test</h1><p>Qwen-Image-2.1 · LoRA update 200 · strength 1 · detailed background v4 · seed 42 in every run.</p><p>Identical source files, prompt and settings. Each repeat starts a fresh seeded generator. Default inference kernels are unchanged; exact reproducibility is measured, not assumed.</p><p>'+state+' · '+str(report['completed'])+'/6 repeats received. <a href="repeat-status.json">Pixel comparisons and checksums</a></p>'
 for original in ['spiderman-poster-01','spiderman-poster-02']:
  page+='<section><h2>'+original+'</h2><div class="pair">'
  items=[('Original',original+'/input.jpeg'),('Previous run · seed 42','outputs/'+original+'.png')]+[(f'Repeat {n} · seed 42','repeats/'+original+f'-repeat-{n}.png') for n in [1,2,3]]
  for title,rel in items:
   page+='<div><b>'+title+'</b><br>'+(('<a href="'+rel+'" target="_blank"><img src="'+rel+'"></a><br><a href="'+rel+'" download>Download</a>') if (out/rel).exists() else '<p>Waiting for generation…</p>')+'</div>'
  page+='</div><details open><summary>Measured repeatability</summary><ul>'
  for name,c in report['comparisons'][original].items():page+='<li>'+html.escape(name.replace('__',' vs ').replace('_',' '))+': '+('pixel-identical' if c['identical_pixels'] else 'different pixels')+(('; mean channel difference '+format(c['mean_absolute_channel_difference'],'.6f')+' / 255') if 'mean_absolute_channel_difference' in c else '')+'</li>'
  page+='</ul></details></section>'
 page+='<details><summary>Exact prompt</summary><pre>'+html.escape(cfg['tasks'][0]['prompt'])+'</pre></details>'
 if state not in ['COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED']:page+='<script>setTimeout(()=>location.reload(),20000)</script>'
 temp=out/'review.html.tmp';temp.write_text(page+'</html>');temp.replace(out/'review.html')
 event=(state,report['completed'])
 if event!=last:print(json.dumps({'state':state,'completed':report['completed'],'total':6}),flush=True);last=event
 if state in ['COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED']:
  if state!='COMPLETED' or report['completed']!=6:raise RuntimeError('Repeat test incomplete: '+state)
  s.put_object(Bucket=b,Key=cfg['prefix']+'/pixel-comparison.json',Body=json.dumps(report,indent=2).encode(),ContentType='application/json');print(json.dumps(report['comparisons']),flush=True);break
 time.sleep(20)
