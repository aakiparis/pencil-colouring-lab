"""Download the two generated test images, verify provenance and publish a local comparison."""
import hashlib,html,json,sys,time
from pathlib import Path
F=Path(__file__).resolve().parent;ROOT=F.parent;sys.path.insert(0,str(ROOT))
from launch import call
sys.path.insert(0,str(ROOT.parent));from storage_ops import client
from botocore.exceptions import ClientError
s,b=client();cfg=json.loads((F/'run.json').read_text());runhash=hashlib.sha256((F/'run.json').read_bytes()).hexdigest();data=Path('/Users/kiparis/Downloads/Pencil colouring dataset');out=data/'inference-tests/spiderman-20261018';(out/'outputs').mkdir(parents=True,exist_ok=True);last=None
while True:
 job=call(['ai','job','get',json.loads((F/'batch-job.json').read_text())['metadata']['id']]);state=job['status']['state'];results={}
 for t in cfg['tasks']:
  try:raw=s.get_object(Bucket=b,Key=cfg['prefix']+'/results/'+t['id']+'.json')['Body'].read()
  except ClientError as ex:
   if ex.response['Error']['Code'] in ['NoSuchKey','404']:continue
   raise
  m=json.loads(raw);assert m['run_sha256']==runhash and m['prompt']==t['prompt'];assert m['additional_loras']==cfg['additional_loras'];assert m['input_sha256']=={n:cfg['input_sha256'][n] for n in t['inputs']}
  if m['status']=='completed':
   p=out/'outputs'/(t['id']+'.png')
   if not p.exists():s.download_file(b,cfg['prefix']+'/results/'+t['id']+'.png',str(p))
   assert hashlib.sha256(p.read_bytes()).hexdigest()==m['output_sha256'];p.with_suffix('.json').write_bytes(raw)
  results[t['id']]=m
 report={'job_id':job['metadata']['id'],'state':state,'completed':sum(m['status']=='completed' for m in results.values()),'total':2,'results':results};(F/'progress.json').write_text(json.dumps(report,indent=2)+'\n');(out/'status.json').write_text(json.dumps(report,indent=2)+'\n');(out/'run.json').write_bytes((F/'run.json').read_bytes())
 page='<html lang="en-GB"><meta charset="utf-8"><title>Pencil colouring · Spider-Man test</title><style>body{font:16px system-ui;margin:24px;background:#f4f5f6}.pair{display:flex;gap:20px}img{width:44vw;max-height:85vh;object-fit:contain;background:white}section{margin-bottom:30px}pre{white-space:pre-wrap}</style><h1>Pencil colouring · Spider-Man test</h1><p>Qwen-Image-2.1 · LoRA update 200 · strength 1 · detailed background v4 · seed 42</p><p>'+state+' · '+str(report['completed'])+'/2 complete</p>'
 for t in cfg['tasks']:
  rel=t['id']+'/input.jpeg';m=results.get(t['id'],{});dest='outputs/'+t['id']+'.png';page+='<section><h2>'+t['id']+'</h2><div class="pair"><a href="'+rel+'"><img src="'+rel+'"></a>'+(('<a href="'+dest+'" download><img src="'+dest+'"></a>') if m.get('status')=='completed' else '<p>Waiting for generation</p>')+'</div><p><a href="'+dest+'" download>Download PNG</a></p><details><summary>Exact prompt</summary><pre>'+html.escape(t['prompt'])+'</pre></details></section>'
 if state not in ['COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED']:page+='<script>setTimeout(()=>location.reload(),20000)</script>'
 (out/'review.html').write_text(page+'</html>')
 event=(state,report['completed'])
 if event!=last:print(json.dumps({'state':state,'completed':report['completed'],'total':2}),flush=True);last=event
 if state in ['COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED']:
  if state!='COMPLETED' or report['completed']!=2:raise RuntimeError('Test incomplete: '+state)
  print('RESULTS '+str(out/'outputs'),flush=True);break
 time.sleep(20)
