"""Download actual training progress and previews; render a local review dashboard."""
import hashlib,html,json,sys
from pathlib import Path
ROOT=Path(__file__).parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from storage_ops import client
OUT=Path('/Users/kiparis/Downloads/Pencil colouring dataset/lora-v1');OUT.mkdir(exist_ok=True)
DATA=OUT.parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def object_or_none(s,b,key):
 try:return s.get_object(Bucket=b,Key=key)['Body'].read()
 except s.exceptions.NoSuchKey:return None
 except Exception as exc:
  from botocore.exceptions import ClientError
  if isinstance(exc,ClientError) and exc.response['Error']['Code'] in ['NoSuchKey','404']:return None
  raise

def sync(run_key,jobs):
 cfg=json.loads((ROOT/'pilot.json').read_text());s,b=client();states={};previews=[]
 for phase in ['calibrate','pilot']:
  prefix=cfg['prefix']+'/'+run_key[:16]+'/'+phase
  info={'job':jobs.get(phase),'metrics':[],'total_updates':cfg['calibration_updates'] if phase=='calibrate' else cfg['training']['max_train_steps']}
  for name in ['completion.json','failure.json','provenance.json','latest-complete.json','preview-index.json']:
   raw=object_or_none(s,b,prefix+'/'+name)
   if raw:
    value=json.loads(raw);info[name[:-5]]=value
    (OUT/(phase+'-'+name)).write_bytes(raw)
  raw=object_or_none(s,b,prefix+'/metrics.jsonl')
  if raw:
   info['metrics']=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
   (OUT/(phase+'-metrics.jsonl')).write_bytes(raw)
  idx=info.get('preview-index',{})
  for item in idx.get('previews',[]):
   key=item['image_key'];name=phase+'-'+key.split('/')[-1];dest=OUT/'previews'/name;dest.parent.mkdir(exist_ok=True)
   expected=item.get('output_sha256')
   if not dest.exists() or expected and digest(dest)!=expected:s.download_file(b,key,str(dest))
   if expected:assert digest(dest)==expected
   previews.append(dict(item,phase=phase,local_path='previews/'+name))
  states[phase]=info
 manifest=json.loads((ROOT/'frozen-dataset/manifest.json').read_text());rows={r['id']:r for r in manifest['rows']}
 state={'run_key':run_key,'instruction':cfg['instruction'],'gpu_count':8,'states':states,'previews':previews,'dataset_splits':{split:sum(r['split']==split for r in manifest['rows']) for split in ['train','validation','test']}}
 (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n');build(state,rows);return state

def build(state,rows):
 e=lambda x:html.escape(str(x),quote=True)
 def photo(url,alt):return '<a target="_blank" href="'+e(url)+'"><img src="'+e(url)+'" alt="'+e(alt)+'"></a>'
 panels=[]
 for phase in ['calibrate','pilot']:
  info=state['states'][phase];records=info['metrics'];steps=max([r.get('step') or 0 for r in records],default=0);job=info['job'] or {};status=job.get('status',{}).get('state','Awaiting previous stage')
  loss=[]
  for r in records:
   values=r.get('metrics',{});v=values.get('loss',values.get('train_loss'))
   if isinstance(v,(float,int)):loss.append((r.get('step',0),v))
  chart=''
  if loss:
   ymin=min(v for _,v in loss);ymax=max(v for _,v in loss);span=max(ymax-ymin,1e-9);xmax=max(x for x,_ in loss) or 1
   points=' '.join(f'{20+560*x/xmax:.1f},{150-125*(v-ymin)/span:.1f}' for x,v in loss)
   chart='<svg viewBox="0 0 600 170" role="img" aria-label="Training loss over updates"><polyline fill="none" stroke="#185473" stroke-width="2" points="'+points+'"/></svg><small>Training loss: '+e(round(loss[-1][1],5))+' · loss alone does not establish image quality.</small>'
  latest=info.get('latest-complete',{}).get('checkpoint','None yet')
  panels.append('<section><h2>'+('Calibration' if phase=='calibrate' else 'Pilot')+'</h2><p><b>'+e(status)+'</b> · '+str(steps)+' / '+str(info['total_updates'])+' logged updates</p><p>Latest complete checkpoint: '+e(latest)+'</p>'+chart+'</section>')
 cards=[]
 for p in sorted(state['previews'],key=lambda x:(0 if x['phase']=='calibrate' else 1,x.get('global_step',x.get('step',0)),x.get('epoch',0))):
  ident=p.get('id','006_pair');row=rows.get(ident)
  if not row:continue
  source=row['source'];inp=source['inputs'][0]['path'];target=source['selected_target']['path'];step=p.get('global_step',p.get('step','?'))
  cards.append('<article><h2>'+e(p['phase'])+' · update '+e(step)+' · '+e(ident)+'</h2><div class="comparison"><div><h3>Original</h3>'+photo('../'+inp,'Original photo')+'</div><div><h3>Selected training target</h3>'+photo('../'+target,'User-selected target')+'</div><div><h3>LoRA · short prompt</h3>'+photo(p['local_path'],'Actual validation preview')+'</div></div><p>Prompt: '+e(state['instruction'])+' · seed '+e(p.get('seed',42))+'</p></article>')
 body=''.join(cards) or '<article><p>No validation image yet. This page shows actual generated previews only. Package installation, data preparation and the first training updates come first.</p></article>'
 css='*{box-sizing:border-box}body{font:15px/1.5 system-ui;background:#f3f5f7;color:#17212b;margin:0}header{background:#142c3e;color:white;padding:20px}main{max-width:1500px;margin:auto;padding:18px}.states{display:grid;grid-template-columns:1fr 1fr;gap:16px}section,article{background:white;padding:16px;border:1px solid #dbe3e9;border-radius:8px;margin-bottom:16px}.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%;height:380px;object-fit:contain}h1{font-size:24px}h2{font-size:19px}h3{font-size:14px}a{color:#185473}header a{color:white}svg{max-height:170px;width:100%}small{color:#657382}@media(max-width:800px){.states{grid-template-columns:1fr}img{height:240px}}'
 page='<!doctype html><html lang="en-GB"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pencil LoRA · live training review</title><style>'+css+'</style><header><h1>Pencil LoRA · live training review</h1><p>One shared adapter · 8 preemptible RTX GPUs · '+e(state['instruction'])+'</p><p>20 calibration updates, then a separate 200-update pilot. One update combines eight source/target examples. This is not a count of unique pictures.</p><p>Periodic previews use held-out validation photos. Final test photos remain separate. Preview generation may pause training briefly.</p><a href="evaluation/review.html">LoRA validation comparison</a> · <a href="../benchmark-v2/review.html">Model comparison and selected targets</a> · <a href="status.json">Saved progress</a><p id="freshness"></p></header><main><div class="states">'+''.join(panels)+'</div>'+body+'</main><script>const opened=Date.now();setInterval(async()=>{try{const r=await fetch("status.json",{cache:"no-store"});const s=await r.json();const current='+json.dumps({p:max([r.get('step') or 0 for r in state['states'][p]['metrics']],default=0) for p in ['calibrate','pilot']})+';const cloudStates='+json.dumps({p:(state['states'][p].get('job') or {}).get('status',{}).get('state') for p in ['calibrate','pilot']})+';const changed=Object.entries(s.states).some(([p,v])=>(v.job?.status?.state||null)!==cloudStates[p])||Object.entries(s.states).some(([p,v])=>Math.max(0,...v.metrics.map(r=>r.step||0))!==current[p])||s.previews.length!=='+str(len(state['previews']))+';if(changed)location.reload();else document.getElementById("freshness").textContent="Progress checked automatically; waiting for the next recorded update.";}catch(_){}},20000);</script></html>'
 tmp=OUT/'review.html.tmp';tmp.write_text(page);tmp.replace(OUT/'review.html')
