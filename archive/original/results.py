"""Download checked baseline results and build an offline image comparison."""
import json,hashlib,html,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).parent
DATA=Path('/Users/kiparis/Downloads/Pencil colouring dataset')
OUT=DATA/'baseline-v1'

def sync_results(quiet=False):
 from storage_ops import client
 s,bucket=client();config=json.loads((ROOT/'run.json').read_text());runhash=hashlib.sha256((ROOT/'run.json').read_bytes()).hexdigest()
 folder=OUT/'outputs';folder.mkdir(parents=True,exist_ok=True)
 keys={o['Key'] for page in s.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix='baseline-v1/batch/') for o in page.get('Contents',[])}
 def one(t):
  key='baseline-v1/batch/'+t['id'];dest=folder/(t['id']+'.json')
  if key+'.json' not in keys:return None
  meta=json.loads(s.get_object(Bucket=bucket,Key=key+'.json')['Body'].read())
  if meta.get('run_sha256')!=runhash:raise RuntimeError('Experiment hash mismatch: '+t['id'])
  if meta['status']=='completed':
   image=folder/(t['id']+'.png')
   if not image.exists() or hashlib.sha256(image.read_bytes()).hexdigest()!=meta['output_sha256']:
    s.download_file(bucket,key+'.png',str(image))
   assert hashlib.sha256(image.read_bytes()).hexdigest()==meta['output_sha256']
  dest.write_text(json.dumps(meta,indent=2)+'\n');return meta
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=[x for x in pool.map(one,config['tasks']) if x]
 for key in keys:
  if 'environment' in key or key.endswith('/completed.json'):
   (OUT/key.rsplit('/',1)[-1]).write_bytes(s.get_object(Bucket=bucket,Key=key)['Body'].read())
 summary={'run_id':config['run_id'],'run_sha256':runhash,'model':config['model'],'additional_loras':[],'total_tasks':len(config['tasks']),'completed':sum(x['status']=='completed' for x in results),'failed':sum(x['status']=='failed' for x in results),'pending':len(config['tasks'])-len(results),'output_checksums_verified':True}
 for name,value in [('summary',summary),('run',config),('results',results)]:(OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 build_review(config,results,summary)
 if not quiet:print(json.dumps(summary,indent=2))
 return summary

def build_review(config,results,summary):
 from PIL import Image,ImageOps
 e=lambda s:html.escape(str(s),quote=True)
 thumbs=OUT/'thumbnails';thumbs.mkdir(exist_ok=True)
 by_id={r['id']:r for r in results};metadata={r['id']:r for r in json.loads((DATA/'manifest.json').read_text())}
 def picture(path,label,link):
  dest=thumbs/(hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.jpg')
  if not dest.exists() or dest.stat().st_mtime<path.stat().st_mtime:
   with Image.open(path) as im:
    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((640,640));im.save(dest,quality=90)
  return '<figure><a href="'+e(link)+'" target="_blank"><img loading="lazy" src="thumbnails/'+dest.name+'"></a><figcaption>'+e(label)+'</figcaption></figure>'
 cards=[]
 for t in config['tasks']:
  r=by_id.get(t['id']);m=metadata[t['id']];status=r['status'] if r else 'pending'
  inputs=''.join(picture(DATA/p,'Input '+str(i+1),'../'+p) for i,p in enumerate(t['inputs']))
  reference=picture(DATA/t['reference'],'Your approved example','../'+t['reference'])
  if r and status=='completed':base=picture(OUT/'outputs'/(t['id']+'.png'),'Qwen base model · no added LoRA','outputs/'+t['id']+'.png')
  else:base='<div class="placeholder">'+('Generation failed: '+e(r.get('error','')) if r else 'Result not downloaded yet')+'</div>'
  note='<p class="dataset-note"><b>Your dataset note:</b> '+e(m['review_notes'])+'</p>' if m.get('review_notes') else ''
  disabled='disabled' if status!='completed' else ''
  cards.append('<article data-id="'+e(t['id'])+'" data-kind="'+t['kind']+'" data-result="'+status+'"><div class="card-title"><h2>'+e(t['id'])+'</h2><span>'+('Multiple references' if t['kind']=='multi_input' else 'Single reference')+'</span></div><div class="comparison"><div class="column"><h3>Original</h3><div class="input-images">'+inputs+'</div></div><div class="column"><h3>Approved example</h3>'+reference+'</div><div class="column"><h3>Base model</h3>'+base+'</div></div>'+note+'<div class="controls"><label>Verdict <select class="verdict" '+disabled+'><option value="pending">Pending</option><option value="good_baseline">Good baseline</option><option value="needs_improvement">Needs improvement</option><option value="reject">Reject</option></select></label><label>Compared with the example <select class="preference" '+disabled+'><option value="not_rated">Not rated</option><option value="reference_better">Example is better</option><option value="similar">Similar</option><option value="baseline_better">Base model is better</option></select></label></div><input class="review-note" placeholder="What to improve: likeness, shading, white areas, missing details…"><details><summary>Prompt and generation details</summary><p>'+e(t['prompt'])+'</p><p>Seed '+str(t['seed'])+' · '+str(config['num_inference_steps'])+' steps · BF16 · CFG '+str(config['true_cfg_scale'])+'</p><p>'+e(' + '.join(m['input_filenames']))+'</p>'+('<p>Generation: '+str(r.get('seconds'))+' seconds · '+str(r.get('width'))+' × '+str(r.get('height'))+' pixels</p>' if r and status=='completed' else '')+'</details></article>')
 css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17212b;font:16px/1.5 system-ui,sans-serif}header{padding:28px max(24px,calc((100vw - 1500px)/2));background:#142c3e;color:white}header h1{margin:0 0 8px;font-size:30px}header p{margin:8px 0;color:#dbe7ef}.badge{display:inline-block;background:#e1eddf;color:#1b4e2b;border-radius:6px;padding:3px 9px;font-weight:650}nav{position:sticky;top:0;background:white;padding:12px 24px;border-bottom:1px solid #d5dce3;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:3}main{max-width:1550px;margin:auto;padding:6px 24px 40px}article{background:white;border:1px solid #dde3e8;border-radius:12px;margin:20px 0;padding:20px}.card-title{display:flex;justify-content:space-between;align-items:center;gap:10px}.card-title h2{font-size:20px;margin:0}.card-title span{color:#647484;font-size:14px}.comparison{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}h3{font-size:15px;text-align:center;color:#536578}figure{margin:0}img{width:100%;height:410px;object-fit:contain;background:#fafafa;border:1px solid #edf0f3}figcaption{text-align:center;font-size:13px;color:#647484;margin-top:5px}.input-images{display:flex;gap:8px}.input-images figure{flex:1;min-width:0}.controls{display:flex;gap:18px;flex-wrap:wrap;margin-top:16px}select,input,button{padding:9px 12px;border:1px solid #b9c5d0;border-radius:7px;font:inherit;background:white}button{cursor:pointer}button.primary{background:#185473;color:white;border-color:#185473}label{font-weight:600}.review-note{width:100%;margin-top:12px}details{margin-top:12px;font-size:14px;color:#536578}summary{cursor:pointer}details p{overflow-wrap:anywhere}.dataset-note{font-size:14px;background:#f5f8fb;padding:10px}.placeholder{height:410px;display:flex;align-items:center;justify-content:center;background:#fff3e4;padding:20px}.nav-progress{margin-left:auto;color:#536578;font-size:14px}.notice{color:#536578;font-size:14px;margin:12px 0}#file{max-width:180px}#status-note{color:#c33}@media(max-width:800px){main{padding:0 10px}article{padding:12px}.comparison{gap:6px}img,.placeholder{height:250px}.controls{gap:10px}header{padding:20px}header h1{font-size:25px}}'''
 js='''const runHash=__RUN_HASH__,model=__MODEL__,key='pencil-baseline-review-'+runHash.slice(0,16),cards=[...document.querySelectorAll('article')];let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}function progress(){const completed=cards.filter(c=>c.dataset.result==='completed'),reviewed=completed.filter(c=>c.querySelector('.verdict').value!=='pending');document.querySelector('#review-progress').textContent=reviewed.length+' / '+completed.length+' reviewed'}function persist(){try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){document.querySelector('#status-note').textContent='Browser storage unavailable. Export decisions before closing.'}progress()}for(const c of cards){const id=c.dataset.id,v=c.querySelector('.verdict'),p=c.querySelector('.preference'),n=c.querySelector('.review-note');if(saved[id]){v.value=saved[id].verdict;p.value=saved[id].preference;n.value=saved[id].notes||''}const changed=()=>{saved[id]={verdict:v.value,preference:p.value,notes:n.value};persist()};v.onchange=changed;p.onchange=changed;n.oninput=changed}function filter(){const kind=document.querySelector('#kind').value,status=document.querySelector('#status').value;for(const c of cards)c.hidden=(kind!=='all'&&c.dataset.kind!==kind)||(status==='pending'&&c.querySelector('.verdict').value!=='pending')||(status==='reviewed'&&c.querySelector('.verdict').value==='pending')}document.querySelector('#kind').onchange=filter;document.querySelector('#status').onchange=filter;document.querySelector('#export').onclick=()=>{const decisions=cards.map(c=>({id:c.dataset.id,verdict:c.querySelector('.verdict').value,preference:c.querySelector('.preference').value,notes:c.querySelector('.review-note').value,result_status:c.dataset.result}));const data={version:1,run_sha256:runHash,model,decisions};const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='pencil-baseline-review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};document.querySelector('#file').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());if(d.run_sha256!==runHash||!Array.isArray(d.decisions))throw Error('This review belongs to another run');for(const row of d.decisions){const c=cards.find(c=>c.dataset.id===row.id);if(!c)continue;c.querySelector('.verdict').value=row.verdict;c.querySelector('.preference').value=row.preference;c.querySelector('.review-note').value=row.notes||'';saved[row.id]=row}persist()}catch(err){alert(err.message)}};progress();'''
 js=js.replace('__RUN_HASH__',json.dumps(summary['run_sha256'])).replace('__MODEL__',json.dumps(config['model']))
 page='<!doctype html><html lang="en-GB"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen baseline - image comparison</title><style>'+css+'</style><header><span class="badge">Base model · No added LoRA</span><h1>Original → approved example → Qwen baseline</h1><p>Qwen-Image-Edit-2511 · BF16 · UK South 2 · preemptible GPU jobs</p><p>'+str(summary['completed'])+' / '+str(summary['total_tasks'])+' results downloaded and verified. '+str(summary['failed'])+' failed; '+str(summary['pending'])+' pending.</p><p>The model received only the original input images and the selected text prompt. Approved examples were never used as generation inputs.</p></header><nav><label>Inputs <select id="kind"><option value="all">All</option><option value="single_input">Single reference</option><option value="multi_input">Multiple references</option></select></label><label>Show <select id="status"><option value="all">All</option><option value="pending">Awaiting review</option><option value="reviewed">Reviewed</option></select></label><button class="primary" id="export">Export review</button><label>Import <input type="file" id="file" accept="application/json"></label><span id="review-progress" class="nav-progress"></span></nav><main><p class="notice">Choose a verdict and optionally compare with your approved example. Click an image for the full-size file. Decisions are saved in this browser; export the review JSON when finished. <span id="status-note"></span></p>'+''.join(cards)+'</main><script>'+js+'</script></html>'
 from review_state import integrate_review
 page=integrate_review(page,config,summary,OUT)
 (OUT/'review.html').write_text(page)
 (OUT/'README.txt').write_text('''Qwen image-edit baseline review

Open review.html to compare originals, approved examples and base-model outputs. Choose a verdict, add optional notes, then Export review. Browser decisions are local; export them to preserve or share them.

The run contains reviewed single-input edits and separately labelled multi-input compositions. Text-only creative tasks are excluded. The model is pinned by revision, with no additional LoRA, no reference-image conditioning and no quantisation. Original inputs were EXIF-oriented and converted to RGB, not cropped. Output dimensions use the first input's aspect ratio at approximately one megapixel; dimensions are rounded to multiples of sixteen. Seed 42, 40 inference steps, CFG 4.0 and guidance_scale 1.0 are fixed. Review run.json for every prompt and exception.

outputs/: downloaded full-resolution PNG results and generation metadata. SHA-256 checksums are verified on download. thumbnails/: display previews only. summary.json: completion counts. results.json: combined generation metadata. run.json: frozen experiment configuration. The experiment hash prevents reviews from being mixed between runs.

The base model preserves scene layout by default; pair 022 requests a plain white background and pair 029 requests completion of the whole Bugatti while retaining the person. Multi-input compositions are a separate task type. Existing imperfect source/reference pairs remain identifiable by dataset IDs and notes.
''')
if __name__=='__main__':sync_results()
