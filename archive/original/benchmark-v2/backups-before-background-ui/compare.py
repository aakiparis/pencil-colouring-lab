"""Verify downloads and render a labelled, durable comparison of both new runs."""
import concurrent.futures
import hashlib
import html
import json
import sys
from pathlib import Path

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent))
from storage_ops import client
DATA=Path('/Users/kiparis/Downloads/Pencil colouring dataset')
OUT=DATA/'benchmark-v2'
MODELS=['qwen21','qwen21_reprompt','flux3']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def sync():
    OUT.mkdir(exist_ok=True);s,bucket=client();state={};configs={}
    for slug in MODELS:
        cfg=json.loads((ROOT/slug/'run.json').read_text());configs[slug]=cfg
        runhash=digest(ROOT/slug/'run.json');folder=OUT/slug/'outputs';folder.mkdir(parents=True,exist_ok=True)
        (OUT/slug/'run.json').write_text(json.dumps(cfg,indent=2)+'\n')
        keys={o['Key'] for page in s.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=cfg['prefix']+'/results/') for o in page.get('Contents',[])}
        def download(t):
            key=cfg['prefix']+'/results/'+t['id']
            if key+'.json' not in keys:return None
            meta=json.loads(s.get_object(Bucket=bucket,Key=key+'.json')['Body'].read())
            if meta['run_sha256']!=runhash:raise RuntimeError('Experiment mismatch: '+t['id'])
            assert meta['model']==cfg['model'] and meta['prompt']==t['prompt']
            for name,sha in meta['input_sha256'].items():assert digest(DATA/name)==sha
            if meta['status']=='completed':
                dest=folder/(t['id']+'.png')
                if not dest.exists() or digest(dest)!=meta['output_sha256']:s.download_file(bucket,key+'.png',str(dest))
                assert digest(dest)==meta['output_sha256']
            (folder/(t['id']+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
            return meta
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results=[r for r in pool.map(download,cfg['tasks']) if r]
        state[slug]={'model':cfg['model'],'run_sha256':runhash,'total':len(cfg['tasks']),
                     'completed':sum(r['status']=='completed' for r in results),'failed':sum(r['status']=='failed' for r in results),
                     'results':{r['id']:r for r in results}}
    (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    build(configs,state)
    return {k:{n:v[n] for n in ['completed','failed','total']} for k,v in state.items()}

def build(configs,state):
    from PIL import Image,ImageOps
    e=lambda x:html.escape(str(x),quote=True)
    thumbs=OUT/'thumbnails';thumbs.mkdir(exist_ok=True)
    feedback=json.loads((DATA/'baseline-v1/review-decisions.json').read_text())
    old={r['id']:r for r in feedback['decisions']}
    qwen_feedback=json.loads((ROOT/'qwen21-feedback.json').read_text())
    qwen_defaults={r['id']:r for r in qwen_feedback['decisions']}
    reprompt_feedback=json.loads((ROOT/'qwen21-reprompt-feedback.json').read_text())
    reprompt_defaults={r['id']:r for r in reprompt_feedback['decisions']}
    selection=json.loads((ROOT/'selection.json').read_text())
    corrections_path=ROOT/'display-corrections.json'
    corrections={r['id']:r for r in json.loads(corrections_path.read_text())} if corrections_path.exists() else {}
    old_slug='qwen_edit_2511'
    def controls(name,row=None):
        row=row or {'verdict':'pending','preference':'not_rated','notes':''}
        def options(values,selected):
            return ''.join('<option value="'+value+'"'+(' selected' if value==selected else '')+'>'+e(label)+'</option>' for value,label in values)
        verdicts=[('pending','Pending'),('good_baseline','Good baseline'),('needs_improvement','Needs improvement'),('reject','Reject')]
        winners=[('not_rated','Not rated'),('reference_better','ChatGPT wins'),('similar','Similar'),('baseline_better',name+' wins')]
        return ('<label>Verdict <select class="verdict">'+options(verdicts,row['verdict'])+'</select></label>'
                '<label hidden>Winner <select class="preference">'+options(winners,row['preference'])+'</select></label>'
                '<label>Review comments<textarea class="notes" rows="6" placeholder="Review comments">'+e(row['notes'])+'</textarea></label>')
    def picture(path,label,link,thumb=True):
        src=link
        if thumb:
            dst=thumbs/(hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.jpg')
            if not dst.exists():
                with Image.open(path) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((600,600));im.save(dst,quality=90)
            src='thumbnails/'+dst.name
        return '<a href="'+e(link)+'" target="_blank"><img loading="lazy" src="'+e(src)+'" alt="'+e(label)+'"></a>'
    cards=[]
    for t in configs['qwen21']['tasks']:
        ident=t['id'];f=old[ident]
        correction=corrections.get(ident)
        display_inputs=correction['display_inputs'] if correction else t['inputs']
        original=''.join(picture(DATA/name,'Original input','../'+name) for name in display_inputs)
        cols='<div><h3>Original</h3><div class="inputs">'+original+'</div></div>'
        if correction:
            cols=cols[:-6]+'<small class="source-correction">Corrected original · '+''.join('<a href="../'+e(n)+'" target="_blank">Previous generation input</a>' for n in correction['generation_inputs'])+'</small></div>'
        cols+='<div><h3>ChatGPT · approved</h3>'+picture(DATA/t['reference'],'ChatGPT approved example','../'+t['reference'])+'<small>Exact ChatGPT model unknown</small></div>'
        provenance='Shared feedback applied at your request' if f['source']=='user_authorised_shared_feedback' else 'Your dictated feedback'
        cols+='<div class="old-model" data-model="'+old_slug+'"><h3>Qwen-Image-Edit-2511</h3>'+picture(DATA/'baseline-v1/outputs'/(ident+'.png'),'Previous Qwen baseline','../baseline-v1/outputs/'+ident+'.png')+'<small>Previous prompt · no added LoRA</small>'+controls('Qwen-Image-Edit-2511',f)+'<small>'+e(provenance)+' · '+e(f['verdict_basis'])+'</small></div>'
        for slug in MODELS:
            name=configs[slug]['model']['id'].split('/')[-1]+(' · reprompt' if slug=='qwen21_reprompt' else '')
            result=state[slug]['results'].get(ident);ready=result and result['status']=='completed'
            result_html=(picture(OUT/slug/'outputs'/(ident+'.png'),name,slug+'/outputs/'+ident+'.png',False)
                         if ready else '<p class="placeholder">'+('Generation failed' if result else 'Awaiting generation')+'</p>')
            cols+='<div class="'+('old-model' if slug=='flux3' else '')+'" data-model="'+slug+'"><h3>'+e(name)+'</h3><div class="result-image">'+result_html+'</div><small>'+('Revised prompt · non-tiled decoding · no LoRA' if slug=='qwen21_reprompt' else 'Previous prompt · no added LoRA')+'</small>'
            cols+=controls(name,qwen_defaults.get(ident) if slug=='qwen21' else reprompt_defaults.get(ident) if slug=='qwen21_reprompt' else None)
            if slug=='qwen21':cols+='<small>Earlier shared user feedback; verdict interpreted from requested improvements.</small>'
            if slug=='qwen21_reprompt':cols+='<small>Selected result · accepted baseline with a remaining background caveat.</small>'
            cols+='</div>'
        provenance='Shared feedback applied at your request' if f['source']=='user_authorised_shared_feedback' else 'Your dictated feedback'
        cards.append('<article id="'+e(ident)+'" data-id="'+e(ident)+'" data-kind="'+t['kind']+'"><h2>'+e(ident)+'</h2><div class="comparison">'+cols+'</div><details class="old-feedback" open><summary>Previous Qwen review · '+e(provenance)+'</summary><p><b>Verdict:</b> '+e(f['verdict'])+' · <b>Winner:</b> '+e(f.get('winner') or 'Not explicitly rated')+'</p><p>'+e(f['notes'])+'</p><small>'+e(f['verdict_basis'])+'</small><a href="../baseline-v1/review.html#'+e(ident)+'"> Edit previous review</a></details><details><summary>Prompts and generation details</summary>'+''.join('<p><b>'+e(configs[m]['model']['id']+(' · reprompt' if m=='qwen21_reprompt' else ''))+'</b> · revision '+e(configs[m]['model']['revision'])+' · '+str(configs[m]['num_inference_steps'])+' steps · seed 42</p><p>'+e(next(x['prompt'] for x in configs[m]['tasks'] if x['id']==ident))+'</p><pre class="expanded" data-expanded="'+m+'"></pre>' for m in MODELS)+'</details></article>')
    hashes={slug:state[slug]['run_sha256'] for slug in MODELS}
    # Keep historical reviews under their actual model column, with the same editable controls.
    cards=[__import__('re').sub(r'<details class="old-feedback" open>.*?</details>','',card,flags=__import__('re').S) for card in cards]
    for i,t in enumerate(configs['qwen21']['tasks']):
        if t['id'] in corrections:
            warning='<p class="source-correction">Original corrected by the user. Historical results use the previous original. The reprompt run uses the corrected original; pair 045 is not a prompt-only comparison.</p>'
            cards[i]=cards[i].replace('<div class="comparison">',warning+'<div class="comparison">',1)
    payload={'version':2,'runs':hashes,'models':{slug:configs[slug]['model'] for slug in MODELS},'progress':state,
             'previous_feedback':feedback,'selected_model':selection['selected_model'],'selection':selection,'qwen_feedback':qwen_feedback,'reprompt_feedback':reprompt_feedback,'prompt_policy':json.loads((ROOT/'prompt-policy.json').read_text()),'display_corrections':list(corrections.values()),'previous_model_slug':old_slug}
    payload['runs'][old_slug]=feedback['run_sha256']
    payload['models'][old_slug]=feedback['model']
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17212b;font:15px/1.5 system-ui,sans-serif}header{background:#142c3e;color:white;padding:24px}h1{font-size:25px;margin:0}header p{margin:6px 0}nav{position:sticky;top:0;background:white;padding:12px 24px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;z-index:2;border-bottom:1px solid #ccd4dc}main{max-width:1900px;margin:auto;padding:12px 18px}article{background:white;padding:16px;border-radius:10px;margin-bottom:20px;border:1px solid #dbe3e9}h2{font-size:19px;margin:0 0 12px}.comparison{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}h3{font-size:14px;text-align:center;margin:0 0 8px;min-height:42px}img,.placeholder{width:100%;height:300px;object-fit:contain;background:#fafafa;border:1px solid #edf0f3}.placeholder{display:flex;align-items:center;justify-content:center;color:#677784;margin:0}.inputs{display:flex;gap:4px}.inputs a{min-width:0;flex:1}small{display:block;color:#657382;font-size:12px}label{display:block;margin-top:8px}select,textarea,button{font:inherit;border:1px solid #b8c5d0;border-radius:6px;padding:7px;background:white}select{max-width:100%;width:100%}textarea{width:100%;margin-top:8px;resize:vertical}button{cursor:pointer;background:#185473;color:white}details{margin-top:12px;font-size:14px}summary{cursor:pointer}.old-feedback{background:#f1f6fa;padding:10px;border-radius:6px}.old-feedback p{margin:5px 0}.shared{background:#e8f0f6;padding:12px;border-radius:8px;margin-bottom:16px}.shared p{margin:6px 0}.expanded{white-space:pre-wrap;font-size:12px}#progress{margin-left:auto;color:#536578}body.hide-old .old-model{display:none}body.hide-old .comparison{grid-template-columns:repeat(4,minmax(0,1fr))}#show-old{width:auto}.top-link{color:white}@media(max-width:900px){main{padding:8px}.comparison{gap:6px}img,.placeholder{height:220px}h3{font-size:12px}select,textarea{font-size:12px;padding:5px}}'''
    title='Qwen-Image-2.1 · original prompt and reprompt · historical models'
    css+='[hidden]{display:none!important}.source-correction{background:#fff3d9;padding:9px;border-radius:6px;color:#704d0d}.shared details{margin:0}textarea.notes{min-height:150px;line-height:1.45}'
    page='<!doctype html><html lang="en-GB"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(title)+'</title><style>'+css+'</style><body class="hide-old"><header><h1>'+e(title)+'</h1><p>Corrected prompts · no added LoRA · Nebius UK South 2 · preemptible RTX PRO 6000</p><p>Selected result: Qwen-Image-2.1 · reprompt. Accepted baseline; a stronger light background is the remaining refinement. Historical models remain available above.</p><p>Reprompt changes: sharper contours, stronger clothing tones, selective scene context and clean monochrome paper. VAE tiling is disabled; no image post-processing.</p><p>Models use their own recommended sampling settings. The previous Qwen run used different prompts.</p><a class="top-link" href="../baseline-v1/review.html">Previous review with saved feedback</a></header><nav><label><input id="show-old" type="checkbox"> Show historical models</label><button id="export">Export new reviews</button><label>Import <input id="import" type="file" accept="application/json"></label><span id="progress"></span></nav><main><section class="shared"><b>Shared criteria from your review</b><p>'+e(feedback['general_feedback']['notes'])+'</p><a href="../baseline-v1/review-decisions.json">Saved previous verdicts, winners and comments</a></section>'+''.join(cards)+'</main><script id="benchmark-data" type="application/json">'+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script><script src="review.js"></script></body></html>'
    page=page.replace('Export new reviews','Export all reviews')
    page=page.replace('<section class="shared"><b>Shared criteria from your review</b>','<section class="shared"><details><summary>Shared criteria from your review</summary>')
    page=page.replace('Saved previous verdicts, winners and comments</a></section>','Saved previous verdicts, winners and comments</a></details></section>')
    tmp=OUT/'review.html.tmp';tmp.write_text(page);tmp.replace(OUT/'review.html')
    (OUT/'review.js').write_text((ROOT/'review.js').read_text())
    (OUT/'review-seed.json').write_text(json.dumps({'version':2,'runs':payload['runs'],'models':payload['models'],'display_corrections':list(corrections.values()),'decisions':[
        dict(row,model=old_slug) for row in feedback['decisions']]+[dict(row) for row in qwen_feedback['decisions']]+[dict(row) for row in reprompt_feedback['decisions']]+[
        {'id':t['id'],'model':slug,'verdict':'pending','preference':'not_rated','notes':''}
        for t in configs['qwen21']['tasks'] for slug in MODELS if slug not in ['qwen21','qwen21_reprompt']]},indent=2)+'\n')

if __name__=='__main__': print(json.dumps(sync(),indent=2))
