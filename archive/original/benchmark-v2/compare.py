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
MODELS=['qwen21','qwen21_reprompt','qwen21_background','flux3','qwen21_lora200','qwen21_lora200_detailed']

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
    labels={'qwen_edit_2511':'Qwen-Image-Edit-2511 · original','qwen21':'Qwen-Image-2.1 · v2','qwen21_reprompt':'Qwen-Image-2.1 · reprompt v3','qwen21_background':'Qwen-Image-2.1 · background v4','flux3':'FLUX 3 Image Klein Base','qwen21_lora200':'Qwen-Image-2.1 LoRA · update 200 · strength 1','qwen21_lora200_detailed':'Qwen-Image-2.1 LoRA · update 200 · strength 1 · detailed background v4'}
    old_slug='qwen_edit_2511'; order=[old_slug]+MODELS
    feedback=json.loads((DATA/'baseline-v1/review-decisions.json').read_text())
    qwen_feedback=json.loads((ROOT/'qwen21-feedback.json').read_text())
    reprompt_feedback=json.loads((ROOT/'qwen21-reprompt-feedback.json').read_text())
    lora_feedback=json.loads((ROOT/'qwen21-lora200-feedback.json').read_text()) if (ROOT/'qwen21-lora200-feedback.json').exists() else None
    selection=json.loads((ROOT/'selection.json').read_text())
    training_targets=json.loads((ROOT/'training-targets.json').read_text())
    target_map={row['id']:row for row in training_targets['targets']}
    defaults={old_slug:{r['id']:r for r in feedback['decisions']},'qwen21':{r['id']:r for r in qwen_feedback['decisions']},'qwen21_reprompt':{r['id']:r for r in reprompt_feedback['decisions']}}
    defaults['qwen21_background']={r['id']:{'id':r['id'],'verdict':'good_baseline','preference':'not_rated','notes':'','verdict_basis':'Explicit user selection as a training target.','training_target':r} for r in training_targets['targets'] if r['selected_variant']=='qwen21_background'}
    if lora_feedback:defaults['qwen21_lora200']={r['id']:r for r in lora_feedback['decisions']}
    old_cfg=json.loads((ROOT.parent/'run.json').read_text()); all_cfg={old_slug:old_cfg,**configs}
    task_maps={m:{t['id']:t for t in cfg['tasks']} for m,cfg in all_cfg.items()}
    correction_file=ROOT/'display-corrections.json'; corrections={r['id']:r for r in json.loads(correction_file.read_text())} if correction_file.exists() else {}
    thumbs=OUT/'thumbnails';thumbs.mkdir(exist_ok=True)
    def picture(path,label,link,thumb=True):
        src=link
        if thumb:
            dst=thumbs/(hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.jpg')
            if not dst.exists():
                with Image.open(path) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((600,600)); im.save(dst,quality=90)
            src='thumbnails/'+dst.name
        return '<a href="'+e(link)+'" target="_blank"><img loading="lazy" src="'+e(src)+'" alt="'+e(label)+'"></a>'
    def controls(name,row):
        row=row or {'verdict':'pending','preference':'not_rated','notes':''}
        def options(values,selected):return ''.join('<option value="'+v+'"'+(' selected' if v==selected else '')+'>'+e(label)+'</option>' for v,label in values)
        verdicts=[('pending','Pending'),('good_baseline','Good baseline'),('needs_improvement','Needs improvement'),('reject','Reject')]
        winners=[('not_rated','Not rated'),('reference_better','ChatGPT wins'),('similar','Similar'),('baseline_better',name+' wins')]
        return '<label>Verdict <select class="verdict">'+options(verdicts,row['verdict'])+'</select></label><label hidden>Winner <select class="preference">'+options(winners,row['preference'])+'</select></label><label>Review comments<textarea class="notes" rows="6" placeholder="Review comments">'+e(row['notes'])+'</textarea></label>'
    def prompt_details(slug,ident):
        cfg=all_cfg[slug];t=task_maps[slug][ident]
        h='<details class="prompt-used"><summary>Prompt used · '+e(t.get('prompt_version','original baseline'))+'</summary><pre>'+e(t['prompt'])+'</pre>'
        if cfg.get('negative_prompt'):h+='<b>Negative prompt</b><pre>'+e(cfg['negative_prompt'])+'</pre>'
        if slug in ['qwen21_lora200','qwen21_lora200_detailed']:
            adapter=cfg['additional_loras'][0]
            h+='<p>Trained LoRA · update '+e(adapter['checkpoint'])+' · strength '+e(adapter['scale'])+' · adapter SHA-256 '+e(adapter['sha256'])+'</p><p>Raw model output; no greyscale conversion. Dataset membership: '+e(t['dataset_split'])+'. Composite inputs were excluded from LoRA training.</p>'
        if slug=='flux3':
            result=state[slug]['results'].get(ident,{})
            h+='<b>Actual FLUX caption after official expansion</b><pre class="expanded" data-expanded="flux3">'+e(result.get('expanded_prompt','Awaiting generation'))+'</pre>'
        h+='<small>Model: '+e(cfg['model']['id'])+' · revision '+e(cfg['model']['revision'])+' · '+str(cfg['num_inference_steps'])+' steps · seed '+str(t.get('seed',42))+' · VAE tiling: '+('enabled' if cfg.get('vae_tiling',True) and slug!='flux3' else 'disabled' if slug!='flux3' else 'pipeline default')+'</small>'
        h+='<a href="'+('../baseline-v1/run.json' if slug==old_slug else slug+'/run.json')+'" target="_blank">Frozen run configuration</a></details>'
        return h
    cards=[]
    for task in configs['qwen21']['tasks']:
        ident=task['id']; correction=corrections.get(ident); inputs=correction['display_inputs'] if correction else task['inputs']
        target=target_map[ident]
        badge='<p class="training-target">Selected training target · user confirmed</p>'
        cols='<div class="always-visible"><h3>Original</h3><div class="inputs">'+''.join(picture(DATA/n,'Original input','../'+n) for n in inputs)+'</div>'
        if correction:cols+='<small>Corrected original · '+''.join('<a href="../'+e(n)+'" target="_blank">Previous generation input</a>' for n in correction['generation_inputs'])+'</small>'
        cols+='</div><div class="always-visible" data-reference="chatgpt"><h3>ChatGPT</h3>'+picture(DATA/task['reference'],'ChatGPT example','../'+task['reference'])+ (badge if target['selected_variant']=='chatgpt' else '')+'<small>Exact ChatGPT model unknown</small><details class="prompt-used"><summary>ChatGPT prompt provenance</summary><p>The exact internal image-generation prompt is unavailable. The historical chat and reconstructed instructions are preserved in the prompt catalogue; they are not an exact tool prompt.</p><a href="../prompt-review/review.html" target="_blank">Historical instructions and reconstructed prompts</a></details></div>'
        for slug in order:
            if slug==old_slug:
                im=picture(DATA/'baseline-v1/outputs'/(ident+'.png'),labels[slug],'../baseline-v1/outputs/'+ident+'.png')
            else:
                result=state[slug]['results'].get(ident);ready=result and result['status']=='completed'
                im=picture(OUT/slug/'outputs'/(ident+'.png'),labels[slug],slug+'/outputs/'+ident+'.png',False) if ready else '<p class="placeholder">'+('Generation failed' if result else 'Awaiting generation')+'</p>'
            caption=('Detailed background v4 prompt with the same LoRA · '+task_maps[slug][ident]['dataset_split']+' · raw output') if slug=='qwen21_lora200_detailed' else ('Primary candidate selected by the user · '+task_maps[slug][ident]['dataset_split']+' · raw output') if slug=='qwen21_lora200' else 'Accepted baseline · remaining background caveat' if slug==selection['selected_model'] else 'User-reviewed background experiment' if slug=='qwen21_background' else 'Historical experiment · no added LoRA'
            cols+='<div data-model="'+slug+'"><h3>'+e(labels[slug])+'</h3><div class="result-image">'+im+'</div>'+(badge if target['selected_variant']==slug else '')+'<small>'+e(caption)+'</small>'+prompt_details(slug,ident)+controls(labels[slug],defaults.get(slug,{}).get(ident))
            if target['selected_variant']==slug:cols+='<small>Good baseline: explicitly selected by the user as a training target. Earlier browser edits are retained.</small>'
            elif slug in defaults:cols+='<small>Saved shared or dictated user feedback. Any inferred verdict is attributed in the feedback file.</small>'
            if slug=='qwen21_lora200' and lora_feedback and ident in defaults.get(slug,{}):cols+='<small>Individual dictated review saved · <a href="qwen21-lora200-feedback.json">Feedback file</a></small>'
            cols+='</div>'
        warning='<p class="source-correction">Original corrected by the user. The old Qwen and FLUX runs use the previous original; Qwen reprompt v3 and background v4 use the corrected original. Pair 045 is not a prompt-only comparison against the earlier runs.</p>' if correction else ''
        cards.append('<article id="'+ident+'" data-id="'+ident+'" data-kind="'+task['kind']+'"><h2>'+ident+'</h2>'+warning+'<div class="comparison">'+cols+'</div></article>')
    from additional_samples import render as render_additional_samples
    additional_html,additional_manifest=render_additional_samples()
    hashes={m:state[m]['run_sha256'] for m in MODELS}; hashes[old_slug]=feedback['run_sha256']
    payload={'version':2,'runs':hashes,'models':{m:all_cfg[m]['model'] for m in order},'labels':labels,'variant_order':order,'progress':state,'previous_feedback':feedback,'previous_model_slug':old_slug,'qwen_feedback':qwen_feedback,'reprompt_feedback':reprompt_feedback,'selected_model':selection['selected_model'],'selection':selection,'display_corrections':list(corrections.values()),'prompt_policy':json.loads((ROOT/'prompt-policy-v4.json').read_text())}
    payload['additional_samples']=additional_manifest
    payload['lora_feedback']=lora_feedback
    if lora_feedback:(OUT/'qwen21-lora200-feedback.json').write_text(json.dumps(lora_feedback,indent=2)+'\n')
    payload['training_targets']=training_targets
    payload['lora_candidate_selection']=json.loads((ROOT/'lora-candidate-selection.json').read_text())
    (OUT/'lora-candidate-selection.json').write_text(json.dumps(payload['lora_candidate_selection'],indent=2)+'\n')
    toggles=''.join('<label class="variant-toggle"><input type="checkbox" data-variant="'+m+'" checked> '+e(labels[m])+'</label>' for m in order)
    css='''*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#f3f5f7;color:#17212b;font:15px/1.5 system-ui,sans-serif}header{background:#142c3e;color:white;padding:20px 24px}h1{font-size:24px;margin:0}header p{margin:6px 0}.top-link{color:white}nav{position:sticky;top:0;background:white;padding:12px 24px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2;border-bottom:1px solid #ccd4dc}.variant-toggle{display:inline-flex;align-items:center;gap:6px;margin:0;font-size:13px}main{max-width:2200px;margin:auto;padding:12px 18px}article{background:white;padding:16px;border-radius:10px;margin-bottom:20px;border:1px solid #dbe3e9}h2{font-size:19px;margin:0 0 12px}.comparison{display:grid;grid-template-columns:repeat(7,minmax(225px,1fr));gap:12px;overflow-x:auto;align-items:start;padding-bottom:8px}h3{font-size:14px;text-align:center;margin:0 0 8px;min-height:42px}img,.placeholder{width:100%;height:300px;object-fit:contain;background:#fafafa;border:1px solid #edf0f3}.placeholder{display:flex;align-items:center;justify-content:center;color:#677784;margin:0}.inputs{display:flex;gap:4px}.inputs a{min-width:0;flex:1}small{display:block;color:#657382;font-size:12px}label{display:block;margin-top:8px}select,textarea,button{font:inherit;border:1px solid #b8c5d0;border-radius:6px;padding:7px;background:white}select{width:100%}textarea{width:100%;min-height:150px;margin-top:8px;resize:vertical;line-height:1.45}button{cursor:pointer;background:#185473;color:white}details{margin-top:10px;font-size:13px}summary{cursor:pointer}.prompt-used pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:250px;overflow:auto;font:12px/1.45 system-ui,sans-serif;background:#f4f7f9;padding:8px}.shared{background:#e8f0f6;padding:12px;border-radius:8px;margin-bottom:16px}.source-correction{background:#fff3d9;padding:9px;border-radius:6px;color:#704d0d}#progress{font-size:12px;color:#536578;flex-basis:100%}@media(max-width:900px){main{padding:8px}img,.placeholder{height:250px}nav{padding:8px;gap:8px}}'''
    css+=' .training-target{background:#dff1e6;border:1px solid #8bb49b;color:#174a2b;padding:8px;border-radius:6px;font-size:13px;font-weight:650}'
    title='Pencil colouring · model and prompt comparison'
    page='<!doctype html><html lang="en-GB"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style><body><header><h1>'+title+'</h1><p>Original and ChatGPT are always shown. Use the checkboxes to compare any model variants.</p><p>Accepted baseline: Qwen-Image-2.1 · reprompt v3. New experiment: background v4, with the same model and settings.</p><p>Prompts, model revisions and sampling settings are recorded beneath each result. Reviews are saved in this browser; export them to keep a portable copy.</p><a class="top-link" href="../baseline-v1/review.html">Previous review archive</a></header><nav>'+toggles+'<button id="export">Export all reviews</button><label>Import <input id="import" type="file" accept="application/json"></label><span id="progress"></span></nav><main><section class="shared"><details><summary>Saved feedback and experiment selection</summary><p>'+e(reprompt_feedback['notes'])+'</p><a href="qwen21-reprompt-feedback.json">Accepted baseline feedback</a> · <a href="selection.json">Selected run</a> · <a href="../baseline-v1/review-decisions.json">Historical reviews</a></details></section>'+''.join(cards)+additional_html+'</main><script id="benchmark-data" type="application/json">'+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script><script src="review.js"></script><script src="additional-samples.js"></script></body></html>'
    page=page.replace('<p>Accepted baseline: Qwen-Image-2.1 · reprompt v3. New experiment: background v4, with the same model and settings.</p>','<p>Training targets confirmed: 44 ChatGPT images and 12 Qwen-Image-2.1 background v4 images. Each selected image has a green badge. Historical reviews remain available.</p><p><a class="top-link" href="training-targets.json">Saved training target selections and provenance</a></p>')
    page=page.replace('</header>','<p>Selected LoRA candidate: update 200, strength 1. Backup: update 100, strength 1. Both LoRA columns contain raw generated images, including any residual colour. The detailed variant reuses the exact background v4 prompts, enabling a same-prompt comparison with and without LoRA. Training and evaluation membership are labelled; this full-set review is not a blind test. <a class="top-link" href="lora-candidate-selection.json">Saved candidate selection</a></p></header>')
    if additional_manifest:page=page.replace('</header>','<p><a class="top-link" href="#additional-samples">Jump to new Spider-Man and Spider-Verse samples</a></p></header>')
    page=page.replace('src="review.js"','src="review.js?v='+digest(ROOT/'review.js')[:12]+'"').replace('src="additional-samples.js"','src="additional-samples.js?v='+digest(ROOT/'additional-samples.js')[:12]+'"')
    temp=OUT/'review.html.tmp';temp.write_text(page);temp.replace(OUT/'review.html');(OUT/'review.js').write_text((ROOT/'review.js').read_text())
    decisions=[]
    for task in configs['qwen21']['tasks']:
        for slug in order:decisions.append(dict(defaults.get(slug,{}).get(task['id'],{'id':task['id'],'verdict':'pending','preference':'not_rated','notes':''}),model=slug))
    (OUT/'review-seed.json').write_text(json.dumps({'version':2,'runs':hashes,'models':payload['models'],'training_targets':training_targets,'lora_feedback':lora_feedback,'display_corrections':list(corrections.values()),'decisions':decisions},indent=2)+'\n')

if __name__=='__main__': print(json.dumps(sync(),indent=2))
