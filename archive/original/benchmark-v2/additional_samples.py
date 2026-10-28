"""Render user-requested new samples without changing historical benchmark runs or reviews."""
import hashlib,html,json
from pathlib import Path
ROOT=Path(__file__).parent;DATA=Path('/Users/kiparis/Downloads/Pencil colouring dataset');OUT=DATA/'benchmark-v2'
def render():
 p=ROOT/'additional-samples.json'
 if not p.exists():return '',None
 m=json.loads(p.read_text());e=lambda x:html.escape(str(x),quote=True)
 (OUT/'additional-samples.json').write_text(json.dumps(m,indent=2)+'\n');(OUT/'additional-samples.js').write_bytes((ROOT/'additional-samples.js').read_bytes())
 h='<section id="additional-samples" class="shared"><h2>'+e(m['section_title'])+'</h2><p>New samples, excluded from training. No ChatGPT images have been supplied.</p><p>'+e(m['comparison_note'])+'</p><p id="additional-status">'+e(m['generation_status'])+'</p>'
 labels={'qwen21_lora200':'Qwen-Image-2.1 LoRA · update 200 · strength 1','qwen21_lora200_detailed':'Qwen-Image-2.1 LoRA · update 200 · strength 1 · detailed background v4 · seed 123'}
 for r in m['rows']:
  h+='<div class="sample-card" id="additional-'+e(r['id'])+'" data-sample-id="'+e(r['id'])+'" style="background:white;padding:16px;margin:16px 0;border-radius:8px"><h2>'+e(r['title'])+'</h2><div class="comparison"><div class="always-visible"><h3>Original</h3><a target="_blank" href="../'+e(r['input'])+'"><img loading="lazy" src="../'+e(r['input'])+'"></a></div><div class="always-visible" data-reference="chatgpt"><h3>ChatGPT</h3><p class="placeholder">No image supplied</p></div>'
  for slug,v in r['variants'].items():
   h+='<div data-model="'+slug+'"><h3>'+e(labels[slug])+'</h3><div class="result-image">'
   if v['status']=='completed':h+='<a target="_blank" href="../'+e(v['image'])+'"><img loading="lazy" src="../'+e(v['image'])+'"></a>'
   else:h+='<p class="placeholder">Awaiting generation</p>'
   h+='</div><small>Seed '+e(v['seed'])+' · raw model output · no greyscale conversion</small><details class="prompt-used"><summary>Exact prompt and provenance</summary><pre>'+e(v['prompt'])+'</pre><p>LoRA checkpoint 200 · strength 1 · input SHA-256 '+e(r['input_sha256'])+'</p><a class="sample-metadata" target="_blank" href="../'+e(v['metadata'])+'"'+(' hidden' if v['status']!='completed' else '')+'>Generation metadata</a></details><label>Verdict<select class="verdict"><option value="pending">Pending</option><option value="good_baseline">Good baseline</option><option value="needs_improvement">Needs improvement</option><option value="reject">Reject</option></select></label><label>Review comments<textarea class="notes" rows="6" placeholder="Review comments"></textarea></label></div>'
  h+='</div></div>'
 return h+'</section>',m
