const data=JSON.parse(document.querySelector('#benchmark-data').textContent);
const cards=[...document.querySelectorAll('article')], key='pencil-benchmark-v2-'+data.runs.qwen21.slice(0,12)+'-'+data.runs.flux3.slice(0,12);
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(_){}
const previousSlug=data.previous_model_slug, previousKey='pencil-baseline-review-'+data.previous_feedback.run_sha256.slice(0,16);
let previousSaved={};try{previousSaved=JSON.parse(localStorage.getItem(previousKey)||'{}')}catch(_){}
for(const card of cards) for(const column of card.querySelectorAll('[data-model]')){
  const id=card.dataset.id, model=column.dataset.model, k=model+'/'+id, row=saved[k]||{};
  for(const field of ['verdict','preference','notes']){
    const el=column.querySelector('.'+field);
    if(model===previousSlug){const old=previousSaved[id]||{},changed=old.fields?old.fields[field]:field==='notes'?Boolean(old.notes):old[field]&&!['pending','not_rated'].includes(old[field]);if(changed)el.value=old[field];}
    if(row[field]!==undefined){
      if(model==='qwen21'&&field==='notes'&&data.qwen_feedback&&!row.feedback_applied&&!row.notes.includes(data.qwen_feedback.notes))el.value=data.qwen_feedback.notes+(row.notes?'\n\nEarlier review: '+row.notes:'');
      else if(model==='qwen21_reprompt'&&field==='notes'&&data.reprompt_feedback&&!row.feedback_applied&&!row.notes.includes(data.reprompt_feedback.notes))el.value=data.reprompt_feedback.notes+(row.notes?'\n\nEarlier review: '+row.notes:'');
      else el.value=row[field];
    }
    el.addEventListener(field==='notes'?'input':'change',()=>{
      saved[k]={...(saved[k]||{}),id,model,[field]:el.value,...(['qwen21','qwen21_reprompt'].includes(model)&&field==='notes'?{feedback_applied:true}:{})};
      try{localStorage.setItem(key,JSON.stringify(saved));
       if(model===previousSlug){previousSaved[id]={...(previousSaved[id]||{}),[field]:el.value,fields:{...(previousSaved[id]?.fields||{}),[field]:true}};localStorage.setItem(previousKey,JSON.stringify(previousSaved));}
      }catch(_){document.querySelector('#progress').textContent='Export your reviews: browser storage is unavailable.'}
    });
  }
}
document.querySelector('#show-old').onchange=e=>document.body.classList.toggle('hide-old',!e.target.checked);
function update(progress){
  let text=[];
  for(const model of Object.keys(data.models)){
    const state=progress[model];if(!state||state.run_sha256!==data.runs[model])continue;
    text.push(data.models[model].id.split('/').pop()+(model==='qwen21_reprompt'?' · reprompt':'')+': '+state.completed+'/'+state.total);
    for(const card of cards){
      const column=card.querySelector('[data-model="'+model+'"]'), result=state.results[card.dataset.id],ready=result?.status==='completed';
      for(const el of column.querySelectorAll('select,textarea'))el.disabled=!ready;
      if(ready&&column.querySelector('img')?.dataset.sha!==result.output_sha256){
        const a=document.createElement('a');a.href=model+'/outputs/'+card.dataset.id+'.png?v='+result.output_sha256.slice(0,12);a.target='_blank';
        const img=document.createElement('img');img.src=a.href;img.dataset.sha=result.output_sha256;img.loading='lazy';img.alt=data.models[model].id;a.append(img);
        column.querySelector('.result-image').replaceChildren(a);
      }
      if(result?.expanded_prompt)card.querySelector('[data-expanded="'+model+'"]').textContent='FLUX expanded caption:\n'+result.expanded_prompt;
      if(result?.status==='failed')column.querySelector('.result-image').textContent='Generation failed: '+result.error;
    }
  }
  document.querySelector('#progress').textContent=text.join(' · ');
}
update(data.progress);
const pipelineStatus=document.createElement('p');pipelineStatus.id='pipeline-status';document.querySelector('header').append(pipelineStatus);
async function refresh(){try{
 const r=await fetch('progress.json',{cache:'no-store'});if(r.ok)update(await r.json());
 const p=await fetch('pipeline-state.json',{cache:'no-store'});if(p.ok){const state=await p.json();pipelineStatus.textContent='Pipeline: '+state.model+' · '+state.stage+' · '+state.state;}
}catch(_){}}
refresh();setInterval(refresh,20000);
document.querySelector('#export').onclick=()=>{
 const decisions=[];
 for(const card of cards)for(const col of card.querySelectorAll('[data-model]'))decisions.push({id:card.dataset.id,model:col.dataset.model,
  verdict:col.querySelector('.verdict').value,preference:col.querySelector('.preference').value,notes:col.querySelector('.notes').value});
 const previous_feedback={...data.previous_feedback,decisions:data.previous_feedback.decisions.map(row=>{
  const edited=decisions.find(d=>d.model===previousSlug&&d.id===row.id);return {...row,...edited,
   winner:{reference_better:'ChatGPT',baseline_better:data.models[previousSlug].id,similar:'tie'}[edited.preference]||null};})};
 const payload={version:2,runs:data.runs,models:data.models,previous_feedback,qwen_feedback:data.qwen_feedback,reprompt_feedback:data.reprompt_feedback,selection:data.selection,selected_model:data.selected_model,display_corrections:data.display_corrections,decisions};
 const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'})),a=document.createElement('a');
 a.href=url;a.download='pencil-benchmark-v2-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.querySelector('#import').onchange=async e=>{
 try{
  const parsed=JSON.parse(await e.target.files[0].text());
  if(!parsed.runs||parsed.runs.qwen21!==data.runs.qwen21||parsed.runs.flux3!==data.runs.flux3||!Object.entries(parsed.runs).every(([m,h])=>data.runs[m]===h)||!Array.isArray(parsed.decisions))throw Error('Review belongs to another experiment');
  const known=new Set(cards.map(c=>c.dataset.id)),seen=new Set();
  for(const r of parsed.decisions){const k=r.model+'/'+r.id;
   if(!known.has(r.id)||!data.models[r.model]||seen.has(k)||!['pending','good_baseline','needs_improvement','reject'].includes(r.verdict)||!['not_rated','reference_better','similar','baseline_better'].includes(r.preference)||typeof r.notes!=='string')throw Error('Invalid review record');
   seen.add(k);
  }
  for(const r of parsed.decisions){saved[r.model+'/'+r.id]=r;const col=cards.find(c=>c.dataset.id===r.id).querySelector('[data-model="'+r.model+'"]');
   for(const field of ['verdict','preference','notes'])col.querySelector('.'+field).value=r[field];
   if(r.model===previousSlug)previousSaved[r.id]={...r,fields:{verdict:true,preference:true,notes:true}};}
  localStorage.setItem(key,JSON.stringify(saved));
  localStorage.setItem(previousKey,JSON.stringify(previousSaved));
 }catch(error){alert(error.message)}
};
