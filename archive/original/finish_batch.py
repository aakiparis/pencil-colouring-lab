"""Watch this finite batch, download each published result and keep the review current."""
import json,time,subprocess
from pathlib import Path
from results import sync_results
from provision import ROOT
job=json.loads((ROOT/'batch-job.json').read_text())['metadata']['id']
last=None
while True:
 p=subprocess.run(['nebius','ai','job','get',job,'--format','json','--no-check-update'],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);(ROOT/'batch-job-status.json').write_text(json.dumps(d,indent=2)+'\n')
 state=d['status']['state']
 summary=sync_results(quiet=True)
 event={'job_id':job,'state':state,'completed':summary['completed'],'failed':summary['failed'],'pending':summary['pending']}
 if event!=last:print('PROGRESS '+json.dumps(event),flush=True);last=event
 workflow=json.loads((ROOT/'workflow-state.json').read_text());workflow.update(status='batch_'+state.lower(),batch_job_id=job,completed_results=summary['completed'],failed_results=summary['failed']);(ROOT/'workflow-state.json').write_text(json.dumps(workflow,indent=2)+'\n')
 if state in ['COMPLETED','FAILED','CANCELLED']:
  if state=='COMPLETED' and summary['completed']==summary['total_tasks'] and not summary['failed']:
   workflow.update(status='baseline_downloaded_ready_for_review',review_file='/Users/kiparis/Downloads/Pencil colouring dataset/baseline-v1/review.html')
   (ROOT/'workflow-state.json').write_text(json.dumps(workflow,indent=2)+'\n');print('READY_FOR_REVIEW',flush=True);break
  raise RuntimeError('Batch ended without all requested outputs: '+json.dumps(event))
 time.sleep(30)
