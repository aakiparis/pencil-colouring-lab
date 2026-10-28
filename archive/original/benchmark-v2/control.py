"""Advance finite warm/smoke/batch jobs, download results, and stop on failure."""
import argparse
import json
import time
from pathlib import Path
from launch import ROOT, call, launch
from compare import sync, OUT

TERMINAL={'COMPLETED','FAILED','CANCELLED','DEADLINE_EXCEEDED'}

def status(slug,mode):
    path=ROOT/slug/(mode+'-job.json')
    if not path.exists():return None
    ident=json.loads(path.read_text())['metadata']['id']
    job=call(['ai','job','get',ident])
    (ROOT/slug/(mode+'-status.json')).write_text(json.dumps(job,indent=2)+'\n')
    return job['status']['state']

def wait(slug,mode):
    previous=None
    while True:
        state=status(slug,mode)
        result=sync()
        event={'model':slug,'stage':mode,'state':state,'results':result}
        (ROOT/'workflow-state.json').write_text(json.dumps(event,indent=2)+'\n')
        (OUT/'pipeline-state.json').write_text(json.dumps(event,indent=2)+'\n')
        if event!=previous:print(json.dumps(event),flush=True);previous=event
        if state=='COMPLETED':return
        if state in TERMINAL:raise RuntimeError(f'{slug} {mode} ended with {state}; inspect saved job status and logs before retrying.')
        time.sleep(30)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','batch']);ap.add_argument('--model',choices=['qwen21','flux3','qwen21_reprompt','qwen21_decodecheck','qwen21_background','all'],default='all');a=ap.parse_args()
    models=['qwen21','flux3'] if a.model=='all' else [a.model]
    for slug in models:
        if a.stage=='smoke':
            if not (ROOT/slug/'warm-job.json').exists():launch(slug,'warm')
            wait(slug,'warm')
            launch(slug,'smoke',1);wait(slug,'smoke')
        else:
            if status(slug,'smoke')!='COMPLETED':raise RuntimeError('A successful smoke job is required before the batch.')
            launch(slug,'batch',8);wait(slug,'batch')
    print('STAGE_COMPLETE',a.stage,flush=True)

if __name__=='__main__':main()
