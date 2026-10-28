"""Provision isolated Nebius resources; never print credentials."""
import json,subprocess,os
from pathlib import Path
ROOT=Path(__file__).parent
PROJECT=json.loads((ROOT/'project.json').read_text())['metadata']['id']
TENANT='tenant-e00bd4z8cgv8v019z8'
def cli(args):
 p=subprocess.run(['nebius',*args,'--format','json','--no-check-update'],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return json.loads(p.stdout)
def ensure(name,args,private=False):
 file=ROOT/(name+'.json')
 if file.exists():return json.loads(file.read_text())
 d=cli(args);file.write_text(json.dumps(d,indent=2));file.chmod(0o600 if private else 0o644)
 print(name,d.get('metadata',{}).get('id','saved'),flush=True);return d
if __name__=='__main__':
 bucket=ensure('bucket',['storage','bucket','create','--parent-id',PROJECT,'--name','kiparis-pencil-baseline-uks2-20261017','--versioning-policy','enabled'])
 sa=ensure('service-account',['iam','service-account','create','--parent-id',PROJECT,'--name','pencil-baseline-storage'])
 group=ensure('storage-group',['iam','group','create','--parent-id',TENANT,'--name','pencil-baseline-storage-20261017'])
 ensure('storage-membership',['iam','group-membership','create','--parent-id',group['metadata']['id'],'--member-id',sa['metadata']['id']])
 ensure('storage-permit',['iam','access-permit','create','--parent-id',group['metadata']['id'],'--resource-id',bucket['metadata']['id'],'--role','storage.editor'])
 ensure('access-key',['iam','v2','access-key','create','--parent-id',PROJECT,'--name','pencil-baseline-s3','--account-service-account-id',sa['metadata']['id'],'--secret-delivery-mode','inline'],True)
 ensure('filesystem',['compute','filesystem','create','--parent-id',PROJECT,'--name','pencil-baseline-model-cache','--type','network_ssd','--size-gibibytes','200'])
