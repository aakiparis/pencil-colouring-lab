"""Private bucket transfers and job secret setup. Credentials never enter logs."""
import json,os,subprocess
from pathlib import Path
import boto3
from botocore.config import Config
ROOT=Path(__file__).parent

def client():
 key=json.loads((ROOT/'access-key.json').read_text())['status']
 bucket=json.loads((ROOT/'bucket.json').read_text())
 return boto3.client('s3',endpoint_url='https://'+bucket['status']['domain_name'],region_name='uk-south2',aws_access_key_id=key['aws_access_key_id'],aws_secret_access_key=key['secret'],config=Config(signature_version='s3v4',request_checksum_calculation='when_required',response_checksum_validation='when_required')),bucket['metadata']['name']

def setup():
 from provision import cli,PROJECT
 key=json.loads((ROOT/'access-key.json').read_text())['status']
 file=ROOT/'runtime-secret.json'
 if not file.exists():
  payload=[{'key':'AWS_ACCESS_KEY_ID','string_value':key['aws_access_key_id']},{'key':'AWS_SECRET_ACCESS_KEY','string_value':key['secret']}]
  secret=cli(['mysterybox','secret','create','--parent-id',PROJECT,'--name','pencil-baseline-s3','--secret-version-payload',json.dumps(payload),'--secret-version-set-primary'])
  file.write_text(json.dumps(secret,indent=2));file.chmod(0o600)
  print('Job storage secret created')
 s,b=client()
 for name in ['dataset.tar.gz','dataset-checksums.json','run.json']:
  s.upload_file(str(ROOT/name),b,'baseline-v1/'+name)
  head=s.head_object(Bucket=b,Key='baseline-v1/'+name)
  assert head['ContentLength']==(ROOT/name).stat().st_size
  print('Uploaded and verified',name,head['ContentLength'],flush=True)
 if __name__=='__main__': pass
if __name__=='__main__':setup()
