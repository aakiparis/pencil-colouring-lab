#!/usr/bin/env python3
"""Download a completed pilot adapter and verify its recorded hash."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--cloud', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    training = args.workspace / 'training'
    cfg = lab.read(training / 'pilot.json')
    identity = lab.module(training / 'run_identity.py', 'adapter_identity')
    key = identity.compute_run_key(training)
    prefix = cfg['prefix'] + '/' + key[:16] + '/pilot/'
    cloud = lab.cloud_config(args.cloud)
    s = lab.storage(cloud)
    report = json.loads(s.get_object(Bucket=cloud['bucket'], Key=prefix + 'completion.json')['Body'].read())
    if report.get('state') != 'complete' or report.get('run_key') != key or report.get('mode') != 'pilot' or report.get('last_logged_update') != cfg['training']['max_train_steps']:
        raise ValueError('No matching completed pilot')
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing adapter')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.download')
    try:
        s.download_file(cloud['bucket'], prefix + 'pytorch_lora_weights.safetensors', str(temporary))
        if lab.sha(temporary) != report['adapter_sha256']:
            raise ValueError('Adapter checksum mismatch')
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print('Verified adapter:', args.output)
    print('Object:', prefix + 'pytorch_lora_weights.safetensors')


if __name__ == '__main__':
    main()
