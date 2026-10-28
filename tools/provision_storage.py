#!/usr/bin/env python3
"""Create a versioned bucket and model cache in an existing Nebius project.

Default: print a plan. --apply creates chargeable storage. Credentials and IAM
permissions are configured separately; see docs/nebius.md.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-id', required=True)
    ap.add_argument('--name', required=True, help='Globally unique bucket name; also used for the filesystem')
    ap.add_argument('--cache-gib', type=int, default=200)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--cli', default='nebius')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch('[a-z0-9][a-z0-9-]{1,61}[a-z0-9]', args.name) or args.cache_gib <= 0:
        ap.error('Use a valid bucket name and positive cache size')
    resources = [('bucket', ['storage', 'bucket'], ['--versioning-policy', 'enabled']), ('filesystem', ['compute', 'filesystem'], ['--type', 'network_ssd', '--size-gibibytes', str(args.cache_gib)])]
    state = json.loads(args.state.read_text()) if args.state.exists() else {'project_id': args.project_id, 'name': args.name, 'cache_gib': args.cache_gib}
    if any(state.get(k) != v for k, v in [('project_id', args.project_id), ('name', args.name), ('cache_gib', args.cache_gib)]):
        ap.error('State belongs to different resource settings; use a new state file')
    for key, service, options in resources:
        command = [args.cli, *service, 'create', '--parent-id', args.project_id, '--name', args.name, *options, '--format', 'json', '--no-check-update']
        if not args.apply:
            print(shlex.join(command))
            continue
        if key in state:
            subprocess.run([args.cli, *service, 'get', state[key]['metadata']['id'], '--format', 'json', '--no-check-update'], check=True, capture_output=True)
            print('Verified existing', key)
            continue
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        state[key] = json.loads(result.stdout)
        if 'metadata' not in state[key]:
            raise RuntimeError('Unexpected create response; inspect the cloud before retrying')
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.state)
        print('Created', key, state[key]['metadata']['id'])
    if not args.apply:
        print('Plan only. Add --apply to create storage. No resources have been allocated.')


if __name__ == '__main__':
    main()
