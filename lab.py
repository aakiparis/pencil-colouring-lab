#!/usr/bin/env python3
"""Portable entry points around the preserved experimental workers. Plans are local."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'archive/original'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def local_file(root, relative):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Expected a dataset-relative path: ' + relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing file or path outside dataset: ' + relative)
    return path


def verify_archive():
    entries = read(ROOT / 'archive/source-inventory.json')
    for row in entries:
        if sha(ARCHIVE / row['path']) != row['sha256']:
            raise ValueError('Archive changed: ' + row['path'])
    print('Verified original source files:', len(entries))


def validate_pairs(manifest, data_root, training=False):
    rows = manifest['pairs']
    if manifest.get('schema_version') != 1 or not rows:
        raise ValueError('Expected schema_version 1 and non-empty pairs')
    seen_ids, groups, identities, hashes = set(), {}, {}, {}
    for row in rows:
        name = row['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in seen_ids:
            raise ValueError('Unsafe or duplicate pair ID: ' + name)
        seen_ids.add(name)
        if not 1 <= len(row['inputs']) <= 2:
            raise ValueError('Inference supports one or two input images')
        paths = [*row['inputs']] + ([row['target']] if row.get('target') else [])
        for path in paths:
            local_file(data_root, path)
        if not training or row.get('train_eligible', True) is False:
            continue
        if len(row['inputs']) != 1 or not row.get('target'):
            raise ValueError('Training requires one input and one approved target')
        split = row.get('split')
        if split not in ('train', 'validation', 'test') or not row.get('group_id'):
            raise ValueError('Training requires an explicit split and source group')
        for key, index in [(row['group_id'], groups)] + [(x, identities) for x in row.get('identities', [])] + [(sha(local_file(data_root, x)), hashes) for x in paths]:
            if key in index and index[key] != split:
                raise ValueError('Split leakage: ' + name)
            index[key] = split
    if training and {r.get('split') for r in rows if r.get('train_eligible', True)} != {'train', 'validation', 'test'}:
        raise ValueError('Training requires non-empty train, validation and test splits')
    return rows


def prepare(args):
    manifest = read(args.manifest)
    rows = validate_pairs(manifest, args.data_root, args.freeze)
    if args.output.exists():
        raise ValueError('Use a new output directory; prepared runs are immutable')
    args.output.mkdir(parents=True)
    out = args.output.resolve()
    asset_root = out / 'assets'
    asset_root.mkdir()
    checks, selection, tasks = [], [], []
    policy = read(ROOT / 'recipes/historical/benchmark-v2/prompt-policy-v4.json')
    for row in rows:
        inputs = []
        target = None
        for field, names in [('input', row['inputs']), ('target', [row['target']] if row.get('target') else [])]:
            for i, name in enumerate(names):
                source = local_file(args.data_root, name)
                relative = f"pairs/{row['id']}/{field}_{i}{source.suffix.lower()}"
                path = asset_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
                item = {'path': relative, 'sha256': sha(path)}
                checks.append(item)
                if field == 'input':
                    inputs.append(item)
                else:
                    target = item
        composite = len(inputs) == 2
        prompt = (policy['composite_policy'] if composite else policy['source_fidelity'] + policy['background_policy']) + policy['shared_style']
        if args.short_prompt:
            prompt = (ROOT / 'prompts/short.txt').read_text().strip()
        tasks.append({'id': row['id'], 'inputs': [x['path'] for x in inputs], 'reference': target['path'] if target else None, 'kind': 'multi_input' if composite else 'single_input', 'prompt_version': 'short' if args.short_prompt else 'background-v4', 'source_group': row.get('group_id', row['id']), 'prompt': prompt, 'seed': args.seed})
        if row.get('train_eligible', True) and target and not composite:
            selection.append({'id': row['id'], 'inputs': inputs, 'selected_target': target, 'group_id': row.get('group_id'), 'split': row.get('split'), 'identities_inferred_from_filenames': row.get('identities', []), 'grouping_confirmed': args.freeze, 'geometry_review': 'approved' if args.freeze else 'pending', 'target_approval': 'approved' if args.freeze else 'pending', 'geometry_normalisation': 'white_letterbox_to_source_aspect'})
    checks.sort(key=lambda x: x['path'])
    write(out / 'dataset-checksums.json', checks)
    # Stable member order, ownership, modes and timestamps; no original filenames in gzip header.
    with (out / 'dataset.tar.gz').open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for item in checks:
            path = asset_root / item['path']
            info = tar.gettarinfo(str(path), arcname=item['path'])
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            info.mode = 0o644
            with path.open('rb') as f:
                tar.addfile(info, f)
    dataset_hash = sha(out / 'dataset.tar.gz')
    original = read(ROOT / 'recipes/historical/benchmark-v2/qwen21_lora200_detailed/run.json')
    keys = ['model', 'pipeline', 'dtype', 'num_inference_steps', 'target_pixels', 'composite_size', 'software', 'true_cfg_scale', 'vae_tiling', 'vae_slicing']
    cfg = {k: original[k] for k in keys}
    cfg.update(tasks=tasks, smoke_ids=[tasks[0]['id']], reuse_ready_model=False, input_sha256={x['path']: x['sha256'] for x in checks}, additional_loras=[], dataset_sha256=dataset_hash)
    if args.adapter:
        adapter_hash = sha(args.adapter)
        shutil.copyfile(args.adapter, out / 'pytorch_lora_weights.safetensors')
        cfg['additional_loras'] = [{'checkpoint': args.checkpoint, 'scale': args.strength, 'sha256': adapter_hash, 'object_key': 'pencil-lab/adapters/' + adapter_hash + '.safetensors'}]
    cfg['runtime_sources'] = {name: sha(path) for name, path in {'worker': ARCHIVE / 'benchmark-v2/qwen21_lora200_detailed/worker.py', 'bootstrap': ROOT / 'runtime/inference_bootstrap.sh', 'requirements': ARCHIVE / 'lora-v1/requirements.txt', 'preparer': ROOT / 'lab.py'}.items()}
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg.update(run_id='pencil-' + fingerprint[:16], prefix='pencil-lab/runs/' + fingerprint)
    write(out / 'run.json', cfg)
    # Independent cache per data/configuration: no historical cache is reused or overwritten.
    cache = '/cache/pencil-lab/' + fingerprint
    worker = (ARCHIVE / 'benchmark-v2/qwen21_lora200_detailed/worker.py').read_text()
    worker = worker.replace('/cache/benchmark-v2', cache).replace('/cache/dataset', cache + '/dataset')
    worker = worker.replace("'baseline-v1/dataset.tar.gz'", "cfg['prefix'] + '/dataset.tar.gz'").replace("'baseline-v1/dataset-checksums.json'", "cfg['prefix'] + '/dataset-checksums.json'")
    (out / 'worker.py').write_text(worker)
    shutil.copyfile(ROOT / 'runtime/inference_bootstrap.sh', out / 'bootstrap.sh')
    shutil.copyfile(ARCHIVE / 'lora-v1/requirements.txt', out / 'requirements.txt')
    write(out / 'selection.json', {'dataset_root': str(asset_root), 'group_review_approved': args.freeze, 'target_selection_approved': args.freeze, 'instruction': 'Create a pencil colouring page.', 'rows': selection})
    if args.freeze:
        module(ARCHIVE / 'lora-v1/freeze_dataset.py', 'freeze').freeze(out / 'selection.json', out / 'frozen-dataset')
    write(out / 'preparation.json', {'manifest_sha256': sha(args.manifest), 'dataset_sha256': dataset_hash, 'run_sha256': sha(out / 'run.json'), 'worker_sha256': sha(out / 'worker.py'), 'bootstrap_sha256': sha(out / 'bootstrap.sh'), 'requirements_sha256': sha(out / 'requirements.txt'), 'archive_inventory_sha256': sha(ROOT / 'archive/source-inventory.json'), 'training_review_confirmed': args.freeze})
    print('Prepared:', out)


def cloud_config(path):
    cfg = read(path)
    for key in ('project_id', 'bucket', 'filesystem_id', 'subnet_id', 'storage_secret_id'):
        if not cfg.get(key) or 'REPLACE_' in cfg[key]:
            raise ValueError('Set cloud configuration: ' + key)
    if cfg['region'] != 'uk-south2':
        raise ValueError('The preserved workers use uk-south2; port region-sensitive code before changing it')
    return cfg


def storage(cfg):
    import boto3
    from botocore.config import Config
    return boto3.client('s3', endpoint_url=cfg['endpoint'], region_name=cfg['region'], config=Config(signature_version='s3v4', request_checksum_calculation='when_required', response_checksum_validation='when_required'))


def infer(args):
    folder = args.workspace.resolve()
    cfg, cloud = read(folder / 'run.json'), cloud_config(args.cloud)
    record = folder / (args.mode + '-job.json')
    if args.submit and record.exists():
        raise ValueError('Job already recorded; inspect it before creating another run')
    for key, name in [('run_sha256', 'run.json'), ('worker_sha256', 'worker.py'), ('bootstrap_sha256', 'bootstrap.sh'), ('requirements_sha256', 'requirements.txt'), ('dataset_sha256', 'dataset.tar.gz')]:
        if sha(folder / name) != read(folder / 'preparation.json')[key]:
            raise ValueError('Prepared file changed: ' + name)
    command = [cloud.get('cli', 'nebius'), 'ai', 'job', 'create', '--parent-id', cloud['project_id'], '--name', cfg['run_id'] + '-' + args.mode, '--image', 'pytorch/pytorch@' + read(ROOT / 'recipes/historical/container.json')['digest'], '--container-command', '/bin/bash', '--args', '/app/benchmark_bootstrap.sh ' + args.mode, '--volume', cloud['filesystem_id'] + ':/cache:rw', '--subnet-id', cloud['subnet_id'], '--restart-policy', 'never', '--disk-size', '100Gi', '--timeout', '4h', '--async']
    env = {'S3_ENDPOINT': cloud['endpoint'], 'S3_BUCKET': cloud['bucket'], 'RUN_SHA256': sha(folder / 'run.json'), 'RUN_PREFIX': cfg['prefix'], 'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '8'}
    for key, value in env.items():
        command += ['--env', key + '=' + value]
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']:
        command += ['--env-secret', key + '=' + cloud['storage_secret_id']]
    if cloud.get('hf_secret_id'):
        command += ['--env-secret', 'HF_TOKEN=' + cloud['hf_secret_id']]
    for local, remote in [('worker.py', 'benchmark_worker.py'), ('bootstrap.sh', 'benchmark_bootstrap.sh'), ('requirements.txt', 'requirements.txt')]:
        command += ['--inject-file', str(folder / local) + ':/app/' + remote]
    if args.mode == 'warm':
        command += ['--platform', 'cpu-d3', '--preset', '8vcpu-32gb']
    else:
        command += ['--platform', cloud['platform'], '--preset', cloud['presets'][str(args.gpus)], '--preemptible', '--follows-spot-price']
    command += ['--format', 'json', '--no-check-update']
    if not args.submit:
        print(shlex.join(command))
        print('Plan only: no uploads or compute. Add --submit to execute.')
        return
    s = storage(cloud)
    for name in ['dataset.tar.gz', 'dataset-checksums.json', 'run.json']:
        s.upload_file(str(folder / name), cloud['bucket'], cfg['prefix'] + '/' + name)
    for adapter in cfg['additional_loras']:
        weight = folder / 'pytorch_lora_weights.safetensors'
        if sha(weight) != adapter['sha256']:
            raise ValueError('Adapter checksum mismatch')
        s.upload_file(str(weight), cloud['bucket'], adapter['object_key'])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    write(record, json.loads(result.stdout))
    print('Submitted; job response:', record)


def stage_training(args):
    folder, cloud = args.workspace.resolve(), cloud_config(args.cloud)
    if not (folder / 'frozen-dataset/manifest.json').exists():
        raise ValueError('Prepare with --freeze after reviewing targets, groups and geometry')
    destination = folder / 'training'
    if destination.exists():
        raise ValueError('Training workspace already exists')
    shutil.copytree(ARCHIVE / 'lora-v1', destination, ignore=shutil.ignore_patterns('evaluation', 'README.txt', 'make_plan.py'))
    shutil.copytree(folder / 'frozen-dataset', destination / 'frozen-dataset')
    shutil.copyfile(folder / 'selection.json', destination / 'selection.json')
    shutil.copyfile(ROOT / 'recipes/historical/lora-v1/source-provenance.json', destination / 'source-provenance.json')
    cfg = read(ROOT / 'recipes/historical/lora-v1/pilot.json')
    cfg['prefix'] = read(folder / 'run.json')['prefix'] + '/training'
    cfg['state'] = 'prepared_not_launched'
    write(destination / 'pilot.json', cfg)
    launch = (destination / 'launch.py').read_text().replace("CLI='/Users/kiparis/.nebius/bin/nebius'", 'CLI=' + repr(cloud.get('cli', 'nebius')))
    launch = launch.replace('https://storage.uk-south2.nebius.cloud', cloud['endpoint'])
    (destination / 'launch.py').write_text(launch)
    for name, resource in [('project', cloud['project_id']), ('filesystem', cloud['filesystem_id']), ('runtime-secret', cloud['storage_secret_id'])]:
        write(folder / (name + '.json'), {'metadata': {'id': resource}})
    write(folder / 'subnets.json', {'items': [{'metadata': {'id': cloud['subnet_id']}}]})
    shutil.copyfile(ROOT / 'recipes/historical/container.json', folder / 'container.json')
    write(folder / 'cloud.json', cloud)
    (folder / 'storage_ops.py').write_text("from pathlib import Path\nimport json\nimport boto3\nfrom botocore.config import Config\ndef client():\n c=json.loads((Path(__file__).parent/'cloud.json').read_text())\n return boto3.client('s3',endpoint_url=c['endpoint'],region_name=c['region'],config=Config(signature_version='s3v4',request_checksum_calculation='when_required',response_checksum_validation='when_required')),c['bucket']\n")
    monitor = (destination / 'monitor.py').read_text()
    monitor = monitor.replace("OUT=Path('/Users/kiparis/Downloads/Pencil colouring dataset/lora-v1');OUT.mkdir(exist_ok=True)", "OUT=ROOT.parent/'training-review';OUT.mkdir(parents=True,exist_ok=True)")
    monitor = monitor.replace("photo('../'+inp", "photo('../assets/'+inp").replace("photo('../'+target", "photo('../assets/'+target")
    (destination / 'monitor.py').write_text(monitor)
    # A portable workspace requires an explicit preview review; do not copy the old automatic controller.
    (destination / 'orchestrate.py').unlink()
    print('Training staged:', destination)
    print('Plan: python ' + shlex.quote(str(destination / 'launch.py')) + ' calibrate')
    print('Submit: append --launch. Review completed calibration before launching pilot.')


def training_status(args):
    folder, cloud = args.workspace.resolve(), cloud_config(args.cloud)
    training = folder / 'training'
    sys.path.insert(0, str(training))
    try:
        identity = module(training / 'run_identity.py', 'training_identity')
        key = identity.compute_run_key(training)
        if args.command == 'approve-calibration':
            if not args.confirm_preview_reviewed:
                raise ValueError('Inspect training-review/review.html, then pass --confirm-preview-reviewed')
            cfg = read(training / 'pilot.json')
            raw = storage(cloud).get_object(Bucket=cloud['bucket'], Key=cfg['prefix'] + '/' + key[:16] + '/calibrate/completion.json')['Body'].read()
            review = {'approved': True, 'run_key': key, 'completion_sha256': hashlib.sha256(raw).hexdigest(), 'basis': 'Explicit local review of the calibration preview and timing.'}
            launcher = module(training / 'launch.py', 'training_launcher')
            launcher.verify_calibration(cfg, key, review, raw)
            write(training / 'calibration-reviewed.json', review)
            print('Calibration evidence verified and review saved. Pilot may now be launched.')
        else:
            monitor = module(training / 'monitor.py', 'training_monitor')
            jobs = {}
            for phase in ['calibrate', 'pilot']:
                record = training / (phase + '-job.json')
                if record.exists():
                    result = read(record)
                    ident = result.get('metadata', {}).get('id', result.get('resource_id'))
                    raw = subprocess.check_output([cloud.get('cli', 'nebius'), 'ai', 'job', 'get', ident, '--format', 'json', '--no-check-update'], text=True)
                    jobs[phase] = json.loads(raw)
            monitor.sync(key, jobs)
            print('Training status and previews:', folder / 'training-review/review.html')
    finally:
        sys.path.pop(0)


def sync(args):
    folder, cloud = args.workspace.resolve(), cloud_config(args.cloud)
    cfg, s = read(folder / 'run.json'), storage(cloud)
    prefix = cfg['prefix'] + '/results/'
    out = folder / 'results'
    count = 0
    for page in s.get_paginator('list_objects_v2').paginate(Bucket=cloud['bucket'], Prefix=prefix):
        for item in page.get('Contents', []):
            relative = Path(item['Key'][len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe object path')
            path = out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            s.download_file(cloud['bucket'], item['Key'], str(path))
            count += 1
    completed = 0
    for task in cfg['tasks']:
        meta, image = out / (task['id'] + '.json'), out / (task['id'] + '.png')
        if meta.exists() and read(meta).get('status') == 'completed':
            result = read(meta)
            if result['run_sha256'] != sha(folder / 'run.json') or sha(image) != result['output_sha256']:
                raise ValueError('Downloaded output provenance mismatch: ' + task['id'])
            completed += 1
    print('Downloaded objects:', count, '; verified completed images:', completed, '/', len(cfg['tasks']))
    render_review(folder, cfg)


def render_review(folder, cfg):
    import html
    from urllib.parse import quote
    esc = html.escape
    def picture(relative):
        if relative and (folder / relative).is_file():
            return '<a href="' + quote(relative) + '"><img loading="lazy" src="' + quote(relative) + '"></a>'
        return '<p>No image</p>'
    sections = []
    for task in cfg['tasks']:
        inputs = ''.join(picture('assets/' + x) for x in task['inputs'])
        target = picture('assets/' + task['reference']) if task.get('reference') else '<p>No image</p>'
        output = picture('results/' + task['id'] + '.png')
        sections.append('<section id="' + esc(task['id']) + '"><h2>' + esc(task['id']) + '</h2><div class="row"><figure><figcaption>Original</figcaption>' + inputs + '</figure><figure><figcaption>Reference target</figcaption>' + target + '</figure><figure><figcaption>Qwen-Image-2.1 ' + ('LoRA' if cfg['additional_loras'] else 'base') + '</figcaption>' + output + '</figure></div><details><summary>Exact prompt and seed</summary><pre>' + esc(json.dumps(task, indent=2)) + '</pre></details></section>')
    (folder / 'review.html').write_text('<!doctype html><html lang="en-GB"><meta charset="utf-8"><title>Pencil comparison</title><style>body{font:16px system-ui;margin:2rem;background:#f5f5f5}.row{display:flex;gap:1rem}figure{flex:1;margin:0;min-width:0}img{width:100%;height:auto}pre{white-space:pre-wrap}section{margin-bottom:3rem}figcaption{font-weight:bold;margin:1rem 0}</style><h1>Pencil comparison</h1><p>Raw outputs; no greyscale conversion or sharpening.</p>' + ''.join(sections) + '</html>')
    print('Review page:', folder / 'review.html')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('verify-archive')
    p = sub.add_parser('prepare')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--checkpoint', type=int, help='Optional known adapter update; not inferred from its filename')
    p.add_argument('--strength', type=float, default=1.0)
    p.add_argument('--short-prompt', action='store_true')
    p.add_argument('--freeze', action='store_true', help='Confirm reviewed targets, identity groups, splits and white-letterbox geometry; export Parquet')
    for name in ['infer', 'stage-training', 'sync', 'status-training', 'approve-calibration']:
        p = sub.add_parser(name)
        p.add_argument('--workspace', type=Path, required=True)
        p.add_argument('--cloud', type=Path, required=True)
        if name == 'approve-calibration':
            p.add_argument('--confirm-preview-reviewed', action='store_true')
        if name == 'infer':
            p.add_argument('--mode', choices=['warm', 'smoke', 'batch'], required=True)
            p.add_argument('--gpus', type=int, choices=[1, 8], default=1)
            p.add_argument('--submit', action='store_true')
    args = ap.parse_args()
    if args.command == 'verify-archive':
        verify_archive()
    else:
        {'prepare': prepare, 'infer': infer, 'stage-training': stage_training, 'sync': sync, 'status-training': training_status, 'approve-calibration': training_status}[args.command](args)


if __name__ == '__main__':
    main()
