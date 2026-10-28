"""Finite independent GPU shards with checked inputs and per-image S3 publication."""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

CACHE = Path('/cache/benchmark-v2')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def storage():
    import boto3
    from botocore.config import Config
    return boto3.client('s3', endpoint_url=os.environ['S3_ENDPOINT'], region_name='uk-south2',
        config=Config(signature_version='s3v4', request_checksum_calculation='when_required', response_checksum_validation='when_required'))

def config():
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (os.environ['RUN_SHA256'] + '.json')
    if not p.exists():
        temp = p.with_suffix('.' + str(os.getpid()) + '.tmp')
        storage().download_file(os.environ['S3_BUCKET'], os.environ['RUN_PREFIX'] + '/run.json', str(temp))
        assert digest(temp) == os.environ['RUN_SHA256'], 'Downloaded configuration mismatch'
        temp.replace(p)
    assert digest(p) == os.environ['RUN_SHA256'], 'Experiment configuration mismatch'
    return json.loads(p.read_text())

def publish(s, cfg, key, value):
    s.put_object(Bucket=os.environ['S3_BUCKET'], Key=cfg['prefix'] + '/' + key,
                 Body=json.dumps(value, indent=2).encode(), ContentType='application/json')

def warm():
    from huggingface_hub import snapshot_download
    import tarfile
    cfg = config(); s = storage(); CACHE.mkdir(parents=True, exist_ok=True)
    # Reuse the previously checked source dataset; verify every referenced file again.
    dataset = Path('/cache/dataset')
    if not dataset.exists():
        archive = CACHE / 'dataset.tar.gz'
        s.download_file(os.environ['S3_BUCKET'], 'baseline-v1/dataset.tar.gz', str(archive))
        dataset.mkdir()
        with tarfile.open(archive) as t: t.extractall(dataset, filter='data')
    checks = json.loads(s.get_object(Bucket=os.environ['S3_BUCKET'], Key='baseline-v1/dataset-checksums.json')['Body'].read())
    hashes = {r['path']: r['sha256'] for r in checks}
    for t in cfg['tasks']:
        for name in t['inputs']: assert digest(dataset / name) == hashes[name], name
    model = cfg['model']
    patterns = (['modular_model_index.json', 'transformer/*', 'text_encoder/*', 'vae/*', '*.json', '*.md']
                if cfg['pipeline'] == 'flux3' else None)
    path = snapshot_download(model['id'], revision=model['revision'], allow_patterns=patterns,
                             ignore_patterns=['assets/*', '*.png', '*.jpg', '*.bin', 'variants/*'], max_workers=8)
    ready = {'model': model, 'path': path, 'input_hashes': hashes, 'run_sha256': os.environ['RUN_SHA256']}
    if cfg.get('prompt_upsampler'):
        pup = cfg['prompt_upsampler']
        ready['pup_path'] = snapshot_download(pup['id'], revision=pup['revision'], max_workers=8)
        # Pin component references to local snapshots, rather than the moving Hub main branch.
        runtime = CACHE / 'flux3-pipeline'
        runtime.mkdir(exist_ok=True)
        for entry in Path(path).iterdir():
            target = runtime / entry.name
            if entry.name != 'modular_model_index.json' and not target.exists(): target.symlink_to(entry)
        index = json.loads((Path(path) / 'modular_model_index.json').read_text())
        for name in ['prompt_upsampler', 'prompt_upsampler_processor']: index.pop(name, None)
        for value in index.values():
            if isinstance(value, list) and len(value) == 3:
                value[2]['pretrained_model_name_or_path'] = path
                value[2]['revision'] = None
        (runtime / 'modular_model_index.json').write_text(json.dumps(index, indent=2))
        ready['path'] = str(runtime)
    (CACHE / (cfg['pipeline'] + '-ready.json')).write_text(json.dumps(ready, indent=2))
    publish(s, cfg, 'warm-ready.json', {'model': model, 'verified_dataset': True, 'run_sha256': os.environ['RUN_SHA256']})
    print('WARM_READY', model['id'], flush=True)

def worker(rank, world, mode):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    import torch
    from PIL import Image, ImageOps
    from importlib.metadata import version
    cfg = config(); s = storage()
    worker_hash = digest(Path(__file__))
    bootstrap_hash = digest(Path('/app/benchmark_bootstrap.sh'))
    tasks = [t for t in cfg['tasks'] if mode != 'smoke' or t['id'] in cfg['smoke_ids']][rank::world]
    if not tasks: return
    ready = json.loads((CACHE / (cfg['pipeline'] + '-ready.json')).read_text())
    assert cfg.get('reuse_ready_model') or ready['run_sha256'] == os.environ['RUN_SHA256']
    assert ready['model'] == cfg['model']
    torch.cuda.set_device(0); torch.set_num_threads(8)
    env = {'gpu': torch.cuda.get_device_name(0), 'rank': rank, 'mode': mode,
           'packages': {n: version(n) for n in ['torch', 'diffusers', 'transformers', 'accelerate', 'huggingface-hub']}}
    publish(s, cfg, f'{mode}/worker-{rank}-environment.json', env)
    if cfg['pipeline'] == 'qwen21':
        from diffusers import QwenImage21Pipeline
        pipe = QwenImage21Pipeline.from_pretrained(ready['path'], torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
        if cfg.get('vae_tiling', True): pipe.vae.enable_tiling()
        else: pipe.vae.disable_tiling()
        pipe.vae.disable_slicing()
        for adapter in cfg.get('additional_loras', []):
            adapter_folder = CACHE / ('adapter-' + adapter['sha256'])
            adapter_folder.mkdir(exist_ok=True)
            weight = adapter_folder / 'pytorch_lora_weights.safetensors'
            if not weight.exists():
                temp = weight.with_name(weight.name + '.' + str(os.getpid()) + '.tmp')
                s.download_file(os.environ['S3_BUCKET'], adapter['object_key'], str(temp))
                assert digest(temp) == adapter['sha256'], 'Adapter checksum mismatch'
                temp.replace(weight)
            assert digest(weight) == adapter['sha256'], 'Adapter checksum mismatch'
            pipe.load_lora_weights(str(adapter_folder), adapter_name='pencil_style')
            pipe.set_adapters('pencil_style', adapter_weights=adapter['scale'])
        pipe.set_progress_bar_config(disable=True)
    else:
        from diffusers.modular_pipelines import Flux3AutoBlocks
        # Run the official caption expander separately to pin its revision and save its output.
        blocks = Flux3AutoBlocks()
        blocks.sub_blocks.pop('prompt_upsampler')
        # The model's component reference points to the pinned local snapshot.
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        pup_model = Qwen3_5ForConditionalGeneration.from_pretrained(ready['pup_path'], dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
        pup_processor = AutoProcessor.from_pretrained(ready['pup_path'], local_files_only=True)
        systems = json.loads((Path(ready['pup_path']) / 'system_prompts.json').read_text())
        pipe = blocks.init_pipeline(ready['path'])
        pipe.load_components(dtype={'vae': torch.float32, 'default': torch.bfloat16})
        pipe.to('cuda')
    output = CACHE / cfg['run_id'] / 'outputs'; output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task in tasks:
        ident = task['id']; key = 'results/' + ident
        try:
            old = json.loads(s.get_object(Bucket=os.environ['S3_BUCKET'], Key=cfg['prefix'] + '/' + key + '.json')['Body'].read())
            if (old.get('status') == 'completed' and old.get('run_sha256') == os.environ['RUN_SHA256']
                and old.get('worker_sha256') == worker_hash and old.get('bootstrap_sha256') == bootstrap_hash):
                print('SKIP', ident, flush=True); continue
        except s.exceptions.NoSuchKey: pass
        images = []
        for name in task['inputs']:
            source = Path('/cache/dataset') / name
            expected = cfg.get('input_sha256', {}).get(name, ready['input_hashes'].get(name))
            assert expected, 'Missing input checksum: ' + name
            if not source.exists() and name in cfg.get('input_objects', {}):
                source.parent.mkdir(parents=True, exist_ok=True)
                temporary = source.with_name(source.name + '.' + str(os.getpid()) + '.tmp')
                s.download_file(os.environ['S3_BUCKET'], cfg['input_objects'][name], str(temporary))
                assert digest(temporary) == expected
                temporary.replace(source)
            assert digest(source) == expected
            with Image.open(Path('/cache/dataset') / name) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
        ratio = images[0].width / images[0].height
        width = max(256, round(math.sqrt(cfg['target_pixels'] * ratio) / 16) * 16)
        height = max(256, round(math.sqrt(cfg['target_pixels'] / ratio) / 16) * 16)
        if task['kind'] == 'multi_input': width, height = cfg['composite_size']
        requested_size = [width, height]
        # Both pipelines accept this common grid; do not rely on silent internal rounding.
        width, height = width // 32 * 32, height // 32 * 32
        meta = {'id': ident, 'status': 'running', 'model': cfg['model'], 'run_sha256': os.environ['RUN_SHA256'],
                'prompt': task['prompt'], 'prompt_version': task['prompt_version'], 'seed': task['seed'],
                'width': width, 'height': height, 'steps': cfg['num_inference_steps'], 'mode': mode,
                'worker_sha256': worker_hash, 'bootstrap_sha256': bootstrap_hash,
                'vae_tiling': cfg.get('vae_tiling', True),
                'requested_size_before_grid_rounding': requested_size,
                'input_sha256': {n: digest(Path('/cache/dataset') / n) for n in task['inputs']}, 'additional_loras': cfg.get('additional_loras', []), 'dataset_split': task.get('dataset_split'), 'post_processing': 'none'}
        start = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                if cfg['pipeline'] == 'qwen21':
                    im = pipe(image=images, prompt=task['prompt'], width=width, height=height, output_resolution=1024,
                              num_inference_steps=cfg['num_inference_steps'], true_cfg_scale=1.0,
                              generator=torch.Generator('cuda').manual_seed(task['seed'])).images[0]
                else:
                    text = 'Reference images follow in order (the first image, the second image, ...).\nUser request: ' + task['prompt'] + '\nOutput the caption.'
                    messages = [{'role':'system','content':systems['i2i']}, {'role':'user','content':
                                [{'type':'image','image':i} for i in images] + [{'type':'text','text':text}]}]
                    inputs = pup_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                              return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')
                    answer = pup_model.generate(**inputs, max_new_tokens=3584, do_sample=False)
                    caption = pup_processor.decode(answer[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    meta['expanded_prompt'] = caption
                    im = pipe(image=images, prompt=caption, width=width, height=height,
                              num_inference_steps=cfg['num_inference_steps'],
                              generator=torch.Generator().manual_seed(task['seed']), output='images')[0]
            if im.mode == 'RGBA':
                white = Image.new('RGBA', im.size, 'white'); white.alpha_composite(im); im = white.convert('RGB')
            meta.update(width=im.width, height=im.height)
            dest = output / (ident + '.png'); im.save(dest)
            s.upload_file(str(dest), os.environ['S3_BUCKET'], cfg['prefix'] + '/' + key + '.png', ExtraArgs={'ContentType':'image/png'})
            meta.update(status='completed', seconds=round(time.time()-start, 3), output_sha256=digest(dest),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())
        except Exception as error:
            meta.update(status='failed', error=repr(error), traceback=traceback.format_exc(), seconds=round(time.time()-start, 3))
            failures += 1; torch.cuda.empty_cache()
        publish(s, cfg, key + '.json', meta)
        print(json.dumps({k: meta.get(k) for k in ['id','status','seconds','error']}), flush=True)
    if failures: raise RuntimeError(f'{failures} failed images')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['warm','smoke','batch']); args = ap.parse_args()
    if args.mode == 'warm': return warm()
    import torch
    import multiprocessing as mp
    world = 1 if args.mode == 'smoke' else min(torch.cuda.device_count(), len(config()['tasks']))
    if not world: raise RuntimeError('No GPU available')
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(rank, world, args.mode)) for rank in range(world)]
    for p in ps: p.start()
    for p in ps: p.join()
    if any(p.exitcode != 0 for p in ps): raise RuntimeError('A GPU worker failed')
    publish(storage(), config(), args.mode + '/completed.json', {'status':'completed','run_sha256':os.environ['RUN_SHA256']})

if __name__ == '__main__': main()
