"""Freeze corrected prompts and two independent model runs without changing baseline-v1."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
PARENT = ROOT.parent
DATA = Path('/Users/kiparis/Downloads/Pencil colouring dataset')

def model_info(name):
    return json.load(urllib.request.urlopen('https://huggingface.co/api/models/' + name, timeout=30))

def main():
    original = json.loads((PARENT / 'run.json').read_text())
    feedback_path = DATA / 'baseline-v1/dictated-feedback.json'
    feedback = json.loads(feedback_path.read_text())
    style = ('Render a realistic black-and-white graphite pencil drawing suitable for children to colour. '
             'Use crisp, clearly defined subject contours, selective deep black accents and a full but restrained tonal range. '
             'Keep large clothing and object surfaces light and open for colouring. '
             'Use natural short pencil strokes and smooth, subtle shading; avoid long parallel stripes, mechanical hatching, '
             'blurred outlines and uniform grey fills. Preserve recognisable identity and fine details of the main subject. ')
    fidelity = ('Reproduce the main subject from the supplied photograph faithfully: keep the original pose, expression, '
                'gaze direction, proportions, clothing, accessories, visible lettering, object shapes and framing. '
                'Keep the same visible extent of the subject; do not invent limbs, people, products or a new scene. ')
    background = ('Remove incidental background people and distracting scenery. Use a clean white or very lightly indicated '
                  'background so the main subject stands out. Do not reconstruct detail in blurred areas. '
                  'Separate the subject clearly from the background with precise contours. ')
    composite = ('Create one coherent drawing with both main subjects from the two supplied images. '
                 'Place the subject from image 1 on the left and the subject from image 2 on the right, '
                 'beside each other at a compatible scale with clear space between them. '
                 'Keep both subjects complete within their original visible framing, recognisable and distinct; '
                 'preserve their identities, poses, expressions and clothing. Neither subject may cover, merge into or replace the other. '
                 'Remove both original backgrounds and all incidental people. Put both subjects on the same clean white background. ')
    tasks = []
    for task in original['tasks']:
        t = dict(task)
        prompt = composite + style if t['kind'] == 'multi_input' else fidelity + background + style
        if t['id'] == '029_pair':
            prompt = ('Draw the whole Bugatti from the photograph, completing only the cropped parts of the car consistently '
                      'with its visible design and perspective. Retain the person with the car, with the same appearance and pose. '
                      'Do not omit the person. Use a clean white background. ' + style)
        elif t['id'] == '053_pair':
            prompt = fidelity + ('Remove the incidental boy in the corner and all other background people. '
                                 'Emphasise only the main photographic subject on a clean white background. ') + style
        elif t['id'] == '055_pair':
            prompt = ('A faithful pencil drawing of only the pair of trainers shown in the reference photograph, on a plain white background. '
                      'Match the exact shoes, their number, arrangement, viewing angle, silhouette, sole construction, mesh, laces and logos. '
                      'Keep the product composition one-to-one with the reference. Do not add a person, legs, a runner or scenery. ' + style)
        elif t['id'] == '048_pair':
            prompt += 'Give the trophy clear, accurate contours and recognisable fine details; keep the face sharply defined.'
        elif t['id'] == '049_pair':
            prompt += 'Keep the ball clearly visible and precisely outlined, separated from the background.'
        t.update(prompt=prompt, prompt_basis='Corrected from the completed user baseline review; task-specific overrides retained.',
                 seed=42, prompt_version='feedback-v2')
        t.pop('prompt_evidence', None)
        tasks.append(t)
    specifications = [
        ('qwen21', 'Qwen/Qwen-Image-2.1', 'qwen21', 40),
        ('flux3', 'black-forest-labs/flux-3-image-klein-base', 'flux3', 50),
    ]
    revision = json.loads((PARENT / 'diffusers-source.json').read_text())['sha']
    for slug, name, pipeline, steps in specifications:
        dest = ROOT / slug
        dest.mkdir(exist_ok=True)
        info = model_info(name)
        config = {'run_id': 'pencil-benchmark-v2-' + slug, 'prefix': 'benchmark-v2/' + slug,
                  'model': {'id': name, 'revision': info['sha']}, 'pipeline': pipeline,
                  'additional_loras': [], 'dtype': 'bfloat16', 'num_inference_steps': steps,
                  'target_pixels': 1024 * 1024, 'composite_size': [1360, 768], 'tasks': tasks,
                  'smoke_ids': ['055_pair', '012_pair', '056_composite'],
                  'prompt_feedback_sha256': hashlib.sha256(feedback_path.read_bytes()).hexdigest(),
                  'input_preprocessing': 'EXIF orientation and RGB conversion; no external crop or enhancement.',
                  'software': {'diffusers_commit': revision, 'transformers': '5.20.1', 'accelerate': '1.15.0',
                               'huggingface_hub': '2.2.0', 'boto3': '1.43.114', 'Pillow': '13.0.0'},
                  'comparison_note': 'Prompts incorporate the user review; differences from baseline-v1 include both model and prompt changes.'}
        if pipeline == 'flux3':
            pup = model_info('black-forest-labs/flux-3-image-klein-pup')
            config['prompt_upsampler'] = {'id': pup['id'], 'revision': pup['sha']}
            config['prompt_upsampling'] = 'Official FLUX caption expansion using the original reference images; expanded captions are saved.'
        else:
            config['true_cfg_scale'] = 1.0
        path = dest / 'run.json'
        if path.exists() and path.read_text() != json.dumps(config, indent=2) + '\n':
            raise RuntimeError('Refusing to overwrite an existing experiment: ' + str(path))
        path.write_text(json.dumps(config, indent=2) + '\n')
        print(slug, name, info['sha'], 'tasks', len(tasks))
    (ROOT / 'prompt-policy.json').write_text(json.dumps({'version': 'feedback-v2', 'shared_style': style,
        'source_fidelity': fidelity, 'background_policy': background, 'composite_policy': composite,
        'provenance': 'User explicitly authorised prompt corrections and both model runs in this chat.',
        'feedback': feedback_path.as_posix()}, indent=2) + '\n')

if __name__ == '__main__':
    main()
