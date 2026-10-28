"""Optional real-image/Parquet integration check; no GPU or cloud access."""
import argparse
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lab
try:
    from PIL import Image
    from datasets import load_dataset
    HAVE_EXPORT_DEPS = True
except ImportError:
    HAVE_EXPORT_DEPS = False


@unittest.skipUnless(HAVE_EXPORT_DEPS, 'Install requirements-local.txt to test real paired export')
class PairedExportTests(unittest.TestCase):
    def test_real_export_load_training_stage_and_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for i, split in enumerate(['train', 'validation', 'test']):
                Image.new('RGB', (96, 128), (30 + i * 50, 80, 140)).save(root / (split + '.png'))
                Image.new('RGB', (128, 128), (60 + i * 50, 60, 60)).save(root / (split + '-target.png'))
                rows.append({'id': split, 'inputs': [split + '.png'], 'target': split + '-target.png', 'group_id': split, 'identities': [split], 'split': split})
            manifest, out = root / 'manifest.json', root / 'prepared'
            lab.write(manifest, {'schema_version': 1, 'pairs': rows})
            with contextlib.redirect_stdout(io.StringIO()):
                lab.prepare(argparse.Namespace(manifest=manifest, data_root=root, output=out, seed=42, adapter=None, checkpoint=None, strength=1, short_prompt=False, freeze=True))
            ds = load_dataset(str(out / 'frozen-dataset'))
            for split in ['train', 'validation', 'test']:
                self.assertEqual(len(ds[split]), 1)
                pair = ds[split][0]
                self.assertEqual(pair['image'].size, pair['cond_image'].size)
                self.assertEqual(pair['caption'], 'Create a pencil colouring page.')
                self.assertEqual(pair['image'].mode, 'RGBA')
            cfg = lab.read(lab.ROOT / 'examples/cloud.example.json')
            for key in ['project_id', 'bucket', 'filesystem_id', 'subnet_id', 'storage_secret_id']:
                cfg[key] = 'fixture-' + key
            lab.write(root / 'cloud.json', cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                lab.stage_training(argparse.Namespace(workspace=out, cloud=root / 'cloud.json'))
            self.assertFalse((out / 'training/orchestrate.py').exists())
            self.assertNotIn('/Users/kiparis', (out / 'training/monitor.py').read_text())
            plan = subprocess.check_output([sys.executable, str(out / 'training/launch.py'), 'calibrate'], text=True)
            self.assertIn('plan_only_no_upload_or_compute', plan)
            for check in ['verify_preparation.py', 'verify_distributed.py']:
                subprocess.run([sys.executable, str(out / 'training' / check)], check=True, capture_output=True, text=True)
