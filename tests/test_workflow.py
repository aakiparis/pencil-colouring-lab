import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lab


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for index, split in enumerate(['train', 'validation', 'test']):
            # Preparation packages bytes; image decoding is exercised by the export test.
            (self.root / (split + '.png')).write_bytes(b'source' + bytes([index]))
            (self.root / (split + '-target.png')).write_bytes(b'target' + bytes([index]))
            self.rows.append({'id': split, 'inputs': [split + '.png'], 'target': split + '-target.png', 'split': split, 'group_id': split, 'identities': [split]})
        self.manifest = {'schema_version': 1, 'pairs': self.rows}
        lab.write(self.root / 'manifest.json', self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, name, **kw):
        args = dict(manifest=self.root / 'manifest.json', data_root=self.root, output=self.root / name, seed=42, adapter=None, checkpoint=None, strength=1.0, short_prompt=False, freeze=False)
        args.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            lab.prepare(argparse.Namespace(**args))
        return self.root / name

    def test_immutable_archive(self):
        with contextlib.redirect_stdout(io.StringIO()):
            lab.verify_archive()

    def test_prepare_is_reproducible_and_keeps_worker_contract(self):
        a, b = self.prepare('a'), self.prepare('b')
        for name in ['dataset.tar.gz', 'dataset-checksums.json', 'run.json', 'worker.py']:
            self.assertEqual(lab.sha(a / name), lab.sha(b / name))
        cfg = lab.read(a / 'run.json')
        self.assertEqual(cfg['tasks'][0]['prompt_version'], 'background-v4')
        self.assertEqual(cfg['tasks'][0]['kind'], 'single_input')
        compile((a / 'worker.py').read_text(), str(a / 'worker.py'), 'exec')
        self.assertNotIn('/cache/dataset', (a / 'worker.py').read_text())
        self.assertIn("cfg['prefix'] + '/dataset.tar.gz'", (a / 'worker.py').read_text())
        with self.assertRaisesRegex(ValueError, 'new output directory'):
            self.prepare('a')

    def test_composite_uses_native_worker_kind(self):
        self.rows[0]['inputs'].append('validation.png')
        lab.write(self.root / 'manifest.json', self.manifest)
        out = self.prepare('composite')
        self.assertEqual(lab.read(out / 'run.json')['tasks'][0]['kind'], 'multi_input')

    def test_seed_changes_run_identity(self):
        a, b = self.prepare('a'), self.prepare('b', seed=123)
        self.assertNotEqual(lab.read(a / 'run.json')['prefix'], lab.read(b / 'run.json')['prefix'])
        self.assertEqual(lab.sha(a / 'dataset.tar.gz'), lab.sha(b / 'dataset.tar.gz'))

    def test_traversal_symlinks_and_unsafe_ids_rejected(self):
        for bad in ['../escape', 'a/b', '.', 'x<script>']:
            cfg = copy.deepcopy(self.manifest)
            cfg['pairs'][0]['id'] = bad
            with self.assertRaises(ValueError):
                lab.validate_pairs(cfg, self.root)
        with self.assertRaises(ValueError):
            lab.local_file(self.root, '../outside.png')
        outside = self.root.parent / (self.root.name + '-outside')
        outside.write_bytes(b'outside')
        try:
            (self.root / 'link').symlink_to(outside)
            with self.assertRaises(ValueError):
                lab.local_file(self.root, 'link')
        finally:
            outside.unlink()

    def test_training_split_leakage(self):
        lab.validate_pairs(self.manifest, self.root, True)
        for key, value in [('group_id', 'train'), ('identities', ['train']), ('target', 'train-target.png'), ('inputs', ['train.png'])]:
            cfg = copy.deepcopy(self.manifest)
            cfg['pairs'][1][key] = value
            with self.assertRaisesRegex(ValueError, 'Split leakage'):
                lab.validate_pairs(cfg, self.root, True)

    def test_plan_does_not_touch_cloud(self):
        out = self.prepare('prepared')
        cfg = lab.read(lab.ROOT / 'examples/cloud.example.json')
        for key in ['project_id', 'bucket', 'filesystem_id', 'subnet_id', 'storage_secret_id']:
            cfg[key] = 'fixture-' + key
        lab.write(self.root / 'cloud.json', cfg)
        with patch.object(lab, 'storage', side_effect=AssertionError('Network accessed')), patch.object(lab.subprocess, 'run', side_effect=AssertionError('CLI invoked')), contextlib.redirect_stdout(io.StringIO()) as stdout:
            lab.infer(argparse.Namespace(workspace=out, cloud=self.root / 'cloud.json', mode='batch', gpus=8, submit=False))
        self.assertIn('--preemptible', stdout.getvalue())
        self.assertIn('8gpu-192vcpu-1744gb', stdout.getvalue())
        self.assertFalse((out / 'batch-job.json').exists())

    def test_modified_prepared_inputs_are_rejected(self):
        out = self.prepare('prepared')
        (out / 'worker.py').write_text('# modified')
        cfg = lab.read(lab.ROOT / 'examples/cloud.example.json')
        for key in ['project_id', 'bucket', 'filesystem_id', 'subnet_id', 'storage_secret_id']:
            cfg[key] = 'fixture-' + key
        lab.write(self.root / 'cloud.json', cfg)
        with self.assertRaisesRegex(ValueError, 'Prepared file changed'):
            lab.infer(argparse.Namespace(workspace=out, cloud=self.root / 'cloud.json', mode='warm', gpus=1, submit=False))

    def test_review_supports_missing_reference(self):
        out = self.prepare('prepared')
        cfg = lab.read(out / 'run.json')
        cfg['tasks'][0]['reference'] = None
        with contextlib.redirect_stdout(io.StringIO()):
            lab.render_review(out, cfg)
        self.assertIn('No image', (out / 'review.html').read_text())


if __name__ == '__main__':
    unittest.main()
