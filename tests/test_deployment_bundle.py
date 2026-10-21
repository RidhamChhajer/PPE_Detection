"""Target setup is tested with fake builders; no local GPU engine is rebuilt."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.registry import sha256, read_registry
from tools.prepare_deployment import bundle_inputs, prepare_engines
from tools.prepare_deployment import verify_engine
from tools.package_deployment import deployment_assets
from app.registry import model_specs
from support import registry_fixture


class DeploymentBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'configs').mkdir()
        registry = registry_fixture(self.root/'models', ('goggles-v1', 'footwear-v1'))
        self.registry = self.root/'configs/models.json'
        registry.replace(self.registry)
        (self.root/'deployment').mkdir()
        image = self.root/'verification/frame.png'
        image.parent.mkdir()
        cv2.imwrite(str(image), np.zeros((48, 64, 3), dtype=np.uint8))
        self.bundle = dict(schema_version=1, registry_sha256=sha256(self.registry), models=[
            dict(id=name, precision=precision, images=[dict(path='verification/frame.png', sha256=sha256(image))])
            for name, precision in [('goggles-v1', 'fp16'), ('footwear-v1', 'fp32')]])
        self.manifest = self.root/'deployment/bundle.json'
        self.manifest.write_text(json.dumps(self.bundle))

    @staticmethod
    def builder(source, target, name, labels, precision):
        target.write_bytes(f'{name}:{precision}'.encode())

    def test_tampered_images_and_registry_fail_before_build(self):
        self.assertEqual(len(bundle_inputs(self.root)), 2)
        self.bundle['models'][0]['images'][0]['path'] = '../outside.png'
        self.manifest.write_text(json.dumps(self.bundle))
        with self.assertRaises(ValueError):
            bundle_inputs(self.root)
        self.registry.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'registry identity'):
            bundle_inputs(self.root)

    def test_setup_commits_both_verified_engines_and_uses_fp32_fallback(self):
        attempts = []
        def verify(reference, candidate, images):
            attempts.append((reference.model_id, candidate.entry['tensorrt']['precision']))
            if candidate.entry['tensorrt']['precision'] == 'fp16':
                raise RuntimeError('parity failed')
            return dict(images=1, exact_classes_and_counts=True)
        with patch('tools.prepare_deployment.importlib.metadata.version', return_value='test'):
            prepare_engines(self.root, self.builder, verify)
        installed = read_registry(self.root/'models/registry.local.json')
        self.assertEqual(len(installed['deployment_verification']['models']), 2)
        self.assertEqual(attempts, [('goggles-v1', 'fp16'), ('goggles-v1', 'fp32'), ('footwear-v1', 'fp32')])
        for entry in installed['models']:
            self.assertEqual(sha256(self.root/'models'/entry['tensorrt']['filename']), entry['tensorrt']['sha256'])

    def test_failed_verification_and_publication_preserve_existing_setup(self):
        installed = self.root/'models/registry.local.json'
        installed.write_bytes(b'previous verified registry')
        def failed(*args):
            raise RuntimeError('parity failed')
        with patch('tools.prepare_deployment.importlib.metadata.version', return_value='test'):
            with self.assertRaisesRegex(RuntimeError, 'parity failed'):
                prepare_engines(self.root, self.builder, failed)
            with patch('tools.prepare_deployment.atomic_json', side_effect=OSError('publication failed')):
                with self.assertRaises(OSError):
                    prepare_engines(self.root, self.builder, lambda *args: dict(images=1))
        self.assertEqual(installed.read_bytes(), b'previous verified registry')
        self.assertFalse(list((self.root/'models').glob('*.engine')))
        self.assertFalse(list((self.root/'models').glob('.build-*')))

    def test_packaged_assets_match_model_hashes_and_exclude_existing_engines(self):
        with self.assertRaises(FileNotFoundError):
            deployment_assets(self.root)
        release = self.root/'release_assets/v2'
        release.mkdir(parents=True)
        document = read_registry(self.registry)
        for entry in document['models']:
            checkpoint = release/entry['checkpoint']['filename']
            checkpoint.write_bytes(b'selected checkpoint')
            entry['checkpoint']['sha256'] = sha256(checkpoint)
            (release/f'{entry["id"]}_rtmdet.py').write_text('model = dict()')
            task = entry['id'].rsplit('-v', 1)[0]
            images = self.root/'data'/task/'images/test'
            images.mkdir(parents=True)
            cv2.imwrite(str(images/'frame.png'), np.zeros((48, 64, 3), dtype=np.uint8))
            annotations = self.root/'data'/task/'annotations'
            annotations.mkdir()
            (annotations/'test.json').write_text(json.dumps(dict(images=[dict(file_name='frame.png')])))
        self.registry.write_text(json.dumps(document))
        (self.root/'models/old.engine').write_bytes(b'not portable')
        assets, bundle = deployment_assets(self.root)
        self.assertEqual(len(assets), 8)
        self.assertFalse(any(name.endswith('.engine') for name in assets))
        self.assertEqual(len(bundle['models']), 2)
        next(path for name, path in assets.items() if name.endswith('.onnx')).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            deployment_assets(self.root)

    def test_parity_check_accepts_matching_outputs_and_rejects_bad_raw_boxes(self):
        manifest = registry_fixture(self.root/'reference', real=True)
        spec = model_specs(self.root/'reference', manifest)[0]
        image = self.root/'verification/frame.png'
        self.assertEqual(verify_engine(spec, spec, [image])['images'], 1)
        from app.runtime import Detector
        expected = np.zeros((1, 8400, 5), dtype=np.float32)
        expected[0, 0] = [10, 10, 30, 30, .8]
        wrong = expected.copy()
        wrong[0, 0, 0] += 10
        with patch.object(Detector, 'infer_raw', side_effect=[expected, wrong]):
            with self.assertRaisesRegex(RuntimeError, 'Raw output parity failed'):
                verify_engine(spec, spec, [image])
