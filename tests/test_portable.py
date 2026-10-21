import copy
import builtins
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from app.registry import model_specs, read_registry, sha256
from app.runtime import Detector, select_runtime, openvino_device
from tools.download_models import download
from support import registry_fixture


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_onnx_single_and_multiclass_structured_outputs(self):
        for labels in (['goggles'],['SAFETY_SHOE','NOT_SAFETY_SHOE']):
            manifest = registry_fixture(self.root,labels=labels,real=True)
            spec = model_specs(self.root,manifest)[0]
            with Detector(spec) as engine:
                results = engine.detect(np.zeros((640,640,3),dtype=np.uint8))
                self.assertEqual([r.class_label for r in results],labels)
                self.assertEqual([r.class_id for r in results],list(range(len(labels))))
                self.assertEqual(results[0].x1,10.)

    def test_openvino_cpu_matches_onnx_and_reuses_the_verified_artifact(self):
        import onnx
        from onnx import helper
        manifest = registry_fixture(self.root, labels=['SAFETY_SHOE', 'NOT_SAFETY_SHOE'], real=True)
        path = self.root/'goggles-v1.onnx'
        model = onnx.load(str(path))
        model.graph.node[0].output[0] = 'boxes'
        # Keep a genuine input dependency: OpenVINO removes unused inputs from constant test graphs.
        model.graph.node.extend([
            helper.make_node('ReduceMean', ['images'], ['mean'], keepdims=0),
            helper.make_node('Constant', [], ['scale'], value=helper.make_tensor('scale', onnx.TensorProto.FLOAT, [], [1e-12])),
            helper.make_node('Mul', ['mean', 'scale'], ['offset']),
            helper.make_node('Add', ['boxes', 'offset'], ['detections']),
        ])
        onnx.save(model, str(path))
        document = read_registry(manifest)
        document['models'][0]['onnx']['sha256'] = sha256(path)
        manifest.write_text(json.dumps(document))
        reference = model_specs(self.root, manifest)[0]
        candidate = model_specs(self.root, manifest, 'openvino', device='CPU')[0]
        self.assertEqual(candidate.path, reference.path)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        with Detector(reference) as baseline, Detector(candidate) as converted:
            self.assertEqual(converted.detect(frame), baseline.detect(frame))
            self.assertIsNone(converted.session)
            self.assertIn('CPU', converted.device_name)
        self.assertIsNone(converted.request)

    def test_openvino_gpu_selection_excludes_non_intel_devices(self):
        core = SimpleNamespace(available_devices=['CPU', 'GPU.0', 'GPU.1'],
                               get_property=lambda name, prop: {'GPU.0': 'NVIDIA GPU', 'GPU.1': 'Intel UHD'}[name])
        self.assertEqual(openvino_device(core, 'GPU'), ('GPU.1', 'Intel UHD'))
        core.available_devices = ['CPU', 'GPU.0']
        with self.assertRaisesRegex(RuntimeError, 'Intel GPU is unavailable'):
            openvino_device(core, 'GPU')

    def test_auto_probes_every_model_and_reports_fallback_without_hiding_explicit_errors(self):
        attempts, closed = [], []
        def specs(directory, manifest, backend, device):
            return [SimpleNamespace(backend=backend, model_id=name) for name in ('goggles', 'footwear')]
        class Probe:
            device_name = 'Intel UHD test GPU'
            def __init__(self, spec): self.spec = spec
            def __enter__(self): return self
            def __exit__(self, *args): closed.append((self.spec.backend, self.spec.model_id))
            def infer_raw(self, tensor):
                attempts.append((self.spec.backend, self.spec.model_id))
                if self.spec.backend == 'tensorrt': raise RuntimeError('incompatible engine')
        with patch('app.runtime.model_specs', side_effect=specs):
            selected, _, status = select_runtime(self.root, factory=Probe)
            self.assertEqual(status['backend'], 'openvino')
            self.assertEqual(status['device'], 'Intel UHD test GPU')
            self.assertIn('incompatible engine', status['fallback_reasons'][0])
            self.assertEqual(attempts, [('tensorrt', 'goggles'), ('openvino', 'goggles'), ('openvino', 'footwear')])
            self.assertEqual(closed, attempts)
        with patch('app.runtime.model_specs', side_effect=ValueError('bad model')):
            with self.assertRaisesRegex(ValueError, 'bad model'):
                select_runtime(self.root, backend='openvino')
            with self.assertRaisesRegex(RuntimeError, 'No usable inference backend'):
                select_runtime(self.root, factory=Probe)

    def test_auto_falls_back_to_cpu_if_both_gpu_paths_fail(self):
        manifest = registry_fixture(self.root, real=True)
        with patch('app.runtime.DEFAULT_REGISTRY', manifest), \
                patch('app.runtime.openvino_device', side_effect=RuntimeError('GPU driver unavailable')):
            _, _, status = select_runtime(self.root)
        self.assertEqual(status['backend'], 'onnx')
        self.assertEqual(status['device'], 'CPU')
        self.assertEqual(len(status['fallback_reasons']), 2)

    def test_registry_missing_hash_label_color_and_version_checks(self):
        manifest = registry_fixture(self.root)
        original = read_registry(manifest)
        for key,value in [('camera_role',None),('camera_role','other'),('labels',['x','x']),('colors',[[999,0,0]]),('input_size',[320,320]),('version','v2'),('confidence_threshold',float('nan'))]:
            bad = copy.deepcopy(original)
            bad['models'][0][key] = value
            manifest.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): read_registry(manifest)
        manifest.write_text(json.dumps(original))
        (self.root/'goggles-v1.onnx').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'): model_specs(self.root,manifest)
        (self.root/'goggles-v1.onnx').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'download_models'): model_specs(self.root,manifest)

    def test_multiple_enabled_models_and_disabled_entries(self):
        manifest = registry_fixture(self.root,('first-v1','second-v1'))
        self.assertEqual(len(model_specs(self.root,manifest)),2)
        document = read_registry(manifest)
        document['models'][1]['enabled'] = False
        manifest.write_text(json.dumps(document))
        self.assertEqual(len(model_specs(self.root,manifest)),1)

    def test_tensorrt_uses_registry_and_never_imports_onnxruntime(self):
        manifest = registry_fixture(self.root)
        document = read_registry(manifest)
        path = self.root/'goggles-v1.engine'
        path.write_bytes(b'test engine')
        document['models'][0]['tensorrt'] = dict(filename=path.name, sha256=sha256(path))
        manifest.write_text(json.dumps(document))
        original_import = builtins.__import__
        def no_ort(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise AssertionError('TensorRT must not import ONNX Runtime')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_ort), patch('app.detector.TensorRTEngine') as engine:
            spec = model_specs(self.root, manifest, 'tensorrt')[0]
            with Detector(spec):
                engine.assert_called_once_with(spec)
            engine.return_value.close.assert_called_once()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            model_specs(self.root, manifest, 'tensorrt')
        (self.root/'.installing').touch()
        with self.assertRaisesRegex(RuntimeError, 'installation'):
            model_specs(self.root, manifest, 'tensorrt')

    def test_cpu_dependency_has_actionable_error(self):
        manifest = registry_fixture(self.root)
        original_import = builtins.__import__
        def no_ort(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ImportError('not installed')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_ort):
            with self.assertRaisesRegex(RuntimeError, 'requirements-cpu.txt'):
                Detector(model_specs(self.root, manifest)[0])

    def test_onnx_embedded_labels_and_input_contract_rejected(self):
        import onnx
        manifest = registry_fixture(self.root,labels=['A','B'],real=True)
        document = read_registry(manifest)
        document['models'][0]['labels'] = ['B','A']
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,'metadata'): Detector(model_specs(self.root,manifest)[0])
        manifest = registry_fixture(self.root,real=True)
        path = self.root/'goggles-v1.onnx'
        model = onnx.load(str(path))
        model.graph.input[0].name = 'wrong'
        onnx.save(model,str(path))
        document = read_registry(manifest)
        document['models'][0]['onnx']['sha256'] = sha256(path)
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,'input contract'): Detector(model_specs(self.root,manifest)[0])

    def test_nonfinite_outputs_rejected(self):
        manifest = registry_fixture(self.root,real=True)
        with Detector(model_specs(self.root,manifest)[0]) as detector:
            with patch.object(detector.session,'run',return_value=[np.full((1,8400,5),np.nan,dtype=np.float32)]):
                with self.assertRaisesRegex(RuntimeError,'non-finite'): detector.detect(np.zeros((640,640,3),dtype=np.uint8))

    def test_download_checksum_atomic_replacement_and_cleanup(self):
        source,target = self.root/'source.onnx',self.root/'target.onnx'
        source.write_bytes(b'new model')
        target.write_bytes(b'working old model')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            download(source.as_uri(),target,'0'*64)
        self.assertEqual(target.read_bytes(),b'working old model')
        self.assertFalse(list(self.root.glob('*.partial')))
        with patch('tools.download_models.os.replace',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError): download(source.as_uri(),target,sha256(source))
        self.assertEqual(target.read_bytes(),b'working old model')
        download(source.as_uri(),target,sha256(source))
        self.assertEqual(target.read_bytes(),source.read_bytes())

    def test_onnx_runtime_never_imports_training_or_tensorrt(self):
        manifest = registry_fixture(self.root,real=True)
        code = "import sys,numpy as np; from app.registry import model_specs; from app.runtime import Detector; d=Detector(model_specs(sys.argv[1],sys.argv[2])[0]); assert d.detect(np.zeros((640,640,3),dtype=np.uint8)); assert not any(m in sys.modules for m in ('torch','mmcv','mmdet','tensorrt')); d.close()"
        subprocess.run([sys.executable,'-c',code,str(self.root),str(manifest)],check=True)


if __name__ == '__main__': unittest.main()
