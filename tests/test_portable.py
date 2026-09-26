import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from app.registry import model_specs, read_registry, sha256
from app.runtime import Detector
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

    def test_registry_missing_hash_label_color_and_version_checks(self):
        manifest = registry_fixture(self.root)
        original = read_registry(manifest)
        for key,value in [('labels',['x','x']),('colors',[[999,0,0]]),('input_size',[320,320]),('version','v2'),('confidence_threshold',float('nan'))]:
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
