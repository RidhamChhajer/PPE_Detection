import unittest
import hashlib
from pathlib import Path
import tempfile
import numpy as np
from app.detector import preprocess, postprocess, model_specs, MEAN, STD


class DetectorMathTests(unittest.TestCase):
    def test_bgr_normalization_and_bottom_padding(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:] = [10, 20, 30]
        tensor, scales = preprocess(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(scales, (3.2, 3.2))
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 0, 0], (np.array([10,20,30])-MEAN)/STD, rtol=1e-6)
        np.testing.assert_allclose(tensor[0, :, 639, 0], (114-MEAN)/STD, rtol=1e-6)

    def test_nms_keeps_different_workers(self):
        raw = np.array([[10, 10, 30, 30, .9], [11, 11, 31, 31, .8], [50, 10, 70, 30, .7]])
        result = postprocess(raw, (1, 1), (100, 100, 3))
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[:, 4], [.9, .7])

    def test_rescaling_clipping_and_threshold(self):
        raw = np.array([[0, 0, 220, 60, .8], [0, 0, 20, 20, .1]])
        result = postprocess(raw, (2, 2), (50, 100, 3))
        np.testing.assert_allclose(result, [[0, 0, 100, 30, .8, 0]])

    def test_empty_and_nonfinite(self):
        self.assertEqual(postprocess(np.zeros((1, 8400, 5)), (1,1), (10,10,3)).shape, (0,6))
        with self.assertRaises(RuntimeError):
            postprocess(np.array([[0,0,1,1,np.nan]]), (1,1), (10,10,3))

    def test_model_provenance_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / 'goggles.engine'
            engine.write_bytes(b'example')
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                model_specs(root)
            digest = hashlib.sha256(engine.read_bytes()).hexdigest()
            (root / 'goggles.toml').write_text(f'label="goggles"\nstatus="sample"\nengine_sha256="{digest}"\n')
            self.assertEqual(model_specs(root)[0].status, 'sample')
            engine.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                model_specs(root)


if __name__ == '__main__':
    unittest.main()
