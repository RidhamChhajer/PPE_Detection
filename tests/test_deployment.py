from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.detector import (postprocess, check_engine_metadata,
                          engine_identity)
from app.video import draw_detections
from tools.export_tensorrt import compare_detections, fp16_layer
from tools.export_models import atomic_copy
from tools.model_run import (best_checkpoint, saved_config, category_ids, sha256,
                             require_evaluation, toml_fields, configure_reference_precision)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_multiclass_nms_preserves_overlapping_different_classes(self):
        raw = np.array([[10,10,30,30,.9,.85], [11,11,31,31,.8,.7], [50,50,60,60,.1,.6]])
        result = postprocess(raw, (1,1), (100,100,3))
        self.assertEqual(result.shape, (3,6))
        np.testing.assert_allclose(result[:,4], [.9,.85,.6])
        np.testing.assert_array_equal(result[:,5], [0,1,1])

    def test_no_scores_above_threshold_and_bad_shape(self):
        self.assertEqual(postprocess(np.zeros((1,8400,6)),(1,1),(100,100,3)).shape, (0,6))
        with self.assertRaises(ValueError):
            postprocess(np.zeros((2,8400,6)),(1,1),(100,100,3))

    def test_degenerate_boxes_do_not_consume_detection_limit(self):
        invalid = np.tile([5,5,5,5,.99,.01], (300,1))
        raw = np.vstack([invalid, [10,10,20,20,.8,.01]])
        result = postprocess(raw,(1,1),(100,100,3))
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[0], [10,10,20,20,.8,0])

    def test_engine_identity_and_output_class_count(self):
        spec = SimpleNamespace(model_id='footwear-v2', labels=('SAFETY_SHOE', 'NOT_SAFETY_SHOE'),
                               entry={'tensorrt': {}})
        name = engine_identity(spec.model_id, spec.labels)
        check_engine_metadata(spec, (1, 8400, 6), name)
        with self.assertRaisesRegex(ValueError, 'class count'):
            check_engine_metadata(spec, (1, 8400, 5), name)
        with self.assertRaisesRegex(ValueError, 'identity'):
            check_engine_metadata(spec, (1, 8400, 6), engine_identity(spec.model_id, spec.labels[::-1]))

    def test_video_draws_actual_class_names(self):
        frame = np.zeros((120,200,3), dtype=np.uint8)
        boxes = np.array([[10,20,30,40,.9,0],[50,60,70,80,.8,1]])
        with patch('app.video.cv2.putText', wraps=__import__('cv2').putText) as draw:
            draw_detections(frame, ('SAFETY_SHOE','NOT_SAFETY_SHOE'), boxes)
            self.assertEqual([call.args[1] for call in draw.call_args_list], ['SAFETY_SHOE 0.90','NOT_SAFETY_SHOE 0.80'])
        with self.assertRaisesRegex(ValueError,'class ID'):
            draw_detections(frame, ('goggles',), boxes)

    def test_class_aware_parity_rejects_wrong_classes_counts_and_scores(self):
        expected = np.array([[1,1,10,10,.8,0],[1,1,10,10,.9,1]])
        self.assertEqual(compare_detections(expected, expected[::-1], 3,.03), (0.,0.))
        for actual in (expected[:1], expected*np.array([1,1,1,1,1,0]), expected+np.array([0,0,0,0,.1,0])):
            with self.assertRaises(RuntimeError):
                compare_detections(expected, actual, 3,.03)

    def test_threshold_crossing_fails_parity_and_stable_score_path_is_fp32(self):
        raw = np.array([[1,1,10,10,.3002046,.01]])
        rounded = raw.copy()
        rounded[0,4] = .2958397
        with self.assertRaisesRegex(RuntimeError, 'count mismatch'):
            compare_detections(postprocess(raw,(1,1),(100,100,3)),
                               postprocess(rounded,(1,1),(100,100,3)), 3,.03)
        for name in ('/backbone/Conv', '/neck/Conv', '/bbox_head/cls_convs.0.0/conv/Conv', '/bbox_head/rtm_cls.0/Conv'):
            self.assertFalse(fp16_layer(name,'stable_scores'))
        self.assertTrue(fp16_layer('/bbox_head/reg_convs.0.0/conv/Conv','stable_scores'))

    def test_fp32_keeps_regression_and_classification_full_precision(self):
        for name in ('/backbone/Conv', '/bbox_head/reg_convs.0.0/conv/Conv', '/bbox_head/rtm_cls.0/Conv'):
            self.assertFalse(fp16_layer(name, 'fp32'))
            self.assertTrue(fp16_layer(name, 'fp16'))

    def test_parity_reports_score_error_even_when_boxes_match(self):
        expected = np.array([[1, 1, 10, 10, .8, 0]])
        actual = expected.copy()
        actual[0, 4] = .9
        with self.assertRaisesRegex(RuntimeError, 'assigned score error=0.100000'):
            compare_detections(expected, actual, 3., .03)

    def test_reference_disables_both_tf32_paths(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('Optional training precision test')
        import torch
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            configure_reference_precision()
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn

    def test_category_ids_are_not_assumed_contiguous_or_one_based(self):
        document = dict(categories=[dict(id=42,name='SHOE'),dict(id=7,name='OTHER')])
        self.assertEqual(category_ids(document, ('SHOE','OTHER')), (42,7))
        with self.assertRaises(ValueError):
            category_ids(document, ('OTHER','SHOE'))

    def test_only_one_best_checkpoint_and_saved_config_selected(self):
        (self.root/'epoch_40.pth').touch()
        with self.assertRaises(RuntimeError):
            best_checkpoint(self.root)
        selected = self.root/'best_coco_bbox_mAP_epoch_28.pth'
        selected.touch()
        self.assertEqual(best_checkpoint(self.root), selected.resolve())
        (self.root/'footwear_rtmdet_l.py').touch()
        self.assertEqual(saved_config(self.root).name, 'footwear_rtmdet_l.py')
        (self.root/'best_other.pth').touch()
        with self.assertRaises(RuntimeError):
            best_checkpoint(self.root)

    def test_export_requires_successful_same_checkpoint_evaluation(self):
        expected = dict(checkpoint_sha256='new',labels=['goggles'])
        with self.assertRaises(RuntimeError):
            require_evaluation(self.root, expected)
        (self.root/'evaluation').mkdir()
        receipt = self.root/'evaluation/verified.toml'
        for status, digest in [('running','new'),('passed','old')]:
            receipt.write_text(toml_fields(dict(status=status,checkpoint_sha256=digest,labels=['goggles'])))
            with self.assertRaises(RuntimeError):
                require_evaluation(self.root, expected)
        receipt.write_text(toml_fields(dict(status='passed',**expected)))
        require_evaluation(self.root, expected)

    def test_atomic_model_copy_preserves_installed_file_on_publication_failure(self):
        source, target = self.root/'candidate.engine', self.root/'installed.engine'
        source.write_bytes(b'new')
        target.write_bytes(b'working')
        with patch('tools.export_models.os.replace', side_effect=OSError('publication failed')):
            with self.assertRaises(OSError):
                atomic_copy(source, target)
        self.assertEqual(target.read_bytes(), b'working')
        self.assertFalse(list(self.root.glob('*.pending')))
        atomic_copy(source, target)
        self.assertEqual(target.read_bytes(), b'new')


if __name__ == '__main__':
    unittest.main()
