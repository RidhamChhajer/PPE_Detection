"""Architecture and Kaggle-to-local migration regression checks."""
import unittest
import importlib.util
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ('torch', 'mmengine', 'mmdet', 'mmcv')):
    raise unittest.SkipTest('Training dependencies are optional in the portable runtime environment')

from mmengine.config import Config
from tools.localize_run import localize_config
from tools.train import TASKS
from tools.check_app import forbidden_runtime_imports
from tools.export_models import portable_config


class LargeModelTests(unittest.TestCase):
    def test_smoke_runtime_isolation_allows_only_selected_backend(self):
        self.assertIn('tensorrt', forbidden_runtime_imports('onnx'))
        self.assertNotIn('tensorrt', forbidden_runtime_imports('tensorrt'))
        self.assertIn('torch', forbidden_runtime_imports('tensorrt'))

    def test_release_keeps_dataset_version_and_qualification(self):
        cfg = Config.fromfile(str(TASKS['footwear']['config']))
        cfg.data_regime = 'sample'
        cfg.dataset_version = 'footwear dataset v1'
        release = portable_config(cfg, 'footwear-v2')
        self.assertEqual(release.dataset_version, cfg.dataset_version)
        self.assertEqual(release.data_regime, 'sample')
        self.assertEqual(release.model.backbone.widen_factor, 1)

    def test_defaults_are_large_and_keep_negative_images(self):
        for task, info in TASKS.items():
            self.assertEqual(info['config'].name, f'{task}_rtmdet_l.py')
            cfg = Config.fromfile(str(info['config']))
            self.assertEqual(cfg.model.backbone.deepen_factor, 1)
            self.assertEqual(cfg.model.backbone.widen_factor, 1)
            self.assertEqual(cfg.model.bbox_head.num_classes, len(info['classes']))
            self.assertFalse(cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt)

    def test_localization_preserves_architecture_and_source(self):
        cfg = Config(dict(model=dict(backbone=dict(deepen_factor=1)),
                          data_regime='sample', load_from='remote.pth'))
        for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            cfg[name] = dict(num_workers=2, persistent_workers=True,
                             dataset=dict(ann_file='/kaggle/old.json',
                                          data_prefix=dict(img='/kaggle/images')))
        cfg.val_evaluator = dict(ann_file='/kaggle/old.json')
        cfg.test_evaluator = dict(ann_file='/kaggle/old.json')
        result = localize_config(cfg, Path.cwd(), 'footwear', 'work_dirs/test')
        self.assertEqual(result.model, cfg.model)
        self.assertEqual(result.data_regime, 'sample')
        self.assertEqual(cfg.val_evaluator.ann_file, '/kaggle/old.json')
        self.assertTrue(result.val_evaluator.ann_file.endswith('footwear/annotations/valid.json'))
        self.assertEqual(result.test_dataloader.num_workers, 0)
        self.assertFalse(result.resume)
        self.assertIsNone(result.load_from)
