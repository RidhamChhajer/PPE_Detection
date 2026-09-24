from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.optim import build_optim_wrapper
from mmdet.utils import register_all_modules
from mmdet.datasets.transforms import RandomCrop
from mmdet.structures.bbox import HorizontalBoxes

from tools.stable_optim import FiniteOptimWrapper
from tools.train import check_run_directory

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs, mode='loss'):
        return dict(loss_cls=[self.weight * inputs], loss_bbox=self.weight * 0)


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    def make_wrapper(self, model):
        return FiniteOptimWrapper(torch.optim.SGD(model.parameters(), lr=.1),
                                  accumulative_counts=2,
                                  clip_grad=dict(max_norm=10, norm_type=2, error_if_nonfinite=True))

    def test_both_resolved_configs_build_fp32_wrapper(self):
        for task in ('goggles', 'footwear'):
            cfg = Config.fromfile(str(ROOT/f'configs/{task}_rtmdet_s.py'))
            model = TinyModel()
            wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
            self.assertIsInstance(wrapper, FiniteOptimWrapper)
            self.assertFalse(hasattr(wrapper, 'loss_scaler'))
            self.assertEqual(wrapper.optimizer.param_groups[0]['lr'], .0001)
            self.assertTrue(cfg.optim_wrapper.clip_grad.error_if_nonfinite)
            self.assertFalse(cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt)
            self.assertTrue(next(t for t in cfg.train_pipeline if t.type == 'RandomCrop').allow_negative_crop)

    def test_real_train_step_accumulation_and_finite_zero_loss(self):
        model = TinyModel()
        wrapper = self.make_wrapper(model)
        model.train_step(dict(inputs=torch.tensor(2.0)), wrapper)
        self.assertEqual(model.weight.item(), 1.)
        model.train_step(dict(inputs=torch.tensor(2.0)), wrapper)
        self.assertAlmostEqual(model.weight.item(), .8, places=6)
        self.assertEqual(len(model._forward_hooks), 0)
        model.train_step(dict(inputs=torch.tensor(0.0)), wrapper)

    def test_nan_and_inf_component_loss_stop_before_backward(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            model = TinyModel()
            wrapper = self.make_wrapper(model)
            with self.assertRaisesRegex(FloatingPointError, 'loss_cls'):
                model.train_step(dict(inputs=torch.tensor(bad)), wrapper)
            self.assertEqual(model.weight.item(), 1.)
            self.assertIsNone(model.weight.grad)
            self.assertEqual(wrapper._inner_count, 0)
            self.assertEqual(len(model._forward_hooks), 0)

    def test_finite_loss_bad_gradient_stops_each_microbatch(self):
        for bad in (float('nan'), float('inf')):
            for failure_batch in (1, 2):
                model = TinyModel()
                wrapper = self.make_wrapper(model)
                if failure_batch == 2:
                    model.train_step(dict(inputs=torch.tensor(1.0)), wrapper)
                model.weight.register_hook(lambda gradient: torch.full_like(gradient, bad))
                with patch.object(wrapper.optimizer, 'step') as step:
                    with self.assertRaisesRegex(FloatingPointError, 'gradient'):
                        model.train_step(dict(inputs=torch.tensor(1.0)), wrapper)
                    step.assert_not_called()
                self.assertEqual(model.weight.item(), 1.)

    def test_norm_overflow_stops_before_update(self):
        model = torch.nn.Linear(2, 1, bias=False)
        wrapper = FiniteOptimWrapper(torch.optim.SGD(model.parameters(), lr=.1),
                                    clip_grad=dict(max_norm=10, norm_type=2, error_if_nonfinite=True))
        model.weight.grad = torch.full_like(model.weight, 1e30)
        with patch.object(wrapper.optimizer, 'step') as step:
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                wrapper.step()
            step.assert_not_called()

    def test_negative_image_and_crop_remain_valid(self):
        transform = RandomCrop(crop_size=(32, 32), allow_negative_crop=True)
        for boxes in ([], [[45, 45, 60, 60]]):
            results = dict(img=np.zeros((64, 64, 3), dtype=np.uint8), img_shape=(64, 64),
                           gt_bboxes=HorizontalBoxes(torch.tensor(boxes).reshape(-1, 4)),
                           gt_bboxes_labels=np.zeros(len(boxes), dtype=np.int64),
                           gt_ignore_flags=np.zeros(len(boxes), dtype=bool))
            with patch.object(transform, '_rand_offset', return_value=(0, 0)):
                output = transform(results)
            self.assertIsNotNone(output)
            self.assertEqual(len(output['gt_bboxes']), 0)

    def test_fresh_directory_guard_for_either_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ('goggles', 'footwear'):
                check_run_directory(root/'new', False, f'{task}_rtmdet_s.py')
            (root/'unrelated.pth').touch()
            with self.assertRaisesRegex(ValueError, 'not empty'):
                check_run_directory(root, False, 'footwear_rtmdet_s.py')


if __name__ == '__main__':
    unittest.main()
