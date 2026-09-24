"""Fail-fast FP32 MMEngine optimizer wrapper, including accumulation batches."""
from contextlib import contextmanager

import torch
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


def require_finite(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f'{name}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f'{name}[{index}]')
    elif isinstance(value, torch.Tensor) and not torch.isfinite(value.detach()).all():
        raise FloatingPointError(f'Non-finite {name}; training stopped before optimizer update.')


@OPTIM_WRAPPERS.register_module()
class FiniteOptimWrapper(OptimWrapper):
    """Use MMEngine's FP32 path; never skip or sanitize invalid updates."""

    @contextmanager
    def optim_context(self, model):
        # MMDetection BaseModel calls forward(mode='loss') inside this context.
        # Check individual FPN loss tensors before parse_losses reduces them.
        def check_losses(module, inputs, output):
            if isinstance(output, dict):
                for name, value in output.items():
                    if 'loss' in name:
                        require_finite(value, name)

        handle = model.register_forward_hook(check_losses)
        try:
            with super().optim_context(model):
                yield
        finally:
            handle.remove()

    def backward(self, loss, **kwargs):
        require_finite(loss, 'total loss')
        super().backward(loss, **kwargs)
        # Must run every microbatch, not just the accumulation/update boundary.
        for group_index, group in enumerate(self.optimizer.param_groups):
            for index, parameter in enumerate(group['params']):
                if parameter.grad is not None:
                    require_finite(parameter.grad, f'gradient group={group_index} parameter={index} microbatch={self._inner_count}')

