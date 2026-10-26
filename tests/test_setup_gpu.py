import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.setup_gpu import verify_gpu


@pytest.mark.parametrize(
    "name,memory,available,target,error",
    [
        ("NVIDIA GeForce RTX 3050", 6, True, True, None),
        ("NVIDIA GeForce RTX 3050", 4, True, True, "Expected RTX 3050"),
        ("NVIDIA GeForce RTX 2050", 4, True, True, "Expected RTX 3050"),
        ("NVIDIA GeForce RTX 2050", 4, True, False, None),
        ("NVIDIA GeForce RTX 3050", 6, False, True, "CUDA unavailable"),
    ],
)
def test_gpu_setup_checks_target_and_calculation(
    monkeypatch, name, memory, available, target, error
):
    calculations = []

    def ones(shape, device):
        assert device == "cuda:0"
        calculations.append(shape)
        return np.ones(shape)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: available,
                get_device_properties=lambda _: SimpleNamespace(
                    name=name, total_memory=memory * 1024**3
                ),
                synchronize=lambda: None,
            ),
            ones=ones,
            version=SimpleNamespace(cuda="test"),
        ),
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            verify_gpu(target)
        assert calculations == []
    else:
        verify_gpu(target)
        assert calculations == [(32, 32)]
