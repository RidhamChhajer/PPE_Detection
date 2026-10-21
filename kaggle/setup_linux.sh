#!/usr/bin/env bash
set -euo pipefail

# Run in a Kaggle GPU Notebook with Internet enabled. Uses a separate environment
# so Kaggle's preinstalled Python 3.12/Torch packages are not changed.
ENV_DIR=/kaggle/working/ppe-py311
if [ ! -x "$ENV_DIR/bin/python" ]; then
  CURRENT_PY=$(command -v python3 || command -v python)
  "$CURRENT_PY" -m pip install --no-cache-dir uv
  UV=$(command -v uv || true)
  if [ -z "$UV" ]; then
    UV=$($CURRENT_PY -c 'import sysconfig; print(sysconfig.get_path("scripts") + "/uv")')
  fi
  if [ ! -x "$UV" ]; then
    echo 'uv installation did not provide an executable.' >&2
    exit 1
  fi
  "$UV" python install 3.11
  "$UV" venv --python 3.11 --seed "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --no-cache-dir 'setuptools==80.9.0'
"$PY" -m pip install --no-cache-dir 'numpy==1.26.4'
"$PY" -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install --no-cache-dir 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl'
"$PY" -m pip install --no-cache-dir 'mmengine==0.10.7' 'mmdet==3.3.0' 'opencv-python-headless==4.10.0.84' 'scipy==1.15.3'
# Some Kaggle image packages pull NumPy 2.x as transitive dependencies. Re-pin
# after the full stack to preserve the NumPy 1.x ABI used by Torch/MMCV wheels.
"$PY" -m pip install --no-cache-dir --force-reinstall 'numpy==1.26.4' 'setuptools==80.9.0'
"$PY" -m pip check
"$PY" -c 'import pkg_resources, numpy, torch, mmcv, mmengine, mmdet; from mmcv.ops import nms; assert numpy.__version__ == "1.26.4", numpy.__version__; assert torch.cuda.is_available(), "CUDA is unavailable"; print("NumPy",numpy.__version__,"Torch",torch.__version__,"CUDA",torch.version.cuda,"GPU",torch.cuda.get_device_name(0),"MMCV",mmcv.__version__,"MMDetection",mmdet.__version__)'
