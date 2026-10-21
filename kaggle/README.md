# Train RTMDet-L on Kaggle

Upload `ppe-rtmdet-l-kaggle.zip` from `artifacts/kaggle` as a **private Kaggle
Dataset**, attach it to a new GPU Notebook, and enable Internet in Notebook
settings. The package contains only training code and the current goggles and
footwear COCO datasets. It does not contain videos, installed models, TensorRT
engines, training logs, previous runs or your separate backup ZIP.

Kaggle mounts attached datasets read-only. In the notebook, run these cells:

```python
from pathlib import Path
from zipfile import ZipFile
matches = list(Path('/kaggle/input').rglob('ppe-rtmdet-l-kaggle.zip'))
assert len(matches) == 1, f'Attach exactly one training package; found {matches}'
project = Path('/kaggle/working/ppe-rtmdet-l')
with ZipFile(matches[0]) as archive:
    project.mkdir(parents=True, exist_ok=True)
    archive.extractall(project)
print(project)
```

```python
import subprocess
subprocess.run(['bash', '/kaggle/working/ppe-rtmdet-l/kaggle/setup_linux.sh'], check=True)
```

The setup uses `uv` to install an isolated Python 3.11 environment because
current Kaggle images use Python 3.12 and may not include Conda. The isolated
version is required by the pinned prebuilt MMCV wheel. It also pins setuptools
80.9.0 because MMEngine 0.10.7 still imports `pkg_resources`.
After installing MMDetection it force-pins NumPy 1.26.4 again, since the
compiled Torch/MMCV wheels require the NumPy 1.x ABI. The runner sets
Matplotlib's backend to `Agg` to avoid inheriting Kaggle's notebook-inline
backend into its subprocess.

```python
import subprocess
python = '/kaggle/working/ppe-py311/bin/python'
script = '/kaggle/working/ppe-rtmdet-l/kaggle/run_training.py'
subprocess.run([python, script, '--task', 'goggles', '--check-only'], check=True)
subprocess.run([python, script, '--task', 'goggles'], check=True)
```

Replace `goggles` with `footwear` to train that detector in a separate run.
Do not run both simultaneously on one GPU. Results appear in
`/kaggle/working/goggles-rtmdet-l-v2` or `footwear-rtmdet-l-v2`.
Download the best checkpoint **and its saved configuration** after training.
Never choose `epoch_40.pth` simply because it is the last epoch. Use the best
validation checkpoint, then evaluate it on the untouched test split.

The recipe uses official RTMDet-L COCO initialization, FP32, a batch of one
with four-step accumulation, gradient clipping and immediate non-finite checks.
It retains negative images and negative random crops. Existing RTMDet-S PPE
checkpoints cannot initialize RTMDet-L layer-for-layer because the architecture
differs; the COCO RTMDet-L weights are the correct starting point.

The included data is the current **dataset v1**. The new RTMDet-L run is named
`v2-l` to distinguish model architecture. If the Notebook session ends, save
the `/kaggle/working` outputs as a Notebook version and reattach them before
using `--resume`; an empty new session cannot resume automatically.

Current datasets are small, especially the negative and test sets. A larger
network may overfit or be slower without improving real-world accuracy. Do not
replace the deployed v1 engines or ONNX files until independent evaluation and
deployment parity pass. Kaggle GPU types, session limits, package versions and
Internet availability can change; setup checks the installed stack and fails
before training if CUDA or compiled MMCV is missing.
