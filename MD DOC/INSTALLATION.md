# Installation

Windows x64 and Python 3.10 are the development platform. Start a fresh environment
on each machine; do not copy the development .venv.

## Auto (default) and Intel integrated graphics

Install Auto dependencies and launch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-auto.txt
.\.venv\Scripts\python.exe -m app.app
```

Auto checks both models: TensorRT/local registry first, OpenVINO Intel GPU next,
then ONNX CPU. The page and terminal disclose the actual selection and fallback
reasons. It does not build TensorRT engines. A supported Intel graphics driver
is required for OpenVINO GPU; the exact graphics name is discovered automatically.
OpenVINO reuses the verified ONNX assets in FP32, with unchanged postprocessing.

Explicit choices never silently fall back:

```powershell
.\.venv\Scripts\python.exe -m app.app --backend openvino
.\.venv\Scripts\python.exe -m app.app --backend openvino --device CPU
.\.venv\Scripts\python.exe -m app.app --backend tensorrt
```

TensorRT additionally needs CUDA bindings, compatible NVIDIA drivers and locally
built engines. The group ZIP setup defaults to Auto; use deployment/setup.ps1
-Backend tensorrt on the recipient NVIDIA PC to build its engines when ready.

**HP EliteDesk 800 G5 TWR / RTX 3050 6 GB setup is deferred.** The driver/runtime
combination, engine builds, camera modes and two-feed performance will be verified
on that computer later. Existing development engines are not qualified for it.
No HP configuration or engine conversion is performed by the application.

## Optional CPU mode

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe tools/download_models.py --base-url "file:///C:/actual/path/to/release_assets/v2"
.\.venv\Scripts\python.exe -m app.app --backend onnx
```

The release URL is unset until publication; supply a real verified asset source.
CPU mode selects configs/models.json automatically. It uses CPU cores, not Intel
graphics. The former ONNX CUDA provider option has been removed. Auto dependencies
include ONNX Runtime for fallback and for OpenVINO's ONNX identity/contract check.

## Optional training/export

Training uses Python 3.10, Torch 2.1.2/CUDA 11.8, torchvision 0.16.2,
MMCV 2.1.0, MMEngine 0.10.7 and MMDetection 3.3.0 in the existing environment.

```powershell
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl"
python -m pip install -r requirements-export.txt
python -m pip check
```

The MMCV wheel command is specific to Windows/CPython 3.10 and that Torch/CUDA
build. These historical development versions are not a new HP deployment
recommendation. Export verification includes ONNX CPU dependencies; training
alone does not require ONNX Runtime. See [Training](TRAINING.md) and
[Export/deployment](EXPORT_AND_DEPLOYMENT.md).
