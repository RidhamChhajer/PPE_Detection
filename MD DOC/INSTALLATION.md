# Installation

Use Python 3.10 x64 and `requirements-runtime.txt`. Runtime needs no PyTorch,
MMCV, MMDetection, TensorRT, checkpoints or datasets. The owner configures the
release URL in `configs/models.json`; the downloader also accepts `--base-url`
for actual HTTPS assets or `file:///` local bundles. No release URL is invented.

## Optional ONNX CUDA

Install runtime requirements first, then replace the CPU package:

```powershell
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu==1.20.1
python -m app.app --provider cuda
```

Never install both ORT distributions together. Match CUDA/cuDNN and expose their
libraries to the OS loader. See [official installation](https://onnxruntime.ai/docs/install/)
and [CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
CPU is tested locally; ORT CUDA needs a deployment-machine smoke test. Missing
CUDA initialization raises an error; choose `--provider cpu` explicitly.

## Optional Windows training/export

Tested: Python 3.10, Torch 2.1.2/CUDA 11.8, torchvision 0.16.2, MMCV 2.1.0,
MMEngine 0.10.7, MMDetection 3.3.0.

```powershell
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl"
python -m pip install -r requirements-training.txt
python -m pip install -r requirements-export.txt
python -m pip check
```

The MMCV command is Windows/CPython 3.10-specific. Other platforms need a matching
compiled MMCV 2.1.0 build; portable requirements contain no Windows wheel URL.

Optional TensorRT on the tested Windows CUDA 12 inference stack:

```powershell
python -m pip install tensorrt-cu12==10.13.3.9 cuda-python==12.9.2 --extra-index-url https://pypi.nvidia.com
```

Build engines on the target GPU. See [Export and deployment](EXPORT_AND_DEPLOYMENT.md).
