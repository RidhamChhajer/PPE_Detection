# AI Based PPE Monitoring & Warning

A local Windows application for camera-based personal protective equipment inspection, built with a React interface and a Python inference service. Two trained YOLO26m models inspect goggles and wrist accessories in the upper view, and footwear in the lower view.

Processing runs on the local PC. An NVIDIA GPU is required for inference; there is no automatic CPU fallback.

## Capabilities

- Live monitoring with separate cameras, one shared camera, or one enabled view.
- Detection of goggles, no goggles, bare wrists, wrist accessories, safety shoes, and other shoes.
- Camera selection, supported driver picture controls, detection thresholds, and optional wrist SAHI slicing.
- Stable, stacked camera panels with full-frame previews and per-camera fullscreen.
- Image and video testing, with annotated, silent H.264 video playback and download.
- Bundled frontend assets, fonts, and logos for offline operation after setup.

Earplug detection, worker counting, violation statistics, audible warnings, and inspection history are not implemented. Their interface placeholders do not represent working detectors or recorded production data. The Sample dashboard uses illustrative data.

## Requirements

- Windows x64 and 64-bit Python 3.10, the version used for local testing. The setup script accepts Python 3.10–3.13.
- An NVIDIA GPU and a driver compatible with PyTorch's CUDA 12.8 runtime.
- Internet access and several GB of available disk space for initial installation.
- A browser and suitable cameras for live monitoring. Cameras are not needed for media tests.

Node.js is needed only when developing or rebuilding the frontend. Local development used an RTX 2050 with 4 GB VRAM; performance on other hardware must be measured.

## Installation

Clone the repository and run these commands in PowerShell:

```powershell
git clone https://github.com/RidhamChhajer/PPE_Detection.git
cd PPE_Detection
py -3.10 tools/setup_gpu.py
.\.venv\Scripts\python.exe tools/download_models.py
```

The setup script creates `.venv`, installs PyTorch 2.9.1 and torchvision 0.24.1 from the CUDA 12.8 wheel index, installs `requirements.txt`, and verifies a real GPU calculation. It also creates a Windows shortcut. It does not install NVIDIA drivers.

The model download command installs `upper.pt` and `lower.pt` into `models/` from the [YOLO Model release](https://github.com/RidhamChhajer/PPE_Detection/releases/tag/yolo-model-v1). It verifies the SHA-256 checksums in [`models/manifest.json`](models/manifest.json), skips files that already match, and refuses to overwrite a different local model. Both weights are release assets rather than files stored in Git history. The downloader uses only the Python standard library.

For a factory PC specifically equipped with an RTX 3050 6 GB, `Setup RTX 3050 6GB.cmd` additionally checks the GPU name and available VRAM. For another supported NVIDIA GPU, use `Setup GPU.cmd` or the Python setup command above.

### Manual dependency installation

If you prefer to install dependencies yourself:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from tools.setup_gpu import verify_gpu; verify_gpu()"
.\.venv\Scripts\python.exe tools/download_models.py
```

Installing `requirements.txt` alone is not the complete CUDA setup. Install the pinned CUDA-enabled PyTorch packages first.

## Run

Double-click `Start PPE Detection System.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe tools/launch.py
```

Open **http://127.0.0.1:8876** if the browser does not open automatically. In **Settings**, select the upper and lower cameras, save, and select **Start detection**. Enable desktop camera access in Windows and close other applications using the devices.

Use **Stop detection** to release the cameras. Closing the browser leaves the service running. Double-click `Stop PPE Detection System.cmd` to stop the service completely.

For a foreground service that stops with Ctrl+C:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8876 --no-access-log
```

## Usage

### Live monitoring

The upper view inspects goggles and wrists; the lower view inspects footwear. Camera capture, preview publishing, and inference run independently. Stale, disconnected, and stopped observations clear the displayed detection evidence.

Camera panels keep their proportions when capture resolution changes. The full frame fits inside each panel without cropping; borders may appear. Browser zoom and small screens use normal scrolling. GPU information and per-model inference timings are available in **Settings > Inference & hardware**.

Stop detection before saving settings. Driver picture controls appear only when supported. Resolution and frame rate are requests; check the actual values after connection. Confidence thresholds and SAHI settings require calibration on representative footage.

### Image and video tests

**Test media** accepts JPEG, PNG, and supported video files up to **250 MB**, using the upper, lower, or both detectors. Video tests process every frame and produce an annotated MP4 without audio. Exports are limited to **1 GB**.

Uploads and exports are temporary. Download results before removing a test, starting another test, or stopping the service. Media tests do not populate worker or violation counters.

## Models and limitations

| Checkpoint | Architecture | Classes |
| --- | --- | --- |
| `upper.pt` | YOLO26m | `goggles`, `no_goggles`, `empty_wrist`, `not_empty_wrist` |
| `lower.pt` | YOLO26m | `SAFETY_SHOE`, `NOT_SAFETY_SHOE` |

The application checks model class names when loading. Generic COCO checkpoints and the older RTMDet ONNX/PTH models cannot replace these weights. To install different trusted weights, stop the service, replace the files, and restart it.

Supplied training reports contain these **validation-set** results, recorded in [`models/metadata.json`](models/metadata.json):

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | ---: | ---: | ---: | ---: |
| Upper | 0.745 | 0.709 | 0.734 | 0.358 |
| Lower | 0.932 | 0.863 | 0.934 | 0.705 |

These are not factory acceptance results or a certification of safety. No independent held-out test report was supplied with these checkpoints. Validate missed detections, false positives, and processing speed with the final cameras, lighting, movement, equipment, and both views operating together.

**No prediction means “Not visible”; it does not establish compliance.** Results describe visible scene evidence. The application does not identify employees, associate every PPE item with an individual, or establish that detected equipment is factory-approved or correctly worn. It does not control an access gate or provide a safety interlock.

## Troubleshooting

| Issue | Action |
| --- | --- |
| CUDA setup fails | Check the NVIDIA driver, internet connection, Python version, and available disk space; rerun setup after resolving the error. |
| Models are missing | Run `tools/download_models.py` with the project Python environment. |
| Model checksum differs | Preserve or move the existing weights before downloading the release version; the downloader will not overwrite them. |
| Camera is unavailable | Check its assignment, Windows camera permissions, and competing camera applications. Stop detection before refreshing controls. |
| Detection is intermittent | Check the model status, lighting, focus, target size, and confidence settings. Empty results do not establish compliance. |
| Processing is slow | Review inference timings in Settings. Reduce SAHI workload or input size, then revalidate detection quality. |
| Application does not start | Inspect `TEMPORARY/server.log` and check whether port 8876 is already in use. |

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check app tools tests
npm --prefix frontend ci
npm --prefix frontend run lint
npm --prefix frontend run build
.\.venv\Scripts\python.exe tests/browser_check.py
```

The browser check requires Microsoft Edge and uses an isolated service with simulated cameras and detector outputs. It exercises the real frontend, API, video encoding, and downloads; it does not measure field detection accuracy. Build output is written to `app/static/` and included in the repository, so normal installation does not require Node.js.

| Directory | Contents |
| --- | --- |
| `app/` | API, camera capture, inference, media processing, and compiled frontend |
| `frontend/` | React source, build configuration, and original assets |
| `models/` | Model manifest and locally downloaded weights |
| `config/` | Local camera settings; device assignments are excluded from Git |
| `tools/` | Setup, model downloads, launch/stop, and optional training utilities |
| `tests/` | Automated regression and browser checks |

`tools/dataset.py` prepares separate upper/lower YOLO datasets from reviewed COCO annotations. `tools/train.py` and the Kaggle notebooks support optional training and evaluation. Training data, recordings, local environments, and temporary outputs are not included in the repository. `DESIGN.md` and `UX-CONTRACT.md` document the maintained interface conventions.

## Previous releases

The earlier RTMDet implementation remains available in the repository history and the existing [v2.0.0 release](https://github.com/RidhamChhajer/PPE_Detection/releases/tag/v2.0.0). Use its matching source and models together. This YOLO version uses a separate model release and does not modify the older release assets.
