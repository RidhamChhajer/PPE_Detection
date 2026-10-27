# AI Based PPE Monitoring & Warning

A local Windows application for camera-based personal protective equipment inspection, built with a React interface and a Python inference service. Two trained YOLO26m models inspect goggles and wrist accessories in the upper view, and footwear in the lower view.

Processing runs locally on the host PC with automatic NVIDIA GPU acceleration (CUDA 12.8) and seamless CPU fallback.

## Capabilities

- **Live monitoring**: Dual-camera inspection (upper and lower views), single camera, or test playback.
- **PPE Class Detection**: Real-time detection of goggles, no goggles, bare wrists, wrist accessories, safety shoes, and non-compliant footwear.
- **Daily Safety Compliance Dashboard**: Dynamic daily telemetry with interactive SVG charts:
  - Circular Donut Gauge for Violation Rate %
  - Sliced Pie Chart for Violation Breakdown by category
  - 24-Hour Non-Compliance Trend Bar Chart highlighting the peak violation period
  - Summary KPI cards & recent safety violation incidents log
- **Automated PDF Audit Reports**: Multi-page PDF generation via ReportLab & Matplotlib with charts, compliance statistics, and facility metadata.
- **Automated Email Dispatcher**: Background daily shift delivery via Gmail SMTP (100% free with direct PDF attachment) or EmailJS REST API, with customizable schedule time and native time picker.
- **Hardware Acceleration**: Automatic CUDA 12.8 acceleration on NVIDIA GPUs with graceful CPU fallback.
- **Image & Video Testing**: Annotation and silent H.264 video rendering for clips up to 250 MB.
- **Offline Ready**: Bundled frontend distribution (`app/static/`), fonts, and UI assets. Node.js is not required to run.

## Requirements

- **Operating System**: 64-bit Windows 10 or 11.
- **Python**: 64-bit Python 3.10 to 3.13.
- **Hardware**:
  - *Recommended*: NVIDIA GPU (GTX 1650, RTX 2050/3050/4060 or higher) for 30+ FPS live inference.
  - *Standard*: Any modern multi-core CPU (Intel Core i5/i7, AMD Ryzen) supported via automatic CPU fallback.
- **Disk Space**: ~2 GB free disk space for dependencies and models.
- **Web Browser**: Chrome, Edge, Brave, or Firefox.

## Quick Installation on Any PC

### Option A: NVIDIA GPU (Recommended)
1. Clone the repository:
   ```powershell
   git clone https://github.com/RidhamChhajer/PPE_Detection.git
   cd PPE_Detection
   ```
2. Double-click **`Setup GPU.cmd`** (or `Setup RTX 3050 6GB.cmd` for RTX 3050 machines).
   *This automatically creates `.venv`, installs PyTorch with CUDA 12.8, installs dependencies, and downloads the trained YOLO models from GitHub releases.*

### Option B: CPU Only (No NVIDIA GPU)
1. Clone the repository:
   ```powershell
   git clone https://github.com/RidhamChhajer/PPE_Detection.git
   cd PPE_Detection
   ```
2. Double-click **`Setup CPU.cmd`**.
   *This creates `.venv`, installs PyTorch CPU runtime, installs dependencies, and downloads the YOLO models.*

---

## Running the Application

1. Double-click **`Start PPE Detection System.cmd`** (or run `.\.venv\Scripts\python.exe tools/launch.py`).
2. The browser will automatically open at **`http://127.0.0.1:8876`**.
3. To configure automated daily email reports, go to the **Dashboard** page, click **Email & Settings**, enter your recipient email address, and click **Save Email Settings** (or copy `config/email_config.example.json` to `config/email_config.json`).
4. To stop the application completely, double-click **`Stop PPE Detection System.cmd`**.

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
