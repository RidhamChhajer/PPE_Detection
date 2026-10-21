# PPE Detection

Detect safety goggles and footwear in recorded videos or live camera feeds using
two trained RTMDet-L models. Detections display bounding boxes, class labels and
confidence scores.

## Features

- Upload a video, review the annotated result, and download it.
- Use a face camera, a footwear camera, one shared camera, or two separate cameras.
- Automatic inference backend selection with the active backend shown in the UI.
- Local processing; uploaded videos preserve their resolution and timing. Output
  videos do not include audio.

## Requirements

- Windows x64 and Python 3.10 x64.
- Git, or a downloaded copy of the repository.
- Internet access for dependency and model downloads.
- A connected camera for live detection; no camera is needed for video uploads.

GPU acceleration requires a supported GPU and compatible drivers. CPU inference
is available when GPU acceleration is unavailable.

## Installation

Clone the repository and open PowerShell in the project folder:

```powershell
git clone https://github.com/RidhamChhajer/PPE_Detection.git
cd PPE_Detection
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-auto.txt
.\.venv\Scripts\python.exe tools/download_models.py
```

The download command installs `goggles-v2.onnx` and `footwear-v2.onnx` into
`models/`, verifies their SHA-256 checksums, and reuses existing verified files.
The release URL is configured in `configs/models.json`.

**Private repository access:** collaborators must accept their invitation. The
downloader does not authenticate with GitHub. If it returns 404, sign into GitHub,
download both ONNX files from the [v2 release](https://github.com/RidhamChhajer/PPE_Detection/releases/tag/v2.0.0),
and place them in `models/`. Training checkpoints (`.pth`) are not needed for inference.

## Run

```powershell
.\.venv\Scripts\python.exe -m app.app
```

Open **http://127.0.0.1:8765**. Keep the terminal open while using the application;
press **Ctrl+C** to stop it.

### Inference backends

Auto mode tries usable TensorRT engines on NVIDIA GPUs, then OpenVINO on Intel
GPUs, then ONNX Runtime on CPU. The interface shows the selection and fallback
reasons. Auto does not install drivers or build TensorRT engines.

To select a backend explicitly:

```powershell
# Intel GPU
.\.venv\Scripts\python.exe -m app.app --backend openvino
# CPU
.\.venv\Scripts\python.exe -m app.app --backend onnx
# NVIDIA GPU: requires TensorRT dependencies and engines built for this computer
.\.venv\Scripts\python.exe -m app.app --backend tensorrt
```

Explicit choices do not silently fall back. See
[Installation](MD%20DOC/INSTALLATION.md) and
[Export and deployment](MD%20DOC/EXPORT_AND_DEPLOYMENT.md) for GPU setup.

## Usage

### Recorded video

Select **Upload video**, choose a file, and click **Process video**. Both detectors
run on the video. The completed result can be played and downloaded in the browser.
Uploads are limited to 512 MB; uploaded inputs are removed after processing and
results expire after 24 hours.

### Live cameras

Select **Live camera**, refresh the device list, and assign the camera roles:

| Mode | Face camera | Feet camera |
| --- | --- | --- |
| Goggles only | Camera A | Not used |
| Footwear only | Not used | Camera A |
| Combined preview | Camera A | Camera A |
| Separate previews | Camera A | Camera B |

Click **Start cameras** to begin and **Stop cameras** before changing assignments
or switching to video processing. A shared camera opens once and runs both models;
separate cameras run only their assigned detector. Live detection skips older
frames to avoid a processing backlog.

## Limitations

Detection quality depends on training data, lighting, visibility and camera
placement. The application can miss PPE or produce false positives; it does not
certify compliance. Earplug detection, worker tracking and missing-PPE alerts are
not implemented. Performance must be validated on the deployment hardware.

## Documentation

- [Model results](MD%20DOC/MODELS.md)
- [Camera integration](MD%20DOC/CAMERA_INTEGRATION.md)
- [Training](MD%20DOC/TRAINING.md)
- [Development and tests](MD%20DOC/DEVELOPMENT.md)
- [Verification](MD%20DOC/VERIFICATION.md)
- [Deployment bundle setup](deployment/START_HERE.md)
