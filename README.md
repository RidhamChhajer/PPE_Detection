# PPE Video Review

A local application for detecting visible goggles and footwear in uploaded videos.
Versioned ONNX models run directly: users need no dataset, training environment
or GPU. Output is silent H.264 with the original frames, resolution and timing.

Current detectors: **goggles-v1** (`goggles`) and **footwear-v1** (`SAFETY_SHOE`,
`NOT_SAFETY_SHOE`). They remain separate because their datasets are not fully
annotated for each other's classes.

## Quick start

Windows 10/11 x64 and Python 3.10 are the tested platform. The ONNX CPU runtime
also supports compatible Linux/macOS installations. Windows may require the
Microsoft Visual C++ runtime.

1. Clone `https://github.com/RidhamChhajer/PPE_Detection.git`; open its folder.
2. Create an environment and install runtime dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

3. Download verified models:

```powershell
.\.venv\Scripts\python.exe tools/download_models.py
```

The owner configures `release_base_url` in [configs/models.json](configs/models.json)
once a release exists. It is intentionally blank until publication. For a local
release bundle, pass `--base-url "file:///C:/actual/path/to/release_assets/v1"`.
For another host, pass its actual HTTPS release-asset directory. Downloads are
SHA256-verified and installed atomically.

4. Start the frontend:

```powershell
.\.venv\Scripts\python.exe -m app.app
```

5. Open **http://127.0.0.1:8765**, select a video and click **Process video**.
   All enabled detectors run. Play/download the result. Ctrl+C stops the server.

## Architecture

A registry supplies model versions, ordered labels, colors, thresholds and
artifact hashes. Shared preprocessing and class-aware NMS serve both backends.
The video service consumes structured detections. Adding a compatible detector
does not require changes to the frontend or video loop.

ONNX Runtime CPU is the default. Optional ONNX CUDA setup is in
[Installation](MD%20DOC/INSTALLATION.md). TensorRT is an optional local build,
selected with `--backend tensorrt --registry models/registry.local.json`.
Hardware-specific engines are never portable release assets.

## Training, adding detectors and upgrading

Training is optional. Install the training dependencies and prepare COCO splits:
see [Training](MD%20DOC/TRAINING.md) and [Datasets](MD%20DOC/DATASETS.md).

```powershell
python tools/train.py --task goggles --work-dir work_dirs/goggles-v2 --model-version v2 --dataset-version v2
```

Use `--pretrained path/to/goggles-v1.pth` to fine-tune the existing detector.
Omit it to initialize from COCO weights. `--resume` continues an interrupted run
with its optimizer and unchanged dataset. Random initialization requires a custom
config and usually substantially more data.

Evaluate/export a new immutable version, install verified files, update the
registry with the app stopped, then restart and smoke-test. Keep the prior version
for rollback. See [Export and deployment](MD%20DOC/EXPORT_AND_DEPLOYMENT.md).
An earplug model can be added through one compatible model/registry entry; see
[Adding a detector](MD%20DOC/ADDING_A_DETECTOR.md).

## Repository and tests

- `app/`: frontend, jobs, video, registry and inference.
- `configs/`: manifest and training recipes.
- `tools/`: download, validation, evaluation, export and verification.
- `tests/`: offline CPU tests plus optional training-stack tests.
- `MD DOC/`: permanent guides and reports.
- `models/`, `data/`, `work_dirs/`, `artifacts/`, `local/`, `release_assets/`:
  local-only files excluded from Git.

Install `requirements-dev.txt`; run `python -m unittest discover -s tests -v`.
Training-specific tests skip when the optional stack is absent. See
[Development](MD%20DOC/DEVELOPMENT.md) and [Verification](MD%20DOC/VERIFICATION.md).

## Roadmap and limitations

Next: earplug detection and two-camera input. See
[Camera integration](MD%20DOC/CAMERA_INTEGRATION.md).
Performance depends on training data and recording conditions. Current models
miss objects and produce false positives. This is not a safety certification
system. See [Model results](MD%20DOC/MODELS.md) and [Limitations](MD%20DOC/LIMITATIONS.md).

## License and publication

A software license has not been selected. MIT is a possible choice requiring
owner approval; no license grant is implied. Model/data redistribution rights
also require review. The configured Git remote identifies the repository; its
portable-model release has not been published and the release URL remains unset.
Source and release bundles are prepared locally; nothing is pushed automatically.
