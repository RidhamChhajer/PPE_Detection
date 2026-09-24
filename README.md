# PPE Detection

Local Windows PPE detection project using RTMDet-S and MMDetection. Scope and
phase acceptance criteria are defined in [IMPLEMENTATION.md](IMPLEMENTATION.md).
Phase 1 passed on 2026-09-23: CUDA and COCO-pretrained RTMDet-S inference were
verified. See [ENVIRONMENT.md](ENVIRONMENT.md) for results.
Phase 2 also passed on 2026-09-23: all 90 sample images and 138 goggles boxes
were validated, normalized, and loaded through MMDetection. See
[DATASET.md](DATASET.md) for results, generated files, and split limitations.
Phase 3's sample training and evaluation are complete; see [TRAINING.md](TRAINING.md).
The current weights are demo-only. Independent datasets from additional recordings
are required for real-world qualification; see [DATA_COLLECTION.md](DATA_COLLECTION.md).
Phases 4–6 are complete: TensorRT conversion, full-video processing, and the
local upload/player/download application are verified. See [VIDEO.md](VIDEO.md)
for acceptance results and [DELIVERY.md](DELIVERY.md) for the file inventory.
A successful smoke test does not establish goggles detection accuracy.

## Run the application

From PowerShell in `D:\PPE DET`:

```powershell
.\.venv\Scripts\python.exe -m app.app
```

Open **http://127.0.0.1:8765**, choose one video, and click **Process video**.
All installed engines run automatically. After complete-output verification,
play or download the silent annotated MP4. Ctrl+C stops the local server.
Use `--port 8766` if the default port is occupied.

The server binds only to this computer. It accepts one job at a time, up to
512 MiB per upload. Input must have one video stream, valid increasing
timestamps, even dimensions (minimum 16 pixels per side), at most 8,294,400
pixels, and frame rate at most 240 FPS. Rotation metadata must be normalized
before upload. Constant and variable frame timing are supported. Invalid or
unsupported input returns an error instead of a partial result.

Uploads are deleted after processing. Results are available for 24 hours while
the server runs; expired files are cleaned on requests/new uploads/startup.
Download a result before restarting the server, which clears job metadata.
Runtime files live in `artifacts/app/`. No detection JSON/CSV or database is written.

The installed goggles engine is **demo-only**. It has no fixed factory ROI,
worker-specific rules, or dependency on the sample video path. A reusable
pipeline cannot compensate for limited training data. Independent recordings
from your team must be trained and evaluated before real-world accuracy can be
claimed. Safety shoes and earplugs remain pending their datasets and camera setup.

## Replace the demo model when data arrives

Follow [DATA_COLLECTION.md](DATA_COLLECTION.md) for COCO labels, independent
source groups, reviewed negatives, and configurable training/evaluation commands.
Then convert the evaluated run:

```powershell
.\.venv\Scripts\python.exe tools/export_tensorrt.py --run-dir work_dirs/goggles_independent
```

Stop the app before publishing a replacement engine, then restart it.
Conversion verifies numerical parity before installing the engine and provenance
file. This does not certify field accuracy. Rebuild and verify TensorRT engines
on the RTX 3050 target instead of copying the RTX 2050 engine.

## Phase 1 environment

Use Windows x64 and Python 3.10.11 with a single project environment, `.venv`.
The selected stack is PyTorch 2.1.2 + CUDA 11.8, torchvision 0.16.2, MMCV 2.1.0,
MMEngine 0.10.7, and MMDetection 3.3.0. NumPy 1.26.4 preserves compatibility
with these compiled extensions. OpenCV is pinned to 4.10.0.84.

Selection references:

- [Official PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/)
- [MMDetection compatibility requirements](https://mmdetection.readthedocs.io/en/main/notes/faq.html)
- [MMCV CUDA 11.8 / PyTorch 2.1 Windows wheels](https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html)
- [Official RTMDet model definitions and weights](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/metafile.yml)

From PowerShell in the project directory, create the environment only if it does
not already exist, then install in this order. Do not run pip installations concurrently.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.3
.\.venv\Scripts\python.exe -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe tools/check_environment.py
```

Run subsequent commands with `.venv\Scripts\python.exe`; activation is optional.
The pip update avoids a dependency-name normalization error in the Python
installation's bundled pip 23.0.1 when using the PyTorch wheel index.
For the full requirements including TensorRT, add
`--extra-index-url https://pypi.nvidia.com` to the requirements installation command.
The first verification downloads the official MMDetection demo image and RTMDet-S
checkpoint. The checkpoint's SHA256 must match its published filename hash prefix.
Later runs reuse these files. Use only trusted model checkpoints.

The driver-reported CUDA version is the maximum CUDA level supported by the
driver, not the locally installed toolkit. PyTorch's selected wheel supplies its
CUDA 11.8 runtime. A separate CUDA toolkit/compiler is not required for Phase 1
because MMCV uses a precompiled Windows wheel. TensorRT 10.13.3.9 / CUDA 12
was installed during Phase 4; see [CONVERSION.md](CONVERSION.md) for its separate
inference runtime and verification process.

## Verification

```powershell
# Inventory without running inference (does not satisfy Phase 1 acceptance):
.\.venv\Scripts\python.exe tools/check_environment.py --inventory-only

# Verify on the first decoded frame of the supplied factory video:
.\.venv\Scripts\python.exe tools/check_environment.py --video "data/sample_media/factory_video.mp4"

# Or supply a local image containing a COCO object, such as a person:
.\.venv\Scripts\python.exe tools/check_environment.py --image "path/to/image.jpg"
```

The verifier returns a nonzero exit code on failure, requires GPU execution,
checks CUDA NMS, verifies finite boxes and confidence scores, and requires at
least one detection with confidence >= 0.30. It saves an annotated image and a
plain-text environment report under `artifacts/phase1/`. These are diagnostic
artifacts, not production inference outputs. First-inference timing includes
initialization overhead and is not a performance benchmark.

Generated models, videos, caches, the environment, and project-owned sample
assets under `data/` are excluded from Git. The application has no dependency
on the user-managed `Previous/` directory.

## Phase 2 dataset validation

```powershell
.\.venv\Scripts\python.exe tools/validate_coco.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The copied sample export is stored in `data/sample_goggles`. Cleaned COCO
annotations, MMDetection ground-truth preview sheets, and a text report are
generated under `artifacts/phase2`. The only class is `goggles` (COCO ID 1,
model label 0). These sample splits share a source video and are suitable only
for proving the pipeline; final evaluation requires independent source groups.
