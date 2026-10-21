# PPE Detection

Local detection of visible goggles and footwear in uploaded videos or one/two live
USB camera feeds. Current models are **RTMDet-L**: goggles-v2 and footwear-v2.

For the shared code+models ZIP, start with [deployment/START_HERE.md](deployment/START_HERE.md).
Its setup script installs Auto/OpenVINO dependencies; NVIDIA engine setup is optional.

## Current setup

- Development: Lenovo LOQ, Intel Core i5-12450HX, 12 GB RAM and RTX 2050.
- Target deployment: HP EliteDesk 800 G5 TWR, RTX 3050 6 GB and 16 GB RAM.
- Intended cameras: two Arducam AR0234 USB cameras, assigned to face and feet.
  One camera also works; two connected cameras are not required.
- Goggles and footwear use separate trained RTMDet-L models. Earplug detection,
  worker tracking/IDs and missing-PPE alerts are not implemented.
- HP TensorRT configuration and sustained two-camera performance testing remain
  deferred until the target hardware is available.

## Run on the current development computer

Open **PowerShell** in the project folder. The commands below use the project's
Python environment directly; you do not need to activate it.

### Auto — default mode

Auto tests both models, preferring usable local TensorRT engines, then an Intel
GPU through OpenVINO, then ONNX Runtime CPU. It does not build engines. The page
and terminal show the selected backend/device and any fallback reasons. Install
Auto dependencies once, then start:

```powershell
Set-Location -LiteralPath 'D:\PPE DET'
.\.venv\Scripts\python.exe -m pip install -r requirements-auto.txt
.\.venv\Scripts\python.exe -m app.app
```

Keep that terminal open, then open **http://127.0.0.1:8765** in your browser.
Press **Ctrl+C** in the terminal to stop the application.

### CPU — optional ONNX mode

Install the CPU dependencies once into the same environment:

```powershell
Set-Location -LiteralPath 'D:\PPE DET'
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

Then launch CPU mode whenever needed:

```powershell
.\.venv\Scripts\python.exe -m app.app --backend onnx
```

Open the same browser address. CPU mode uses processor cores, not Intel
integrated graphics, and can be slower than GPU mode.

### Force a GPU backend

```powershell
# Intel integrated graphics (requires a supported Intel graphics driver)
.\.venv\Scripts\python.exe -m app.app --backend openvino
# NVIDIA GPU (requires locally built TensorRT engines and its dependencies)
.\.venv\Scripts\python.exe -m app.app --backend tensorrt
```

Explicit choices fail clearly instead of changing devices. OpenVINO uses the
same ONNX weights and FP32 precision; no retraining or extra model export is needed.
Auto is a compatibility policy, not a benchmark that guarantees the fastest device.
When Auto and a manual command select the same backend, device and settings,
inference follows the same path. Supplying a GPU name does not itself improve speed.
The exact GPU name is discovered automatically; compatible drivers are still needed.
Initial OpenVINO model compilation can take several seconds before previews start.

### First run without an existing Python environment

Install **Python 3.10 x64** with the Python launcher, open PowerShell in the
extracted project folder, then create the environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-auto.txt
.\.venv\Scripts\python.exe -m app.app
```

This requires both included ONNX files in models/. A source-only copy without
model files cannot run inference. NVIDIA TensorRT mode additionally needs a compatible
TensorRT installation and engines built for that computer. Do not copy the
development .venv or assume its engines will run on the HP RTX 3050. For an
extracted deployment bundle, follow its [setup guide](deployment/START_HERE.md).

TensorRT selects models/registry.local.json; OpenVINO/CPU select configs/models.json.
Only Auto falls back, with reasons shown. ONNX Runtime validates the ONNX contract
for OpenVINO; TensorRT inference itself does not use it. See [Installation](MD%20DOC/INSTALLATION.md).

## Inputs

### Upload and detect

1. Select **Upload video**, choose one video and click **Process video**.
2. Wait for processing to finish; both goggles and footwear models run on it.
3. Play the result in the browser or click **Download annotated video**.

Output is silent H.264 with the original frames, resolution and timing.
Uploads are deleted after processing; results expire after 24 hours.

### Live cameras — four arrangements

| Arrangement | Face camera selection | Feet camera selection | Detection output |
| --- | --- | --- | --- |
| Face only | Your camera | Not used | Goggles |
| Feet only | Not used | Your camera | Footwear |
| Same camera for both | Camera A | Camera A | One combined goggles + footwear preview |
| Two separate cameras | Camera A | Camera B | Goggles on A; footwear on B |

1. Connect your camera(s) and select **Live camera**.
2. Click **Refresh camera list**. Choose Face only, Feet only, the same camera
   for both, or two different cameras. **Not used** disables a role.
3. Click **Start cameras** to display the annotated preview(s).
4. Click **Stop cameras** before switching to video upload or reconnecting
   a camera.

Face-only runs goggles; feet-only runs footwear. A shared camera opens once and
runs both models on each selected original frame in one combined preview. Two
different cameras stay role-specific: goggles never runs on the feet feed. Recheck
assignments after USB reconnects because device indices can change. Earplug
inference will join the face route when a trained model is available.

Each camera has one latest-frame buffer. A single inference worker loads each
model once and processes the two roles in turn. Old live frames are skipped;
uploaded videos still process every frame. A disconnected camera clears its
preview while the other can continue. Stop cameras before reconnecting/restarting.
Live capture and upload processing are mutually exclusive.

The displayed latency measures host frame-read completion through preview
encoding. It excludes exposure, USB buffering and browser delivery. Stale
previews are hidden. Final camera modes and end-to-end latency must be measured
on the target machine.

### Common startup problems

- **Python executable not found:** create .venv using the first-run commands above.
- **Missing ONNX models:** supply models/goggles-v2.onnx and models/footwear-v2.onnx;
  installing dependencies does not install model weights.
- **TensorRT engine missing or incompatible:** configure/build engines on that
  computer later, or use Auto to try Intel graphics and then CPU. An explicit
  --backend tensorrt command does not fall back.
- **OpenVINO Intel GPU unavailable:** check the Intel graphics driver. Use Auto
  for CPU fallback, or --backend onnx to force CPU. Installing OpenVINO alone
  does not install the graphics driver.
- **DirectShow warning while previews work:** older discovery code tried unused
  camera indices, which could emit warnings without breaking active cameras.
  Current discovery enumerates Windows devices. If a selected preview fails,
  check the connection, permissions and other apps using that camera.
- **Port already in use:** stop the previous server, or run
  `.\.venv\Scripts\python.exe -m app.app --port 8766` and open
  http://127.0.0.1:8766. Add --backend onnx if using CPU mode.
- **Camera missing/disconnected:** stop cameras, close other apps using
  them, check Windows camera permissions, reconnect and refresh the list.
  Device discovery enumerates connected devices instead of probing unused indices.

## Models and development

Ordered labels, hashes and camera roles live in configs/models.json. Local engine
metadata lives in models/registry.local.json. Live mode supports either role,
one shared camera or two cameras; upload runs all enabled models. Shoe pairing remains
a display heuristic and can join the wrong shoes or leave them separate.

The release model URL is not published yet. For a verified local release bundle:

```powershell
python tools/download_models.py --base-url "file:///C:/actual/path/to/release_assets/v2"
```

See [Training](MD%20DOC/TRAINING.md), [Export/deployment](MD%20DOC/EXPORT_AND_DEPLOYMENT.md),
[Adding a detector](MD%20DOC/ADDING_A_DETECTOR.md), and
[Development](MD%20DOC/DEVELOPMENT.md). Install requirements-dev.txt and run:

```powershell
python -m unittest discover -s tests -v
```

App inference imports no training stack. Training, evaluation and export tools
remain separate from the application.

## Group deployment bundle

Extract **release_assets/industry/PPE_Detection_Windows.zip**, open PowerShell in
the extracted PPE_Detection folder, and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\setup.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\start.ps1
```

Setup installs dependencies and verifies the included models. Start uses Auto.
To force Intel graphics after setup:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\start.ps1 -Backend openvino
```

When the NVIDIA deployment computer is ready, run setup with -Backend tensorrt
**on that computer** to build and verify its engines. This step has not been
performed for the HP. See the [complete setup guide](deployment/START_HERE.md).

To build a fresh delivery from the development repository, choose a new filename:

```powershell
python tools/package_deployment.py release_assets/industry/PPE_Detection_Windows_new.zip
```

The packager refuses to overwrite an existing ZIP; it verifies included file hashes.
Share release_assets/industry/PPE_Detection_Windows.zip. It includes code, both
trained ONNX models, selected .pth checkpoints/configs, engine-verification
images and setup/launch scripts. Dependencies are installed after extraction;
the .venv and current machine's engines are excluded. Auto/OpenVINO is the default
setup. Optional TensorRT setup builds/checks engines on the recipient's GPU later.
The former source-only delivery was removed.
Original factory videos, training datasets, credentials and Git history stay local.

See [Cleanup](MD%20DOC/CLEANUP.md) and [Verification](MD%20DOC/VERIFICATION.md).
No software license has been selected, and redistribution rights still require
owner review. Nothing is committed, pushed or published automatically.

Current models can miss PPE and produce false positives. This application does
not certify compliance or match a worker across cameras. See
[Model results](MD%20DOC/MODELS.md) and [Limitations](MD%20DOC/LIMITATIONS.md).
Passing runtime/conversion tests does not establish 95% industrial detection accuracy.
