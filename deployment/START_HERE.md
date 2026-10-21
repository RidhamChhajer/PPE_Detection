# Run the shared project on Windows

This bundle includes the application, both trained RTMDet-L ONNX models, selected
PyTorch checkpoints/configs, and a small engine-verification image set. It does
not include Python, installed dependencies, a virtual environment, training
datasets, original factory videos, or engines compiled on the sender's GPU.

## Start now: Auto / Intel integrated graphics / CPU

Install Python **3.10 x64** with the Python launcher. Extract the whole ZIP and
open PowerShell in PPE_Detection. Keep Internet access for dependency installation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\setup.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\start.ps1
```

Setup creates .venv, installs OpenVINO/ONNX CPU dependencies and verifies included
model inputs. Auto tests both models and uses compatible installed TensorRT
engines first, otherwise OpenVINO on a supported Intel GPU, otherwise CPU. The
terminal and page show the selection and fallback reasons. Auto does not build
engines or guarantee identical speed across different hardware. Install a supported
Intel graphics driver for iGPU use; no GPU-name configuration is required.
OpenVINO runs FP32 on the same ONNX weights. It may compile for several seconds
when starting inference. Windows may require the Visual C++ 2015-2022 x64 runtime.

Force Intel GPU: deployment/start.ps1 -Backend openvino. This reports an error,
not CPU fallback, if Intel graphics is unavailable. A normal CPU-only machine
can use the default Auto mode or the explicit CPU instructions below.

## Later: NVIDIA GPU / HP EliteDesk with RTX 3050 6 GB

1. Install Python **3.10 x64**, including the Python launcher, and an NVIDIA
   driver compatible with CUDA 12/TensorRT 10.13.3. NVIDIA documents Windows
   driver r537 or later as the minimum for this TensorRT release. Install a
   current compatible driver if setup reports that CUDA cannot initialize.
   Windows may also need the Microsoft Visual C++ 2015-2022 x64 runtime.
2. Extract the whole ZIP. Open PowerShell in the extracted PPE_Detection folder.
   Keep Internet access available for the dependency installation.
3. Run setup once **on the computer that will run inference**:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\setup.ps1 -Backend tensorrt
```

Setup creates a local .venv, installs the pinned dependencies, builds one model
at a time on that PC's GPU and compares its output with the included ONNX model
on every bundled verification image. Goggles tries FP16 with FP32 fallback if
parity fails; footwear uses FP32. Expect the first build to take several minutes.
Existing verified engines are replaced only after both new models pass checks.
ONNX Runtime is used for setup verification; normal GPU inference uses TensorRT.
No model training or MMDetection/PyTorch installation is needed to run the app.

4. Start the application:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\start.ps1 -Backend tensorrt
```

Open **http://127.0.0.1:8765**. Upload a video to run both detectors, or connect
one or two cameras, refresh the camera list, assign Face and/or Feet, and start.
Choose Not used for a disabled role. Face-only detects goggles; feet-only detects
footwear; the same camera for both opens once and displays both detectors together;
two different cameras run only their assigned detectors. Upload always runs both.
Earplugs need a trained model later.
Stop the server with Ctrl+C. Recheck camera assignments after USB reconnects.

## CPU alternative

No NVIDIA GPU or driver is needed for CPU inference:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\setup.ps1 -Cpu
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\start.ps1 -Cpu
```

CPU means processor cores, not Intel integrated graphics, and may process slowly.
The -Cpu option remains supported; -Backend onnx is equivalent.

## Contents and verification

- models/: ONNX files needed for inference. Local TensorRT engines/registry are
  created during GPU setup; they are deliberately absent from the ZIP.
- checkpoints/: the selected trained .pth files and resolved configurations for
  future training/export. Inference does not load these .pth files.
- verification/: held-out example images for conversion parity, not a training
  dataset or proof of industrial accuracy.
- deployment/bundle.json: model/input identities and dependency-setup profile.
- BUNDLE_SHA256SUMS.txt: integrity hashes for all delivered files.

If verification fails, setup stops and reports the failing model/image. Do not
lower validation tolerances to force installation. CPU mode remains available.
One/two-camera resolution/FPS and end-to-end latency still need measurement on the
actual HP/cameras. Building engines verifies conversion, not detection accuracy.

References: [NVIDIA compatibility matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html),
[TensorRT 10.13.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html).
Intel setup: [OpenVINO GPU device requirements](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html).
