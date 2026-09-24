# TensorRT conversion

The selected best checkpoint is exported through ONNX to a TensorRT FP16 engine.
The runtime uses TensorRT, NumPy, OpenCV, and CUDA driver bindings; it does not
import PyTorch. PyTorch remains necessary for training, evaluation, and export.

## Pinned conversion stack

- ONNX 1.16.2, opset 17.
- TensorRT CUDA 12 package 10.13.3.9.
- CUDA Python 12.9.2 (driver API).
- Existing driver 581.86 supports the CUDA 12 inference stack alongside the
  PyTorch CUDA 11.8 training runtime. No system toolkit replacement is needed.

Official references: [TensorRT installation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-pip.html),
[10.13.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html),
[CUDA Python 12.9.2](https://pypi.org/project/cuda-python/12.9.2/).

```powershell
.\.venv\Scripts\python.exe -m pip install onnx==1.16.2 tensorrt-cu12==10.13.3.9 cuda-python==12.9.2 --extra-index-url https://pypi.nvidia.com
.\.venv\Scripts\python.exe tools/export_tensorrt.py --run-dir work_dirs/goggles_independent
# Existing sample: preserve explicit demo status throughout conversion.
.\.venv\Scripts\python.exe tools/export_tensorrt.py --run-dir work_dirs/goggles_rtmdet_s --sample
```

The command environment used by the coding tool omitted Windows's
`PROCESSOR_ARCHITECTURE`, causing NVIDIA's installer to reject the machine.
For that environment only, it was supplied as `AMD64` after confirming the
Python executable is Windows x64. A normal Windows terminal supplies this value.

## Contract and acceptance

- Input: float32 BGR-normalized NCHW `[1, 3, 640, 640]`; bilinear resize with
  aspect ratio preserved, right/bottom padding of 114, then the model's mean/std.
- Output: float32 `[1, 8400, 5]`, with decoded `x1,y1,x2,y2,confidence`.
- Internal builder FP16 enabled; float32 IO retained for stable postprocessing.
- Confidence 0.30, NMS IoU 0.65, maximum 300 detections, original-frame scaling.
- Workspace cap: 512 MiB. Engine building runs separately from ONNX export to
  release PyTorch GPU allocations first.
- Verify every test image against the PyTorch export adapter and native MMDetection.
- Limits: raw box difference <= 2 input pixels and confidence difference <= 0.03
  for candidates with reference confidence >= 0.10; exact detection count match
  at 0.30. The adapter must agree with native MMDetection boxes within 0.05 pixels.
- Final post-NMS boxes must match one-to-one within 3 original-frame pixels and
  confidence within 0.03; candidate count equality alone is insufficient.
- Latency: 50 warmed engine calls, including CPU/GPU copies, reported as median
  and p95. This is not full-video application latency.

Candidates remain in `artifacts/phase4/`. Only a verified engine is published to
`models/goggles.engine`; the verification report records checkpoint and engine
SHA256 hashes. `models/goggles.toml` records the matching engine hash and sample
or field-validation-required status. Numerical parity never certifies accuracy.
Without explicit `--sample`, conversion checks independent-source provenance.
Engine files are trusted local build artifacts. Rebuild and run
the same verification on the RTX 3050 target; do not copy the RTX 2050 engine.

## Status

Passed on 2026-09-23 using TensorRT 10.13.3.9 on the RTX 2050. All nine test
images matched detection counts at confidence 0.30. Maximum raw box difference:
0.108780 input pixels; confidence difference: 0.012006; post-NMS box difference:
0.113129 original pixels. The PyTorch export adapter agreed with native
MMDetection within 0.000031 pixels. Median latency was 6.37 ms; p95 6.87 ms
over 50 warmed calls including memory copies.

A separate process loaded and executed the engine with `torch` absent from
`sys.modules`, confirming TensorRT-only runtime operation. The installed model
has **sample / demo-only** status. These checks concern conversion correctness,
not generalization or industrial accuracy.
