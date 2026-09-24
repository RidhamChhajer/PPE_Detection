# Phase 1 environment record

Initial inventory on 2026-09-19; GPU verification passed on 2026-09-23
on the development Windows machine.

| Component | Before setup | Selected Phase 1 version |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 2050, 4 GB | Existing hardware |
| NVIDIA driver | 581.86 | Existing driver |
| Driver-supported CUDA maximum | 13.0 | Not the PyTorch runtime version |
| CUDA toolkit (`nvcc`) | Not found in PATH or default toolkit directory | Not required for prebuilt wheels |
| Python | 3.10.11, x64 | 3.10.11, `.venv` |
| PyTorch | Not installed | 2.1.2+cu118 |
| torchvision | Not installed | 0.16.2+cu118 |
| PyTorch CUDA runtime | Not installed | 11.8, bundled with PyTorch |
| MMEngine | Not installed | 0.10.7 |
| MMCV | Not installed | 2.1.0, CUDA-enabled Windows wheel |
| MMDetection | Not installed | 3.3.0 |
| TensorRT Python package | Not installed | Deferred to Phase 4 |

The compatible version set was selected before package installation, using the
official references linked in README.md. Only one project environment, `.venv`,
was created. No system Python packages or GPU drivers were changed.

## Acceptance

Phase 1 is complete. The user ran the verification commands on 2026-09-23;
the saved `artifacts/phase1/environment.txt` report confirms the successful run.

- `pip check`: no broken requirements found (user-supplied terminal output).
- PyTorch CUDA runtime and MMCV compiled CUDA: both 11.8.
- GPU: NVIDIA GeForce RTX 2050, 4.00 GiB.
- MMCV CUDA NMS: PASS.
- Official COCO-pretrained RTMDet-S inference on `demo.jpg`: PASS.
- Detections with confidence >= 0.30: 22.
- First inference: 1770.4 ms, including startup overhead; not a throughput benchmark.
- Peak GPU tensor allocation: 192.8 MiB; not total process GPU memory.
- Annotated result: `artifacts/phase1/rtmdet_s_smoke.jpg`.
- Checkpoint SHA256: `387a891e157cf0ab57d76b3ffc17bf77247089d672532427930b3140f9e789d6`.

The terminal also reported two unexpected checkpoint keys,
`data_preprocessor.mean` and `data_preprocessor.std`, and a future
`torch.meshgrid` API warning. Neither prevented this inference test from passing;
these messages are retained here for reproducibility.

TensorRT remains uninstalled and is deferred to Phase 4. This test establishes
environment readiness only; it does not evaluate PPE accuracy or industrial
deployment readiness. Phase 2 has not started.
