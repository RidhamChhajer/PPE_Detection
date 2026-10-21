# Verification

## Auto, OpenVINO and flexible cameras - 21 October 2026

App/full-video commands now default to Auto: probe both models on TensorRT,
otherwise OpenVINO Intel graphics, otherwise ONNX CPU. Explicit backend choices
do not fall back. Model weights, thresholds and shoe-pairing logic are unchanged.
Default shared-bundle setup installs Auto/OpenVINO; NVIDIA builds remain an
explicit later action on the recipient computer. No HP setup was performed.

- OpenVINO 2025.4.1 installed in the local project environment; pip check passed.
  Intel GPU was discovered automatically. FP32 inference passed all 5 goggles
  and 34 footwear test images against ONNX, with exact final classes/counts.
  Maximum final box/score errors: goggles .000489 pixels/.00000489;
  footwear .00293 pixels/.00000474. No parity tolerances were changed.
- Real connected webcam/shared mode ran both models on Intel GPU, produced a
  single combined preview and released the camera cleanly. First processed-frame
  host capture-to-encoding latency was 1563 ms; frame age was 1672 ms. This is
  a smoke measurement, not sustained performance or sensor-to-display latency.
- Browser tests passed face-only, feet-only, same-camera and separate-camera
  modes, keyboard tabs, disconnect/stop handling, 390px layout and invalid upload.
  Camera UI fixtures are simulated; real camera capture was checked separately.
  The actual full 754-frame factory upload also passed playback/download checks.
- Automatic TensorRT selection and subsequent real inference passed for both
  models; the shared logger stays alive across probes/jobs. No engines rebuilt.
- New tests cover OpenVINO CPU parity, Intel-only device selection, automatic
  fallback/errors, single/shared capture ownership and Windows backend matching.
  Full Python regression suite: **92 tests passed**.
- Python compilation, JavaScript syntax, PowerShell script parsing, design
  YAML/token checks and source link checks passed. Strict UI scanner ran; its
  external-handler/native-select false positives remain as documented below.

Two physical Arducams, sustained load, independent industrial accuracy and the
final HP/RTX 3050 environment still require validation on that hardware.

## Group Windows deployment bundle - 21 October 2026

The requested replacement delivery is release_assets/industry/PPE_Detection_Windows.zip.
It contains both hash-verified ONNX models, selected .pth checkpoints and resolved
configs, 39 verification images (5 goggles, 34 footwear), application source,
dependency manifests and deployment/setup.ps1 + start.ps1. Installed dependencies,
the sender's virtual environment and current GPU engines are excluded.

The optional -Backend tensorrt setup builds/checks engines on the recipient
GPU only. It checks all bundled images against ONNX with the existing raw-output
and one-to-one final-detection tolerances. Goggles FP16 may fall back to FP32;
footwear starts FP32. Both candidates must pass before the local registry changes.
Failed setup preserves the prior registry and engines. This setup needs no training
dataset, MMDetection or PyTorch. CPU setup validates the included ONNX models.

All **83 Python tests passed**, including engine setup failure/commit behavior,
FP32 fallback, raw parity rejection, bundle input tampering and asset selection.
PowerShell setup/start parsing and Python compilation also passed. Tests use fake
GPU builders: no actual HP build or development-PC engine rebuild occurred.
The archive builder hashes every delivered file and rechecks each ZIP entry.
Recipient-PC driver compatibility and physical camera performance remain untested.

## Two-camera software and source handoff - 21 October 2026

Both RTMDet-L models and existing engine files are unchanged. The local registry
now carries face/feet routing. At that stage app/CLI defaulted to TensorRT; CPU was explicit and
uses optional requirements-cpu.txt. No HP TensorRT configuration was performed.

- Full Python suite, including training-stack tests: **78 tests passed** in the
  final run, including handoff/default-selection and worker-start cleanup checks.
  Windows sandbox denied atomic replacement of temporary files, so verification
  was rerun outside the sandbox and passed. Validation was not weakened.
- Browser smoke: tabs/keyboard switching, duplicate device rejection, simultaneous
  previews, per-feed failure, stop controls, 390px layout and invalid upload passed.
  Camera API responses were simulated for UI tests; Python tests separately exercise
  two capture workers and one inference worker with distinguishable fake frames.
- Real TensorRT browser upload of the complete Previous/input_videos factory clip
  passed processing, player display and download. Both models ran. Saved output
  is artifacts/app/6d48a181ac4d43dd991d496697bad8c1/output.mp4 (local-only, expires).
  The processing function verified all 754 frames, dimensions and presentation
  timestamps; output is silent H.264. The browser test did not certify visual
  accuracy. Screenshots are artifacts/browser/two-cameras-desktop.png and
  two-cameras-mobile.png and use simulated camera images.
- Python compilation, JavaScript syntax and pip dependency checks passed.
- Source allowlist/link/history audit passed: no historical binary/private/large
  file candidates and no historical credential-pattern matches. Historical path
  references remain in Git; the source-only ZIP excludes Git history. Current
  path findings were reviewed as examples or standard system executable paths.
- Strict premium UI audit ran. Four reported actionless buttons are static-tool
  false positives: this scanner recognizes inline HTML handlers but not the
  external addEventListener bindings required by our CSP. Browser tests exercise
  tabs, refresh and stop actions. Native select ownership is explicitly recorded
  in CAMERA_INTEGRATION.md. No inline-handler workaround was introduced.
- The external DESIGN.md npm validator was blocked by automatic approval review
  for unverified elevated package execution. Local YAML/token/section checks and
  actual browser checks are used instead; official validator success is not claimed.

Physical two-Arducam testing, HP driver/runtime setup, engine rebuild/parity and
end-to-end <=3-second measurements remain deferred. No missing-PPE alerts or
cross-camera worker matching was added.

The source-only delivery was superseded by the code+models Windows deployment
bundle. See deployment/START_HERE.md for recipient-PC setup. Existing development
engines remain unchanged; target-PC engine building is deferred until extraction.

## RTMDet-L v2 verification - 27 September 2026

Both trained Kaggle models are installed. No retraining occurred. All paths below are repository-relative. Original ZIPs in Previous, datasets, historical checkpoints and logs were preserved.

## Selected models and hashes

| Model | Best checkpoint | SHA-256 |
| --- | --- | --- |
| goggles-v2 | work_dirs/goggles_rtmdet_l_v2/best_coco_bbox_mAP_epoch_28.pth | f5862021fc0101f82b5fd8bb97ec6ec795ed6574e8e894412615758bed322231 |
| footwear-v2 | work_dirs/footwear_rtmdet_l_v2/best_coco_bbox_mAP_epoch_14.pth | 50d4d203742e72c9a410fbd0000737e1b713b0e5e0c304ee8395b1346e26009b |

| Installed asset | SHA-256 |
| --- | --- |
| models/goggles-v2.onnx | fddd119d3865ecda103b8ba65e79497d8f49780e93161493941fa3bb3cb456d8 |
| models/goggles-v2.engine | 9114527ddc2b69a73ea18227aa88d9fb90a08cb6c087262da8dd42222529d2ac |
| models/footwear-v2.onnx | 3af7410bec4e2424ca2a764017a5d81cf74d8a4df5331092b4dc28fd7bc84a8e |
| models/footwear-v2.engine | 12466da0a3ec3b215a304a0f5a92682ea0b4e6e3772bb224c2529f2c1676813b |

Public metadata: configs/models.json. Local TensorRT metadata: models/registry.local.json. Goggles uses FP16; footwear uses FP32 with TF32 disabled. Footwear FP16 failed detection-count parity; mixed stable_scores failed final box/confidence matching. Neither failed candidate was installed. Tolerances were not relaxed.

## Evaluation and parity

Validation/test evaluation and previews are in each new run's evaluation directory. Full overall and per-class results are in [Models](MODELS.md). Test mAP50-95: goggles 0.1563; footwear 0.1761; SAFETY_SHOE 0.2273; NOT_SAFETY_SHOE 0.1249. These development results do not establish field accuracy. Both overall scores are below the earlier RTMDet-S results.

Both ONNX and TensorRT passed every test image (5 goggles, 34 footwear), with exact final class IDs/counts and class-aware NMS. Counts: 9 goggles, 68 SAFETY_SHOE, 7 NOT_SAFETY_SHOE.

| Model/backend | Max raw box error | Raw score error | Final box error | Final score error |
| --- | ---: | ---: | ---: | ---: |
| goggles-v2/onnx | 0.00163269 | 0.00000617 | 0.00027466 | 0.00000283 |
| goggles-v2/tensorrt | 1.00042725 | 0.00527817 | 0.14144897 | 0.00382590 |
| footwear-v2/onnx | 0.00231934 | 0.00002739 | 0.00274658 | 0.00000262 |
| footwear-v2/tensorrt | 0.00148010 | 0.00002889 | 0.00195312 | 0.00000259 |

Full provenance: release_assets/v2/*-provenance.json and work_dirs/*_rtmdet_l_v2/evaluation/*-parity.json.

## Automated verification

- 64 automated tests passed. pip check: no broken requirements.
- Both RTMDet-L configuration and input-pipeline checks passed without fitting weights. Goggles train/valid/test: 148/28/5; footwear: 161/35/34. Negative images and negative crops remain supported.
- All 603 local dataset files matched the Kaggle package manifest hashes.
- Runtime-only ONNX inference, generated video and frontend smoke passed without importing PyTorch, MMDetection, MMCV or TensorRT.
- Source packaging and documentation links passed. Training dependency tests remain optional for runtime-only installations.

## Full-video verification

```text
PASS: 754 frames; 480x864; 60.014062 FPS; 12.563722 seconds; 1960 detections; elapsed 98.6s; no PyTorch.
Preserved all presentation timestamps: True
Codec: h264; audio streams: 0; detections by class: {'goggles': 826, 'SAFETY_SHOE': 1092, 'NOT_SAFETY_SHOE': 42}
Output: outputs\factory_rtmdet_l_v2_20260927.mp4
```

98.6 seconds is measured end-to-end processing time on the current machine, not realtime throughput. Output preserves all source frames and timestamps.

## Frontend and API verification

```text
PASS: frontend labels [['goggles'], ['SAFETY_SHOE', 'NOT_SAFETY_SHOE']]; permanent versioned workflow.
Upload 202; completed job 9162c815ca3f47ca874c64ed90b08369; range playback 206; download 200/hash identical.
754 frames; 480x864; 60.014062 FPS; 12.563722 s; all timestamps preserved; silent H.264.
No PyTorch imported; input removed; output: artifacts\api_smoke_9207ffae587a45ffa037aadaa8257f09\9162c815ca3f47ca874c64ed90b08369\output.mp4
```

This checks frontend HTML and the actual API workflow. Interactive browser playback controls were not separately exercised. The smoke-test service is closed; no server was left running.

## Changed files and cleanup

- configs/*_rtmdet_l.py, tools/train.py and tools/check_environment.py: RTMDet-L defaults; obsolete RTMDet-S training configs removed.
- configs/models.json: v2 identifiers, RTMDet-L architecture, development qualification and verified hashes. Old v1 release URL cleared so downloads cannot silently use old models.
- tools/localize_run.py: preserves original Kaggle configs while adapting local paths and workers. Original configs remain under each new run/original/.
- tools/export_models.py and tools/export_tensorrt.py: architecture-neutral release filenames, preserved dataset version/qualification, explicit FP32 fallback, recorded precision/hashes and actionable parity diagnostics.
- tools/check_app.py: TensorRT API smoke support while rejecting training-framework imports.
- tools/prepare_repository.py: v2 release defaults and Kaggle scripts in source packages.
- tests/test_rtmdet_l.py, tests/test_deployment.py and tests/test_training_stability.py: architecture, migration, qualification, precision fallback and parity regressions.
- README.md and the Models, Training, Export and Deployment, Verification and Cleanup guides updated.

Removed obsolete v1 engines/ONNX files from models, artifacts/versioned_export/goggles-v1, artifacts/versioned_export/footwear-v1 and artifacts/v1. Cleanup inventory: artifacts/rtmdet_l_cleanup_inventory.json. Historical training runs and release_assets/v1 remain as records of the previous trained/released models; they are not loaded by the application.

## Remaining manual work

- Review annotated footage and independent recordings; measure false positives and missed PPE under real camera conditions. Larger architecture did not improve these small test splits.
- Rebuild and verify TensorRT engines on the final RTX 3050. Existing engines are specific to the current GPU/TensorRT installation.
- Publish a new GitHub release for v2 artifacts and then set release_base_url to its real asset URL. Nothing was committed, pushed or published by this migration.
