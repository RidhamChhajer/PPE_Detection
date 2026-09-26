# v1 verification — 26 September 2026

ONNX Runtime CPU is the default; CUDA ORT and local TensorRT are optional.
No retraining occurred. Paths below are repository-relative.

## Selected checkpoints and installed assets

| Selected original | SHA-256 |
| --- | --- |
| `work_dirs/goggles_merged_fp32/best_coco_bbox_mAP_epoch_32.pth` | `337acd8decbce98623bb82c710bc7e6b19741a9e147407bab3e03d8c1a89d22b` |
| `work_dirs/footwear_v1_fp32/best_coco_bbox_mAP_epoch_28.pth` | `9835f6d588292a166877c62ff8567c5b7b7b26f8df232df90c51f623e5ea34d2` |

Originals remain unchanged; release checkpoint copies are byte-identical.
Neither deployment uses epoch 40.

| Installed asset | SHA-256 |
| --- | --- |
| `models/goggles-v1.onnx` | `9ded9b32518962346b4a4f2fbb5878de6ced68e03b44062c2265e23721214827` |
| `models/footwear-v1.onnx` | `dfa96fdce8525ceb90ceb116ebfb8b761c96e91f6e5e58d5e8da30a9eca9ec96` |
| `models/goggles-v1.engine` | `e1726835e1abb93c127c86e9b5ca5348aeb24e4bc9f4148f11822da0ab8de8aa` |
| `models/footwear-v1.engine` | `89576983cd514d836b0dc5d1b9e3ec8389a0591c4e3bd0f5eebfa233ac7b8e97` |

Registry: `configs/models.json`; local TensorRT registry:
`models/registry.local.json`. Existing RTX 2050 engines were reused with fresh
parity checks, not rebuilt: FP16 goggles and mixed-precision stable-score footwear.
Engines are not release assets. Rebuild and verify on the final RTX 3050.

## Evaluation and parity

Both selected checkpoints were evaluated on their validation and test splits in
FP32 with TF32 disabled. Reports, previews and `verified.toml` remain in each
run's `evaluation/`. Complete metrics are in [Models](MODELS.md).
Test AP50 / COCO mAP: goggles **0.6747 / 0.2452**; footwear overall
**0.4971 / 0.2851**; SAFETY_SHOE **0.6617 / 0.3995**;
NOT_SAFETY_SHOE **0.3324 / 0.1707**. Small splits do not establish field accuracy.

ONNX checker passed both models. All 39 test images passed PyTorch comparisons,
including raw scores/boxes and final class-aware NMS. Classes/counts matched
exactly: goggles 5 detections over 5 images; footwear 53 SAFETY_SHOE and 6
NOT_SAFETY_SHOE over 34 images.

| Backend/model | Max raw box error (px) | Raw score error | Final box error (px) | Final score error |
| --- | ---: | ---: | ---: | ---: |
| ONNX goggles | 0.000504 | 0.00001729 | 0.0000611 | 0.00001431 |
| ONNX footwear | 0.004700 | 0.00001395 | 0.001709 | 0.00000814 |
| TensorRT goggles | 0.478241 | 0.003753 | 0.094147 | 0.002864 |
| TensorRT footwear | 0.531388 | 0.00014192 | 1.689209 | 0.00010592 |

All configured tolerances passed before installation. Detailed records:
`release_assets/v1/*-provenance.json` and each run's `evaluation/*-parity.json`.

## Automated and clean-install checks

- Full suite: **58 tests passed**, 4.362 seconds.
- Main and isolated runtime-only environments: `pip check` passed.
- Both configurations and input pipelines passed without training. Goggles
  train/valid/test: 148/28/5 images; footwear: 161/35/34.
- Goggles training retains 5 negative images; footwear negatives: 2/1/1.
  Negative crops remain supported. FP32, learning rate 0.0001, clipping 10 and
  immediate non-finite loss/gradient rejection remain in training.
- Clean source snapshot installed runtime requirements only, downloaded both
  hash-verified ONNX models, ran real inference, a three-frame video and frontend
  GET. No datasets/checkpoints needed; no PyTorch/MMDetection/MMCV/TensorRT imports.

## Full-video and frontend checks

`local/media/factory_video.mp4` → `artifacts/v1/factory_onnx.mp4`:
**754 frames, 480 × 864, 60.014062 FPS, 12.563722 seconds**, silent H.264.
Every presentation timestamp matched. Detections: goggles 504, SAFETY_SHOE 549,
NOT_SAFETY_SHOE 27 (1080 total). CPU elapsed approximately 353 seconds;
this is not realtime throughput.

API independently completed the entire video with both models: upload 202,
completed job, range playback 206, download 200 with matching hash, upload cleanup.
Report: `artifacts/api_smoke_6b6a7b74041a434c9ea4d3c97df53dfd/smoke.txt`.

The stale frontend server was restarted. Browser now shows both versioned models
and the permanent limitation notice. Actual browser upload completed with playback
and download: 3 frames, 64 × 48, 25 FPS, 0.12 s;
job `35cd12f2e509457aa74b83ee2709a6b0`. Server remains at localhost:8765.

## Cleanup and publication

See [Cleanup](CLEANUP.md) for removed and preserved paths. Release assets, source
bundle, suggested notes and SHA256SUMS are under `release_assets/v1`.
Inventory: `local/release_inventory.json`. No commit, push or history rewrite.

Read-only audit found zero historical large/model/private-media objects and zero
supported credential-pattern matches. This is not proof that all secrets are absent.
Five historical absolute-path references remain: `DATA_COLLECTION.md` lines 35/38,
`README.md` line 20, `TRAINING_STABILITY.md` line 66,
`tools/check_environment.py` line 76. Path-only audit: `local/publication_audit.json`.
`.gitignore` does not remove history. No sensitive-history cleanup was identified.

Repository: https://github.com/RidhamChhajer/PPE_Detection. Before public release,
choose a license (MIT is an option; none assigned), review redistribution rights,
authorize publishing, and set the registry's single `release_base_url` to the real
published release. Local file-URL download is verified. Unseen workers/sites,
camera conditions and final hardware still require field validation. This is not
a safety certification system.
