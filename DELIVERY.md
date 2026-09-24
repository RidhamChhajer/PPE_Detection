# Delivery file inventory

The subsequent FP32 training-stability change is inventoried in
[TRAINING_STABILITY.md](TRAINING_STABILITY.md), including its checks and fresh
goggles run command. Existing deployed demo weights are unchanged.

Current delivery: phases 1–6 implemented and verified, with sample-only model
accuracy status. Generated files are ignored by Git. Required sample assets were
copied into project-owned `data/`; `Previous/` remains unchanged and can now be
deleted by the user. No source files were removed.

## Source and documentation created or updated

| File | Purpose |
| --- | --- |
| `IMPLEMENTATION.md` | Phase scope and explicit generalization requirement |
| `.gitignore` | Exclude environments, datasets, generated models/videos and caches |
| `requirements.txt` | Pinned training, export, runtime and local web dependencies |
| `README.md` | Setup, launch, limitations and model replacement workflow |
| `ENVIRONMENT.md` | Phase 1 CUDA/model smoke-test record |
| `DATASET.md` | Phase 2 sample validation and limitations |
| `TRAINING.md` | Sample training and measured evaluation results |
| `DATA_COLLECTION.md` | Team collection/labeling and independent-source training contract |
| `CONVERSION.md` | TensorRT contract, parity tolerances and verified conversion results |
| `VIDEO.md` | Complete-video/browser acceptance and operational boundaries |
| `DELIVERY.md` | This consolidated handoff inventory |
| `configs/goggles_rtmdet_s.py` | Configurable RTMDet-S training recipe with general image augmentation |
| `tools/check_environment.py` | CUDA/MMCV/pretrained inference verification |
| `tools/validate_coco.py` | Annotation/image validation, normalization and split provenance checks |
| `tools/train.py` | Configurable training, explicit sample mode and input-pipeline check |
| `tools/evaluate.py` | Saved-run evaluation and optional complete-video qualitative review |
| `tools/export_tensorrt.py` | Separate ONNX export/build/parity stages and verified engine installation |
| `tools/process_video.py` | Standalone complete TensorRT video processing CLI |
| `app/__init__.py` | Application package |
| `app/detector.py` | TensorRT runtime, engine integrity, preprocessing and NMS |
| `app/video.py` | Every-frame multi-engine inference, drawing and verified video encoding |
| `app/app.py` | Local upload/job/progress/video/download service and cleanup |
| `app/templates/index.html` | Single-page interface with visible model status |
| `app/static/style.css` | Responsive page layout and controls |
| `app/static/app.js` | File selection, progress polling, playback and download flow |
| `tests/test_validate_coco.py` | Dataset and independent-source regression checks |
| `tests/test_evaluation.py` | Detection metric matching checks |
| `tests/test_detector.py` | TensorRT data contract and model provenance checks |
| `tests/test_video.py` | CFR/VFR completeness, engine sequencing and failure cleanup |
| `tests/test_app.py` | Upload/job protection, streaming/download and expiry checks |

## Generated local artifacts

- `artifacts/phase1/`: downloaded official pretrained checkpoint/demo and environment verification outputs.
- `artifacts/phase2/`: cleaned train/valid/test annotations, preview sheets and validation report.
- `artifacts/phase3/`, `artifacts/phase3_training.log`: initial sample evaluation/preview and training record.
- `work_dirs/goggles_rtmdet_s/`: resolved historical sample config, checkpoints, logs and reevaluation outputs. The saved recipe is retained to reproduce the already-trained sample weights.
- `artifacts/phase4/goggles.onnx`, `goggles.engine`, `conversion.txt`: export/build candidates and parity record.
- `models/goggles.engine`, `models/goggles.toml`: verified installed engine and matching SHA256/sample status.
- `artifacts/phase5/factory_annotated.mp4`: verified complete CLI result.
- `artifacts/app/<job-id>/output.mp4`: verified browser job result, subject to expiry. Uploaded input is automatically removed.
- `data/sample_goggles/`: byte-verified project copy of the 90-image sample COCO export.
- `data/sample_media/factory_video.mp4`: byte-verified project copy of the required sample video.
- `.venv/`: single project environment, extended with the required conversion/video/web dependencies.

Temporary test files are managed by temporary directories and removed after
tests. Failed partial outputs and completed uploads are removed automatically.
No experimental alternative runtime or unused future detector modules remain.
