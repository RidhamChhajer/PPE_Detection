# Goggles model training and evaluation

For recovery of the merged dataset run that developed NaNs, use
[TRAINING_STABILITY.md](TRAINING_STABILITY.md). Both current task configurations
use fail-fast FP32 training; the AMP recipe below describes the historical
original sample run, not the corrected rerun.

This is a sample pipeline model, not an industrially validated detector. The
sample splits share a recording; reported metrics cannot establish performance
on new workers, sessions, or cameras. No compliance decisions are made.

## Completed sample training recipe

- COCO-pretrained RTMDet-S, one `goggles` class.
- 640 x 640 input, preserving aspect ratio with right/bottom padding.
- Batch size 2, gradient accumulation 2, AMP, zero loader subprocesses on Windows.
- AdamW, initial learning rate 0.0004, warmup then cosine decay, 40 epochs.
- First backbone stage frozen, backbone batch-normalization statistics frozen.
- Color augmentation before horizontal flipping; this order keeps OpenCV's
  in-place color conversion input contiguous.
- No mosaic/mixup on this tiny dataset: avoid further shrinking small targets.
- Validate every two epochs; retain the best validation mAP checkpoint and the
  latest resumable training checkpoint. Test data never selects the checkpoint.
- Seed 2026. GPU operations are not forced deterministic.

```powershell
# Re-evaluate the saved sample run; video is optional and never used for fitting:
.\.venv\Scripts\python.exe tools/evaluate.py --run-dir work_dirs/goggles_rtmdet_s --video "data/sample_media/factory_video.mp4"
```

The classifier shape mismatch when loading COCO's 80-class checkpoint is
expected: its output layers are replaced for the single goggles class. Other
compatible weights are retained for fine-tuning. Do not substitute an arbitrary
checkpoint at deployment; use the best checkpoint selected by validation.

Precision/recall use confidence >= 0.30, IoU >= 0.50, and one-to-one matching.
AP50 and mAP50-95 use the COCO evaluator over confidence-ranked detections.
The completed sample video review covered all 754 frames of the provided clip, with representative
prediction images saved for inspection. It is qualitative because this clip
does not have complete frame-level ground truth.

## Outputs

- `work_dirs/goggles_rtmdet_s/`: resolved config, logs, latest checkpoint, best checkpoint.
- `artifacts/phase3/evaluation.txt`: validation/test metrics and full-clip statistics.
- `artifacts/phase3/valid_predictions.jpg`, `test_predictions.jpg`, and
  `factory_predictions.jpg`: prediction previews.

The generalized evaluation command now writes under the selected run's
`evaluation/` directory and reads paths from its saved config. Original sample
results above are retained as the historical smoke-test record.

## Subsequent independent-data runs

See [DATA_COLLECTION.md](DATA_COLLECTION.md). Training paths, run directory,
pretrained checkpoint, and epoch count are configurable. Default training
requires disjoint source-group metadata and reviewed negative images. Explicit
`--sample` is available only for demo workflows. New runs include moderate
random scale/crop augmentation, color augmentation, and horizontal flips;
augmentation does not replace diverse real recordings. The already-trained
sample weights and their saved configuration have not been changed or tuned
against the factory clip.

## Prepared goggles and footwear datasets

The merged goggles pool is split by complete `source_group`: 148 training
images, 28 validation images, and 5 test images. The validation and test groups
do not occur in training. Only training currently contains reviewed negative
images, and the test group is too small for a production accuracy claim.

The footwear export retains its supplied 161/35/34-image split and two classes,
`SAFETY_SHOE` and `NOT_SAFETY_SHOE`. Its filenames indicate source-video
overlap and its COCO image records lack `source_group`; treat it as sample data.

Run the input checks before fitting weights:

```powershell
.\.venv\Scripts\python.exe tools/train.py --task goggles --sample --check-only
.\.venv\Scripts\python.exe tools/train.py --task footwear --sample --check-only
```

Then use separate run directories:

```powershell
.\.venv\Scripts\python.exe tools/train.py --task goggles --sample --work-dir work_dirs/goggles_merged
.\.venv\Scripts\python.exe tools/train.py --task footwear --sample --work-dir work_dirs/footwear_v1
```

`--sample` is required for these datasets because they do not satisfy the
independent-session and negative-coverage requirements for qualification.
Do not run the two GPU training jobs concurrently.

## Status

Phase 3 pipeline acceptance passed on 2026-09-23. Training completed 40 epochs;
the best validation checkpoint is `best_coco_bbox_mAP_epoch_36.pth`.

| Split | Precision | Recall | AP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.8214 | 0.8214 | 0.8288 | 0.5713 |
| Test | 1.0000 | 0.8667 | 0.9424 | 0.4752 |

Test counts at the declared operating point: TP=13, FP=0, FN=2. Complete-video
evaluation processed 754 frames in 22.1 seconds, producing 548 detections on 286
frames. Test-image and full-video preview sheets were inspected: several eye
regions are localized sensibly, but distant targets are missed and duplicate
boxes occur in some frames. This is sufficient to prove the sample pipeline,
not to approve industrial accuracy. Improve representative data and perform
independent validation before operational use.
