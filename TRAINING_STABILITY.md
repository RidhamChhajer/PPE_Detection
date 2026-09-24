# FP32 recovery of the merged goggles training run

## Diagnosis from the saved run

Inspected `work_dirs/goggles_merged/goggles_rtmdet_s.py` and
`20260924_172448/20260924_172448.log`, without modifying either.

- Saved configuration: `AmpOptimWrapper`, dynamic loss scaling, AdamW LR
  0.0004, accumulation 2, gradient clipping norm 10, no invalid-loss hook.
- First logged non-finite gradient norm: epoch 1, batch 16/74, while losses
  remained finite. Epoch 2 also reported infinite gradient norms.
- Epoch 10, batch 48/74 was still finite: loss 0.7024, classification 0.2475,
  bbox 0.4549, gradient norm 13.7684. Batch 64/74 reported all NaN at LR
  0.00037576. Logs are sampled/smoothed, so they do not identify the exact
  failing microbatch or arithmetic operation.
- Subsequent losses remained NaN through epoch 40. Validation reported empty
  detections; repeated earlier AP values after this failure are stale, not
  evidence that those later weights remain valid.
- The selected epoch-8 checkpoint had validation mAP 0.250 and AP50 0.509.
  It is the only recovery checkpoint used by the command below.

The evidence establishes numerical failure in the AMP training run and absence
of fail-fast protection. FP16 overflow/unstable gradients are plausible, not a
proven single-operation root cause. Dynamic scaling can skip invalid gradients
while training continues; clipping with its default `error_if_nonfinite=False`
does not terminate the run. The logs cannot prove that LR or a negative image
caused the failure. The saved head has `exp_on_reg=False`, so exponential box
decoding is not an explanation for this run. Current COCO/image validation
and transformed-input checks passed; stochastic augmentation histories cannot
be reconstructed from these logs.

## Safeguards applied to both tasks

Both configurations now use a registered `FiniteOptimWrapper` extending
MMEngine's **FP32** `OptimWrapper`, with no autocast or GradScaler. AdamW LR
is reduced fourfold to 0.0001; cosine minimum is reduced proportionally to
0.000005. The existing 100-iteration warmup starts at 0.00001. Batch size 2,
accumulation 2, weight decay and norm-10 clipping are retained.

The wrapper inspects each returned loss tensor before reduction/backward,
checks total loss before backward, and checks parameter gradients immediately
after every backward, including the first accumulation microbatch. A non-finite
value raises `FloatingPointError` before an optimizer update. The temporary
forward hook is removed even when an exception occurs. Norm clipping also sets
`error_if_nonfinite=True`, catching non-finite aggregate norms even when every
gradient element is finite. Exceptions propagate out of training; invalid
batches are never silently skipped, sanitized or saved as a new checkpoint.

Verified against installed MMEngine 0.10.7 `BaseModel.train_step`,
`OptimWrapper.update_params/backward/step`, and MMDetection 3.3.0. The built-in
`CheckInvalidLossHook` runs after `train_step`/optimizer updates and checks only
total loss; by itself it cannot provide the requested pre-update gradient
protection. The extension uses MMEngine's documented wrapper API:
[OptimWrapper](https://mmengine.readthedocs.io/en/v0.10.4/tutorials/optim_wrapper.html).

Negative images remain enabled with `filter_empty_gt=False`. Random crops
retain `allow_negative_crop=True`. Zero bbox loss on an all-negative batch is
valid and must not trigger a failure. No deployment/export code was changed.

## Start the corrected goggles run

Run from PowerShell. This starts a **new** sample run from the selected epoch-8
model weights, resetting optimizer and schedule. Do not add `--resume`:

```powershell
cd "D:\PPE DET"
.\.venv\Scripts\python.exe tools\train.py `
  --task goggles `
  --annotations "data\goggles\annotations" `
  --images "data\goggles\images" `
  --work-dir "work_dirs\goggles_merged_fp32" `
  --pretrained "work_dirs\goggles_merged\best_coco_bbox_mAP_epoch_8.pth" `
  --epochs 40 `
  --sample
```

The fresh directory is not created by preflight checks. A nonempty destination
is refused for either task. Resuming an old AMP run with this new optimizer
policy is refused. `epoch_40.pth` is never loaded. Existing checkpoints,
saved configurations, logs and application models are preserved.

## Expected output and checks

Startup prints `FP32 training; AdamW lr=0.0001; clipping max_norm=10;` and
`data regime=sample`. Training should log finite numerical `loss`, `loss_cls`,
`loss_bbox` and `grad_norm`. Exact values and monotonic improvement cannot be
promised. Logged gradient norm is **before clipping**, so values above 10 are
normal; NaN or infinity are not. FP32 may use more GPU memory and run slower.
If a guard fires, retain the error and last finite checkpoint for diagnosis;
do not resume the failing process with checks disabled.

Checks run without full model training, evaluation or export:

- 35 unit tests, including real MMEngine `BaseModel.train_step` on a tiny CPU
  test model, injected NaN/Inf losses, finite-loss/non-finite gradients on each
  accumulation position, clipping norm overflow, negative images/crops and
  fresh-directory protection.
- Both resolved configs built the registered FP32 optimizer successfully.
- Complete COCO validation and all-image train/valid/test input-pipeline checks
  through `tools/train.py --task <task> --sample --check-only`.
- Goggles: 148/28/5 images; 220/29/12 boxes; 5/0/0 source negatives.
- Footwear: 161/35/34 images; 218/47/46 boxes; 2/1/1 source negatives.
- Read-only epoch-8 checkpoint inspection found no non-finite state tensors.
- Before/after SHA256 checks matched for every file in the failed run,
  `models/`, and `artifacts/phase4/`, including both existing checkpoints.

Sample status remains required. Goggles validation/test have no negative
images; these development results cannot establish operational false-alarm
performance. No full training run was started, including footwear. Numerical
stability over future epochs remains to be verified during the requested rerun.

## File inventory for this change

- Updated `configs/goggles_rtmdet_s.py` and `configs/footwear_rtmdet_s.py`:
  FP32 wrapper registration, lower LR/minimum and strict clipping.
- Created `tools/stable_optim.py`: loss and microbatch-gradient safeguards.
- Updated `tools/train.py`: task-aware destination protection, optimizer-policy
  resume guard, startup status and all-image finite input/box checks.
- Created `tests/test_training_stability.py`: seven targeted regression tests.
- Created `TRAINING_STABILITY.md`: diagnosis, exact rerun command and results.
- Updated `TRAINING.md` and `DELIVERY.md`: link to this recovery procedure.
- Removed no files. No dataset, failed-run artifact, checkpoint or TensorRT
  engine was replaced; the application still uses its existing demo engine.
