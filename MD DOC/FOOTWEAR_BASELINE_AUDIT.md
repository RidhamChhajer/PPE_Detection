# Footwear v2 baseline audit

Status: Phase 1 started; not yet representative enough to close its gate.

## Evidence used

- Active model: `footwear-v2` (RTMDet-L), TensorRT FP32, confidence 0.30.
- Existing held-out test report: [model results](MODELS.md).
- One unique 754-frame, 12.56-second factory corridor video. The copies at
  `Previous/factory video.mp4` and `Previous/input_videos/WhatsApp Video 2026-08-27 at 8.46.56 PM.mp4`
  have the same SHA256 digest, so they are **one recording**, not independent
  evidence.
- Six sampled original frames at 5, 6, 7, 8, 9, and 10 seconds were processed
  by the current footwear TensorRT engine. Raw shoe boxes and the current
  proximity-paired boxes are side by side in the local-only review folder
  `artifacts/footwear_audit/factory_v2_current/`.
- The separate face-camera screenshot supplied in chat shows a large
  `NOT_SAFETY_SHOE 0.54` false positive, but the unannotated source frame is
  not available here for repeatable measurement.

## Sampled video results

| Time | Raw shoe boxes | Display boxes after proximity pairing | Review note |
| ---: | ---: | ---: | --- |
| 5 s | 0 | 0 | Workers are distant; footwear is too small to judge reliably in this sample. |
| 6 s | 0 | 0 | Workers remain distant; no displayed shoe boxes. |
| 7 s | 7 | 7 | Several closely spaced and overlapping low-confidence shoe boxes; no pair formed. |
| 8 s | 5 | 4 | One pair formed in a crowded scene; ownership needs reviewed ground truth. |
| 9 s | 6 | 5 | One union box spans shoes positioned under adjacent workers. This is a likely cross-worker pairing error. |
| 10 s | 0 | 0 | Feet are largely outside the frame; absence of boxes is not a miss. |

At 9 seconds, two predictions overlap the same left shoe area, and two overlap
the same right shoe area. The pair display also joins boxes from different
horizontal worker positions. This demonstrates two separate problems:
duplicate/wrong shoe predictions and an ownership error in the proximity rule.
The rule cannot solve the first problem, and a single screenshot cannot set a
safe distance threshold for the second.

The current small footwear test split reports overall precision 0.347 and
recall 0.565 at confidence 0.30, with `NOT_SAFETY_SHOE` recall 0.118. Raising
the confidence threshold to hide these samples would likely miss more real
non-safety footwear; it has not been changed.

The prior complete-video run with both RTMDet-L detectors processed 754 frames
in 98.6 seconds, approximately 7.6 frames per second including decoding,
encoding, and output verification. This is an offline throughput observation,
not a measured live-camera latency or a final RTX 3050 benchmark.

## Dataset provenance check

The current footwear COCO files have 161/35/34 train/validation/test images.
None of the image records includes `source_group`. Filenames beginning with
`oldvideo1`, `video2`, `phone`, `internet`, and `negative` occur in all three
splits. These names suggest related images may cross splits, but filenames
alone do not prove recording-level independence or leakage. Original recording
IDs must be established before the final test set is trusted.

The project's strict training check was run against the current data and stops
at `train image 3: source_group recording/session metadata is required`.
The RTX 2050 and CUDA training environment are available; provenance and
independent factory footage are the current gates, not missing GPU support.

Annotation previews also show a strong mix of isolated shoe product photos and
small shoes in corridor frames. Only 2/161 train, 1/35 validation, and 1/34
test images have no footwear annotations. Files named `negative_shoe_*` often
contain labeled `NOT_SAFETY_SHOE` objects; they are **not** necessarily empty
background negatives. This leaves very little verified no-footwear material
for measuring face/background false positives. Some visually similar black
footwear is labeled safe in the sample preview; its protective classification
requires confirmation from the dataset owner, not a guess from appearance.

## Phase 1 gate and next input

This audit confirms a pairing failure and shows why recognition errors must be
measured separately. It does **not** establish factory-wide error rates or
identify the winning pair strategy. The only available original factory video
is one short, already used corridor recording; the live face screenshot lacks
its raw frame.

To close Phase 1 and start a defensible Phase 2, obtain several independent
original factory recordings, including a face-only negative recording and
crowded feet views from the intended cameras. Record which video/session each
image came from. Review and annotate sampled frames for individual shoes,
worker ownership for evaluation, and whether both shoes are visible. Keep at
least one complete recording untouched for final testing.

## Current implementation check

After this baseline, the display-only pair rule was replaced with duplicate
suppression and conservative reciprocal-neighbor matching. It keeps uncertain
shoes separate and does not store worker IDs. The model and inference thresholds
are unchanged. On the same recording, the 8-second left/middle shoes no longer
form a cross-worker box, while the middle worker's two visible detections at
9 seconds form one box. This is a qualitative check, not pair-level accuracy.

The complete input from `Previous/input_videos` was processed with both
TensorRT detectors. The output `outputs/footwear_pairing_test.mp4` has all 754
frames, the original presentation timestamps, 480x864 resolution, H.264 video,
and 12.563722-second duration. Processing took 77.0 seconds on this machine;
this measures offline throughput, not live-camera latency. Independent footage
and reviewed pair labels are still required for Phases 2-6 and field acceptance.
