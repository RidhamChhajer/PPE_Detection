# Footwear detection and shoe-pair output plan

Status: the available-video audit and a conservative display-only pairing
improvement are implemented. See the [baseline audit](FOOTWEAR_BASELINE_AUDIT.md).
The RTMDet-L weights, inference thresholds, and class labels are unchanged.
Phases requiring new labels, independent recordings, or held-out evaluation
remain open; this is not an industrial accuracy qualification.

## Goal and boundaries

Improve two different outcomes:

1. **Recognition:** correctly find and classify visible safety and non-safety
   footwear in factory footage, without detecting faces, floors, or other objects
   as shoes.
2. **Presentation:** show one box around the two shoes belonging to one worker
   when a pair can be identified reliably, including when several workers walk
   close together.

The current application accepts a complete uploaded video or one live camera,
runs goggles-v2 and footwear-v2 (RTMDet-L), and draws annotated frames. Keep the
upload workflow, live workflow, goggles detector, and TensorRT deployment path.
Do not add worker IDs, identity storage, missing-PPE alarms, or earplug logic as
part of this work.

## Current baseline

The footwear model has two **individual-shoe** classes:
`SAFETY_SHOE` and `NOT_SAFETY_SHOE`. The dataset currently contains 161 train,
35 validation, and 34 test images, with 218, 47, and 46 boxes respectively.
Its recorded test results at confidence 0.30 and IoU 0.50 are:

| Class | Precision | Recall | AP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `SAFETY_SHOE` | 0.353 | 0.828 | 0.557 | 0.227 |
| `NOT_SAFETY_SHOE` | 0.286 | 0.118 | 0.215 | 0.125 |

The overall footwear test result is precision 0.347, recall 0.565, and
mAP50-95 0.176. These small test splits are not a reliable field estimate.
Source: [measured model results](MODELS.md).

The revised display in `app/video.py` removes near-duplicate shoe boxes and
joins reciprocal, clearly separated same-class neighbors only. It does not
know which worker owns each shoe; uncertain matches remain individual boxes.
Pair grouping cannot correct a missed or false shoe detection.

## Phase 1 - Record and classify current errors

Use a fixed set of **complete, original factory videos** representative of the
intended camera positions. Preserve their original files. Select recordings
with one worker, multiple adjacent workers, crossed legs, partial occlusion,
distance changes, blur, lighting changes, ordinary footwear, and frames with
no visible feet. Include the face-camera example that produced a false
`NOT_SAFETY_SHOE` box.

For each sampled frame, record:

- Visible left/right shoes and their worker association for evaluation only.
- Actual footwear class for each visible shoe, or `uncertain` if appearance
  cannot establish the class.
- Whether both shoes of a worker are visible enough for a pair box.
- Detector outcomes: correct, missed, duplicate, wrong class, wrong location,
  and wrong cross-worker pair.
- Source video/session, camera, lighting, approximate distance, and occlusion.

Save a compact error report and a small set of representative annotated frames.
Do not tune against the final held-out test recordings. Reproduce the baseline
with the existing checkpoint, ONNX model, TensorRT engine, preprocessing, and
confidence threshold before changing anything.

**Gate:** the team can point to the dominant errors with counts, not just
screenshots. In particular, determine whether the crowded example is caused
mostly by duplicate boxes, missed boxes, wrong class labels, or wrong pairing.

## Phase 2 - Improve the footwear dataset

Collect additional **real factory footage** from the intended views. Prioritize
new workers, shoes, camera sessions, backgrounds, and lighting over many near
identical frames from one clip. Include:

- Both safety and non-safety shoes, including hard lookalikes.
- Several workers side by side and crossing paths.
- Feet near and far from the camera, partly hidden, or motion blurred.
- Faces, torsos, hands, floor markings, equipment, and empty scenes that must
  not become footwear detections.
- Close footwear views if the eventual feet camera will produce them.

Keep the existing two classes and **individual-shoe boxes** for the first
retraining experiment. Review every image for complete annotation of visible
target shoes. An image with no target shoes is a valid negative with an empty
annotation list. Do not label uncertain protective footwear as definitely safe
or unsafe; set it aside for review or exclude it from the class evaluation.

Split by complete source recording, session, and preferably worker/shoe group.
No neighboring frames from the same passage may be divided across train,
validation, and test. Freeze the independent test recordings before training.
Run `tools/validate_training_dataset.py` and inspect sampled annotations.
Record counts per class, source, camera, negative scene, and split.

**Gate:** the revised dataset has reviewed labels, documented provenance, hard
negatives, and an independent test set. A larger image count alone does not
satisfy this gate.

## Phase 3 - Retrain and measure shoe recognition

Fine-tune the existing RTMDet-L footwear checkpoint in a **new versioned run**;
do not overwrite v2. Keep preprocessing, class order, and evaluation protocol
consistent initially so the comparison is meaningful. Benchmark RTMDet-S as a
fallback if RTMDet-L cannot meet live-camera latency on the target GPU; do not
assume the larger architecture is more accurate, as v2 did not beat the earlier
RTMDet-S test score.

Evaluate on the frozen factory test recordings:

- Per-class precision, recall, AP50, and mAP50-95.
- False footwear detections per minute on negative/face-only footage.
- Missed visible shoes and wrong class assignments per worker passage.
- Duplicate predictions for a single physical shoe.
- Accuracy by distance, crowding, occlusion, and camera view.

Choose operating thresholds using validation data only. Do not raise a
threshold merely to hide a single false positive: that can lower recall,
especially for `NOT_SAFETY_SHOE`, whose recorded test recall is already low.

**Gate:** compare against v2 on the same frozen footage. Continue only if the
new model produces a meaningful reduction in false positives without an
unacceptable increase in missed shoes, and the class-level metrics support it.
If results do not improve, revise data/labels before adding postprocessing.

## Phase 4 - Choose one shoe-pair strategy

Use the Phase 1 audit and the improved detector to compare these approaches.
Implement **only the selected approach** in the final runtime.

### Candidate A: direct pair-box detector

Create a separate COCO annotation version with one tight box around the two
visible shoes of each worker. The box must not include a neighboring worker's
shoe. Define pair-class rules before labeling: a `SAFETY_SHOE_PAIR` requires
both visible shoes to be confidently safety footwear; a
`NOT_SAFETY_SHOE_PAIR` requires clearly visible non-safety footwear. Mixed,
one-shoe-visible, and uncertain cases need an explicit annotation/evaluation
policy rather than being silently treated as negative examples.

Train a new footwear model version on these pair boxes and compare it with the
individual-shoe model on the same held-out recordings. This route gives one
box directly and avoids a separate person detector, but may struggle when only
one shoe is visible or workers overlap.

### Candidate B: associate individual shoes with people

Keep the improved individual-shoe model. Add lightweight per-frame person or
lower-body localization, then assign shoe boxes to the same visible person
before drawing their union. Do not persist a worker ID. If the assignment is
ambiguous, show the individual shoe boxes rather than joining two workers'
shoes. Benchmark the additional detector and association step on the RTX 2050
and later RTX 3050.

This route preserves useful single-shoe detections and handles partial
visibility more explicitly, but costs more inference and implementation work.

**Selection gate:** choose the simplest candidate that meets pair-level
accuracy and latency on crowded held-out footage. Do not select based only on
how clean one screenshot looks. If neither candidate passes, retain individual
shoe boxes and collect the missing data.

## Phase 5 - Validate pair output

Evaluate the final annotated video, not just raw model predictions. At minimum
measure:

- Correct pair boxes at IoU >= 0.50 against reviewed pair annotations.
- Wrong cross-worker pairings per 100 visible worker passages.
- Duplicate pair boxes per passage.
- Pair detection rate when both shoes are visible.
- Behavior when one shoe is hidden or a pair cannot be assigned.
- End-to-end live processed FPS and frame latency with goggles and footwear
  running together; uploaded video must retain its full duration/frame count.

Set numerical acceptance targets with the industrial guide using these named
metrics. A request for `95% accuracy` is not testable until the specific metric,
test set, and visibility rules are agreed. Review errors on complete footage,
including the face false-positive case and the crowded shoe example.

**Gate:** pair-level results improve over the current display without raising
cross-worker pair mistakes or materially reducing footwear recall.

## Phase 6 - Deploy and clean up

Export only the selected checkpoint to ONNX, then build a new local TensorRT
engine. Compare its detections with the validated PyTorch/ONNX output before
switching the model registry. Keep v2 available for rollback until v3 passes
video and live smoke tests on the intended GPU. Do not assume FP16 is valid for
footwear: the current footwear-v2 engine uses FP32 because its FP16 candidate
failed parity checks.

Replace the conservative pairing function if direct pair detection wins,
or replace it with person-associated pairing if Candidate B wins. Remove the
superseded rule and its tests in the same change. Keep only the model, data
version, evaluation report, inference code, and tests used by the chosen path.
Do not introduce a warning panel as part of this change.

**Final acceptance:** the upload and live workflows still run both available
PPE detectors automatically; annotated footwear boxes reflect the chosen pair
policy; TensorRT parity, held-out video metrics, live latency, and rollback
instructions are recorded.
