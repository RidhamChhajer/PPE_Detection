# Data required for generalization

The current 90-image export and trained checkpoint are demonstration assets.
They must not be described as industrially validated. The application processes
full frames and has no worker identities, fixed eye locations, factory-specific
ROIs, or background rules. Broader training data, not special-case code for the
factory clip, is required to improve real-world performance.

## Collection and labeling

- Collect independent recordings across workers, cameras, days, locations,
  backgrounds, distances, lighting, head poses, and occlusion.
- Include different goggles designs and both small/distant and large/near targets.
- Include representative negative images: people without goggles, empty scenes,
  and confusing objects. These images need complete review; missing labels do
  not make an image a valid negative.
- Agree on a consistent visual definition for `goggles` and annotate every
  visible instance. Decide how ordinary eyeglasses, partially visible goggles,
  and goggles held away from the face are treated before labeling.
- Preserve recording/session provenance in every COCO image record as a string
  `source_group`, e.g. `site-b_camera-2_session-2026-09-23-am`. Do not use a new
  group per frame. This is dataset metadata, not runtime worker tracking.
- Assign complete recording/session groups to train, validation, or test before
  extracting frames or making augmented versions. A group cannot cross splits.
  When the intended claim is unseen workers/cameras/sites, separate those too.

## Reusable training interface

Normalized annotations are `train.json`, `valid.json`, and `test.json`; images
live in corresponding `train`, `valid`, and `test` subdirectories of an image root.
Paths are configurable; neither training nor evaluation requires the sample
factory filenames.

```powershell
.\.venv\Scripts\python.exe tools/train.py --annotations "D:/datasets/goggles/annotations" --images "D:/datasets/goggles/images" --work-dir "work_dirs/goggles_independent"
.\.venv\Scripts\python.exe tools/evaluate.py --run-dir "work_dirs/goggles_independent"
# An optional video is evaluated as a whole, never used to fit the model:
.\.venv\Scripts\python.exe tools/evaluate.py --run-dir "work_dirs/goggles_independent" --video "D:/evaluation/unseen_session.mp4"
```

Default training requires source-group metadata, disjoint groups, multiple
training recordings/sessions, and negative images in each split. These are
minimum structural checks, not a guarantee that a dataset is sufficiently
large or diverse. Final dataset size and acceptance thresholds must follow
measured performance across the intended operating conditions.

Use validation data for model choices. Keep test recordings untouched until
those choices are fixed. Report per-condition precision/recall and error
examples in addition to aggregate AP; assess false alarms on negative scenes.
Do not tune against the demonstration factory clip.

## Current demo model

Explicit `--sample` mode permits a pipeline smoke test on incomplete sample data.
Legacy sample checkpoints also retain sample status during conversion. Exported
engine provenance identifies sample models, and the frontend must display that
status. Passing TensorRT numerical parity does not establish model accuracy.
An independent-source model still requires field validation before operational
deployment; no automatic accuracy certification is performed.
