# Goggles sample dataset — Phase 2

Source: `data/sample_goggles/`, copied byte-for-byte from the user-managed
`Previous/PPE Googles.coco/` export so `Previous/` can be removed independently.
Cleaned annotations and validation artifacts are generated in `artifacts/phase2/`
and ignored by Git. The JSON files are COCO training inputs, not application
detection outputs.

## Normalization and validation

The single annotated category `goggles on face` is renamed to `goggles`, with
COCO category ID 1. The unannotated `PPE-Googles-and-Boots` project category is
removed. MMDetection maps category ID 1 to model label 0.

The validator checks image decoding and dimensions, unique IDs, image/category
references, finite bounding boxes inside image boundaries, positive box sizes,
and agreement between box area and annotation area. Invalid data fails validation;
boxes are not silently clipped and annotations are not silently dropped.

| Split | Images | Goggles boxes | Empty images | Boxes with area < 32² px | Minimum width / height |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 63 | 95 | 0 | 67 | 14 / 7 px |
| valid | 18 | 28 | 0 | 20 | 16 / 7 px |
| test | 9 | 15 | 0 | 14 | 18 / 7 px |
| Total | 90 | 138 | 0 | 101 | 14 / 7 px |

Minimum width and height are measured independently. All annotations have
`iscrowd=0`. This sample has no negative images, and many goggles are very small.
Final data collection must include representative negatives and varied workers,
lighting, poses, occlusion, and camera conditions.

## Split limitation

All three splits contain filenames identifying the same source video:
`WhatsApp Video 2026-09-15 at 3_00_09 PM_mp4`. This implies source overlap;
nearby video frames cannot establish independent generalization. Preserve these
sample splits for pipeline development only. Final train/validation/test splits
must be separated by source video, worker, or recording session. Reliable group
metadata is required; filenames and exact image hashes cannot certify independence.

## Reproduce

```powershell
.\.venv\Scripts\python.exe tools/validate_coco.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The command validates all splits before writing cleaned annotations. It then
loads every image and annotation through MMDetection's `CocoDataset`,
`LoadImageFromFile`, `LoadAnnotations`, and `PackDetInputs`. It checks that all
boxes and labels survive loading and uses `DetLocalVisualizer` to produce three
ground-truth examples per split.

| Cleaned annotation | Image directory | Ground-truth preview |
| --- | --- | --- |
| `artifacts/phase2/train.json` | `data/sample_goggles/train/` | `artifacts/phase2/train_preview.jpg` |
| `artifacts/phase2/valid.json` | `data/sample_goggles/valid/` | `artifacts/phase2/valid_preview.jpg` |
| `artifacts/phase2/test.json` | `data/sample_goggles/test/` | `artifacts/phase2/test_preview.jpg` |

Each annotation's `file_name` remains relative to its copied image directory.
`validation.txt` records split statistics, exact-image
overlap, source-video warnings, and the final acceptance result. A failed rerun
may leave earlier generated artifacts, so the report must say PASS before use.

## Acceptance

Passed on 2026-09-23. MMDetection loaded all 90 images and 138 annotations,
preserving box coordinates and mapping every goggles label to 0. All three
preview sheets (nine images total) were visually inspected: boxes align with
the annotated goggles/eye regions and display the normalized class name.
This is a sample visual inspection, not an exhaustive labeling-quality audit.
There are no identical decoded images across splits; the shared-video warning
still applies. Six regression tests passed. Phase 3 has not started.

During verification, the execution sandbox blocked YAPF's user-cache creation;
the MMDetection verification completed successfully with approved cache access.
The visualizer also requires conversion from `HorizontalBoxes` to tensors,
which the tool performs only at the visualization boundary.

## File changes

- Created `tools/validate_coco.py`: validation, normalization, loader verification,
  split statistics, overlap checks, and MMDetection previews.
- Created `tests/test_validate_coco.py`: six regression tests for corruption,
  invalid references/geometry, and preserving the source during normalization.
- Created `DATASET.md`: dataset results, limitations, and reproduction commands.
- Updated `README.md`: Phase 2 completion and verification commands.
- Generated the three cleaned JSON files and three preview images listed above,
  plus `artifacts/phase2/validation.txt`: ignored, reproducible validation outputs.
- Removed no files; source images and annotations remain unchanged.
