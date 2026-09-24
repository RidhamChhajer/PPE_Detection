# PPE Detection Project - Implementation Plan

## 1. Current deliverable

Build a local Windows application that:

1. Accepts one prerecorded full-frame video.
2. Runs every currently available PPE detector automatically.
3. Draws a green bounding box, class name, and confidence score for every detection.
4. Produces one complete annotated video that can be played and downloaded in the frontend.

The first available detector will be for `goggles`. Safety-shoe and earplug detection will be added only when their datasets are available.

## 2. Fixed technical decisions

- Detector: RTMDet-S
- Training framework: MMDetection
- Dataset format: COCO
- Production inference: TensorRT FP16
- Development machine: Windows, RTX 2050, 12 GB RAM
- Final target machine: RTX 3050; TensorRT engines must be rebuilt on that machine
- Input: one uploaded video
- Output: one annotated video
- Multiple workers may appear in the same frame
- No tracking or worker IDs
- No ByteTrack
- No compliance/non-compliance decision in the first version
- No JSON, CSV, or database output
- No face/feet model selector in the frontend

## 3. Model arrangement

Use separate models for separate camera tasks:

- Goggles model: available first
- Safety-shoes model: add when its dataset is available
- Earplug model: add later, with ROI/SAHI only if testing proves it is necessary

For the current single-video application, all available TensorRT engines run automatically on the same frames and their detections are combined before drawing.

For the later two-camera system:

- Face camera runs the goggles detector and future earplug detector.
- Feet camera runs the safety-shoes detector.

Do not create empty shoe, earplug, SAHI, camera, tracking, or compliance modules before they are needed.

## 4. Implementation phases

Status on 2026-09-23: phases 1–6 passed their engineering acceptance checks.
Training/conversion used the explicitly labeled sample model; real-world
generalization remains unverified pending independent team data. Phase 7 awaits
safety-shoe data; phase 8 awaits earplug data and cameras. See `VIDEO.md` for
the complete-video/browser results and `DELIVERY.md` for the file inventory.

### Generalization requirement (user clarification, 2026-09-23)

The supplied goggles dataset and factory video are sample pipeline assets, not
the definition of the operating environment. Keep dataset locations configurable,
process full frames without worker/position-specific rules, and do not tune the
model against the factory demonstration clip. Keep sample-trained engines visibly
marked as demo-only. The user's team is preparing broader datasets.

Normal training must check independent recording/session groups and reviewed
negative-image coverage. Broader data and unseen-source/field evaluation are
required before claiming real-world effectiveness. Numerical TensorRT agreement
and a working frontend do not qualify a model for industrial deployment. Continue
the reusable software pipeline with explicitly labeled demonstration weights
while the broader datasets are being collected.

### Phase 1 - Environment check

Record the installed versions of:

- NVIDIA driver
- CUDA
- Python
- PyTorch
- MMEngine, MMCV, and MMDetection
- TensorRT

Choose one compatible version set before installing packages. Create only one project virtual environment.

**Done when:** CUDA is visible to PyTorch and a pretrained RTMDet-S model can run one test inference.

### Phase 2 - Dataset validation

Validate the sample goggles COCO dataset:

- Images referenced by JSON exist.
- Bounding boxes are within image boundaries.
- The real class is renamed consistently to `goggles`.
- The unused zero-annotation project category is removed.
- Class IDs are valid.
- Train, validation, and test statistics are printed.

The sample dataset is only for proving the pipeline. The final dataset must be split by source video, worker, or recording session, not by nearby frames from the same video.

**Done when:** MMDetection can load and visualize the cleaned annotations correctly.

### Phase 3 - Train and evaluate goggles model

Fine-tune COCO-pretrained RTMDet-S on the goggles dataset. Keep the configuration conservative for RTX 2050 memory and use AMP where supported.

Evaluate at least:

- Precision
- Recall
- AP50
- mAP50-95
- Detection behavior on the provided complete factory video

Do not claim industrial accuracy from the small sample dataset.

**Done when:** the best checkpoint produces sensible goggles detections in PyTorch on unseen video frames.

### Phase 4 - TensorRT conversion

Convert only the selected best checkpoint:

```text
MMDetection checkpoint -> ONNX -> TensorRT FP16 engine
```

Compare TensorRT output against PyTorch output on the same test images. Record inference latency and confirm that confidence scores and boxes remain acceptably close.

**Done when:** the TensorRT engine returns correct goggles detections without using PyTorch for production inference.

### Phase 5 - Video inference

Implement one video processor that:

1. Opens the uploaded video.
2. Preserves its resolution and playback FPS.
3. Runs all available TensorRT detector engines on every selected processing frame.
4. Combines their detections.
5. Draws green boxes with `class confidence` labels.
6. Writes a complete browser-playable output video without audio.

Start by processing every frame. Add frame skipping only if measurement shows it is required. Do not add tracking to compensate for skipped frames.

**Done when:** the provided full input video produces a complete annotated output video with the same duration and frame count.

### Phase 6 - Minimal frontend

Create one local page containing:

- Video upload
- Process button
- Progress/status display
- Output video player
- Download button

There is no model selector. The backend automatically runs every installed PPE engine.

**Done when:** a user can upload a video, wait for processing, play the result, and download it.

### Phase 7 - Add safety shoes

Begin only after a safety-shoes COCO dataset is supplied.

Train and export a separate RTMDet-S TensorRT engine. Register it in the existing inference pipeline so it runs automatically alongside the goggles engine. Do not redesign the frontend.

### Phase 8 - Add earplugs and live cameras

Begin only after an earplug dataset and the required camera setup are available.

Benchmark direct face-camera detection first. Add head ROI processing and SAHI only if direct detection is insufficient. Live two-camera capture and latency control belong to this phase, not the prerecorded-video phase.

## 5. Expected repository

Create files only as their phase begins. The expected repository after the goggles version is complete is:

```text
PPE DET/
|-- IMPLEMENTATION.md
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- goggles_rtmdet_s.py
|-- tools/
|   |-- validate_coco.py
|   |-- train.py
|   `-- export_tensorrt.py
|-- app/
|   |-- app.py
|   |-- detector.py
|   |-- video.py
|   `-- templates/
|       `-- index.html
|-- models/
|   `-- goggles.engine
`-- tests/
    |-- test_detector.py
    `-- test_video.py
```

Generated checkpoints, ONNX files, uploaded videos, output videos, caches, and virtual environments must be ignored by Git. They must not be committed as project source.

## 6. Useful-code and cleanup rules

These rules apply to every phase:

1. Implement only the active phase and current requirement.
2. Do not add placeholder abstractions for possible future features.
3. Project size is not restricted. Keep all code required for correct behavior, validation, error handling, testing, maintainability, and the stated architecture.
4. Never remove working or required logic merely to reduce file count or line count.
5. Prefer one clear implementation when alternatives provide no current benefit, but keep separate components when they have distinct responsibilities.
6. Temporary diagnostic scripts and extracted frames must be deleted when the phase finishes unless they become documented, reusable project tools.
7. Replaced code must be removed in the same change; do not leave `old`, `backup`, or `v2` copies.
8. Remove only demonstrably unused imports, dependencies, configuration values, dead functions, and superseded code.
9. Do not keep both experimental and final production inference paths after TensorRT is validated. PyTorch remains required for training, evaluation, and export.
10. Add a dependency only when current project code imports it or a documented build step requires it.
11. Refactoring must preserve behavior. Run the relevant tests or verification command before deleting the replaced implementation.
12. Before each handoff, list every created, changed, and removed file and explain the reason.
13. Do not start the next phase until the current phase has a working verification result.

## 7. Acceptance criteria for the first complete version

- One full video can be uploaded without selecting a PPE type.
- The complete video is processed, not a short preview.
- All workers' visible goggles are eligible for detection in every frame.
- Output boxes are green and show class name plus confidence.
- Output resolution, duration, and frame count match the input.
- The saved video plays in the browser and can be downloaded.
- Production detection uses the TensorRT FP16 engine.
- No tracking, worker IDs, compliance decisions, or unused future modules are present.
- Temporary files are cleaned automatically or removed after processing.
