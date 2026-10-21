# Camera routing and automatic inference

Software implementation completed on 21 October 2026. HP TensorRT deployment
and physical two-camera performance checks remain deferred.

## Inputs and routing

| Input | Models |
| --- | --- |
| Uploaded full video | All enabled detectors, currently goggles and footwear |
| Face camera | Entries with camera_role face, currently goggles-v2 |
| Feet camera | Entries with camera_role feet, currently footwear-v2 |
| One camera assigned to both roles | Both models, one physical capture and combined preview |

Roles are explicit registry metadata, validated on discovery and preserved by
export. The goggles engine is never invoked on feet-camera pixels. The current
goggles model has no earplug class. Add a trained earplug model with role face
later; there is no placeholder detector.

Select face only, feet only, one camera for both, or two different cameras.
Not used disables a role. At least one role must be selected. Only active models
are loaded; a shared physical device opens once and both models see original
pixels before annotations. USB indices may change across reconnects; recheck
assignments. Windows discovery lists devices without opening nonexistent indices;
selected-device failures still report connection/permission errors. Backend fallback
matches device paths, never assumes two backend indices refer to the same camera.

## Runtime and lifecycle

Auto is the default app/full-video backend. It probes both models, preferring
local TensorRT, then OpenVINO on an enumerated Intel GPU, then ONNX CPU. It never
builds engines. Selection and fallback reasons appear on the page and terminal.
Explicit --backend tensorrt/openvino/onnx never silently changes backend.
OpenVINO consumes the same ONNX assets after contract/hash validation, runs FP32,
and uses its own inference request (not ONNX Runtime for per-frame inference).
ONNX Runtime checks the model contract and provides CPU fallback. CPU means cores.

One capture worker per unique camera maintains a latest-frame slot. One inference worker
owns all detector contexts, loads each model once and alternates roles fairly.
Live processing drops superseded frames; upload processes every original frame.
Annotations are applied to original pixels after the selected detectors run.

Each feed has a separate MJPEG endpoint, status and metrics. Disconnect clears
that feed's output while a healthy feed continues. Frames older than three
seconds are hidden. Stop joins workers and releases owned handles/contexts.
A blocked driver read can delay stop; the session stays reserved until it returns.
Use Stop cameras, reconnect, then Start cameras to retry. The app does not reconnect
devices automatically or guess new USB assignments.

Upload and live use the same mode lock and cannot run together. Discovery is
also blocked while either is active. API POST requests keep CSRF/origin checks.

## API

- GET /api/cameras: available device indices.
- POST /api/live/start: face/feet integer indices or null for Not used; duplicate indices allowed.
- POST /api/live/stop: stop active cameras.
- GET /api/live/status: session state/generation and unique feeds, each with its active roles.
- GET /api/live/face/stream and /api/live/feet/stream: active preview endpoints.

A shared feed uses the first assigned role's endpoint and one visible preview.
Unassigned endpoints return 409. No capture/model work is duplicated.

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
| --- | --- | --- | --- | --- |
| Select/Listbox | Native camera select | DESIGN.md, app/templates/index.html | native | tests/browser_smoke.cjs |
| Form | Upload form and camera assignment | app/static/app.js, app/app.py | upload / single / shared / two cameras | tests/test_app.py, tests/browser_smoke.cjs |
| Scrollbar | Global stylesheet | app/static/style.css, DESIGN.md | forced colors | browser inspection |
| Toast | Inline alerts and status | app/templates/index.html | error / status | tests/browser_smoke.cjs |

Native OS popup geometry is accepted. No separate UI library, toast framework,
worker identities, missing-PPE alarms, or cross-camera person matching is added.
The existing display-only shoe pairing remains unchanged.

## Deferred: HP deployment

Target: HP EliteDesk 800 G5 TWR, RTX 3050 6 GB, 16 GB RAM and two Arducam AR0234
USB cameras. Identify actual OS, CPU, driver and USB/camera modes on that PC.
Install a compatible GPU runtime and rebuild/verify engines there. Start from the
previously validated choices: goggles FP16, footwear FP32. Footwear FP16 must
pass parity independently before it can replace FP32.

Measure negotiated camera modes, USB throughput, sustained per-feed FPS and
visible end-to-end latency with both feeds active. The host latency metric
excludes exposure, driver buffers and browser delivery. Verify the <=3-second
target with a timed scene. CPU fallback speed also needs measurement. Hardware
checks and industrial accuracy validation are not implied by software tests.
