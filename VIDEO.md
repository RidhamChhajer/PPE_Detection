# Full video and application verification

Phases 5 and 6 passed on 2026-09-23 on Windows / RTX 2050.
The model remains sample/demo-only; these are engineering acceptance results.

## Complete video

```powershell
.\.venv\Scripts\python.exe tools/process_video.py "data/sample_media/factory_video.mp4" artifacts/phase5/factory_annotated.mp4
```

The tool refuses to overwrite an existing output; choose a new filename on a rerun.

| Property | Verified result |
| --- | --- |
| Input and output frames | 754 |
| Dimensions | 480 × 864 |
| Average frame rate | 60.014062 FPS |
| Duration | 12.563722 seconds |
| Presentation timestamps | All normalized timestamps matched |
| Output | H.264, yuv420p, MP4 fast-start, no audio |
| Detections | 549, including repeated appearances across frames |
| CLI processing time | 26.1 seconds including initialization and verification |
| Runtime | TensorRT; PyTorch absent from the process |

The application does not skip frames, track identities, or decide compliance.
Every installed engine sees the same original frame before green annotations
are drawn. Distant/small goggles are missed by the sample model; successful
encoding and conversion must not be interpreted as adequate detection coverage.

Output is written to a partial file, decoded again, and checked against the
input's dimensions, frame count, timing, duration, and codec. Only then is it
published as complete. Failures remove partial output. Input timing is preserved
with PyAV timestamp/duration handling, including variable-frame-rate sources.

## Browser acceptance

The complete supplied clip was uploaded through the local browser interface.
Progress reached 100%; the page reported all 754 frames. The browser decoded
the result at 480 × 864 with duration 12.563722 seconds, played it successfully,
and emitted a download event from the download button. No browser console
errors were observed. HTTP byte-range serving is covered by automated tests.

The app uses Waitress on loopback, one background job, a 512 MiB upload limit,
same-origin/CSRF checks, safe generated job paths, and expiry cleanup restricted
to application-owned files. Status is process-local; a server restart clears
job lookup. This is a local application, not a remotely deployed multiuser service.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

25 tests cover dataset validation/provenance, precision/recall matching, model
integrity, preprocessing/postprocessing, complete CFR/VFR video preservation,
multiple engines, failure cleanup, upload rejection, single-job exclusion,
CSRF/origin checks, range/download responses, and expiry ownership boundaries.
Injected processing failures deliberately exercise the error path.

## Remaining prerequisites

Generalization validation needs the independent datasets being prepared by the
team. Phase 7 requires a labeled safety-shoes dataset. Phase 8 requires earplug
data and the actual camera setup. No placeholder models or simulated accuracy
claims have been added for those phases.
