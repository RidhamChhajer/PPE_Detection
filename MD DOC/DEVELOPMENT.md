# Development and release preparation

Install `requirements-dev.txt`, then run:

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

Training-stack tests run when those optional packages are installed and skip in
a runtime-only environment. CPU tests generate tiny ONNX/video fixtures and use
local/mocked downloads; tests need no Internet. Real end-to-end checks use
`tools/check_app.py --source <video>` and `tools/process_video.py <source> <output>`.
These commands default to Auto (usable TensorRT, then OpenVINO Intel GPU, then
ONNX CPU). Pass --backend openvino, --backend tensorrt or --backend onnx to force one.
The browser smoke test is `node tests/browser_smoke.cjs <server-url> [input-video]`
with Playwright available to Node and Chrome installed. PPE_CHROME can point to
a different Chrome executable. It simulates camera responses for UI state checks;
the optional input video uses the real server backend. Screenshots stay under
ignored artifacts/browser. The Python live tests exercise actual session logic
using two fake camera devices and independent role-tagged frames.

The manifest is source; ONNX/checkpoints are Release assets; engines remain local.
`tools/prepare_repository.py` audits intended source and Git history read-only,
prepares a source bundle and release checksums. Private audit details stay under
ignored `local/`. No commit, remote push or history rewrite is performed.

For the code+models group handoff, use tools/package_deployment.py with a new ZIP
destination. It includes only the selected, hash-verified model artifacts and
verification images in addition to the source allowlist. It deliberately excludes
development engines. Default deployment/setup.ps1 installs Auto/OpenVINO; use
-Backend tensorrt to build engines on the recipient PC when ready. Never run
that GPU setup on the sender merely to make the ZIP.

Before publication: assign repository/release URL, select a license, review
model/data redistribution rights and source inventory, then publish manually.
`.gitignore` cannot remove committed private files. Obtain approval before a
narrow `git filter-repo --path AFFECTED_PATH --invert-paths` cleanup and coordinate
any later force-push. Rotate exposed credentials independently. A reviewed clean
source bundle can start a new public repository while preserving local history.

See [Verification](VERIFICATION.md) for actual results and [Cleanup](CLEANUP.md)
for removed and preserved material.
