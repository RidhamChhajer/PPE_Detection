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

The manifest is source; ONNX/checkpoints are Release assets; engines remain local.
`tools/prepare_repository.py` audits intended source and Git history read-only,
prepares a source bundle and release checksums. Private audit details stay under
ignored `local/`. No commit, remote push or history rewrite is performed.

Before publication: assign repository/release URL, select a license, review
model/data redistribution rights and source inventory, then publish manually.
`.gitignore` cannot remove committed private files. Obtain approval before a
narrow `git filter-repo --path AFFECTED_PATH --invert-paths` cleanup and coordinate
any later force-push. Rotate exposed credentials independently. A reviewed clean
source bundle can start a new public repository while preserving local history.

See [Verification](VERIFICATION.md) for actual results and [Cleanup](CLEANUP.md)
for removed and preserved material.
