Run `.venv\Scripts\python.exe tools/download_models.py` to install the release's hash-verified models here:

- `upper.pt`: detection classes exactly `goggles`, `no_goggles`, `empty_wrist`, `not_empty_wrist`.
- `lower.pt`: detection classes exactly `SAFETY_SHOE`, `NOT_SAFETY_SHOE`.

Use custom YOLO26 detection checkpoints. Generic pretrained YOLO26 COCO weights do not detect these PPE states. RTMDet `.pth` files are incompatible. Train with `tools/train.py`; evaluate on independent recordings before installation. Stop the service, replace the files, then start the service again to load the new models.
