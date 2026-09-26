# Export and deployment

Register the version, class order and selected checkpoint hash, then evaluate:

```powershell
python tools/evaluate.py --model-id goggles-v1 --run-dir work_dirs/goggles_merged_fp32
python tools/evaluate.py --model-id footwear-v1 --run-dir work_dirs/footwear_v1_fp32
python tools/export_models.py --model-id goggles-v1 --run-dir work_dirs/goggles_merged_fp32
python tools/export_models.py --model-id footwear-v1 --run-dir work_dirs/footwear_v1_fp32
```

Only the unique best checkpoint is selected. An explicit different checkpoint
is rejected; epoch_40 is never substituted. Successful evaluation must match
the checkpoint, saved config, annotation hashes and reference precision.
The exporter runs ONNX checker and every test image before installing candidates.
Original checkpoint contents and saved training configs remain unchanged.
Downloaded checkpoints can be evaluated with `--checkpoint` and `--config`
pointing to the matching release files (and local datasets at the config paths).
The checkpoint SHA256 must match the registry. No dataset is needed for inference.

## Contract

- Input `images`: float32 `[1,3,640,640]`, BGR; aspect-preserving bilinear resize
  with rounded dimensions; right/bottom padding 114; mean `[103.53,116.28,123.675]`,
  std `[57.375,57.12,58.395]`; contiguous NCHW.
- Output `detections`: float32 `[1,8400,4+C]`, input-pixel xyxy then all sigmoid
  class scores in registry order. Class IDs are zero-based score-column indices.
- Embedded ONNX metadata binds model ID, labels and contract to the registry.
  Runtime validates hash, names/shapes/dtypes, metadata and finite outputs.
- Structured detections include box, confidence, class ID and label. Class-aware
  NMS uses IoU .65, confidence .30 and max 300/model. Zero-area boxes are removed
  before the cap; coordinates are rescaled/clipped to the source frame.

Parity requires exact final class/count results and one-to-one matches: raw boxes
<=2 input pixels for either score >=.10; every raw score <=.03; final boxes <=3
original pixels and scores <=.03. Native MMDetection/adapter agreement is
<=.05 pixels/.0001 score. Both PyTorch TF32 paths and builder TF32 are disabled.

## TensorRT, optional and local

Stop the app; build one model at a time on the target GPU:

```powershell
python tools/export_models.py --model-id goggles-v1 --run-dir work_dirs/goggles_merged_fp32 --build-tensorrt
python tools/export_models.py --model-id footwear-v1 --run-dir work_dirs/footwear_v1_fp32 --build-tensorrt --precision stable_scores
python -m app.app --backend tensorrt --registry models/registry.local.json
```

Goggles uses FP16. Footwear needs `stable_scores`: FP32 shared/classification
layers, FP16 regression convolutions. All-FP16 crossed the confidence threshold
and failed parity; tolerances were not relaxed. Hashes/embedded labels are checked.
Failure preserves installed assets. Engines are excluded from releases; rebuild
and reverify on the final RTX 3050. See [Installation](INSTALLATION.md).

## Upgrading

Use immutable version filenames/IDs such as goggles-v2. Stop the server, install
verified files, atomically replace the manifest to enable v2/disable v1, restart
and smoke-test. Keep prior manifest/files for rollback. Ordinary users download
ONNX only; training users may also download checkpoints/configs. Downloads use
temporary files and hash verification before atomic replacement.
