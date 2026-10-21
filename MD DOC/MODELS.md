# RTMDet-L models and measured results

The active model revision is v2, trained on Kaggle with RTMDet-L. Class order remains goggles; SAFETY_SHOE, NOT_SAFETY_SHOE. These development datasets do not establish field accuracy.

Selected checkpoints (unmodified copies of the archives in Previous):
- Goggles: work_dirs/goggles_rtmdet_l_v2/best_coco_bbox_mAP_epoch_28.pth
- Footwear: work_dirs/footwear_rtmdet_l_v2/best_coco_bbox_mAP_epoch_14.pth

Original Kaggle configurations are retained in each run's original folder. Local configurations change dataset paths and loader workers; model architecture, preprocessing and weights are preserved. Archive logs are retained. The epoch 38/40 checkpoints remain inside the original ZIPs.

Evaluation uses FP32 with TF32 disabled, confidence 0.30 and matching IoU 0.50. AP50 and mAP50-95 are COCO metrics. Test splits contain five goggles and 34 footwear images.

## Goggles

```text
Model: goggles-v2; dataset: goggles dataset v1
Checkpoint: work_dirs/goggles_rtmdet_l_v2/best_coco_bbox_mAP_epoch_28.pth
Performance depends on training data. This is not a safety certification system.
Checkpoint SHA256: f5862021fc0101f82b5fd8bb97ec6ec795ed6574e8e894412615758bed322231
Ordered labels: ('goggles',)
Reference arithmetic: FP32, CUDA matmul and cuDNN TF32 disabled.
valid: precision=0.9231, recall=0.4138, TP=12, FP=1, FN=17 (confidence>=0.30, IoU>=0.50); AP50=0.5017; mAP50-95=0.2807
valid/goggles (COCO ID 1): precision=0.9231, recall=0.4138, TP=12, FP=1, FN=17; AP50=0.5017; mAP50-95=0.2807
test: precision=0.5556, recall=0.4167, TP=5, FP=4, FN=7 (confidence>=0.30, IoU>=0.50); AP50=0.6552; mAP50-95=0.1563
test/goggles (COCO ID 1): precision=0.5556, recall=0.4167, TP=5, FP=4, FN=7; AP50=0.6552; mAP50-95=0.1563
Evaluation completed. Visual review is required before model qualification.
```

## Footwear

```text
Model: footwear-v2; dataset: footwear dataset v1
Checkpoint: work_dirs/footwear_rtmdet_l_v2/best_coco_bbox_mAP_epoch_14.pth
Performance depends on training data. This is not a safety certification system.
Checkpoint SHA256: 50d4d203742e72c9a410fbd0000737e1b713b0e5e0c304ee8395b1346e26009b
Ordered labels: ('SAFETY_SHOE', 'NOT_SAFETY_SHOE')
Reference arithmetic: FP32, CUDA matmul and cuDNN TF32 disabled.
valid: precision=0.4474, recall=0.7234, TP=34, FP=42, FN=13 (confidence>=0.30, IoU>=0.50); AP50=0.6017; mAP50-95=0.3196
valid/SAFETY_SHOE (COCO ID 1): precision=0.4237, recall=0.8621, TP=25, FP=34, FN=4; AP50=0.6823; mAP50-95=0.3183
valid/NOT_SAFETY_SHOE (COCO ID 2): precision=0.5294, recall=0.5000, TP=9, FP=8, FN=9; AP50=0.5211; mAP50-95=0.3208
test: precision=0.3467, recall=0.5652, TP=26, FP=49, FN=20 (confidence>=0.30, IoU>=0.50); AP50=0.3861; mAP50-95=0.1761
test/SAFETY_SHOE (COCO ID 1): precision=0.3529, recall=0.8276, TP=24, FP=44, FN=5; AP50=0.5573; mAP50-95=0.2273
test/NOT_SAFETY_SHOE (COCO ID 2): precision=0.2857, recall=0.1176, TP=2, FP=5, FN=15; AP50=0.2148; mAP50-95=0.1249
Evaluation completed. Visual review is required before model qualification.
```

The larger architecture did not improve these local test results: previous RTMDet-S test mAP50-95 was 0.2452 for goggles and 0.2851 for footwear. Review false positives, missed objects and representative independent recordings before deployment.
