# Models and measured results

The [registry](../configs/models.json) defines goggles-v1 and footwear-v1,
ordered labels, BGR colors, thresholds and full checkpoint/ONNX SHA256 hashes.
Dataset versions are goggles dataset v1 and footwear dataset v1.

| Model | Selected checkpoint under work_dirs |
| --- | --- |
| goggles-v1 | goggles_merged_fp32/best_coco_bbox_mAP_epoch_32.pth |
| footwear-v1 | footwear_v1_fp32/best_coco_bbox_mAP_epoch_28.pth |

Evaluation uses saved configs and true FP32 with both TF32 paths disabled.
Precision/recall use confidence 0.30 and matching IoU 0.50. COCO AP uses
maxDets=100 and averages classes; overall precision/recall are micro-averaged.

| Model/class | Split | Precision | Recall | AP50 | mAP50-95 | TP/FP/FN |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Goggles | valid | .9412 | .5517 | .6582 | .3193 | 16/1/13 |
| Goggles | test | .8000 | .3333 | .6747 | .2452 | 4/1/8 |
| Footwear overall | valid | .4923 | .6809 | .6099 | .3591 | 32/33/15 |
| SAFETY_SHOE | valid | .4630 | .8621 | .7529 | .4403 | 25/29/4 |
| NOT_SAFETY_SHOE | valid | .6364 | .3889 | .4669 | .2779 | 7/4/11 |
| Footwear overall | test | .4576 | .5870 | .4971 | .2851 | 27/32/19 |
| SAFETY_SHOE | test | .4528 | .8276 | .6617 | .3995 | 24/29/5 |
| NOT_SAFETY_SHOE | test | .5000 | .1765 | .3324 | .1707 | 3/3/14 |

Goggles has five test images; footwear has 34. Reports and class-colored previews
are under each selected run's `evaluation/`. Private previews are not published.
Runtime releases contain ONNX; optional training assets contain byte-identical
checkpoints and portable resolved configs/provenance. Hardware-specific TensorRT
metadata stays in `models/registry.local.json`. See [Limitations](LIMITATIONS.md).
