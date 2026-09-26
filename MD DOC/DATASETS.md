# Dataset contract

Use `data/<detector>/annotations/{train,valid,test}.json` and corresponding
`images/{train,valid,test}/`. COCO category IDs may be noncontiguous; category
array order must match saved model class order. IDs must be unique, image
dimensions accurate and boxes finite, positive-area and inside the image.
Negative images have no annotations; there is no background category.

```powershell
python tools/validate_training_dataset.py --annotations data/goggles/annotations --images data/goggles/images --classes goggles
python tools/validate_training_dataset.py --annotations data/footwear/annotations --images data/footwear/images --classes SAFETY_SHOE NOT_SAFETY_SHOE
```

Goggles dataset v1 training has 148 images, 220 boxes and five negatives. Fully
label every visible target class. Collect varied workers, lighting, camera
views, small/distant targets, occlusions, ordinary glasses and confusing footwear.
Review negatives and ambiguous labels. Near-identical frames do not replace variety.

Keep whole recordings/sessions together across splits and prefer independent
validation/test sources. Optional strict checks use `source_group` (or
`extra.source_group`) and reviewed-negative metadata. Filenames cannot prove
independence. Do not merge detectors' training sets without fully cross-annotating
every class; unannotated PPE otherwise becomes background.

Raw data, worker images/videos and private previews stay outside Git/releases.
Review data redistribution rights separately from software licensing.
