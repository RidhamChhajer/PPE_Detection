# Training and continuation

Install the optional stack from [Installation](INSTALLATION.md) and prepare
[COCO data](DATASETS.md). Training is never required for ordinary inference.

```powershell
python tools/train.py --task goggles --work-dir work_dirs/goggles-v2 --model-version v2 --dataset-version v2 --check-only
python tools/train.py --task goggles --work-dir work_dirs/goggles-v2 --model-version v2 --dataset-version v2
python tools/train.py --task footwear --work-dir work_dirs/footwear-v2 --model-version v2 --dataset-version v2
```

`--check-only` validates and exercises every image through its input pipeline
without fitting. Use `--annotations`/`--images` for custom data, or `--config`
for another compatible detector recipe (classes come from metainfo).
Structural validation is always mandatory. `--strict-provenance` optionally
requires independent source groups and reviewed negative coverage.

The default initializes from official COCO weights. Use
`--pretrained release_assets/v1/goggles-v1.pth` to fine-tune existing weights in a
fresh run. Keep class order/head shape compatible; changed classes need deliberate
head adaptation. A custom config with `load_from=None` trains from random weights.

For interrupted runs, use identical arguments plus `--resume`. The tool selects
the run's `last_checkpoint`, verifies its location and checks dataset/optimizer
compatibility. It never substitutes initialization weights. Fresh runs refuse
nonempty work directories.

Recipes use FP32, AdamW lr=0.0001, accumulation of two microbatches and clipping
norm 10. `FiniteOptimWrapper` stops before updates on non-finite losses,
gradients or overflowing norms. Negative images and empty random crops remain
supported. Never resume a known damaged checkpoint.

The earlier unstable goggles run developed non-finite losses after epoch 10.
Selected v1 weights came from the subsequent stable FP32 runs. No retraining
occurred during application packaging. Legacy saved configs stay readable;
new runs record model/dataset versions in the permanent workflow.
