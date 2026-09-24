"""Fine-tune RTMDet-S in the single project environment; supports resuming."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASKS = {
    'goggles': {
        'classes': ('goggles',),
        'config': ROOT / 'configs/goggles_rtmdet_s.py',
    },
    'footwear': {
        'classes': ('SAFETY_SHOE', 'NOT_SAFETY_SHOE'),
        'config': ROOT / 'configs/footwear_rtmdet_s.py',
    },
}


def check_run_directory(work_dir, resume, config_name):
    """Refuse nonempty destinations for fresh runs, regardless of task name."""
    saved = work_dir / config_name
    if resume:
        if not saved.is_file():
            raise ValueError('Resume requires a saved configuration in the run directory')
    elif work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError('Run directory is not empty; choose a fresh --work-dir')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(TASKS), default='goggles')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--pretrained', type=Path, default=ROOT / 'artifacts/phase1/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--sample', action='store_true', help='Explicit pipeline smoke test; not an independent-data training run')
    parser.add_argument('--check-only', action='store_true', help='Exercise the complete training input pipeline without fitting weights')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    task = TASKS[args.task]
    annotations = (args.annotations or ROOT / f'data/{args.task}/annotations').resolve()
    images = (args.images or ROOT / f'data/{args.task}/images').resolve()
    work_dir = (args.work_dir or ROOT / f'work_dirs/{args.task}').resolve()
    existing = check_run_directory(work_dir, args.resume, task['config'].name)
    from tools.validate_coco import SPLITS, check_independent_splits
    from tools.validate_training_dataset import validate_dataset
    if args.epochs <= 4:
        raise ValueError('Choose more than four epochs for the warmup/cosine schedule')
    documents, statistics = validate_dataset(annotations, images, task['classes'])
    if args.check_only:
        print(f'{args.task}: COCO validation complete.', flush=True)
    if not args.sample:
        check_independent_splits(documents)
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    if not args.check_only and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    if args.check_only:
        print(f'{args.task}: MMDetection registered.', flush=True)
    cfg = Config.fromfile(str(task['config']))
    if args.check_only:
        print(f'{args.task}: configuration loaded.', flush=True)
    for split, name in zip(SPLITS, ('train_dataloader', 'val_dataloader', 'test_dataloader')):
        cfg[name].dataset.ann_file = str(annotations / f'{split}.json')
        cfg[name].dataset.data_prefix.img = str(images / split)
    cfg.val_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.test_evaluator.ann_file = cfg.test_dataloader.dataset.ann_file
    cfg.work_dir = str(work_dir)
    cfg.load_from = str(args.pretrained.resolve())
    cfg.data_regime = 'sample' if args.sample else 'independent_sources'
    cfg.max_epochs = cfg.train_cfg.max_epochs = args.epochs
    cfg.param_scheduler[1].end = args.epochs
    cfg.param_scheduler[1].T_max = args.epochs - 4
    print('FP32 training; AdamW lr=0.0001; clipping max_norm=10; '
          'non-finite losses/gradients stop before update; '
          f'data regime={cfg.data_regime}.', flush=True)
    if args.check_only:
        from mmdet.registry import DATASETS
        checked = []
        for split, name in zip(SPLITS, ('train_dataloader', 'val_dataloader', 'test_dataloader')):
            dataset = DATASETS.build(cfg[name].dataset)
            print(f'{args.task}: checking {split} pipeline ({len(dataset)} images).', flush=True)
            if len(dataset) != len(documents[split]['images']):
                raise RuntimeError(f'{split} dataset unexpectedly filtered images')
            negative_count = 0
            for index in range(len(dataset)):
                packed = dataset[index]
                if packed['inputs'].ndim != 3 or packed['inputs'].shape[0] != 3:
                    raise RuntimeError(f'{split} pipeline returned an unexpected image tensor')
                if not torch.isfinite(packed['inputs']).all():
                    raise RuntimeError(f'{split} pipeline returned non-finite pixels')
                boxes = packed['data_samples'].gt_instances.bboxes
                boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
                if not torch.isfinite(boxes).all() or (len(boxes) and not (boxes[:, 2:] > boxes[:, :2]).all()):
                    raise RuntimeError(f'{split} pipeline returned invalid boxes')
                negative_count += len(boxes) == 0
            print(f'{split}: source negatives={statistics[split]["empty_images"]}; '
                  f'empty targets after transforms={negative_count}', flush=True)
            checked.append(f'{split}={len(dataset)}')
        print(f'{args.task} input pipelines: PASS ({", ".join(checked)}); no weights fitted.')
        return
    # A completed run must not be silently overwritten with another dataset.
    if existing.exists():
        if not args.resume:
            raise ValueError('Run directory exists; use --resume or a new --work-dir')
        previous = Config.fromfile(str(existing))
        if previous.optim_wrapper != cfg.optim_wrapper:
            raise ValueError('Cannot resume with a different optimizer/stability policy; use a fresh work directory')
        for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            if previous[name].dataset != cfg[name].dataset:
                raise ValueError('Cannot resume with a different dataset or pipeline')
    cfg.resume = args.resume
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
