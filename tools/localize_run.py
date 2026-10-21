"""Copy a saved Kaggle configuration with only local dataset/runtime paths changed."""
from copy import deepcopy
from pathlib import Path


def localize_config(cfg, root, task, run_dir):
    cfg = deepcopy(cfg)
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    for split, name in [('train', 'train_dataloader'), ('valid', 'val_dataloader'), ('test', 'test_dataloader')]:
        dataset = cfg[name].dataset
        dataset.data_root = ''
        dataset.ann_file = (root / 'data' / task / 'annotations' / f'{split}.json').as_posix()
        dataset.data_prefix.img = (root / 'data' / task / 'images' / split).as_posix()
        cfg[name].num_workers = 0
        cfg[name].persistent_workers = False
    cfg.val_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.test_evaluator.ann_file = cfg.test_dataloader.dataset.ann_file
    cfg.work_dir = run_dir.as_posix()
    cfg.load_from = None
    cfg.resume = False
    return cfg
