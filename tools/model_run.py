"""Shared, strict selection and provenance for completed RTMDet runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_reference_precision():
    """Use true FP32 for evaluation/export parity, matching TRT's TF32 policy."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def best_checkpoint(run_dir=None):
    candidates = list(Path(run_dir or ROOT/'work_dirs/goggles').glob('best_*.pth'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one selected best checkpoint, found {len(candidates)}')
    return candidates[0].resolve()


def saved_config(run_dir):
    candidates = list(Path(run_dir).glob('*.py'))
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one saved RTMDet configuration')
    return candidates[0].resolve()


def category_ids(document, labels):
    categories = document['categories']
    if len({c['id'] for c in categories}) != len(categories) or len({c['name'] for c in categories}) != len(categories):
        raise ValueError('Duplicate COCO category identity')
    # MMDetection COCO get_cat_ids follows annotation category order.
    selected = [c for c in categories if c['name'] in labels]
    if tuple(c['name'] for c in selected) != tuple(labels):
        raise ValueError('COCO category order disagrees with saved model labels')
    return tuple(c['id'] for c in selected)


def load_run(run_dir, checkpoint=None, config=None):
    from mmengine.config import Config
    run_dir = Path(run_dir).resolve()
    selected = Path(checkpoint).resolve() if config is not None and checkpoint is not None else best_checkpoint(run_dir)
    if checkpoint is not None and Path(checkpoint).resolve() != selected:
        raise ValueError('Only the selected best checkpoint in this run is allowed')
    config_path = Path(config).resolve() if config is not None else saved_config(run_dir)
    cfg = Config.fromfile(str(config_path))
    labels = tuple(cfg.metainfo.classes)
    if not labels or len(set(labels)) != len(labels) or cfg.model.bbox_head.num_classes != len(labels):
        raise ValueError('Saved class metadata disagrees with model head')
    for split, name in [('train','train_dataloader'), ('valid','val_dataloader'), ('test','test_dataloader')]:
        dataset = cfg[name].dataset
        if tuple(dataset.metainfo.classes) != labels:
            raise ValueError(f'{split} labels disagree with saved model')
        category_ids(json.loads(Path(dataset.ann_file).read_text(encoding='utf-8')), labels)
    return selected, config_path, cfg, labels


def identity(checkpoint, config_path, cfg, labels):
    return dict(reference_precision='fp32_tf32_disabled',
                checkpoint_sha256=sha256(checkpoint), config_sha256=sha256(config_path),
                labels=list(labels), valid_sha256=sha256(cfg.val_dataloader.dataset.ann_file),
                test_sha256=sha256(cfg.test_dataloader.dataset.ann_file))


def require_evaluation(run_dir, expected):
    import tomli
    receipt = Path(run_dir)/'evaluation/verified.toml'
    if not receipt.is_file():
        raise RuntimeError('Successful evaluation required before export')
    record = tomli.loads(receipt.read_text(encoding='utf-8'))
    if record.get('status') != 'passed' or any(record.get(k) != v for k,v in expected.items()):
        raise RuntimeError('Evaluation provenance does not match selected checkpoint/configuration/data')


def toml_fields(fields):
    return '\n'.join(f'{key} = {json.dumps(value)}' for key, value in fields.items()) + '\n'
