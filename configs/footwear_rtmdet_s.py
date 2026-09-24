_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'
custom_imports = dict(imports=['tools.stable_optim'], allow_failed_imports=False)

# Two-class footwear detector for the same single-GPU Windows environment.
model = dict(
    backbone=dict(init_cfg=None, frozen_stages=1, norm_eval=True, norm_cfg=dict(type='BN')),
    neck=dict(norm_cfg=dict(type='BN')),
    bbox_head=dict(num_classes=2, norm_cfg=dict(type='BN')),
)
metainfo = dict(
    classes=('SAFETY_SHOE', 'NOT_SAFETY_SHOE'),
    palette=[(0, 255, 0), (0, 0, 255)],
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomResize', scale=(640, 640), ratio_range=(0.75, 1.25), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640), allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    batch_size=2, num_workers=0, persistent_workers=False, pin_memory=True,
    dataset=dict(data_root='', ann_file='data/footwear/annotations/train.json',
                 data_prefix=dict(img='data/footwear/images/train/'),
                 metainfo=metainfo, pipeline=train_pipeline,
                 filter_cfg=dict(filter_empty_gt=False, min_size=1)))
val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root='', ann_file='data/footwear/annotations/valid.json',
                 data_prefix=dict(img='data/footwear/images/valid/'), metainfo=metainfo))
test_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root='', ann_file='data/footwear/annotations/test.json',
                 data_prefix=dict(img='data/footwear/images/test/'), metainfo=metainfo))
val_evaluator = dict(ann_file='data/footwear/annotations/valid.json', metric='bbox',
                     proposal_nums=(100, 300, 1000))
test_evaluator = dict(ann_file='data/footwear/annotations/test.json', metric='bbox',
                      proposal_nums=(100, 300, 1000))

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, val_interval=2, dynamic_intervals=[])
optim_wrapper = dict(
    _delete_=True, type='FiniteOptimWrapper', accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2, error_if_nonfinite=True),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(type='CosineAnnealingLR', eta_min=0.000005, begin=4, end=max_epochs,
         T_max=max_epochs-4, by_epoch=True, convert_to_iter_based=True),
]
custom_hooks = []
default_hooks = dict(
    logger=dict(interval=16),
    checkpoint=dict(interval=2, max_keep_ckpts=1, save_best='coco/bbox_mAP', rule='greater'))
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
randomness = dict(seed=2026, deterministic=False)
auto_scale_lr = dict(enable=False)
load_from = 'artifacts/phase1/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
work_dir = 'work_dirs/footwear'
