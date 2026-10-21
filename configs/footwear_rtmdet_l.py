_base_ = './goggles_rtmdet_l.py'

# Same RTMDet-L architecture and stability policy, with two footwear classes.
model = dict(bbox_head=dict(num_classes=2))
metainfo = dict(classes=('SAFETY_SHOE', 'NOT_SAFETY_SHOE'),
                palette=[(0, 255, 0), (0, 0, 255)])
train_dataloader = dict(dataset=dict(
    ann_file='data/footwear/annotations/train.json',
    data_prefix=dict(img='data/footwear/images/train/'), metainfo=metainfo))
val_dataloader = dict(dataset=dict(
    ann_file='data/footwear/annotations/valid.json',
    data_prefix=dict(img='data/footwear/images/valid/'), metainfo=metainfo))
test_dataloader = dict(dataset=dict(
    ann_file='data/footwear/annotations/test.json',
    data_prefix=dict(img='data/footwear/images/test/'), metainfo=metainfo))
val_evaluator = dict(ann_file='data/footwear/annotations/valid.json')
test_evaluator = dict(ann_file='data/footwear/annotations/test.json')
dataset_version = 'footwear dataset v1'
