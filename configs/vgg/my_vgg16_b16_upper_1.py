_base_ = ['../_base_/datasets/voc_bs16.py', '../_base_/default_runtime.py']

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=10),
    neck=None,
    head=dict(
        type='MultiLabelClsHead',
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# dataset settings
dataset_type = 'PedestrianUpper'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_prefix = 'data/train1A'
ann_file_train = 'data/labels/upper_train.csv'
ann_file_val = 'data/labels/upper_val.csv'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_file_train,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_file_val,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_file_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, save_best="MF1", greater_keys=['MF1'], metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'MF1', 'OF1'])

# load model pretrained on imagenet
load_from = 'checkpoints/vgg16_batch256_imagenet_20210208-db26f1a5.pth'
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=20, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=40)