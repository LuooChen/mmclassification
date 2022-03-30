_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        _delete_=True,
        type='MultiLabelLinearClsHead',
        num_classes=24,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# dataset settings
dataset_type = 'PedestrianMain7Props'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), adaptive_side='long'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Pad', pad_to_square=True, pad_val=(128,128,128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), adaptive_side='long'),
    dict(type='Pad', pad_to_square=True, pad_val=(128,128,128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_prefix = 'data/train22/train2_new'
ann_file_train = 'data/labels/main_7_prop_train.csv'
ann_file_val = 'data/labels/main_7_prop_val.csv'
data = dict(
    samples_per_gpu=64,
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

# Training schedule config
# lr is set for a batch size of 64
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[20,40])
runner = dict(type='EpochBasedRunner', max_epochs=60)
log_config = dict(interval=80)