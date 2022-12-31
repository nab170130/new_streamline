_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=10)))

CLASSES = ('Car', 'Van', 'Pedestrian', 'Truck', 'Misc', 'Tram', 'Person_sitting', 'Cyclist')

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train_fog=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/coco_annotations_foggy.json',
            img_prefix='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/',
            pipeline=train_pipeline,
            classes=CLASSES)),
    train_rain=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/coco_annotations_rain.json',
            img_prefix='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/',
            pipeline=train_pipeline,
            classes=CLASSES)),
    val_fog=dict(
        type=dataset_type,
        ann_file='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/coco_annotations_foggy.json',
        img_prefix='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/',
        pipeline=test_pipeline,
        classes=CLASSES),
    val_rain=dict(
        type=dataset_type,
        ann_file='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/coco_annotations_rain.json',
        img_prefix='/home/013/s/sn/snk170001/streamline/data/kitti/data_object/training/',
        pipeline=test_pipeline,
        classes=CLASSES))
evaluation = dict(interval=50, metric='bbox')
checkpoint_config = dict(interval=10)

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=100)  # actual epoch = 4 * 3 = 12