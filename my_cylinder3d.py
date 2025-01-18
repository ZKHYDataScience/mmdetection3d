backend_args = None
class_names = [
    'duiliao', 'dabi', 'meipeng', 'yunmeiche', 'dangmeiqiang', 'guidao','NoName1','NoName2'
                    'dimian', 'fenchen', 'neimen', 'shiwai', 'madao',
                    'neiqiang', 'menji', 'bian',
]
data_root = '/home/yiyi/GitHub/mmdetection3d/data/mySemKitti/'
dataset_type = 'MySemKittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
grid_shape = [
    480,
    360,
    32,
]
input_modality = dict(use_camera=False, use_lidar=True)
labels_map = dict({
    0: 19,
    1: 19,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 19,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 19
})
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.001
metainfo = dict(
    classes=[
        'duiliao', 'dabi', 'meipeng', 'yunmeiche', 'dangmeiqiang', 'guidao','NoName1','NoName2'
                    'dimian', 'fenchen', 'neimen', 'shiwai', 'madao',
                    'neiqiang', 'menji', 'bian',
    ],
    max_label=259,
    seg_label_mapping=dict({
        0: 19,
        1: 19,
        10: 0,
        11: 1,
        13: 4,
        15: 2,
        16: 4,
        18: 3,
        20: 4,
        252: 0,
        253: 6,
        254: 5,
        255: 7,
        256: 4,
        257: 4,
        258: 3,
        259: 4,
        30: 5,
        31: 6,
        32: 7,
        40: 8,
        44: 9,
        48: 10,
        49: 11,
        50: 12,
        51: 13,
        52: 19,
        60: 8,
        70: 14,
        71: 15,
        72: 16,
        80: 17,
        81: 18,
        99: 19
    }))
model = dict(
    backbone=dict(
        base_channels=32,
        grid_size=[
            480,
            360,
            32,
        ],
        input_channels=16,
        norm_cfg=dict(eps=1e-05, momentum=0.1, type='BN1d'),
        type='Asymm3DSpconv'),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            grid_shape=[
                480,
                360,
                32,
            ],
            max_num_points=-1,
            max_voxels=-1,
            point_cloud_range=[
                0,
                -3.14159265359,
                -4,
                50,
                3.14159265359,
                2,
            ]),
        voxel_type='cylindrical'),
    decode_head=dict(
        channels=128,
        loss_ce=dict(
            class_weight=None,
            loss_weight=1.0,
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_lovasz=dict(loss_weight=1.0, reduction='none', type='LovaszLoss'),
        num_classes=17,
        type='Cylinder3DHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=None,
    type='Cylinder3D',
    voxel_encoder=dict(
        feat_channels=[
            64,
            128,
            256,
            256,
        ],
        feat_compression=16,
        in_channels=6,
        return_point_feats=False,
        type='SegVFE',
        with_voxel_center=True))
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            30,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='mySemKitti_infos_val.pkl',
        backend_args=None,
        data_root=data_root,
        ignore_index=19,
        metainfo=dict(
            classes=[
                'duiliao', 'dabi', 'meipeng', 'yunmeiche', 'dangmeiqiang', 'guidao','NoName1','NoName2'
                    'dimian', 'fenchen', 'neimen', 'shiwai', 'madao',
                    'neiqiang', 'menji', 'bian',
            ],
            max_label=259,
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                backend_args=None,
                dataset_type='semantickitti',
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='MySemKittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='SegMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        backend_args=None,
        dataset_type='semantickitti',
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=128, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='mySemKitti_infos_train.pkl',
        backend_args=None,
        data_root=data_root,
        ignore_index=19,
        metainfo=dict(
            classes=[
                'duiliao', 'dabi', 'meipeng', 'yunmeiche', 'dangmeiqiang', 'guidao','NoName1','NoName2'
                    'dimian', 'fenchen', 'neimen', 'shiwai', 'madao',
                    'neiqiang', 'menji', 'bian',
            ],
            max_label=259,
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                backend_args=None,
                dataset_type='semantickitti',
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                rot_range=[
                    -0.78539816,
                    0.78539816,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0.1,
                    0.1,
                    0.1,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        type='MySemKittiDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        backend_args=None,
        dataset_type='semantickitti',
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0.1,
            0.1,
            0.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
tta_model = dict(type='Seg3DTTAModel')
tta_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        backend_args=None,
        dataset_type='semantickitti',
        seg_3d_dtype='np.int32',
        seg_offset=65536,
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        transforms=[
            [
                dict(
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=0.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=1.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=0.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    flip_ratio_bev_horizontal=1.0,
                    flip_ratio_bev_vertical=1.0,
                    sync_2d=False,
                    type='RandomFlip3D'),
            ],
            [
                dict(
                    rot_range=[
                        -0.78539816,
                        -0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        0.95,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        -0.78539816,
                        -0.78539816,
                    ],
                    scale_ratio_range=[
                        1.0,
                        1.0,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        -0.78539816,
                        -0.78539816,
                    ],
                    scale_ratio_range=[
                        1.05,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        0.95,
                        0.95,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        1.0,
                        1.0,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        1.05,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        0.95,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        1.0,
                        1.0,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    rot_range=[
                        0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        1.05,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
            ],
            [
                dict(keys=[
                    'points',
                ], type='Pack3DDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='mySemKitti_infos_val.pkl',
        backend_args=None,
        data_root=data_root,
        ignore_index=19,
        metainfo=dict(
            classes=[
                'duiliao', 'dabi', 'meipeng', 'yunmeiche', 'dangmeiqiang', 'guidao','NoName1','NoName2'
                    'dimian', 'fenchen', 'neimen', 'shiwai', 'madao',
                    'neiqiang', 'menji', 'bian',
            ],
            max_label=259,
            seg_label_mapping=dict({
                0: 19,
                1: 19,
                10: 0,
                11: 1,
                13: 4,
                15: 2,
                16: 4,
                18: 3,
                20: 4,
                252: 0,
                253: 6,
                254: 5,
                255: 7,
                256: 4,
                257: 4,
                258: 3,
                259: 4,
                30: 5,
                31: 6,
                32: 7,
                40: 8,
                44: 9,
                48: 10,
                49: 11,
                50: 12,
                51: 13,
                52: 19,
                60: 8,
                70: 14,
                71: 15,
                72: 16,
                80: 17,
                81: 18,
                99: 19
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                backend_args=None,
                dataset_type='semantickitti',
                seg_3d_dtype='np.int32',
                seg_offset=65536,
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='MySemKittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='SegMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

