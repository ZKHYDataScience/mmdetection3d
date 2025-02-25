model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        batch_first=False,
        max_voxels=80000,#80000,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-5, -5, -5, 150, 300, 30],#[-100, -100, -20, 100, 100, 20],
            voxel_size=[1, 1, 0.1],#[0.05, 0.05, 0.05]
            )),
    backbone=dict(
        type='SPVCNNBackbone',
        in_channels=4,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='torchsparse',
        drop_ratio=0.3),
    decode_head=dict(
        type='MinkUNetHead',
        channels=96,
        num_classes=14,#19
        dropout_ratio=0,
        # loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True),
        loss_decode=dict(reduction='none', type='LovaszLoss'),
        ignore_index=14),#19
    train_cfg=dict(),
    test_cfg=dict())
