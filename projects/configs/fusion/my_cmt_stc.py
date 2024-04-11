_base_ = ['../default_runtime.py']
custom_imports = dict(
    imports=['projects.mmdet3d_plugin'], allow_failed_imports=False)


# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 0.0002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1, decay_mult=5),
            'img_neck': dict(lr_mult=0.1)
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=120,
        end=120,
        by_epoch=True,
        eta_min=lr*5e-3,
        convert_to_iter_based=True)
]
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# dataset settings
dataset_type = 'STCrowdDataset'
data_root = 'data/STCrowd/'
class_names = ['Pedestrian']
point_cloud_range = [0, -18, -5, 36, 18, 1]
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
backend_args = None

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
    
ida_aug_conf = {
    "resize_lim": (0.5, 0.65),
    "final_dim": (480, 960),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "rand_flip": True,
}

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(
        type='LoadMultiViewImageFromFilesKitti',
        to_float32=True,
        color_type='color',
        backend_args=backend_args
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTransAll',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='CustomRandomFlip3D',
        flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
    ),
    dict(
        type='LoadMultiViewImageFromFilesKitti',
        to_float32=True,
        color_type='color',
        backend_args=backend_args
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1000, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='', img=''),
        ann_file='STCrowd_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='', img=''),
        ann_file='STCrowd_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='', img=''),
        ann_file='STCrowd_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'STCrowd_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'STCrowd_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')


voxel_size = [0.075, 0.075, 0.15]
out_size_factor = 8

model = dict(
    type='CmtDetector',
    use_grid_mask=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        bgr_to_rgb=False),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    pts_voxel_layer=dict(
        num_point_features=4,
        max_num_points=20,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[16, 16],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=16,
        sparse_shape=[41, 480, 480],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmtHead',
        num_query=900,
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        scalar=10,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        tasks=[
            dict(num_class=len(class_names), class_names=class_names),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-10.0, -30.0, -6.0, 50.0, 30.0, 2.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=len(class_names)), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
            )),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.5),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='kitti',
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.5),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[480, 480, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='kitti',
            grid_size=[480, 480, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=10),
    changestrategy=dict(
        type='ChangeStrategyHook',
        change_epoch=[-1, -1],
        change_strategy=['remove_GTSample', 'remove_DN'])
    )

load_from='models/nuim_r50.pth'

# resume = True