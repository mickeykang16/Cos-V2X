import os
# ================ base config ===================
# Stage 2 V2X config: det + map + motion + planning
# Loads from stage1 V2X checkpoint and fine-tunes with motion/planning heads.
#
# Key differences from stage1:
#   - with_motion_plan = True
#   - ego_fut_ts = 12 (12 future ego timestamps for planning)
#   - total_batch_size = 12 (motion head is memory-intensive)
#   - num_epochs = 10 (fine-tuning on top of stage1)
#   - backbone lr_mult = 0.1 (freeze backbone more in stage2)
#   - load_from points to stage1 V2X checkpoint
#   - val/test samples_per_gpu = 1 (required for tracking)
version = 'trainval'

length = {'trainval': 12263, 'mini': 323}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
find_unused_parameters = False
log_level = "INFO"
work_dir = "work_dirs/6cams_both_infra_v6_v2x_stage2_v3"

total_batch_size = 24
num_gpus = 3
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 30
checkpoint_epoch_interval = 5

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)


# ================== model ========================
with_infra_cam = True
infra_cam_select = 0
num_veh_cams = 4
num_infra_cams = 2
num_cams = num_veh_cams + num_infra_cams  # 6
kmeans_folder = "data/kmeans/"

class_names = [
    "car",
    "pedestrian",
]
map_class_names = [
    'intersection',
    'lane',
    'ped_crossing',
]
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 12   # ← stage2: full 12-step future horizon for planning
ego_fut_mode = 6
queue_length = 4  # history + current

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=True,   # ← enabled in stage2
)

# Vehicle backbone/neck
_veh_backbone_cfg = dict(
    type="ResNet",
    depth=50,
    num_stages=4,
    frozen_stages=-1,
    norm_eval=False,
    style="pytorch",
    with_cp=True,
    out_indices=(0, 1, 2, 3),
    norm_cfg=dict(type="BN", requires_grad=True),
    pretrained="ckpt/resnet50-19c8e357.pth",
)
_veh_neck_cfg = dict(
    type="FPN",
    num_outs=num_levels,
    start_level=0,
    out_channels=embed_dims,
    add_extra_convs="on_output",
    relu_before_extra_convs=True,
    in_channels=[256, 512, 1024, 2048],
)

# Infrastructure backbone/neck
_infra_backbone_cfg = dict(
    type="ResNet",
    depth=50,
    num_stages=4,
    frozen_stages=-1,
    norm_eval=False,
    style="pytorch",
    with_cp=True,
    out_indices=(0, 1, 2, 3),
    norm_cfg=dict(type="BN", requires_grad=True),
    pretrained="ckpt/resnet50-19c8e357.pth",
)
_infra_neck_cfg = dict(
    type="FPN",
    num_outs=num_levels,
    start_level=0,
    out_channels=embed_dims,
    add_extra_convs="on_output",
    relu_before_extra_convs=True,
    in_channels=[256, 512, 1024, 2048],
)

# Shared instance bank (same as stage1 – weights loaded from checkpoint)
_shared_bank_cfg = dict(
    type="InstanceBank",
    num_anchor=900,
    embed_dims=embed_dims,
    anchor=kmeans_folder + "kmeans_det_900.npy",
    anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
    num_temp_instances=700 if temporal else -1,
    confidence_decay=0.95,
    feat_grad=False,
)

model = dict(
    type="SparseDriveV2X",
    num_veh_cams=num_veh_cams,
    num_infra_cams=num_infra_cams,
    freeze_infra=True,  # freeze all infra modules during stage2 training
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=_veh_backbone_cfg,
    img_neck=_veh_neck_cfg,
    infra_img_backbone=_infra_backbone_cfg,
    infra_img_neck=_infra_neck_cfg,
    depth_branch=None,
    head=dict(
        type="SparseDriveHeadV2X",
        task_config=task_config,
        freeze_infra=True,  # skip infra det/map loss (frozen modules)

        shared_instance_bank=_shared_bank_cfg,

        # ============================================================
        # Vehicle det head  (num_cams = 4)
        # ============================================================
        veh_det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=kmeans_folder + "kmeans_det_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=700 if temporal else -1,
                confidence_decay=0.95,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * num_single_frame_decoder
                + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ) if temporal else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 4,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups,
                num_levels=num_levels, num_cams=num_veh_cams,
                attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[[0,0,0],[0.45,0,0],[-0.45,0,0],
                               [0,0.45,0],[0,-0.45,0],[0,0,0.45],[0,0,-0.45]],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims, num_cls=num_classes,
                refine_yaw=True, with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0, num_temp_dn_groups=0,
                dn_noise_scale=[2.0]*3 + [0.5]*7, max_dn_gt=32, add_neg_dn=True,
                cls_weight=2.0, box_weight=0.25,
                reg_weights=[2.0]*3 + [0.5]*3 + [0.0]*4, cls_wise_reg_weights={},
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0]*3 + [1.0]*5 + [2.0]*2,
        ),

        # ============================================================
        # Infra det head  (num_cams = 2)
        # ============================================================
        infra_det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=kmeans_folder + "kmeans_det_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=700 if temporal else -1,
                confidence_decay=0.95,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * num_single_frame_decoder
                + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ) if temporal else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 4,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups,
                num_levels=num_levels, num_cams=num_infra_cams,
                attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[[0,0,0],[0.45,0,0],[-0.45,0,0],
                               [0,0.45,0],[0,-0.45,0],[0,0,0.45],[0,0,-0.45]],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims, num_cls=num_classes,
                refine_yaw=True, with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0, num_temp_dn_groups=0,
                dn_noise_scale=[2.0]*3 + [0.5]*7, max_dn_gt=32, add_neg_dn=True,
                cls_weight=2.0, box_weight=0.25,
                reg_weights=[2.0]*3 + [0.5]*3 + [0.0]*4, cls_wise_reg_weights={},
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0]*3 + [1.0]*5 + [2.0]*2,
        ),

        # ============================================================
        # Vehicle map head  (num_cams = 4)
        # ============================================================
        veh_map_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100, embed_dims=embed_dims,
                anchor=kmeans_folder + "kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6, feat_grad=True,
            ),
            anchor_encoder=dict(type="SparsePoint3DEncoder", embed_dims=embed_dims, num_sample=num_sample),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * num_single_frame_decoder_map
                + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ) if temporal_map else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 4,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups,
                num_levels=num_levels, num_cams=num_veh_cams,
                attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims, num_sample=num_sample,
                    num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims, num_sample=num_sample, num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes, num_sample=num_sample, roi_size=roi_size,
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=num_sample, roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels", gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id", with_instance_id=False, task_prefix='map',
        ),

        # ============================================================
        # Infra map head  (num_cams = 2)
        # ============================================================
        infra_map_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100, embed_dims=embed_dims,
                anchor=kmeans_folder + "kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6, feat_grad=True,
            ),
            anchor_encoder=dict(type="SparsePoint3DEncoder", embed_dims=embed_dims, num_sample=num_sample),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * num_single_frame_decoder_map
                + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ) if temporal_map else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 4,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups,
                num_levels=num_levels, num_cams=num_infra_cams,
                attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims, num_sample=num_sample,
                    num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims, num_sample=num_sample, num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes, num_sample=num_sample, roi_size=roi_size,
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=num_sample, roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels", gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id", with_instance_id=False, task_prefix='map',
        ),        shared_instance_bank=_shared_bank_cfg,

        # ============================================================
        # Fusion det head  (same as stage1)
        # ============================================================
        fusion_det_head=dict(
            type="FusionDetModule",
            embed_dims=embed_dims,
            num_veh_instances=900,
            num_infra_instances=900,
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=embed_dims,
                mode="add",
                output_fc=True,
                in_loops=1,
                out_loops=2,
            ),
            veh_cross_attn=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims, num_heads=num_groups,
                batch_first=True, dropout=drop_out,
            ),
            infra_cross_attn=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims, num_heads=num_groups,
                batch_first=True, dropout=drop_out,
            ),
            self_attn=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims, num_heads=num_groups,
                batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 4,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims, num_cls=num_classes,
                refine_yaw=True, with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0, num_temp_dn_groups=0,
                dn_noise_scale=[2.0]*3 + [0.5]*7, max_dn_gt=32, add_neg_dn=True,
                cls_weight=2.0, box_weight=0.25,
                reg_weights=[2.0]*3 + [0.5]*3 + [0.0]*4, cls_wise_reg_weights={},
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0]*3 + [1.0]*5 + [2.0]*2,
            task_prefix="fused_det",
        ),

        # ============================================================
        # Fusion map head  (same as stage1)
        # ============================================================
        fusion_map_head=dict(
            type="FusionMapModule",
            embed_dims=embed_dims,
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            veh_cross_attn=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims, num_heads=num_groups,
                batch_first=True, dropout=drop_out,
            ),
            infra_cross_attn=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims, num_heads=num_groups,
                batch_first=True, dropout=drop_out,
            ),
            self_attn=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims, num_heads=num_groups,
                batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 4,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims, num_sample=num_sample, num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes, num_sample=num_sample, roi_size=roi_size,
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=num_sample, roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            task_prefix='fused_map',
        ),

        # ============================================================
        # Motion + Planning head  (stage2 only)
        #
        # Input:
        #   det  → fused_det_output  (900 slots) → top-50
        #   map  → fused_map_output  (100 slots) → top-10
        # Both are in ego coordinate frame and have instance_feature +
        # anchor_embed from fusion heads.
        # ============================================================
        motion_plan_head=dict(
            type='MotionPlanningHead',
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            motion_anchor=f'{kmeans_folder}kmeans_motion_{fut_mode}.npy',
            plan_anchor=f'{kmeans_folder}kmeans_plan_{ego_fut_mode}.npy',
            embed_dims=embed_dims,
            decouple_attn=decouple_attn_motion,
            instance_queue=dict(
                type="InstanceQueue",
                embed_dims=embed_dims,
                queue_length=queue_length,
                tracking_threshold=0.1,
                feature_map_scale=(
                    input_shape[1] / strides[-1],
                    input_shape[0] / strides[-1],
                ),
            ),
            operation_order=(
                ["temp_gnn", "gnn", "norm", "cross_gnn", "norm", "ffn", "norm"] * 3
                + ["refine"]
            ),
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            cross_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups, batch_first=True, dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims, pre_norm=dict(type="LN"),
                embed_dims=embed_dims, feedforward_channels=embed_dims * 2,
                num_fcs=2, ffn_drop=drop_out, act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(
                type="MotionPlanningRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            motion_sampler=dict(type="MotionTarget"),
            motion_loss_cls=dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                loss_weight=0.2,
            ),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.4),
            planning_sampler=dict(
                type="PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            plan_loss_cls=dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                loss_weight=0.5,
            ),
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
            # GT ego_status is directly injected as input (with_gt_ego_status=True),
            # so predicting it again is circular. Disable the status prediction loss.
            plan_loss_status=dict(type='L1Loss', loss_weight=0.0),
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),
            planning_decoder=dict(
                type="HierarchicalPlanningDecoder",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
                use_rescore=True,
            ),
            # top-k selection from fused outputs
            num_det=50,   # top-50 from fused_det_output (900 slots)
            num_map=10,   # top-10 from fused_map_output (100 slots)
            # Collision-awareness loss weight.
            # Creates a direct gradient path:
            #   perception (fused det) → motion prediction quality
            #   → collision penalty accuracy → planning improvement
            # Set to 0.0 to disable (grad_norm instability observed).
            plan_loss_collision_weight=0.0,
            # Inject GT ego status (speed, yaw-rate, etc.) into ego_feature.
            with_gt_ego_status=True,
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"
data_root = os.getenv("V2XREAL_DATA_ROOT", "data/v2xreal/")
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR", load_dim=4, use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(class_names)),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size, simplify=False, normalize=False,
        sample_num=num_sample, permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img", "timestamp", "projection_mat", "image_wh",
            "gt_depth", "focal",
            "gt_bboxes_3d", "gt_labels_3d",
            'gt_map_labels', 'gt_map_pts',
            'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
            'ego_status',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=["img", "timestamp", "projection_mat", "image_wh", 'ego_status', 'gt_ego_fut_cmd'],
        meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],
    ),
]
eval_pipeline = [
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(class_names)),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type='VectorizeMap', roi_size=roi_size, simplify=True, normalize=False),
    dict(
        type='Collect',
        keys=[
            'vectors', "gt_bboxes_3d", "gt_labels_3d",
            'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
            'fut_boxes',
        ],
        meta_keys=['token', 'timestamp'],
    ),
]

input_modality = dict(
    use_lidar=False, use_camera=True,
    use_radar=False, use_map=False, use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
    use_valid_flag=True,
    with_infra_cam=with_infra_cam,
    infra_cam_select=infra_cam_select,
)
eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + 'nuscenes_infos_test.pkl',
    pipeline=eval_pipeline,
    test_mode=True,
)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_test.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
        samples_per_gpu=1,  # CRITICAL: must be 1 for tracking!
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_test.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
        samples_per_gpu=1,  # CRITICAL: must be 1 for tracking!
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=3e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            # Freeze backbones more aggressively in stage2 (fine-tuning)
            "img_backbone": dict(lr_mult=0.1),
            "infra_img_backbone": dict(lr_mult=0.1),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=True,
    with_motion=True,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval*3,
    eval_mode=eval_mode,
)

# ================== pretrained model ========================
# Load stage1 V2X checkpoint – all det/map/fusion weights are pre-trained.
# Only motion_plan_head weights are freshly initialised.
load_from = 'work_dirs/6cams_both_infra_v6_v2x/latest.pth'
resume_from = None
