from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.ops import feature_maps_format

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        plan_loss_collision_weight=0.0,
        with_gt_ego_status=False,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map
        self.plan_loss_collision_weight = plan_loss_collision_weight
        self.with_gt_ego_status = with_gt_ego_status
        # GT ego status injection: encode current-frame ego kinematics
        # (speed, yaw rate, etc.) and add to ego_feature before planning.
        # ego_status is 10-dim: [vx, vy, ax, ay, yaw_rate, ...] in ego frame.
        if with_gt_ego_status:
            self.ego_status_encoder = nn.Sequential(
                nn.Linear(10, embed_dims),
                nn.ReLU(),
                nn.LayerNorm(embed_dims),
                nn.Linear(embed_dims, embed_dims),
            )

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
        infra_det_output=None,
        data=None,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== GT ego status injection ===========
        # Encode current-frame ground-truth ego kinematics and add to
        # ego_feature so the planner has direct access to speed/yaw-rate.
        # At inference time this can be replaced with CAN-bus measurements.
        if self.with_gt_ego_status and data is not None and "ego_status" in data:
            ego_status_embed = self.ego_status_encoder(
                data["ego_status"].float()
            )  # (bs, embed_dims)
            ego_feature = ego_feature + ego_status_embed.unsqueeze(1)

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) 
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        # When GT ego_status is injected as input (with_gt_ego_status=True),
        # store GT directly in the temporal queue instead of the predicted
        # plan_status (which receives no gradient and would be random).
        if self.with_gt_ego_status and data is not None and "ego_status" in data:
            _cache_status = data["ego_status"].float().detach().unsqueeze(1)  # (bs,10)->(bs,1,10) matches plan_status shape
        else:
            _cache_status = plan_status
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], _cache_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs,
        planning_model_outs,
        data,
        motion_loss_cache,
        det_output=None,
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(
            planning_model_outs, data,
            motion_model_outs=motion_model_outs,
            det_output=det_output,
        )
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]

        # Downsample GT if needed
        gt_trajs = data["gt_agent_fut_trajs"]
        gt_masks = data["gt_agent_fut_masks"]
        
        # Check if GT temporal dim matches fut_ts (60 vs 12)
        if gt_trajs[0].shape[-2] != self.fut_ts:
             # Assume GT is 60 (10Hz) and model is 12 (2Hz), so downsample by sum
             # Or generic: downsample by factor
             gt_steps = gt_trajs[0].shape[-2]
             factor = gt_steps // self.fut_ts
             new_gt_trajs = []
             new_gt_masks = []
             for i in range(len(gt_trajs)):
                 # (N, 60, 2) -> (N, 12, 5, 2) -> (N, 12, 2)
                 traj = gt_trajs[i].reshape(gt_trajs[i].shape[0], self.fut_ts, factor, 2).sum(dim=2)
                 mask = gt_masks[i].reshape(gt_masks[i].shape[0], self.fut_ts, factor).min(dim=2).values
                 new_gt_trajs.append(traj)
                 new_gt_masks.append(mask)
             gt_trajs = new_gt_trajs
             gt_masks = new_gt_masks

        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                gt_trajs,
                gt_masks,
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data,
                      motion_model_outs=None, det_output=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}

        gt_ego_fut_trajs = data['gt_ego_fut_trajs']
        gt_ego_fut_masks = data['gt_ego_fut_masks']

        if gt_ego_fut_trajs.shape[1] > self.ego_fut_ts:
            bs, ts_gt, d = gt_ego_fut_trajs.shape
            factor = ts_gt // self.ego_fut_ts
            gt_ego_fut_trajs = gt_ego_fut_trajs.reshape(bs, self.ego_fut_ts, factor, d).sum(dim=2)
            
            bs, ts_gt = gt_ego_fut_masks.shape
            gt_ego_fut_masks = gt_ego_fut_masks.reshape(bs, self.ego_fut_ts, factor).sum(dim=2) > 0
        # if gt_ego_fut_trajs.shape[1] == 12 and self.ego_fut_ts == 6:
        #     # Slice the first 6 steps (3 seconds) if dataset returns 12 steps (6 seconds)
        #     gt_ego_fut_trajs = gt_ego_fut_trajs[:, :6, :]
        #     gt_ego_fut_masks = gt_ego_fut_masks[:, :6]
            
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                gt_ego_fut_trajs,
                gt_ego_fut_masks,
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        # ── Collision-awareness loss ──────────────────────────────────────
        # Uses predicted agent trajectories (from motion head) to penalise
        # ego planning modes that approach agents too closely.
        # Gradient path: perception quality → motion prediction quality
        #                → collision detection accuracy → planning quality.
        # Only active when plan_loss_collision_weight > 0 and both
        # motion_model_outs and det_output are provided.
        if (
            self.plan_loss_collision_weight > 0
            and motion_model_outs is not None
            and det_output is not None
            and len(reg_preds) > 0
        ):
            collision_loss = self._plan_collision_loss(
                reg_preds[-1], motion_model_outs, det_output, data
            )
            output["planning_loss_collision"] = (
                collision_loss * self.plan_loss_collision_weight
            )

        return output

    def _plan_collision_loss(
        self,
        plan_reg_all,
        motion_model_outs,
        det_output,
        data,
        margin=2.0,
        conf_thresh=0.3,
    ):
        """Soft collision penalty between ego planning trajectories and
        predicted agent trajectories.

        Args:
            plan_reg_all: (bs, 3*ego_fut_mode, ts, 2)  – all-cmd ego deltas
            motion_model_outs: dict  – motion head output
            det_output: dict  – detection head output (agent boxes & classes)
            data: dict  – batch data (contains gt_ego_fut_cmd)
            margin: float  – minimum clearance distance in metres
            conf_thresh: float  – ignore agents below this detection confidence

        Returns:
            scalar collision loss
        """
        bs = plan_reg_all.shape[0]
        bs_idx = torch.arange(bs, device=plan_reg_all.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)  # (bs,)

        # Select the GT-command's planning modes: (bs, ego_fut_mode, ts, 2)
        plan_reg = plan_reg_all.reshape(
            bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2
        )
        plan_reg = plan_reg[bs_idx, cmd]           # (bs, ego_fut_mode, ts, 2)
        ego_abs  = plan_reg.cumsum(dim=-2)         # absolute ego positions

        # Agent predicted trajectories
        # motion_model_outs["prediction"][-1] is over the top-K agents
        # selected inside MotionPlanningHead.forward() (num_det = self.num_det).
        # We must select the same top-K from det_output to get matching xy.
        motion_reg = motion_model_outs["prediction"][-1]        # (bs, K, mode, ts_m, 2)
        motion_cls = motion_model_outs["classification"][-1].sigmoid()  # (bs, K, mode)

        # Select top-K from det_output to match motion head selection
        det_cls_full   = det_output["classification"][-1].sigmoid()           # (bs, N_full, C)
        det_conf_full  = det_cls_full.max(dim=-1).values                       # (bs, N_full)
        det_anch_full  = det_output["prediction"][-1]                          # (bs, N_full, A)
        K = motion_reg.shape[1]
        _, topk_idx = torch.topk(det_conf_full, K, dim=1)                     # (bs, K)
        bs_idx2 = torch.arange(bs, device=topk_idx.device)[:, None].expand_as(topk_idx)
        det_anchors = det_anch_full[bs_idx2, topk_idx]                        # (bs, K, A)
        det_conf    = det_conf_full[bs_idx2, topk_idx]                        # (bs, K)

        N    = K
        ts_m = motion_reg.shape[3]
        ts   = min(self.ego_fut_ts, ts_m)

        # Best motion mode per agent: (bs, N, ts_m, 2)
        best_idx    = motion_cls.argmax(dim=-1)           # (bs, N)
        best_motion = motion_reg.gather(
            2,
            best_idx[:, :, None, None, None].expand(-1, -1, 1, ts_m, 2),
        ).squeeze(2)                                       # (bs, N, ts_m, 2)

        # Absolute agent positions = current xy + cumulative delta
        agent_xy  = det_anchors[..., :2]                  # (bs, N, 2)
        agent_abs = (
            best_motion[..., :ts, :].cumsum(dim=-2)
            + agent_xy.unsqueeze(2)
        )                                                  # (bs, N, ts, 2)

        # Pairwise distance: (bs, ego_fut_mode, N, ts)
        diff = (
            ego_abs[:, :, None, :ts, :]                   # (bs, M, 1, ts, 2)
            - agent_abs[:, None, :, :ts, :]               # (bs, 1, N, ts, 2)
        )                                                  # (bs, M, N, ts, 2)
        dist = torch.norm(diff, dim=-1)                    # (bs, M, N, ts)

        # Hinge penalty: push ego away when closer than margin
        penalty = F.relu(margin - dist)                    # (bs, M, N, ts)

        # Mask low-confidence agents
        conf_mask = (det_conf > conf_thresh).float()       # (bs, N)
        penalty   = penalty * conf_mask[:, None, :, None]

        # Normalise: per sample, per mode
        num_valid = conf_mask.sum(dim=-1).clamp(min=1.0)   # (bs,)
        loss = penalty.sum(dim=(1, 2, 3)) / num_valid / self.ego_fut_mode / ts
        return loss.mean()

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result