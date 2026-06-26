"""
SparseDriveHeadV2X: separate det+map heads for vehicle and infra camera streams,
with optional attention-based fusion (FusionDetModule).

Architecture (with fusion)
--------------------------
  vehicle features  (4 cams) ──► veh_det_head  ──► veh_det (top-600) ──┐
                                                                          ├─ FusionDetModule ──► fused_det ──► motion_plan_head
  infra   features  (2 cams) ──► infra_det_head ──► infra_det (top-300) ─┘

  vehicle features  (4 cams) ──► veh_map_head  ──► veh_map_output  ──┐
                                                                       ├── merge ──► motion_plan_head (stage-2)
  infra   features  (2 cams) ──► infra_map_head ──► infra_map_output ─┘

Both det heads are trained against the full GT (same objects, different viewpoints).
Loss keys are prefixed with "veh_" / "infra_" to avoid collisions.

When `fusion_det_head` is provided:
  - FusionDetModule runs bidirectional cross-attn + self-attn + FFN + refinement/cls
  - The fused output (top-K from each stream, then 900 merged) is used for final det
  - Loss includes: veh_det_loss + infra_det_loss + fused_det_loss
  - post_process uses the fused output directly
"""

from typing import List, Optional, Union

import copy
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS, build_head
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


def _merge_det_outputs(veh_det: dict, infra_det: dict) -> dict:
    """Merge two det_output dicts by concatenating along the anchor dim.

    Both dicts share the same keys; tensors in list fields are cat-ed
    element-wise.
    """
    merged = {}
    # list fields: classification, prediction, quality
    for key in ("classification", "prediction", "quality"):
        veh_list = veh_det[key]
        inf_list = infra_det[key]
        merged[key] = [
            torch.cat([v, i], dim=1) if (v is not None and i is not None) else (v if v is not None else i)
            for v, i in zip(veh_list, inf_list)
        ]
    # tensor fields
    for key in ("instance_feature", "anchor_embed"):
        merged[key] = torch.cat([veh_det[key], infra_det[key]], dim=1)
    # dn fields (only if present in veh; infra's dn fields are dropped for
    # motion planning – the vehicle head is the motion anchor reference)
    for key in veh_det:
        if key not in merged:
            merged[key] = veh_det[key]
    return merged


def _merge_map_outputs(veh_map: dict, infra_map: dict) -> dict:
    """Merge two map_output dicts (same structure as det)."""
    merged = {}
    for key in ("classification", "prediction"):
        veh_list = veh_map[key]
        inf_list = infra_map[key]
        merged[key] = [
            torch.cat([v, i], dim=1)
            for v, i in zip(veh_list, inf_list)
        ]
    for key in ("instance_feature", "anchor_embed"):
        merged[key] = torch.cat([veh_map[key], infra_map[key]], dim=1)
    for key in veh_map:
        if key not in merged:
            merged[key] = veh_map[key]
    return merged


def _merge_feature_maps(veh_feat, infra_feat):
    """Concatenate two multi-level feature map lists along the camera dim.

    Handles both:
    - Raw list format: list of (bs, num_cams, C, H, W) per level
    - Deformable packed format: [col_feats, spatial_shape, scale_start_index]
      produced by feature_maps_format() when use_deformable_func=True.
      col_feats:        (bs, total_tokens, C)  — global token storage
      spatial_shape:    (num_cams, num_levels, 2)
      scale_start_index:(num_cams, num_levels)  — global start pos in col_feats
      → infra indices must be offset by veh's total token count.
    """
    if isinstance(veh_feat, (list, tuple)) and len(veh_feat) == 3 and isinstance(veh_feat[0], torch.Tensor):
        # Deformable packed format
        veh_col, veh_sp, veh_ssi = veh_feat
        inf_col, inf_sp, inf_ssi = infra_feat
        veh_total = veh_col.shape[1]
        merged_col = torch.cat([veh_col, inf_col], dim=1)
        merged_sp = torch.cat([veh_sp, inf_sp], dim=0)
        merged_ssi = torch.cat([veh_ssi, inf_ssi + veh_total], dim=0)
        return [merged_col, merged_sp, merged_ssi]
    if isinstance(veh_feat, torch.Tensor):
        return torch.cat([veh_feat, infra_feat], dim=1)
    return [torch.cat([v, i], dim=1) for v, i in zip(veh_feat, infra_feat)]


@HEADS.register_module()
class SparseDriveHeadV2X(BaseModule):
    """Detection head with independent vehicle / infra perception branches.

    When ``shared_instance_bank`` is provided (recommended), both
    ``veh_det_head`` and ``infra_det_head`` operate on the **same** set of
    temporal anchors from a single shared InstanceBank.  This ensures that
    slot indices are identical across streams, enabling element-wise
    fusion and stable instance-ID assignment for downstream motion planning.

    Args:
        task_config (dict): Keys: with_det, with_map, with_motion_plan.
        veh_det_head (dict): Config for vehicle detection head (num_cams=4).
        veh_map_head (dict): Config for vehicle map head (num_cams=4).
        infra_det_head (dict): Config for infra detection head (num_cams=2).
        infra_map_head (dict): Config for infra map head (num_cams=2).
        shared_instance_bank (dict, optional): When provided, a single
            InstanceBank is built and injected into **both** det heads,
            replacing whatever ``instance_bank`` they would build on their
            own.  The shared bank drives temporal caching; per-head banks
            from the config are discarded.
        fusion_det_head (dict, optional): Config for FusionDetModule.
            With shared bank, ``forward_aligned()`` is called instead of
            the original ``forward()`` (no top-k selection needed).
        motion_plan_head (dict, optional): Stage-2 motion / planning head.
    """

    def __init__(
        self,
        task_config: dict,
        veh_det_head: dict,
        veh_map_head: dict,
        infra_det_head: dict,
        infra_map_head: dict,
        shared_instance_bank: Optional[dict] = None,
        fusion_det_head: Optional[dict] = None,
        fusion_map_head: Optional[dict] = None,
        motion_plan_head: Optional[dict] = None,
        freeze_infra: bool = False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.task_config = task_config
        self.with_fusion = fusion_det_head is not None
        self.with_map_fusion = fusion_map_head is not None
        self.use_shared_bank = shared_instance_bank is not None
        self.freeze_infra = freeze_infra
        # Store cam split indices – set from SparseDriveV2X after build, or
        # override via set_cam_split().  Default assumes 4 veh + 2 infra.
        self._num_veh_cams = 4
        self._num_infra_cams = 2

        # ── shared InstanceBank (optional) ─────────────────────────────
        if self.use_shared_bank:
            self.shared_instance_bank = build_from_cfg(
                shared_instance_bank, PLUGIN_LAYERS
            )
            # Inject the pre-built bank into both det-head configs so that
            # Sparse4DHead.__init__() skips building its own bank.
            veh_det_head = copy.deepcopy(veh_det_head)
            veh_det_head["instance_bank"] = self.shared_instance_bank
            infra_det_head = copy.deepcopy(infra_det_head)
            infra_det_head["instance_bank"] = self.shared_instance_bank

        if task_config["with_det"]:
            self.veh_det_head = build_head(veh_det_head)
            self.infra_det_head = build_head(infra_det_head)
            if self.use_shared_bank:
                for _head in (self.veh_det_head, self.infra_det_head):
                    del _head._modules["instance_bank"]
                    object.__setattr__(_head, "instance_bank", self.shared_instance_bank)
        if self.with_fusion:
            from .fusion_det import FusionDetModule
            self.fusion_det_head = FusionDetModule(**{k: v for k, v in fusion_det_head.items() if k != "type"})
        if self.with_map_fusion:
            from .fusion_det import FusionMapModule
            self.fusion_map_head = FusionMapModule(**{k: v for k, v in fusion_map_head.items() if k != "type"})
        if task_config["with_map"]:
            self.veh_map_head = build_head(veh_map_head)
            self.infra_map_head = build_head(infra_map_head)
        if task_config.get("with_motion_plan", False):
            assert motion_plan_head is not None, (
                "motion_plan_head config is required when with_motion_plan=True"
            )
            self.motion_plan_head = build_head(motion_plan_head)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_weights(self):
        if self.task_config["with_det"]:
            self.veh_det_head.init_weights()
            self.infra_det_head.init_weights()
        if self.with_fusion:
            self.fusion_det_head.init_weights()
        if self.task_config["with_map"]:
            self.veh_map_head.init_weights()
            self.infra_map_head.init_weights()
        if self.with_map_fusion:
            self.fusion_map_head.init_weights()
        if self.task_config.get("with_motion_plan", False):
            self.motion_plan_head.init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_metas(metas, cam_start: int, cam_end: int):
        """Return a shallow copy of metas with camera-indexed fields sliced.

        Slices ``projection_mat`` and ``image_wh`` along the camera axis so
        each det head only sees its own cameras.

        Args:
            metas (dict): full metas containing all cameras
            cam_start (int): first camera index (inclusive)
            cam_end (int): last camera index (exclusive)

        Returns:
            dict: metas with sliced projection_mat / image_wh
        """
        m = dict(metas)  # shallow copy – safe for nested tensors
        if "projection_mat" in m:
            m["projection_mat"] = m["projection_mat"][:, cam_start:cam_end]
        if "image_wh" in m:
            m["image_wh"] = m["image_wh"][:, cam_start:cam_end]
        return m

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _shared_bank_forward(self, veh_features, infra_features, metas):
        """Forward path used when ``use_shared_bank=True``.

        1. Fetch bank state ONCE from the shared InstanceBank.
        2. Both det heads run ``forward_shared()`` (same slots, no bank side
           effects).
        3. Element-wise aligned fusion via ``FusionDetModule.forward_aligned()``.
        4. Cache fused features back to the shared bank once.
        5. Assign instance IDs from the shared bank.
        """
        batch_size = veh_features[0].shape[0]
        bank = self.shared_instance_bank

        # Reset stale DN metas before calling bank.get()
        for head in (self.veh_det_head, self.infra_det_head):
            if (
                head.sampler.dn_metas is not None
                and head.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
            ):
                head.sampler.dn_metas = None

        # Single bank.get() for both heads
        bank_state = bank.get(
            batch_size, metas,
            dn_metas=self.veh_det_head.sampler.dn_metas,
        )

        # Split metas so each head sees only its own cameras
        nv = self._num_veh_cams
        ni = self._num_infra_cams
        metas_veh   = self._split_metas(metas, 0, nv)
        metas_infra = self._split_metas(metas, nv, nv + ni)

        # Both heads see the same temporal anchors.
        # Detach bank state for the infra head so that the shared bank's
        # learnable anchor parameter (self.anchor) only appears once in the
        # computation graph (via veh head).  This avoids DDP "parameter marked
        # ready twice" errors while still using the same anchor positions.
        bank_state_detached = tuple(
            x.detach() if isinstance(x, torch.Tensor) else x
            for x in bank_state
        )

        # ── Anchor-only bank sync for infra ────────────────────────────
        # In a real V2X deployment, transmitting the full cached_feature
        # (700×256 float32 ≈ 717 KB/frame) from vehicle to infra is the
        # BPS bottleneck.  Instead we transmit only cached_anchor
        # (700×11 float32 ≈ 31 KB/frame, 96% reduction) and reconstruct
        # a surrogate feature via anchor_encoder on the infra side.
        # The infra's temp_gnn thus attends to positional embeddings
        # rather than full content features — acceptable because the
        # deformable attention layer re-samples actual content from the
        # infra images each frame.
        (inst_feat_d, anchor_d, cached_feat_d, cached_anchor_d, ti_d) = bank_state_detached
        if cached_anchor_d is not None and cached_feat_d is not None:
            # Reconstruct surrogate feature from anchor embedding only
            surrogate_feat = self.veh_det_head.anchor_encoder(cached_anchor_d)
            infra_bank_state = (inst_feat_d, anchor_d, surrogate_feat, cached_anchor_d, ti_d)
        else:
            infra_bank_state = bank_state_detached

        veh_det_output, veh_feat, veh_anchor, veh_cls = (
            self.veh_det_head.forward_shared(veh_features, metas_veh, bank_state)
        )
        infra_det_output, infra_feat, infra_anchor, infra_cls = (
            self.infra_det_head.forward_shared(infra_features, metas_infra, infra_bank_state)
        )

        # Element-wise aligned fusion → single fused output (same N slots)
        if self.with_fusion:
            fused_det_output = self.fusion_det_head.forward_aligned(
                veh_det_output, infra_det_output
            )
            cache_feat = fused_det_output["instance_feature"]
            cache_anchor = fused_det_output["prediction"][-1]
            cache_cls = fused_det_output["classification"][-1]
        else:
            # Fallback: use vehicle head output for cache
            fused_det_output = None
            cache_feat = veh_feat
            cache_anchor = veh_anchor
            cache_cls = veh_cls

        # Single cache() call – preserves stable slot IDs across frames
        bank.cache(cache_feat, cache_anchor, cache_cls, metas, veh_features)

        # Assign instance IDs from the shared bank
        instance_id = bank.get_instance_id(
            cache_cls,
            cache_anchor,
            self.veh_det_head.decoder.score_threshold,
        )
        if fused_det_output is not None:
            fused_det_output["instance_id"] = instance_id
        else:
            veh_det_output["instance_id"] = instance_id

        return veh_det_output, infra_det_output, fused_det_output

    def forward(
        self,
        feature_maps,   # tuple: (veh_features, infra_features)
        metas: dict,
    ):
        veh_features, infra_features = feature_maps

        # ── per-stream perception ──────────────────────────────────────
        if self.use_shared_bank and self.task_config["with_det"]:
            veh_det_output, infra_det_output, fused_det_output = (
                self._shared_bank_forward(veh_features, infra_features, metas)
            )
        else:
            nv = self._num_veh_cams
            ni = self._num_infra_cams
            metas_veh   = self._split_metas(metas, 0, nv)
            metas_infra = self._split_metas(metas, nv, nv + ni)
            veh_det_output = self.veh_det_head(veh_features, metas_veh) if self.task_config["with_det"] else None
            infra_det_output = self.infra_det_head(infra_features, metas_infra) if self.task_config["with_det"] else None

            # ── optional fusion (cross-attn + self-attn + FFN + refine) ──
            fused_det_output = None
            if self.with_fusion and veh_det_output is not None and infra_det_output is not None:
                fused_det_output = self.fusion_det_head(veh_det_output, infra_det_output)

        nv = self._num_veh_cams
        ni = self._num_infra_cams
        metas_veh   = self._split_metas(metas, 0, nv)
        metas_infra = self._split_metas(metas, nv, nv + ni)
        veh_map_output = self.veh_map_head(veh_features, metas_veh) if self.task_config["with_map"] else None
        # Detach infra_features for the map head so that the infra backbone
        # (with_cp=True) only receives gradient from the det head path.
        # This avoids DDP "ready twice" caused by reentrant checkpoint backward.
        infra_features_map = [f.detach() for f in infra_features]
        infra_map_output = self.infra_map_head(infra_features_map, metas_infra) if self.task_config["with_map"] else None

        # ── map fusion (cross-attn + confidence-weighted merge + refine) ─
        fused_map_output = None
        if self.with_map_fusion and veh_map_output is not None and infra_map_output is not None:
            fused_map_output = self.fusion_map_head.forward_aligned(
                veh_map_output, infra_map_output
            )

        # ── motion / planning (stage-2, fused det + merged map) ────────
        motion_output, planning_output = None, None
        if self.task_config.get("with_motion_plan", False):
            if fused_det_output is not None:
                det_for_motion = fused_det_output
                anchor_encoder_ref = self.fusion_det_head.anchor_encoder
            else:
                det_for_motion = _merge_det_outputs(veh_det_output, infra_det_output)
                anchor_encoder_ref = self.veh_det_head.anchor_encoder

            # Prefer fused map output (confidence-weighted, 100 slots) over
            # naive concatenation (200 slots) when FusionMapModule is present.
            if self.with_map_fusion and fused_map_output is not None:
                map_for_motion = fused_map_output
            else:
                map_for_motion = _merge_map_outputs(veh_map_output, infra_map_output)
            merged_features = _merge_feature_maps(veh_features, infra_features)
            bank_ref = self.shared_instance_bank if self.use_shared_bank else self.veh_det_head.instance_bank
            motion_output, planning_output = self.motion_plan_head(
                det_for_motion,
                map_for_motion,
                merged_features,
                metas,
                anchor_encoder_ref,
                bank_ref.mask,
                bank_ref.anchor_handler,
                infra_det_output=infra_det_output,
                data=metas,
            )

        return (
            (veh_det_output, infra_det_output, fused_det_output),
            (veh_map_output, infra_map_output, fused_map_output),
            motion_output,
            planning_output,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, model_outs, data):
        (veh_det_output, infra_det_output, fused_det_output), (veh_map_output, infra_map_output, fused_map_output), motion_output, planning_output = model_outs
        losses = {}

        if self.task_config["with_det"]:
            veh_det_loss = self.veh_det_head.loss(veh_det_output, data)
            losses.update({f"veh_{k}": v for k, v in veh_det_loss.items()})
            if not self.freeze_infra:
                infra_det_loss = self.infra_det_head.loss(infra_det_output, data)
                losses.update({f"infra_{k}": v for k, v in infra_det_loss.items()})

        if self.with_fusion and fused_det_output is not None:
            fusion_loss = self.fusion_det_head.loss(fused_det_output, data)
            losses.update(fusion_loss)

        if self.task_config["with_map"]:
            veh_map_loss = self.veh_map_head.loss(veh_map_output, data)
            losses.update({f"veh_{k}": v for k, v in veh_map_loss.items()})
            if not self.freeze_infra:
                infra_map_loss = self.infra_map_head.loss(infra_map_output, data)
                losses.update({f"infra_{k}": v for k, v in infra_map_loss.items()})

        if self.with_map_fusion and fused_map_output is not None:
            fusion_map_loss = self.fusion_map_head.loss(fused_map_output, data)
            losses.update(fusion_map_loss)

        if self.task_config.get("with_motion_plan", False):
            # Use fused_det sampler indices when fusion is active so that the
            # Hungarian-matched slots (pred_idx) align with the detection output
            # that the motion head actually receives (fused_det_output).
            # Using veh_det indices on fused_det output causes GT trajectory
            # targets to be placed at slots that may not be in the motion
            # head's top-50 selection, silently zeroing out motion loss.
            if self.with_fusion and fused_det_output is not None:
                motion_indices = self.fusion_det_head.sampler.indices
            else:
                motion_indices = self.veh_det_head.sampler.indices
            motion_loss_cache = dict(indices=motion_indices)
            # Pass det_output so that loss_planning can compute the
            # collision-awareness loss using predicted agent trajectories.
            # This creates a gradient path: perception → motion → planning.
            det_for_motion_loss = (
                fused_det_output
                if (self.with_fusion and fused_det_output is not None)
                else veh_det_output
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, planning_output, data, motion_loss_cache,
                det_output=det_for_motion_loss,
            )
            losses.update(loss_motion)

        return losses

    # ------------------------------------------------------------------
    # Post-process
    # ------------------------------------------------------------------

    def post_process(self, model_outs, data):
        (veh_det_output, infra_det_output, fused_det_output), (veh_map_output, infra_map_output, fused_map_output), motion_output, planning_output = model_outs

        batch_size = None

        if self.task_config["with_det"]:
            if self.with_fusion and fused_det_output is not None:
                # Use the fused output as the final detection result
                fused_det_result = self.fusion_det_head.post_process(fused_det_output)
                batch_size = len(fused_det_result)
            else:
                # Fall back to per-stream merge if no fusion
                veh_det_result = self.veh_det_head.post_process(veh_det_output)
                infra_det_result = self.infra_det_head.post_process(infra_det_output)
                batch_size = len(veh_det_result)

        if self.task_config["with_map"]:
            veh_map_result = self.veh_map_head.post_process(veh_map_output)
            infra_map_result = self.infra_map_head.post_process(infra_map_output)
            if self.with_map_fusion and fused_map_output is not None:
                fused_map_result = self.fusion_map_head.post_process(fused_map_output)
            batch_size = len(veh_map_result)

        if self.task_config.get("with_motion_plan", False):
            if self.with_fusion and fused_det_output is not None:
                det_for_motion = fused_det_output
            else:
                det_for_motion = _merge_det_outputs(veh_det_output, infra_det_output)
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_for_motion, motion_output, planning_output, data
            )

        results = [dict() for _ in range(batch_size)]
        for i in range(batch_size):
            if self.task_config["with_det"]:
                if self.with_fusion and fused_det_output is not None:
                    results[i].update(fused_det_result[i])
                else:
                    results[i].update(
                        _merge_det_results(veh_det_result[i], infra_det_result[i])
                    )
            if self.task_config["with_map"]:
                if self.with_map_fusion and fused_map_output is not None:
                    results[i].update(fused_map_result[i])
                else:
                    results[i].update(
                        _merge_map_results(veh_map_result[i], infra_map_result[i])
                    )
            if self.task_config.get("with_motion_plan", False):
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results


# ------------------------------------------------------------------
# Result merging helpers
# ------------------------------------------------------------------

def _merge_det_results(veh: dict, infra: dict) -> dict:
    """Merge two per-sample detection result dicts.

    Concatenates boxes/scores/labels; NMS is left to the evaluator.
    Tracking IDs from the vehicle head take precedence; infra IDs are
    offset by a large constant to avoid collision.
    """
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    merged = {}
    for key in ("boxes_3d", "scores_3d", "labels_3d"):
        if key not in veh:
            continue
        v_val = veh[key]
        i_val = infra[key]
        if isinstance(v_val, LiDARInstance3DBoxes):
            merged_tensor = torch.cat(
                [v_val.tensor, i_val.tensor], dim=0
            )
            merged[key] = v_val.__class__(
                merged_tensor, box_dim=v_val.tensor.shape[-1]
            )
        else:
            merged[key] = torch.cat([v_val, i_val], dim=0)

    # Track scores / IDs: keep vehicle scores; infra appended
    for key in ("track_scores", "track_ids"):
        if key not in veh:
            continue
        v_val = veh[key]
        i_val = infra[key]
        if key == "track_ids":
            # Offset infra IDs so they don't clash with vehicle IDs
            i_val = i_val + 100_000
        merged[key] = torch.cat([v_val, i_val], dim=0)

    return merged


def _merge_map_results(veh: dict, infra: dict) -> dict:
    """Map is in the ego (vehicle) coordinate frame.

    Vehicle map is more reliable for local navigation; infra map may extend
    the observable range.  We simply concatenate instance predictions and
    let the downstream evaluator handle duplicates.
    """
    merged = {}
    for key in veh:
        if key not in infra:
            merged[key] = veh[key]
            continue
        v_val = veh[key]
        i_val = infra[key]
        if isinstance(v_val, torch.Tensor) and isinstance(i_val, torch.Tensor) and v_val.dim() == i_val.dim():
            merged[key] = torch.cat([v_val, i_val], dim=0)
        else:
            # Keep vehicle result for non-tensor / incompatible fields
            merged[key] = v_val
    return merged
