"""
FusionDetModule: fuses vehicle det output (top-N_veh) and infra det output
(top-N_infra) via bidirectional cross-attention → self-attention → FFN →
refinement + classification, producing a single fused detection output.

Architecture per forward pass:
  veh_det_output (900 anchors) ─→ top-600 ──┐
                                             ├─ cross-attn (bidirectional)
  infra_det_output (900 anchors) ─→ top-300 ─┘
         │
         └─ concat (900) → self-attn → FFN → refine → fused_det_output
"""

from typing import List, Optional

import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
    POSITIONAL_ENCODING,
    PLUGIN_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.models import LOSSES
from mmdet.core.bbox.builder import BBOX_CODERS, BBOX_SAMPLERS
from mmdet.core import reduce_mean

__all__ = ["FusionDetModule", "FusionMapModule"]


class FusionDetModule(BaseModule):
    """Attention-based fusion of vehicle and infra detection outputs.

    Args:
        embed_dims (int): Feature dimension. Default: 256.
        num_veh_instances (int): Top-k instances selected from vehicle det.
        num_infra_instances (int): Top-k instances selected from infra det.
        anchor_encoder (dict): Config for box → embed projection.
        veh_cross_attn (dict): Cross-attn: vehicle queries ← infra KV.
        infra_cross_attn (dict): Cross-attn: infra queries ← vehicle KV.
        self_attn (dict): Self-attn over all fused instances.
        norm_layer (dict): Normalisation layer (applied after each attn / ffn).
        ffn (dict): Feed-forward network (AsymmetricFFN with pre_norm).
        refine_layer (dict): SparseBox3DRefinementModule.
        loss_cls / loss_reg (dict): Classification / regression losses.
        sampler (dict): Hungarian matcher (SparseBox3DTarget, no DN).
        decoder (dict): SparseBox3DDecoder for post-processing.
        reg_weights (list): Per-dim regression weight.
        task_prefix (str): Prefix for loss keys.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_veh_instances: int = 600,
        num_infra_instances: int = 300,
        anchor_encoder: Optional[dict] = None,
        veh_cross_attn: Optional[dict] = None,
        infra_cross_attn: Optional[dict] = None,
        self_attn: Optional[dict] = None,
        norm_layer: Optional[dict] = None,
        ffn: Optional[dict] = None,
        refine_layer: Optional[dict] = None,
        loss_cls: Optional[dict] = None,
        loss_reg: Optional[dict] = None,
        sampler: Optional[dict] = None,
        decoder: Optional[dict] = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: Optional[List] = None,
        task_prefix: str = "fused_det",
        infra_topk: Optional[int] = None,
        fusion_mode: str = "full",
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_veh = num_veh_instances
        self.num_infra = num_infra_instances
        self.infra_topk = infra_topk  # None = full fusion, int = selective top-k fusion
        # fusion_mode controls ablation:
        #   "full"           – cross-attn  +  confidence-weighted merge  (default)
        #   "no_cross_attn" – skip cross-attn, keep confidence-weighted merge
        #   "no_conf_weight"– keep cross-attn,  use simple 0.5/0.5 average
        #   "no_both"       – skip cross-attn  +  use simple 0.5/0.5 average
        self.fusion_mode = fusion_mode
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.task_prefix = task_prefix
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 10

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)

        # ── bidirectional cross-attention ────────────────────────────────
        # veh_cross_attn : vehicle queries attend to infra KV
        self.veh_cross_attn = build(veh_cross_attn, ATTENTION)
        self.norm_veh = build(norm_layer, NORM_LAYERS)

        # infra_cross_attn : infra queries attend to vehicle KV
        self.infra_cross_attn = build(infra_cross_attn, ATTENTION)
        self.norm_infra = build(norm_layer, NORM_LAYERS)

        # ── self-attention over all fused instances ───────────────────────
        self.self_attn = build(self_attn, ATTENTION)
        self.norm_self = build(norm_layer, NORM_LAYERS)

        # ── FFN (AsymmetricFFN already has pre_norm + residual inside) ────
        self.ffn = build(ffn, FEEDFORWARD_NETWORK)
        self.norm_ffn = build(norm_layer, NORM_LAYERS)

        # ── final refine + classify ───────────────────────────────────────
        self.refine_layer = build(refine_layer, PLUGIN_LAYERS)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _topk(self, det_output, k: int):
        """Select top-k instances by max sigmoid class score."""
        feat = det_output["instance_feature"]          # (bs, N, dim)
        anchor = det_output["prediction"][-1]           # (bs, N, anchor_dim)
        cls = det_output["classification"][-1].sigmoid()  # (bs, N, num_cls)
        confidence = cls.max(dim=-1).values            # (bs, N)

        actual_k = min(k, confidence.shape[1])
        idx = confidence.topk(actual_k, dim=1).indices  # (bs, k)

        feat = feat.gather(1, idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))
        anchor = anchor.gather(1, idx.unsqueeze(-1).expand(-1, -1, anchor.shape[-1]))
        return feat, anchor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_aligned(self, veh_det_output, infra_det_output):
        """Element-wise fusion for aligned (same-slot) outputs.

        Both streams share the same N slots (slot i = same object in both).
        Unlike the top-k ``forward()``, no cross-stream instance matching is
        needed; instead bidirectional cross-attention enriches each stream
        before a confidence-weighted per-slot merge.

        Pipeline:
          1. Anchor-embed both streams.
          2. Bidirectional cross-attention (same API as forward()).
          3. Confidence-weighted per-slot feature & anchor average.
          4. Self-attention → FFN → refine.

        Args:
            veh_det_output (dict): output from veh_det_head.forward_shared()
            infra_det_output (dict): output from infra_det_head.forward_shared()

        Returns:
            dict: fused output with keys
                  'classification', 'prediction', 'quality',
                  'instance_feature', 'anchor_embed'
        """
        veh_feat   = veh_det_output["instance_feature"]      # (bs, N, D)
        infra_feat = infra_det_output["instance_feature"]    # (bs, N, D)
        veh_anchor   = veh_det_output["prediction"][-1]      # (bs, N, A)
        infra_anchor = infra_det_output["prediction"][-1]    # (bs, N, A)

        # 1. Positional embeddings
        veh_embed   = self.anchor_encoder(veh_anchor)    # (bs, N, D)
        infra_embed = self.anchor_encoder(infra_anchor)  # (bs, N, D)

        # Confidence scores (used for both top-k selection and weighted fusion)
        veh_cls   = veh_det_output["classification"][-1].sigmoid()    # (bs, N, C)
        infra_cls = infra_det_output["classification"][-1].sigmoid()  # (bs, N, C)
        veh_conf   = veh_cls.max(dim=-1, keepdim=True).values    # (bs, N, 1)
        infra_conf = infra_cls.max(dim=-1, keepdim=True).values  # (bs, N, 1)

        # Ablation flags derived from fusion_mode
        use_cross_attn  = self.fusion_mode in ("full", "no_conf_weight")
        use_conf_weight = self.fusion_mode in ("full", "no_cross_attn")

        if self.infra_topk is not None and self.infra_topk < infra_conf.shape[1]:
            # Bandwidth-limited V2X: select top-k infra slots BEFORE cross-attn
            # so that only top-k infra features are "transmitted" to vehicle.
            topk_idx = infra_conf.squeeze(-1).topk(
                self.infra_topk, dim=-1
            ).indices  # (bs, K)
            idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, veh_feat.shape[-1])   # (bs, K, D)
            idx_anch = topk_idx.unsqueeze(-1).expand(-1, -1, veh_anchor.shape[-1]) # (bs, K, A)

            infra_feat_sel   = infra_feat.gather(1, idx_feat)    # (bs, K, D)
            infra_anchor_sel = infra_anchor.gather(1, idx_anch)  # (bs, K, A)
            infra_embed_sel  = infra_embed.gather(1, idx_feat)   # (bs, K, D)

            if use_cross_attn:
                # 2. Bidirectional cross-attention with top-k infra only
                #    veh(900) attends to infra_sel(K) as K/V
                veh_feat = self.norm_veh(
                    self.veh_cross_attn(
                        veh_feat,
                        key=infra_feat_sel,
                        value=infra_feat_sel,
                        query_pos=veh_embed,
                        key_pos=infra_embed_sel,
                    )
                )
                #    infra_sel(K) attends to veh(900) as K/V
                infra_feat_sel = self.norm_infra(
                    self.infra_cross_attn(
                        infra_feat_sel,
                        key=veh_feat,
                        value=veh_feat,
                        query_pos=infra_embed_sel,
                        key_pos=veh_embed,
                    )
                )
            else:
                # Dummy zero-contribution pass so DDP sees all params in backward.
                # Detached inputs → grad to cross-attn params = 0, no effect on output.
                veh_feat = veh_feat + self.norm_veh(
                    self.veh_cross_attn(
                        veh_feat.detach(),
                        key=infra_feat_sel.detach(),
                        value=infra_feat_sel.detach(),
                        query_pos=veh_embed.detach(),
                        key_pos=infra_embed_sel.detach(),
                    )
                ) * 0
                infra_feat_sel = infra_feat_sel + self.norm_infra(
                    self.infra_cross_attn(
                        infra_feat_sel.detach(),
                        key=veh_feat.detach(),
                        value=veh_feat.detach(),
                        query_pos=infra_embed_sel.detach(),
                        key_pos=veh_embed.detach(),
                    )
                ) * 0

            # 3. Fusion for top-k slots only
            if use_conf_weight:
                total_conf_sel = veh_conf.gather(1, topk_idx.unsqueeze(-1)) \
                               + infra_conf.gather(1, topk_idx.unsqueeze(-1)) + 1e-6
                veh_w_sel   = veh_conf.gather(1, topk_idx.unsqueeze(-1))   / total_conf_sel
                infra_w_sel = infra_conf.gather(1, topk_idx.unsqueeze(-1)) / total_conf_sel
            else:
                # Simple average (0.5 / 0.5) — confidence-weighting ablated
                veh_w_sel   = veh_feat.new_full((veh_feat.shape[0], self.infra_topk, 1), 0.5)
                infra_w_sel = veh_feat.new_full((veh_feat.shape[0], self.infra_topk, 1), 0.5)

            fused_sel_feat   = (veh_w_sel * veh_feat.gather(1, idx_feat)
                                + infra_w_sel * infra_feat_sel)
            fused_sel_anchor = (veh_w_sel * veh_anchor.gather(1, idx_anch)
                                + infra_w_sel * infra_anchor_sel)

            # 800 remaining slots: veh-only (no infra info transmitted)
            fused_feat   = veh_feat.clone()
            fused_anchor = veh_anchor.clone()
            fused_feat.scatter_(1, idx_feat, fused_sel_feat)
            fused_anchor.scatter_(1, idx_anch, fused_sel_anchor)
        else:
            if use_cross_attn:
                # 2. Bidirectional cross-attention (full 900 infra slots)
                veh_feat = self.norm_veh(
                    self.veh_cross_attn(
                        veh_feat,
                        key=infra_feat,
                        value=infra_feat,
                        query_pos=veh_embed,
                        key_pos=infra_embed,
                    )
                )
                infra_feat = self.norm_infra(
                    self.infra_cross_attn(
                        infra_feat,
                        key=veh_feat,
                        value=veh_feat,
                        query_pos=infra_embed,
                        key_pos=veh_embed,
                    )
                )
            else:
                # Dummy zero-contribution pass so DDP sees all params in backward.
                veh_feat = veh_feat + self.norm_veh(
                    self.veh_cross_attn(
                        veh_feat.detach(),
                        key=infra_feat.detach(),
                        value=infra_feat.detach(),
                        query_pos=veh_embed.detach(),
                        key_pos=infra_embed.detach(),
                    )
                ) * 0
                infra_feat = infra_feat + self.norm_infra(
                    self.infra_cross_attn(
                        infra_feat.detach(),
                        key=veh_feat.detach(),
                        value=veh_feat.detach(),
                        query_pos=infra_embed.detach(),
                        key_pos=veh_embed.detach(),
                    )
                ) * 0

            # 3. Fusion (all N slots)
            if use_conf_weight:
                total_conf = veh_conf + infra_conf + 1e-6
                veh_w   = veh_conf   / total_conf
                infra_w = infra_conf / total_conf
            else:
                # Simple average for ablation
                veh_w   = torch.full_like(veh_conf, 0.5)
                infra_w = torch.full_like(infra_conf, 0.5)
            fused_feat   = veh_w * veh_feat   + infra_w * infra_feat    # (bs, N, D)
            fused_anchor = veh_w * veh_anchor + infra_w * infra_anchor  # (bs, N, A)
        fused_embed  = self.anchor_encoder(fused_anchor)             # (bs, N, D)

        # 4. Self-attention → FFN → refine
        fused_feat = self.norm_self(
            self.self_attn(
                fused_feat,
                key=fused_feat,
                value=fused_feat,
                query_pos=fused_embed,
            )
        )
        fused_feat = self.norm_ffn(self.ffn(fused_feat))

        fused_anchor_new, cls, qt = self.refine_layer(
            fused_feat, fused_anchor, fused_embed, return_cls=True
        )

        return {
            "classification": [cls],
            "prediction": [fused_anchor_new],
            "quality": [qt],
            "instance_feature": fused_feat,
            "anchor_embed": self.anchor_encoder(fused_anchor_new),
        }

    def forward(self, veh_det_output, infra_det_output):
        """
        Args:
            veh_det_output   (dict): Output from vehicle det head.
            infra_det_output (dict): Output from infra det head.

        Returns:
            dict: fused_det_output with keys
                  'classification', 'prediction', 'quality',
                  'instance_feature', 'anchor_embed'.
        """
        # 1. Select top-K instances from each stream
        veh_feat, veh_anchor = self._topk(veh_det_output, self.num_veh)    # (bs,600,dim)
        infra_feat, infra_anchor = self._topk(infra_det_output, self.num_infra)  # (bs,300,dim)

        # 2. Positional embeddings (anchor → dim)
        veh_embed = self.anchor_encoder(veh_anchor)      # (bs, 600, dim)
        infra_embed = self.anchor_encoder(infra_anchor)  # (bs, 300, dim)

        # 3. Bidirectional cross-attention
        #    veh queries attend to infra KV  (MultiheadFlashAttention has residual internally)
        veh_feat = self.norm_veh(
            self.veh_cross_attn(
                veh_feat,
                key=infra_feat,
                value=infra_feat,
                query_pos=veh_embed,
                key_pos=infra_embed,
            )
        )
        #    infra queries attend to veh KV  (uses updated veh_feat as KV)
        infra_feat = self.norm_infra(
            self.infra_cross_attn(
                infra_feat,
                key=veh_feat,
                value=veh_feat,
                query_pos=infra_embed,
                key_pos=veh_embed,
            )
        )

        # 4. Concatenate all instances
        fused_feat = torch.cat([veh_feat, infra_feat], dim=1)       # (bs, 900, dim)
        fused_anchor = torch.cat([veh_anchor, infra_anchor], dim=1)  # (bs, 900, anchor_dim)
        fused_embed = torch.cat([veh_embed, infra_embed], dim=1)     # (bs, 900, dim)

        # 5. Self-attention over all fused instances
        fused_feat = self.norm_self(
            self.self_attn(
                fused_feat,
                key=fused_feat,
                value=fused_feat,
                query_pos=fused_embed,
            )
        )

        # 6. FFN  (AsymmetricFFN has pre_norm + residual inside)
        fused_feat = self.norm_ffn(self.ffn(fused_feat))

        # 7. Final refinement + classification
        fused_anchor_new, cls, qt = self.refine_layer(
            fused_feat, fused_anchor, fused_embed, return_cls=True
        )

        return {
            "classification": [cls],
            "prediction": [fused_anchor_new],
            "quality": [qt],
            "instance_feature": fused_feat,
            "anchor_embed": self.anchor_encoder(fused_anchor_new),
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @force_fp32(apply_to=("model_outs",))
    def loss(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}

        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.reg_weights)]

            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls, reg, data[self.gt_cls_key], data[self.gt_reg_key]
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1),
                cls_target.flatten(end_dim=1),
                avg_factor=num_pos,
            )

            mask = mask.reshape(-1)
            reg_weights = (reg_weights * reg.new_tensor(self.reg_weights)).flatten(end_dim=1)[mask]
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg_target = torch.where(reg_target.isnan(), reg.new_tensor(0.0), reg_target)
            reg = reg.flatten(end_dim=1)[mask]
            cls_target_masked = cls_target.flatten(end_dim=1)[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target_masked,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        return output

    # ------------------------------------------------------------------
    # Post-process
    # ------------------------------------------------------------------

    @force_fp32(apply_to=("model_outs",))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )


class FusionMapModule(BaseModule):
    """Attention-based fusion of vehicle and infra map outputs.

    Both map heads are initialised from the same ``kmeans_map_100.npy``
    anchor file and operate in the ego (vehicle) coordinate frame, so a
    confidence-weighted per-slot merge after bidirectional cross-attention
    is well-defined.

    Pipeline (forward_aligned):
      1. Encode map-point anchors → positional embeddings.
      2. Bidirectional cross-attention (veh ← infra, infra ← veh).
      3. Confidence-weighted per-slot feature & anchor average.
      4. Self-attention → FFN → SparsePoint3DRefinementModule.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        anchor_encoder: Optional[dict] = None,
        veh_cross_attn: Optional[dict] = None,
        infra_cross_attn: Optional[dict] = None,
        self_attn: Optional[dict] = None,
        norm_layer: Optional[dict] = None,
        ffn: Optional[dict] = None,
        refine_layer: Optional[dict] = None,
        loss_cls: Optional[dict] = None,
        loss_reg: Optional[dict] = None,
        sampler: Optional[dict] = None,
        decoder: Optional[dict] = None,
        gt_cls_key: str = "gt_map_labels",
        gt_reg_key: str = "gt_map_pts",
        reg_weights: Optional[List] = None,
        task_prefix: str = "fused_map",
        fusion_mode: str = "full",
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.task_prefix = task_prefix
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 40
        self.fusion_mode = fusion_mode  # same semantics as FusionDetModule

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder  = build(anchor_encoder, POSITIONAL_ENCODING)
        self.loss_cls        = build(loss_cls, LOSSES)
        self.loss_reg        = build(loss_reg, LOSSES)
        self.sampler         = build(sampler, BBOX_SAMPLERS)
        self.decoder         = build(decoder, BBOX_CODERS)

        # bidirectional cross-attention
        self.veh_cross_attn   = build(veh_cross_attn, ATTENTION)
        self.norm_veh         = build(norm_layer, NORM_LAYERS)
        self.infra_cross_attn = build(infra_cross_attn, ATTENTION)
        self.norm_infra       = build(norm_layer, NORM_LAYERS)
        # self-attention + FFN
        self.self_attn  = build(self_attn, ATTENTION)
        self.norm_self  = build(norm_layer, NORM_LAYERS)
        self.ffn        = build(ffn, FEEDFORWARD_NETWORK)
        self.norm_ffn   = build(norm_layer, NORM_LAYERS)
        # refine
        self.refine_layer = build(refine_layer, PLUGIN_LAYERS)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_aligned(self, veh_map_output, infra_map_output):
        """Confidence-weighted per-slot fusion for map outputs.

        Both map heads share the same kmeans anchor init and operate in the
        ego coordinate frame, so slot-level weighted fusion is meaningful.

        Args:
            veh_map_output (dict): output from veh_map_head
            infra_map_output (dict): output from infra_map_head

        Returns:
            dict: keys 'classification', 'prediction',
                  'instance_feature', 'anchor_embed'
        """
        veh_feat   = veh_map_output["instance_feature"]    # (bs, N, D)
        infra_feat = infra_map_output["instance_feature"]  # (bs, N, D)
        veh_anchor   = veh_map_output["prediction"][-1]    # (bs, N, 40)
        infra_anchor = infra_map_output["prediction"][-1]  # (bs, N, 40)

        # 1. Positional embeddings
        veh_embed   = self.anchor_encoder(veh_anchor)    # (bs, N, D)
        infra_embed = self.anchor_encoder(infra_anchor)  # (bs, N, D)

        # Ablation flags
        use_cross_attn  = self.fusion_mode in ("full", "no_conf_weight")
        use_conf_weight = self.fusion_mode in ("full", "no_cross_attn")

        if use_cross_attn:
            # 2. Bidirectional cross-attention
            veh_feat = self.norm_veh(
                self.veh_cross_attn(
                    veh_feat,
                    key=infra_feat,
                    value=infra_feat,
                    query_pos=veh_embed,
                    key_pos=infra_embed,
                )
            )
            infra_feat = self.norm_infra(
                self.infra_cross_attn(
                    infra_feat,
                    key=veh_feat,
                    value=veh_feat,
                    query_pos=infra_embed,
                    key_pos=veh_embed,
                )
            )
        else:
            # Dummy zero-contribution pass so DDP sees all params in backward.
            veh_feat = veh_feat + self.norm_veh(
                self.veh_cross_attn(
                    veh_feat.detach(),
                    key=infra_feat.detach(),
                    value=infra_feat.detach(),
                    query_pos=veh_embed.detach(),
                    key_pos=infra_embed.detach(),
                )
            ) * 0
            infra_feat = infra_feat + self.norm_infra(
                self.infra_cross_attn(
                    infra_feat.detach(),
                    key=veh_feat.detach(),
                    value=veh_feat.detach(),
                    query_pos=infra_embed.detach(),
                    key_pos=veh_embed.detach(),
                )
            ) * 0

        # 3. Per-slot fusion
        veh_cls   = veh_map_output["classification"][-1].sigmoid()    # (bs, N, C)
        infra_cls = infra_map_output["classification"][-1].sigmoid()  # (bs, N, C)
        veh_conf   = veh_cls.max(dim=-1, keepdim=True).values    # (bs, N, 1)
        infra_conf = infra_cls.max(dim=-1, keepdim=True).values  # (bs, N, 1)
        if use_conf_weight:
            total_conf = veh_conf + infra_conf + 1e-6
            veh_w   = veh_conf   / total_conf
            infra_w = infra_conf / total_conf
        else:
            # Simple average for ablation
            veh_w   = torch.full_like(veh_conf, 0.5)
            infra_w = torch.full_like(infra_conf, 0.5)

        fused_feat   = veh_w * veh_feat   + infra_w * infra_feat    # (bs, N, D)
        fused_anchor = veh_w * veh_anchor + infra_w * infra_anchor  # (bs, N, 40)
        fused_embed  = self.anchor_encoder(fused_anchor)             # (bs, N, D)

        # 4. Self-attention → FFN → refine
        fused_feat = self.norm_self(
            self.self_attn(
                fused_feat,
                key=fused_feat,
                value=fused_feat,
                query_pos=fused_embed,
            )
        )
        fused_feat = self.norm_ffn(self.ffn(fused_feat))

        fused_anchor_new, cls, _ = self.refine_layer(
            fused_feat, fused_anchor, fused_embed, return_cls=True
        )

        return {
            "classification": [cls],
            "prediction": [fused_anchor_new],
            "instance_feature": fused_feat,
            "anchor_embed": self.anchor_encoder(fused_anchor_new),
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @force_fp32(apply_to=("model_outs",))
    def loss(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds  = model_outs["prediction"]
        output = {}

        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            reg = reg[..., : len(self.reg_weights)]

            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls, reg, data[self.gt_cls_key], data[self.gt_reg_key]
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1),
                cls_target.flatten(end_dim=1),
                avg_factor=num_pos,
            )

            mask = mask.reshape(-1)
            reg_weights = (
                reg_weights * reg.new_tensor(self.reg_weights)
            ).flatten(end_dim=1)[mask]
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg = reg.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        return output

    # ------------------------------------------------------------------
    # Post-process
    # ------------------------------------------------------------------

    @force_fp32(apply_to=("model_outs",))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
