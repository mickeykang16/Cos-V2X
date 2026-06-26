"""
SparseDriveV2X: separate backbone+neck for vehicle cameras and infra cameras.

Vehicle cameras (first num_veh_cams) and infrastructure cameras (last
num_infra_cams) are encoded through independent backbone+neck modules,
then concatenated along the camera dimension before being passed to the
detection head (which still sees num_cams = num_veh_cams + num_infra_cams).
"""

from inspect import signature

import torch

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_backbone, build_neck

from .sparsedrive import SparseDrive

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except Exception:
    DAF_VALID = False

__all__ = ["SparseDriveV2X"]


@DETECTORS.register_module()
class SparseDriveV2X(SparseDrive):
    """SparseDrive with independent perception modules for vehicle / infra cameras.

    Args:
        img_backbone (dict): Config for the *vehicle* camera backbone.
        img_neck (dict, optional): Config for the *vehicle* camera neck.
        infra_img_backbone (dict, optional): Config for the infra camera backbone.
            When ``None`` the vehicle backbone weights are *shared*.
        infra_img_neck (dict, optional): Config for the infra camera neck.
            When ``None`` the vehicle neck weights are *shared*.
        num_veh_cams (int): Number of vehicle cameras (default: 4).
            These correspond to the *first* ``num_veh_cams`` channels of ``img``.
        num_infra_cams (int): Number of infra cameras (default: 2).
            These correspond to the *last* ``num_infra_cams`` channels of ``img``.
        **kwargs: Passed directly to :class:`SparseDrive`.
    """

    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        infra_img_backbone=None,
        infra_img_neck=None,
        num_veh_cams: int = 4,
        num_infra_cams: int = 2,
        freeze_infra: bool = False,
        **kwargs,
    ):
        super().__init__(
            img_backbone=img_backbone,
            head=head,
            img_neck=img_neck,
            **kwargs,
        )
        self.num_veh_cams = num_veh_cams
        self.num_infra_cams = num_infra_cams
        self.freeze_infra = freeze_infra
        # Propagate cam split info to the head so it can slice metas correctly
        if hasattr(self.head, '_num_veh_cams'):
            self.head._num_veh_cams = num_veh_cams
            self.head._num_infra_cams = num_infra_cams
        # Propagate freeze_infra to head so it can skip infra loss computation
        if hasattr(self.head, 'freeze_infra'):
            self.head.freeze_infra = freeze_infra

        # Independent infra backbone — build_backbone() creates a SEPARATE nn.Module
        # instance (separate weight tensors) even if the same config dict is passed.
        if infra_img_backbone is not None:
            self.infra_img_backbone = build_backbone(infra_img_backbone)
        else:
            self.infra_img_backbone = None  # weight-sharing fallback

        # Independent infra neck
        if infra_img_neck is not None:
            self.infra_img_neck = build_neck(infra_img_neck)
        else:
            self.infra_img_neck = None  # weight-sharing fallback

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_weights(self):
        """Initialise vehicle modules (via parent) then infra modules."""
        super().init_weights()
        if self.infra_img_backbone is not None:
            self.infra_img_backbone.init_weights()
        if self.infra_img_neck is not None:
            if isinstance(self.infra_img_neck, torch.nn.Sequential):
                for m in self.infra_img_neck:
                    m.init_weights()
            else:
                self.infra_img_neck.init_weights()
        if self.freeze_infra:
            self._freeze_infra_modules()

    def _freeze_infra_modules(self):
        """Freeze all infra-related parameters (backbone, neck, det/map heads)."""
        modules_to_freeze = []
        if self.infra_img_backbone is not None:
            modules_to_freeze.append(self.infra_img_backbone)
        if self.infra_img_neck is not None:
            modules_to_freeze.append(self.infra_img_neck)
        # Also freeze infra-only heads inside SparseDriveHeadV2X.
        # fusion_det_head / fusion_map_head are NOT frozen: they receive
        # gradient from the planning head and should continue to learn.
        head = self.head
        for attr in ("infra_det_head", "infra_map_head"):
            if hasattr(head, attr):
                modules_to_freeze.append(getattr(head, attr))
        for mod in modules_to_freeze:
            for param in mod.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Override to keep infra modules frozen even after .train() calls."""
        super().train(mode)
        if self.freeze_infra:
            self._freeze_infra_modules()
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_stream(self, img, num_cams, backbone, neck):
        """Run backbone+neck for one camera stream.

        Args:
            img (Tensor): shape (bs, num_cams, C, H, W)
            num_cams (int): number of cameras in this stream
            backbone: backbone module
            neck: neck module (may be None)

        Returns:
            list[Tensor]: per-level feature maps, each (bs, num_cams, C, H, W)
        """
        bs = img.shape[0]
        img = img.flatten(end_dim=1)  # (bs*num_cams, C, H, W)

        if self.use_grid_mask:
            img = self.grid_mask(img)

        if "metas" in signature(backbone.forward).parameters:
            feature_maps = backbone(img, num_cams)
        else:
            feature_maps = backbone(img)

        if neck is not None:
            feature_maps = list(neck(feature_maps))

        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])

        return feature_maps

    # ------------------------------------------------------------------
    # Override extract_feat  (returns MERGED features for depth branch)
    # ------------------------------------------------------------------

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        veh_features, infra_features = self.extract_feat_split(img)

        # Merge for depth supervision (applied to merged features)
        merged = [torch.cat([v, i], dim=1) for v, i in zip(veh_features, infra_features)]

        depths = None
        if return_depth and self.depth_branch is not None:
            focal = metas.get("focal") if metas is not None else None
            depths = self.depth_branch(merged, focal)

        if return_depth:
            return merged, depths
        return merged

    def extract_feat_split(self, img):
        """Extract features separately for vehicle and infra streams.

        Returns:
            (veh_features, infra_features): two lists of per-level feature
            maps, each element shaped (bs, num_cams_in_stream, C, H, W).
            Features are always returned in fp32, matching the behaviour of
            ``SparseDrive.extract_feat`` which uses ``out_fp32=True``.
        """
        if img.dim() != 5:
            raise ValueError("extract_feat_split requires 5-D img tensor")

        veh_neck = self.img_neck if hasattr(self, "img_neck") else None

        # ── vehicle cameras (first num_veh_cams) ─────────────────────────
        veh_features = self._extract_stream(
            img[:, : self.num_veh_cams], self.num_veh_cams, self.img_backbone, veh_neck
        )

        # ── infra cameras (last num_infra_cams) ──────────────────────────
        infra_backbone = self.infra_img_backbone or self.img_backbone
        infra_neck = self.infra_img_neck or veh_neck
        infra_features = self._extract_stream(
            img[:, self.num_veh_cams :], self.num_infra_cams, infra_backbone, infra_neck
        )

        # The FPN neck is decorated with @auto_fp16() which enables autocast
        # and produces fp16 outputs when fp16_enabled=True (set by
        # wrap_fp16_model during training).  Cast back to fp32 to match the
        # behaviour of SparseDrive.extract_feat(out_fp32=True).
        veh_features = [f.float() for f in veh_features]
        infra_features = [f.float() for f in infra_features]

        if self.use_deformable_func:
            veh_features = feature_maps_format(veh_features)
            infra_features = feature_maps_format(infra_features)

        return veh_features, infra_features

    # ------------------------------------------------------------------
    # Override forward_train / forward_test to use split features
    # ------------------------------------------------------------------

    def forward_train(self, img, **data):
        veh_features, infra_features = self.extract_feat_split(img)

        # Depth supervision on merged features
        if self.depth_branch is not None and "gt_depth" in data:
            if self.use_deformable_func:
                depths = None  # deformable packed format; skip depth on merged
            else:
                merged = [torch.cat([v, i], dim=1) for v, i in zip(veh_features, infra_features)]
                depths = self.depth_branch(merged, data.get("focal"))
        else:
            depths = None

        model_outs = self.head((veh_features, infra_features), data)
        output = self.head.loss(model_outs, data)

        if depths is not None:
            output["loss_dense_depth"] = self.depth_branch.loss(depths, data["gt_depth"])
        return output

    def simple_test(self, img, **data):
        veh_features, infra_features = self.extract_feat_split(img)
        model_outs = self.head((veh_features, infra_features), data)
        results = self.head.post_process(model_outs, data)
        return [{"img_bbox": result} for result in results]

    def aug_test(self, img, **data):
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
