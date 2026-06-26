# ================ Ablation: No Cross-Attention ========================
# Removes bidirectional cross-attention from both FusionDetModule and
# FusionMapModule.  Infra features are merged into vehicle slots via
# confidence-weighted average ONLY, without any cross-stream attention.
#
# Comparison:
#   full model : cross-attn (veh←infra, infra←veh)  +  conf-weighted avg
#   this model : (skip cross-attn)                  +  conf-weighted avg
#
# All other hyper-parameters are identical to the stage2 baseline.
# Train from stage1 checkpoint for a fair comparison.
# ======================================================================

_base_ = ['./sparsedrive_small_stage2_6cams_v2x_top100.py']

work_dir = "work_dirs/6cams_both_infra_v8_v2x_stage2_top100_abl_no_cross_attn_trial2"

# Start from stage1 (same as the baseline stage2 training)
load_from = "work_dirs/6cams_both_infra_v8_v2x_stage1_top100_fix/latest.pth"
resume_from = None

model = dict(
    head=dict(
        # Det fusion: skip bidirectional cross-attention, keep conf-weight
        fusion_det_head=dict(fusion_mode="no_cross_attn"),
        # Map fusion: same ablation
        fusion_map_head=dict(fusion_mode="no_cross_attn"),
    )
)
