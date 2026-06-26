# ================ Ablation: No Confidence-Weighted Fusion =============
# Removes confidence-weighted averaging from both FusionDetModule and
# FusionMapModule.  Cross-attention still runs; features are merged with
# a simple 0.5 / 0.5 average regardless of per-slot confidence.
#
# Comparison:
#   full model : cross-attn (veh←infra, infra←veh)  +  conf-weighted avg
#   this model : cross-attn (veh←infra, infra←veh)  +  simple 0.5/0.5 avg
#
# All other hyper-parameters are identical to the stage2 baseline.
# Train from stage1 checkpoint for a fair comparison.
# ======================================================================

_base_ = ['./sparsedrive_small_stage2_6cams_v2x_top100.py']

work_dir = "work_dirs/6cams_both_infra_v8_v2x_stage2_top100_abl_no_conf_weight"

# Start from stage1 (same as the baseline stage2 training)
load_from = "work_dirs/6cams_both_infra_v8_v2x_stage1_top100_fix/latest.pth"
resume_from = None

model = dict(
    head=dict(
        # Det fusion: keep cross-attn, replace conf-weight with 0.5/0.5
        fusion_det_head=dict(fusion_mode="no_conf_weight"),
        # Map fusion: same ablation
        fusion_map_head=dict(fusion_mode="no_conf_weight"),
    )
)
