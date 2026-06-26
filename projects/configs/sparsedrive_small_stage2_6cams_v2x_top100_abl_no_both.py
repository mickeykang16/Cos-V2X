# ================ Ablation: No Cross-Attn + No Conf-Weighted Fusion ==
# Removes BOTH bidirectional cross-attention AND confidence-weighted
# averaging.  This is the simplest possible fusion: a plain 0.5/0.5
# element-wise average of vehicle and infra features (after topk selection
# for the det head), followed by self-attn → FFN → refine.
#
# Comparison:
#   full model : cross-attn (veh←infra, infra←veh)  +  conf-weighted avg
#   this model : (skip cross-attn)                  +  simple 0.5/0.5 avg
#
# All other hyper-parameters are identical to the stage2 baseline.
# Train from stage1 checkpoint for a fair comparison.
# ======================================================================

_base_ = ['./sparsedrive_small_stage2_6cams_v2x_top100.py']

work_dir = "work_dirs/6cams_both_infra_v8_v2x_stage2_top100_abl_no_both_trial2"

# Start from stage1 (same as the baseline stage2 training)
load_from = "work_dirs/6cams_both_infra_v8_v2x_stage1_top100_fix/latest.pth"
resume_from = None

model = dict(
    head=dict(
        # Det fusion: skip cross-attn AND use simple 0.5/0.5 average
        fusion_det_head=dict(fusion_mode="no_both"),
        # Map fusion: same ablation
        fusion_map_head=dict(fusion_mode="no_both"),
    )
)
