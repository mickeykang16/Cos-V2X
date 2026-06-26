#!/usr/bin/env bash
# Ablation: No Cross-Attention
# Removes bidirectional cross-attn from FusionDetModule and FusionMapModule.
# Confidence-weighted fusion is kept; only cross-attn is ablated.
# Trains from stage1 checkpoint (load_from set in config).
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x_top100_abl_no_cross_attn.py \
    4 \
    --deterministic \
    "$@"
