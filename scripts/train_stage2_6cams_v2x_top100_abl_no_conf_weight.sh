#!/usr/bin/env bash
# Ablation: No Confidence-Weighted Fusion
# Replaces confidence-weighted averaging with a simple 0.5/0.5 average.
# Bidirectional cross-attn is kept; only conf-weighting is ablated.
# Trains from stage1 checkpoint (load_from set in config).
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x_top100_abl_no_conf_weight.py \
    4 \
    --deterministic \
    "$@"
