#!/usr/bin/env bash
# Ablation: No Cross-Attention + No Confidence-Weighted Fusion
# Simplest possible fusion: plain 0.5/0.5 element-wise average only.
# Both cross-attn and conf-weighting are removed.
# Trains from stage1 checkpoint (load_from set in config).
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x_top100_abl_no_both.py \
    4 \
    --deterministic \
    "$@"
