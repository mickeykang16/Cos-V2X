#!/usr/bin/env bash
# Test: Ablation — No Cross-Attention + No Confidence-Weighted Fusion
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x_top100_abl_no_both.py \
    work_dirs/6cams_both_infra_v8_v2x_stage2_top100_abl_no_both/latest.pth \
    4 \
    --deterministic \
    --eval bbox \
    "$@"
