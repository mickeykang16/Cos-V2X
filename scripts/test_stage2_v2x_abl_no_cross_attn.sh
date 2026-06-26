#!/usr/bin/env bash
# Test: Ablation — No Cross-Attention
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x_top100_abl_no_cross_attn.py \
    work_dirs/6cams_both_infra_v8_v2x_stage2_top100_abl_no_cross_attn/latest.pth \
    4 \
    --deterministic \
    --eval bbox \
    "$@"
