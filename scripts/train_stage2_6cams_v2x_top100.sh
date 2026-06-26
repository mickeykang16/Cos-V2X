#!/usr/bin/env bash
# Stage 2 V2X top-100 bandwidth-limited fusion training: det + map + motion + planning
# Loads from work_dirs/6cams_both_infra_v6_v2x_top100/latest.pth (stage1 top100 checkpoint)
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x_top100.py \
    4 \
    --deterministic \
    "$@"
