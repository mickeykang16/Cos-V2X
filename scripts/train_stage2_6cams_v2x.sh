#!/usr/bin/env bash
# Stage 2 V2X training: det + map + motion + planning
# Loads from work_dirs/6cams_both_infra_v6_v2x/latest.pth (stage1 checkpoint)
set -e
cd "$(dirname "$0")/.."

bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2_6cams_v2x.py \
    3 \
    --deterministic \
    "$@"
