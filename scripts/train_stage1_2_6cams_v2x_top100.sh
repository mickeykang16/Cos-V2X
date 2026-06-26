#!/usr/bin/env bash
# Full V2X top-100 training pipeline: stage1 → stage2 (sequential)
# Stage2 automatically loads from stage1's latest checkpoint.
set -e
cd "$(dirname "$0")/.."

STAGE1_CFG="projects/configs/sparsedrive_small_stage1_6cams_v2x_top100.py"
STAGE2_CFG="projects/configs/sparsedrive_small_stage2_6cams_v2x_top100.py"
STAGE1_WORKDIR="work_dirs/6cams_both_infra_v8_v2x_stage1_top100_fix"
NUM_GPUS=4

echo "========================================"
echo "  [1/2] Starting Stage 1 (det + map)"
echo "========================================"
bash ./tools/dist_train.sh \
    "${STAGE1_CFG}" \
    "${NUM_GPUS}" \
    --deterministic \
    "$@"

STAGE1_CKPT="${STAGE1_WORKDIR}/latest.pth"
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage1 checkpoint not found at ${STAGE1_CKPT}"
    exit 1
fi

echo ""
echo "========================================"
echo "  [2/2] Starting Stage 2 (+ motion/plan)"
echo "  load_from: ${STAGE1_CKPT}"
echo "========================================"
bash ./tools/dist_train.sh \
    "${STAGE2_CFG}" \
    "${NUM_GPUS}" \
    --deterministic \
    --cfg-options load_from="${STAGE1_CKPT}" \
    "$@"

echo ""
echo "========================================"
echo "  Training complete."
echo "========================================"
