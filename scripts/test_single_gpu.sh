#!/usr/bin/env bash

# Single GPU test for proper tracking
# InstanceBank needs sequential processing on ONE GPU

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/sparsedrive_small_stage2/latest.pth \
    --eval bbox \
    --deterministic
