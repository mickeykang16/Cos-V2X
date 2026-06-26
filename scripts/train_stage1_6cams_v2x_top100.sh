## stage1 – V2X top-100 bandwidth-limited fusion
## Only the top-100 infra anchors (by detection confidence) are transmitted.
## Corresponding 100 vehicle slots are fused; remaining 800 vehicle slots unchanged.
## work_dir: work_dirs/6cams_both_infra_v6_v2x_top100
## num_gpus : 3  (set in config: num_gpus = 3)
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1_6cams_v2x_top100.py \
   3 \
   --deterministic \
   "$@"
