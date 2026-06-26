## stage1 – V2X (4 vehicle cams + 2 infra cams, shared InstanceBank)
## work_dir: work_dirs/6cams_both_infra_v6_v2x
## num_gpus : 3  (set in config: num_gpus = 3)
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1_6cams_v2x.py \
   3 \
   --deterministic \
   "$@"
