bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2_6cams.py \
    work_dirs/4cams_only_vehicle_v8_stage2_trial2/iter_10210.pth\
    4 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl


# bash ./tools/dist_test.sh \
#     projects/configs/sparsedrive_small_stage1_6cams.py \
#     work_dirs/6cams_both_infra_v6/latest.pth\
#     4 \
#     --deterministic \
#     --eval bbox

# bash ./tools/dist_test.sh \
#     projects/configs/sparsedrive_small_stage2.py \
#     ckpt/sparsedrive_stage2.pth \
#     4 \
#     --deterministic \
#     --eval bbox
#     # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl