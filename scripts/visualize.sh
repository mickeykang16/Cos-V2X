export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/sparsedrive_small_stage2_6cams_v2x_top100.py \
	--result-path work_dirs/6cams_both_infra_v8_v2x_stage2_top100_fix/results.pkl