#!/usr/bin/env bash
# CoS-V2X cooperative data preparation (V2X-Real).
#
# Converts raw V2X-Real (SPD format) -> nuScenes-format -> sparse infos (.pkl)
# that the training / open-loop-eval configs read from ./data/infos/.
# Run from the repo root, in the cos_v2x env (needs mmcv/mmdet3d + nuscenes-devkit
# incl. the can_bus API + shapely).
#
# Prerequisites (place these yourself; they are git-ignored):
#   ./datasets/v2xreal/                              raw V2X-Real data (SPD layout)
#   ./data/split_datas_V2XREAL/split_datas_V2XREAL_coop.json   cooperative split
#   ./data/nuscenes/                                 nuScenes can_bus assets
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

DATA_ROOT=./datasets/v2xreal
SPLIT=./data/split_datas_V2XREAL/split_datas_V2XREAL_coop.json

echo "[1/5] SPD -> UniAD-format infos"
python tools/spd_data_converter/spd_to_uniad_REAL.py \
    --data-root "$DATA_ROOT" --save-root ./data/infos/v2xreal \
    --v2x-side cooperative --skip-noinfra True --split-file "$SPLIT"

echo "[2/5] SPD -> nuScenes-format (creates $DATA_ROOT/cooperative/)"
python tools/spd_data_converter/spd_to_nuscenes_REAL.py \
    --data-root "$DATA_ROOT" --save-root "$DATA_ROOT" \
    --v2x-side cooperative --skip-noinfra True --split-file "$SPLIT"

echo "[3/5] maps -> nuScenes-format"
python tools/spd_data_converter/map_spd_to_nuscenes_REAL.py \
    --maps-root "$DATA_ROOT/maps_final" --save-root "$DATA_ROOT" --v2x-side cooperative

echo "[4/5] build sparse infos -> nuscenes_infos_{train,val,test}.pkl"
python tools/sparse_data_converter/sparse_converter_w_map_parallel.py nuscenes \
    --root-path "$DATA_ROOT/cooperative/" --canbus ./data/nuscenes \
    --out-dir ./data/infos_sparse_cooperative --extra-tag nuscenes --version v1.0 \
    --infra-root-path "$DATA_ROOT/cooperative/inf_in_coop/"

echo "[5/5] place infos where the configs read them (./data/infos/)"
mkdir -p ./data/infos
cp ./data/infos_sparse_cooperative/nuscenes_infos_train.pkl ./data/infos/
cp ./data/infos_sparse_cooperative/nuscenes_infos_val.pkl   ./data/infos/
cp ./data/infos_sparse_cooperative/nuscenes_infos_test.pkl  ./data/infos/

echo "Done -> ./data/infos/nuscenes_infos_{train,val,test}.pkl"
