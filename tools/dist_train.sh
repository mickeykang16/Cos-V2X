# #!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-28651}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

# Use a random port in 29500-65535 if PORT is not set in the environment
if [ -z "$PORT" ]; then
    PORT=$((29500 + $RANDOM % 36035))
fi
echo "Using port: $PORT"

# Explicitly set the GPUs to use, from 0 up to GPUS
# e.g. GPUS=8 -> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_IDS=$(seq -s, 0 $((GPUS-1)))

PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH" \
CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}