#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

PORT="${PORT:-$(
python3 - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)}"

# Safe even if PYTHONPATH is unset
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
  "$(dirname "$0")/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:4}"