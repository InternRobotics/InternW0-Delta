#!/usr/bin/env bash
# Start one launcher per node; the scheduler allocates resources.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TASK="${1:-pretrain}"
if [[ "$TASK" == --help || "$TASK" == -h ]]; then
  echo 'Usage: bash run.sh <pretrain|4D_distillation|robotwin|libero|robodojo|ebench|ebench_joint_ee|rtc> [Hydra overrides]'
  echo 'Environment: NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, RUN_ID'
  echo 'Optional: WAM_ACCELERATE_CONFIG, WAM_OUTPUT_ROOT, WAM_TRAIN_ENTRY, DRY_RUN=1'
  exit 0
fi
if (( $# )); then shift; fi
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
for value in "$NNODES" "$NODE_RANK" "$NPROC_PER_NODE"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Node and process counts must be integers.' >&2; exit 2; }
done
(( NNODES > 0 && NPROC_PER_NODE > 0 && NODE_RANK < NNODES )) || {
  echo 'Need visible GPUs, a positive process count, and 0 <= NODE_RANK < NNODES.' >&2; exit 2;
}
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set the rank-0 hostname/IP on every node}"
  : "${RUN_ID:?Set the same RUN_ID on every node}"
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DEFAULT_ACCELERATE_CONFIG=scripts/accelerate_configs/accelerate_zero1_ds.yaml
if [[ "$TASK" == robotwin || "$TASK" == 4D_distillation ]]; then
  DEFAULT_ACCELERATE_CONFIG=scripts/accelerate_configs/accelerate_zero2_ds.yaml
fi
if [[ "$TASK" == rtc ]]; then
  DEFAULT_ACCELERATE_CONFIG=scripts/accelerate_configs/accelerate_zero3_ds.yaml
fi
LAUNCH=(accelerate launch
  --config_file "${WAM_ACCELERATE_CONFIG:-$DEFAULT_ACCELERATE_CONFIG}"
  --num_machines "$NNODES" --num_processes "$(( NNODES * NPROC_PER_NODE ))"
  --machine_rank "$NODE_RANK" --main_process_ip "${MASTER_ADDR:-127.0.0.1}"
  --main_process_port "${MASTER_PORT:-29500}")
# Each node launches its own workers; no SSH hostfile or scheduler plugin is needed.
LAUNCH+=(--deepspeed_multinode_launcher standard)
LAUNCH+=("${WAM_TRAIN_ENTRY:-scripts/train.py}" "task=$TASK" "output_dir=${WAM_OUTPUT_ROOT:-runs}/$TASK/$RUN_ID" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${LAUNCH[@]}"
  printf '\n'
else
  exec "${LAUNCH[@]}"
fi
