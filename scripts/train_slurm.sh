#!/usr/bin/env bash
#SBATCH --job-name=internw0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%j.out
# Override resources with sbatch options; activate your environment before sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export NNODES="$SLURM_NNODES"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RUN_ID="${RUN_ID:-$SLURM_JOB_ID}"
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c \
  'export NODE_RANK="$SLURM_PROCID"; exec bash run.sh "$@"' bash "$@"
