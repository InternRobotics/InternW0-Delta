#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
seed=$1
gpu=$(robodojo_gpu "$2")
port=$3
policy_env="${ROBODOJO_POLICY_ENV}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PATH="${policy_env}/bin:${PATH}"
export PYTHONPATH="${INTERNW0_ROOT}:${INTERNW0_ROOT}/src:${ROBODOJO_ROOT}/XPolicyLab"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export TRITON_CACHE_DIR="${ROBODOJO_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${ROBODOJO_CACHE_ROOT}/inductor"
if [[ -x "${policy_env}/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="${policy_env}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${policy_env}/bin/x86_64-conda-linux-gnu-g++"
fi
exec "${policy_env}/bin/python" -m eval.robodojo.server --seed "${seed}" --port "${port}" --host localhost
