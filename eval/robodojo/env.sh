#!/usr/bin/env bash
# Paths are relative to the InternW0-delta checkout unless absolute.
set -euo pipefail
export INTERNW0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${INTERNW0_ROOT}"
absolute_path() {
  case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${INTERNW0_ROOT}" "$1" ;; esac
}
export ROBODOJO_ROOT="$(absolute_path "${ROBODOJO_ROOT:-third_party/RoboDojo}")"
export ROBODOJO_POLICY_ENV="$(absolute_path "${ROBODOJO_POLICY_ENV:-.venv}")"
export ROBODOJO_SIM_ENV="$(absolute_path "${ROBODOJO_SIM_ENV:-.venv-robodojo}")"
export ROBODOJO_OUTPUT_ROOT="$(absolute_path "${ROBODOJO_OUTPUT_ROOT:-runs/eval/robodojo}")"
export ROBODOJO_RESULT_ROOT="${ROBODOJO_OUTPUT_ROOT}/results"
export ROBODOJO_CACHE_ROOT="${ROBODOJO_OUTPUT_ROOT}/cache"
export ROBODOJO_CHECKPOINT="$(absolute_path "${ROBODOJO_CHECKPOINT:-${WAM_CHECKPOINT_ROOT:-checkpoints}/robodojo.pt}")"
export WAM_WAN_PATH="$(absolute_path "${WAM_WAN_PATH:-${WAM_CHECKPOINT_ROOT:-checkpoints}/Wan2.2-TI2V-5B}")"
export WAM_VLM_PATH="$(absolute_path "${WAM_VLM_PATH:-${WAM_CHECKPOINT_ROOT:-checkpoints}/RynnBrain1.1-2B}")"
export ROBODOJO_NORM_STATS="$(absolute_path "${ROBODOJO_NORM_STATS:-assets/stats/robodojo.json}")"
export ROBODOJO_POLICY_NAME=internw0
export ROBODOJO_CHECKPOINT_NAME=robodojo
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,::1"
# Resolve logical GPU numbers inside the scheduler's visibility allocation.
robodojo_gpu() {
  local logical=$1
  local -a visible
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible <<< "${CUDA_VISIBLE_DEVICES}"
    [[ "${logical}" -lt "${#visible[@]}" ]] || { echo "GPU ${logical} is outside CUDA_VISIBLE_DEVICES" >&2; return 2; }
    printf '%s\n' "${visible[$logical]}"
  else
    printf '%s\n' "${logical}"
  fi
}
