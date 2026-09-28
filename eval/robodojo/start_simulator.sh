#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
task=$1
seed=$2
gpu=$(robodojo_gpu "$3")
port=$4
sim_env="${ROBODOJO_SIM_ENV}"
additional_info="ckpt_name=robodojo,action_type=joint,replan_steps=10"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PATH="${sim_env}/bin:${PATH}"
export CONDA_PREFIX="${sim_env}"
# Keep the policy environment's libraries and Python packages out of Isaac Sim.
unset PYTHONHOME LD_PRELOAD
export PYTHONPATH="${ROBODOJO_ROOT}:${ROBODOJO_ROOT}/XPolicyLab"
IFS=: read -r -a library_paths <<< "${LD_LIBRARY_PATH:-}"
sim_libraries="${sim_env}/lib"
for entry in "${library_paths[@]}"; do
  case "${entry}" in ""|"${ROBODOJO_POLICY_ENV}"/*) continue ;; esac
  sim_libraries="${sim_libraries}:${entry}"
done
export LD_LIBRARY_PATH="${sim_libraries}"
if [[ -f "${sim_env}/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="${sim_env}/lib/libstdc++.so.6"
fi
exec bash "${ROBODOJO_ROOT}/scripts/eval_policy.sh" \
  --root_dir "${ROBODOJO_ROOT}" --task_name "${task}" --env_cfg_type arx_x5 \
  --device_id 0 --policy_name internw0 --port "${port}" --host localhost \
  --protocol ws --eval_batch false --additional_info "${additional_info}" --seed "${seed}"
