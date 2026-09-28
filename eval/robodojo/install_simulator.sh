#!/usr/bin/env bash
# Run after setup.py --dependencies. Requires a CUDA 12.8 toolkit and C++ compiler.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
[[ -x "${conda_exe}" ]] || { echo 'Install Conda/Miniforge or set CONDA_EXE' >&2; exit 2; }
[[ -d "${ROBODOJO_ROOT}/third_party/IsaacLab/source" && -d "${ROBODOJO_ROOT}/third_party/curobo" ]] || {
  echo 'First run: python -m eval.robodojo.setup --dependencies' >&2; exit 2;
}
if [[ ! -x "${ROBODOJO_SIM_ENV}/bin/python" ]]; then
  "${conda_exe}" create -y -p "${ROBODOJO_SIM_ENV}" python=3.11 pip
fi
"${conda_exe}" install -y -p "${ROBODOJO_SIM_ENV}" -c conda-forge libstdcxx-ng
sim_python="${ROBODOJO_SIM_ENV}/bin/python"
"${sim_python}" -m pip install --upgrade pip setuptools wheel
"${sim_python}" -m pip install 'isaacsim[all,extscache]==5.1.0.0' --extra-index-url https://pypi.nvidia.com
"${sim_python}" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${sim_python}" -m pip install -r eval/robodojo/requirements-sim.txt
for package in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  "${sim_python}" -m pip install --no-deps -e "${ROBODOJO_ROOT}/third_party/IsaacLab/source/${package}"
done
"${sim_python}" -m pip install --no-deps --no-build-isolation -e "${ROBODOJO_ROOT}/third_party/curobo"
echo "Simulator environment installed at ${ROBODOJO_SIM_ENV}"
