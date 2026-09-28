#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/env.sh"
evaluation_id="$(date -u +%Y-%m-%d_%H-%M-%S)"
for seed in 0 1 2; do
  bash "${script_dir}/run.sh" --seed "${seed}" --run-id "${evaluation_id}" "$@"
done
"${ROBODOJO_POLICY_ENV}/bin/python" -m eval.robodojo.summarize_results
