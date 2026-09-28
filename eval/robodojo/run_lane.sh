#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || { echo "Usage: $0 SLOT RUN_ID" >&2; exit 2; }
readonly slot=$1
readonly run_id=$2
[[ "${slot}" =~ ^([0-9]|[12][0-9]|3[01])$ ]] || { echo "SLOT must be 0..31" >&2; exit 2; }
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }

readonly eval_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${eval_root}/env.sh"

readonly plan="${eval_root}/plan/lanes.tsv"
IFS=$'\t' read -r _ seed global_lane estimated tasks < <(sed -n "$((slot + 1))p" "${plan}")
[[ -n "${tasks:-}" ]] || { echo "empty lane ${slot}" >&2; exit 2; }

export ROBODOJO_LAUNCH_STAGGER_S=1
export ROBODOJO_WATCHDOG_POLL_S=10
export ROBODOJO_SIM_START_TIMEOUT_S=180
export ROBODOJO_LAUNCH_TIMEOUT_S=660
export ROBODOJO_NO_PROGRESS_TIMEOUT_S=1800
export ROBODOJO_WATCHDOG_MAX_ATTEMPTS=40
export ROBODOJO_REQUEST_TIMEOUT_S=900
export ROBODOJO_FAST_LOG_ROOT="${ROBODOJO_OUTPUT_ROOT}/logs/${run_id}/slot-${slot}"

readonly lane_run_id="${run_id}-s${seed}-lane${global_lane}"
echo "ROBODOJO_LANE_START slot=${slot} seed=${seed} lane=${global_lane} estimated_s=${estimated} tasks=${tasks}"
exec bash "${eval_root}/run.sh" \
  --seed "${seed}" --policy-gpus 0 --env-gpus 0 \
  --tasks "${tasks}" --eval-num native --run-id "${lane_run_id}" --resume
