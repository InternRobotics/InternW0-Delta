#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/env.sh"
readonly storage_root="${ROBODOJO_OUTPUT_ROOT}"
readonly robodojo_root="${ROBODOJO_ROOT}"
readonly policy_name=internw0
readonly checkpoint_name=robodojo
readonly result_config="ckpt_name=${checkpoint_name},action_type=joint,replan_steps=10"
readonly model_env="${ROBODOJO_POLICY_ENV}"
readonly sim_env="${ROBODOJO_SIM_ENV}"
readonly client_launcher="${script_dir}/start_simulator.sh"
readonly start_detector="${script_dir}/simulation_state.py"
readonly rotate_layout="${script_dir}/rotate_layout.py"

seed=0
policy_gpus="0"
env_gpus="0"
tasks="all"
tasks_file=""
eval_num="native"
run_id="$(date -u +%Y-%m-%d_%H-%M-%S)"
dry_run=false
resume=false
launch_stagger_s="${ROBODOJO_LAUNCH_STAGGER_S:-20}"
poll_s="${ROBODOJO_WATCHDOG_POLL_S:-10}"
sim_start_timeout_s="${ROBODOJO_SIM_START_TIMEOUT_S:-180}"
launch_timeout_s="${ROBODOJO_LAUNCH_TIMEOUT_S:-660}"
no_progress_timeout_s="${ROBODOJO_NO_PROGRESS_TIMEOUT_S:-1800}"
max_attempts="${ROBODOJO_WATCHDOG_MAX_ATTEMPTS:-40}"

usage() {
    cat <<'EOF'
Usage: run.sh [options]

Runs one official RoboDojo seed with persistent InternW0-delta replan=10 policy servers and
independent Isaac Sim clients. The default is one colocated policy/simulator worker.

Options:
  --seed N                 Official seed: 0, 1, or 2 (default: 0)
  --policy-gpus IDS        Policy GPU ids (default: 0)
  --env-gpus IDS           Isaac Sim GPU ids (default: 0)
  --tasks all|A,B          Task subset (default: all 54 runnable configs)
  --tasks-file PATH        Additional newline-separated task subset
  --eval-num native|N      Native 25/50 counts or fixed positive count
  --run-id ID              Stable output identity for resume
  --resume                 Skip already-complete task results for this run id
  --dry-run                Verify, plan, and print without launching GPUs
  -h, --help               Show this help

Examples:
  run.sh --seed 0 --eval-num native
  run.sh --seed 0 --tasks align_blocks,general_pickup --eval-num 1
  run.sh --seed 0 --policy-gpus 0,2,4,6,8,10,12,14 \
    --env-gpus 1,3,5,7,9,11,13,15 --eval-num native
EOF
}

need_value() {
    [[ $# -ge 2 && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) need_value "$@"; seed=$2; shift 2 ;;
        --policy-gpus) need_value "$@"; policy_gpus=$2; shift 2 ;;
        --env-gpus) need_value "$@"; env_gpus=$2; shift 2 ;;
        --tasks) need_value "$@"; tasks=$2; shift 2 ;;
        --tasks-file) need_value "$@"; tasks_file=$2; shift 2 ;;
        --eval-num) need_value "$@"; eval_num=$2; shift 2 ;;
        --run-id) need_value "$@"; run_id=$2; shift 2 ;;
        --resume) resume=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${seed}" =~ ^[0-2]$ ]] || { echo "--seed must be 0, 1, or 2" >&2; exit 2; }
[[ "${eval_num}" == "native" || "${eval_num}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--eval-num must be native or a positive integer" >&2
    exit 2
}
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Unsafe --run-id" >&2; exit 2; }
for value in "${launch_stagger_s}" "${poll_s}" "${sim_start_timeout_s}" \
    "${launch_timeout_s}" "${no_progress_timeout_s}" "${max_attempts}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Timeout/count values must be positive integers" >&2; exit 2; }
done
[[ -x "${model_env}/bin/python" ]] || { echo "Missing model environment: ${model_env}" >&2; exit 2; }
[[ -x "${sim_env}/bin/python" ]] || { echo "Missing simulator environment: ${sim_env}" >&2; exit 2; }
[[ -f "${client_launcher}" ]] || { echo "Missing no-video client launcher: ${client_launcher}" >&2; exit 2; }

IFS=',' read -r -a policy_gpu_list <<< "${policy_gpus}"
IFS=',' read -r -a env_gpu_list <<< "${env_gpus}"
[[ "${#policy_gpu_list[@]}" -eq "${#env_gpu_list[@]}" && "${#policy_gpu_list[@]}" -gt 0 ]] || {
    echo "Policy and simulator GPU lists must have the same nonzero length" >&2
    exit 2
}
declare -A seen_policy_gpu=()
declare -A seen_env_gpu=()
for gpu in "${policy_gpu_list[@]}" "${env_gpu_list[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: ${gpu}" >&2; exit 2; }
done
for gpu in "${policy_gpu_list[@]}"; do
    [[ -z "${seen_policy_gpu[${gpu}]:-}" ]] || { echo "Duplicate policy GPU id: ${gpu}" >&2; exit 2; }
    seen_policy_gpu["${gpu}"]=1
done
for gpu in "${env_gpu_list[@]}"; do
    [[ -z "${seen_env_gpu[${gpu}]:-}" ]] || { echo "Duplicate simulator GPU id: ${gpu}" >&2; exit 2; }
    seen_env_gpu["${gpu}"]=1
done

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export ROBODOJO_DISABLE_VIDEO=1
export ROBODOJO_REQUEST_TIMEOUT_S="${ROBODOJO_REQUEST_TIMEOUT_S:-900}"
export TMPDIR="${storage_root}/cache/tmp"
export XDG_CACHE_HOME="${storage_root}/cache/xdg"
export CUDA_CACHE_PATH="${storage_root}/cache/cuda"
export TORCH_HOME="${storage_root}/cache/robodojo/torch"
export HF_HOME="${storage_root}/cache/robodojo/huggingface"

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,::1"

readonly log_root="${ROBODOJO_FAST_LOG_ROOT:-${storage_root}/logs/${run_id}-s${seed}}"
mkdir -p "${log_root}"
exec 9>"${log_root}/.lock"
flock -n 9 || { echo "Run already active: ${log_root}" >&2; exit 2; }
readonly plan_dir="${log_root}/plan"
readonly status_dir="${log_root}/status"
mkdir -p "${plan_dir}" "${status_dir}" "${TMPDIR}" "${XDG_CACHE_HOME}" \
    "${CUDA_CACHE_PATH}" "${TORCH_HOME}" "${HF_HOME}" "${storage_root}/cache/ov"

"${model_env}/bin/python" -m eval.robodojo.validate --run-dir "${log_root}" --seed "${seed}"

plan_command=(
    "${sim_env}/bin/python" "${script_dir}/plan_tasks.py"
    --robodojo-root "${robodojo_root}"
    --weights "${script_dir}/runtime_weights.json"
    --workers "${#policy_gpu_list[@]}"
    --tasks "${tasks}"
    --output-dir "${plan_dir}"
)
[[ -z "${tasks_file}" ]] || plan_command+=(--tasks-file "${tasks_file}")
"${plan_command[@]}" > "${plan_dir}/planner.stdout.json"
cat "${plan_dir}/assignment.json"

if [[ "${dry_run}" == true ]]; then
    for lane in "${!policy_gpu_list[@]}"; do
        task_text=$(paste -sd, "${plan_dir}/worker_${lane}.tasks")
        echo "ROBODOJO_FAST_DRY_RUN lane=${lane} policy_gpu=${policy_gpu_list[$lane]} env_gpu=${env_gpu_list[$lane]} tasks=${task_text}"
    done
    echo "ROBODOJO_FAST_DRY_RUN_OK run_id=${run_id} seed=${seed}"
    exit 0
fi


process_group_alive() {
    local pgid=$1
    ps -eo pgid=,stat= | awk -v target="${pgid}" '
        $1 == target && $2 !~ /^Z/ { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

stop_process_group() {
    local pgid=$1
    local index
    process_group_alive "${pgid}" || return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for index in {1..20}; do
        process_group_alive "${pgid}" || return 0
        sleep 0.5
    done
    kill -KILL -- "-${pgid}" 2>/dev/null || true
    for index in {1..10}; do
        process_group_alive "${pgid}" || return 0
        sleep 0.2
    done
    ! process_group_alive "${pgid}"
}

run_lane() {
    local lane=$1
    local policy_gpu=$2
    local env_gpu=$3
    local task_file=$4
    local lane_root="${log_root}/lane${lane}"
    local server_pid=""
    local client_pid=""
    local server_port
    local server_generation=0
    mkdir -p "${lane_root}"
    server_port=$(bash "${robodojo_root}/XPolicyLab/utils/get_free_port.sh")

    cleanup_lane() {
        trap - EXIT INT TERM
        if [[ -n "${client_pid}" ]]; then
            stop_process_group "${client_pid}" || true
            wait "${client_pid}" 2>/dev/null || true
            client_pid=""
        fi
        if [[ -n "${server_pid}" ]]; then
            stop_process_group "${server_pid}" || true
            wait "${server_pid}" 2>/dev/null || true
            server_pid=""
        fi
    }
    trap cleanup_lane EXIT
    trap 'cleanup_lane; exit 130' INT TERM

    start_server() {
        local bootstrap_task=$1
        local server_log
        server_generation=$((server_generation + 1))
        server_log="${lane_root}/server-${server_generation}.log"
        echo "ROBODOJO_FAST_SERVER_START lane=${lane} generation=${server_generation} task=${bootstrap_task} gpu=${policy_gpu} port=${server_port} log=${server_log}"
        setsid bash "${script_dir}/start_policy.sh" \
                "${seed}" "${policy_gpu}" "${server_port}" \
                >"${server_log}" 2>&1 &
        server_pid=$!
        if ! bash "${robodojo_root}/XPolicyLab/utils/wait_for_policy_server.sh" \
            localhost "${server_port}" "${server_pid}" "${policy_name} persistent lane ${lane}" 600; then
            tail -80 "${server_log}" >&2 || true
            stop_process_group "${server_pid}" || true
            wait "${server_pid}" 2>/dev/null || true
            server_pid=""
            return 1
        fi
        echo "ROBODOJO_FAST_SERVER_READY lane=${lane} generation=${server_generation} pid=${server_pid}"
    }

    restart_server() {
        local bootstrap_task=$1
        echo "ROBODOJO_FAST_SERVER_RESTART lane=${lane} task=${bootstrap_task}"
        if [[ -n "${server_pid}" ]]; then
            stop_process_group "${server_pid}" || true
            wait "${server_pid}" 2>/dev/null || true
            server_pid=""
        fi
        start_server "${bootstrap_task}"
    }

    task_progress() {
        "${sim_env}/bin/python" "${script_dir}/result_tools.py" progress "$1" "$2"
    }

    run_task() {
        local task=$1
        local expected=$2
        local task_run_id="${run_id}-${task}-s${seed}"
        local seed_root="${ROBODOJO_RESULT_ROOT}/RoboDojo/${task}/${policy_name}/arx_x5/${seed}_${result_config}"
        local resume_path="${seed_root}/_resume_${task_run_id}.json"
        local result_path="${seed_root}/${task_run_id}/_result.json"
        local initial_progress current_progress attempt attempt_root client_log omni_root
        local attempt_started last_change now start_state pending_since saw_start rc reason

        if [[ "${resume}" == true && -s "${result_path}" ]] && \
            "${sim_env}/bin/python" "${script_dir}/result_tools.py" validate "${result_path}" "${expected}" >/dev/null; then
            echo "ROBODOJO_FAST_TASK_SKIP lane=${lane} task=${task} result=${result_path}"
            printf '%s\t%s\t%s\n' "${task}" SKIP "${result_path}" >> "${status_dir}/lane${lane}.tsv"
            return 0
        fi

        initial_progress=$(task_progress "${resume_path}" "${result_path}")
        echo "ROBODOJO_FAST_TASK_START lane=${lane} task=${task} expected=${expected} initial=${initial_progress} server_generation=${server_generation}"
        for ((attempt=1; attempt<=max_attempts; attempt++)); do
            if [[ -z "${server_pid}" ]] || ! kill -0 "${server_pid}" 2>/dev/null; then
                restart_server "${task}"
            fi
            attempt_root="${lane_root}/${task}/attempt-$(printf '%03d' "${attempt}")"
            client_log="${attempt_root}/client.log"
            omni_root="${attempt_root}/omniverse"
            mkdir -p "${omni_root}"
            printf '[paths]\ncache_root = "%s"\ndata_root = "%s"\nlogs_root = "%s"\n' \
                "${storage_root}/cache/ov" "${omni_root}/data" "${omni_root}/logs" \
                > "${omni_root}/omniverse.toml"
            echo "ROBODOJO_FAST_ATTEMPT lane=${lane} task=${task} attempt=${attempt}/${max_attempts} progress=${initial_progress} log=${client_log}"
            setsid env \
                ROBODOJO_RUN_ID="${task_run_id}" \
                ROBODOJO_DISABLE_VIDEO=1 \
                ROBODOJO_FATAL_RESTART_COUNT=0 \
                ROBODOJO_MAX_BASH_RETRIES=1 \
                EVAL_NUM="${expected}" \
                OMNI_CONFIG_PATH="${omni_root}" \
                bash "${client_launcher}" \
                    "${task}" "${seed}" "${env_gpu}" "${server_port}" >"${client_log}" 2>&1 &
            client_pid=$!
            attempt_started=$(date +%s)
            last_change=${attempt_started}
            pending_since=""
            saw_start=0
            reason=""

            while kill -0 "${client_pid}" 2>/dev/null; do
                sleep "${poll_s}"
                now=$(date +%s)
                current_progress=$(task_progress "${resume_path}" "${result_path}")
                if (( current_progress > initial_progress )); then
                    echo "ROBODOJO_FAST_PROGRESS lane=${lane} task=${task} eval_time=${current_progress}/${expected}"
                    initial_progress=${current_progress}
                    last_change=${now}
                fi
                if [[ -z "${server_pid}" ]] || ! kill -0 "${server_pid}" 2>/dev/null; then
                    reason=server-exit
                    break
                fi
                start_state=$("${sim_env}/bin/python" "${start_detector}" "${client_log}")
                case "${start_state}" in
                    pending)
                        saw_start=1
                        if [[ -z "${pending_since}" ]]; then
                            pending_since=${now}
                        elif (( now - pending_since >= sim_start_timeout_s )); then
                            reason=simulation-startup
                        fi
                        ;;
                    ready) saw_start=1; pending_since="" ;;
                    not-started) ;;
                    *) reason=detector-error ;;
                esac
                if [[ -z "${reason}" && "${saw_start}" -eq 0 && $((now-attempt_started)) -ge "${launch_timeout_s}" ]]; then
                    reason=launch
                fi
                if [[ -z "${reason}" && $((now-last_change)) -ge "${no_progress_timeout_s}" ]]; then
                    reason=no-progress
                fi
                [[ -z "${reason}" ]] || break
            done

            if [[ -n "${reason}" ]]; then
                echo "ROBODOJO_FAST_STALL lane=${lane} task=${task} reason=${reason} progress=${initial_progress}/${expected} attempt=${attempt}" >&2
                stop_process_group "${client_pid}" || true
                wait "${client_pid}" 2>/dev/null || true
                client_pid=""
                if [[ "${reason}" == simulation-startup && "${task}" == make_toast_random && "${initial_progress}" -lt "${expected}" ]]; then
                    "${sim_env}/bin/python" "${rotate_layout}" "${task}" "${seed}" "${task_run_id}" --expected "${expected}" || return 3
                elif [[ "${reason}" != simulation-startup ]]; then
                    restart_server "${task}"
                fi
                continue
            fi

            set +e
            wait "${client_pid}"
            rc=$?
            set -e
            stop_process_group "${client_pid}" || true
            client_pid=""
            current_progress=$(task_progress "${resume_path}" "${result_path}")
            if (( rc == 0 )); then
                "${sim_env}/bin/python" "${script_dir}/result_tools.py" validate "${result_path}" "${expected}"
                if find "${seed_root}/${task_run_id}" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' \) -print -quit | grep -q .; then
                    echo "No-video invariant failed for ${task}" >&2
                    return 1
                fi
                echo "ROBODOJO_FAST_TASK_OK lane=${lane} task=${task} server_generation=${server_generation} result=${result_path}"
                printf '%s\t%s\t%s\n' "${task}" PASS "${result_path}" >> "${status_dir}/lane${lane}.tsv"
                return 0
            fi
            if (( current_progress > initial_progress )); then
                initial_progress=${current_progress}
                echo "ROBODOJO_FAST_PARTIAL_RESTART lane=${lane} task=${task} progress=${initial_progress}/${expected} rc=${rc} server_reused=1"
                continue
            fi
            echo "ROBODOJO_FAST_CLIENT_FAILED lane=${lane} task=${task} rc=${rc} progress=${current_progress}/${expected} attempt=${attempt}" >&2
            tail -60 "${client_log}" >&2 || true
            restart_server "${task}"
        done
        printf '%s\t%s\t%s\n' "${task}" FAIL "${result_path}" >> "${status_dir}/lane${lane}.tsv"
        echo "ROBODOJO_FAST_TASK_EXHAUSTED lane=${lane} task=${task}" >&2
        return 1
    }

    local bootstrap_task task expected
    bootstrap_task=$(head -n 1 "${task_file}")
    start_server "${bootstrap_task}"
    while IFS= read -r task || [[ -n "${task}" ]]; do
        [[ -n "${task}" ]] || continue
        if [[ "${eval_num}" == native ]]; then
            expected=$("${sim_env}/bin/python" "${script_dir}/result_tools.py" native-count \
                "${robodojo_root}/task/RoboDojo/config/_task.yml" "${task}")
        else
            expected=${eval_num}
            native=$("${sim_env}/bin/python" "${script_dir}/result_tools.py" native-count \
                "${robodojo_root}/task/RoboDojo/config/_task.yml" "${task}")
            (( expected <= native )) || { echo "Requested ${expected} episodes, but ${task} supports ${native}" >&2; return 2; }
        fi
        run_task "${task}" "${expected}"
    done < "${task_file}"
    echo "ROBODOJO_FAST_LANE_OK lane=${lane} server_starts=${server_generation}"
    : > "${status_dir}/lane${lane}.done"
    cleanup_lane
}

declare -a lane_pids=()
cleanup_all() {
    trap - EXIT INT TERM
    local pid
    for pid in "${lane_pids[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
    for pid in "${lane_pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup_all EXIT
trap 'cleanup_all; exit 130' INT TERM

for lane in "${!policy_gpu_list[@]}"; do
    : > "${status_dir}/lane${lane}.tsv"
    run_lane "${lane}" "${policy_gpu_list[$lane]}" "${env_gpu_list[$lane]}" \
        "${plan_dir}/worker_${lane}.tasks" >"${log_root}/lane${lane}.manager.log" 2>&1 &
    lane_pids+=("$!")
    if (( lane + 1 < ${#policy_gpu_list[@]} )); then sleep "${launch_stagger_s}"; fi
done

echo "ROBODOJO_FAST_START run_id=${run_id} seed=${seed} workers=${#lane_pids[@]} logs=${log_root}"
while :; do
    running=0
    for pid in "${lane_pids[@]}"; do kill -0 "${pid}" 2>/dev/null && running=$((running + 1)); done
    (( running > 0 )) || break
    passed=$(awk -F '\t' '$2 == "PASS" || $2 == "SKIP" {count++} END {print count+0}' "${status_dir}"/lane*.tsv)
    echo "ROBODOJO_FAST_HEARTBEAT running=${running} completed=${passed}"
    sleep 30
done

overall_rc=0
for lane in "${!lane_pids[@]}"; do
    if ! wait "${lane_pids[$lane]}"; then
        overall_rc=1
        echo "ROBODOJO_FAST_LANE_FAILED lane=${lane} log=${log_root}/lane${lane}.manager.log" >&2
        tail -100 "${log_root}/lane${lane}.manager.log" >&2 || true
    fi
done
trap - EXIT INT TERM
passed=$(awk -F '\t' '$2 == "PASS" || $2 == "SKIP" {count++} END {print count+0}' "${status_dir}"/lane*.tsv)
planned=$(find "${plan_dir}" -type f -name 'worker_*.tasks' -exec cat {} + | sed '/^$/d' | wc -l)
[[ "${overall_rc}" -eq 0 && "${passed}" -eq "${planned}" ]] || {
    echo "ROBODOJO_FAST_FAILED completed=${passed}/${planned} logs=${log_root}" >&2
    exit 1
}
echo "ROBODOJO_FAST_OK run_id=${run_id} seed=${seed} completed=${passed}/${planned} logs=${log_root}"
