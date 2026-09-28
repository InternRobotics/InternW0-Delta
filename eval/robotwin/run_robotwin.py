"""Distribute RoboTwin tasks across GPUs with isolated resumable sessions."""

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
import yaml
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = Path(__file__).with_name("single.py")
MODEL_SERVER_ENTRY = Path(__file__).with_name("server.py")
TASK_FILE = Path(__file__).with_name("tasks.txt")
POLL_INTERVAL_SEC = 2
TERMINATE_TIMEOUT_SEC = 15
PHASE_TO_TASK_CONFIG = {"clean": "demo_clean", "random": "demo_randomized"}


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _load_all_tasks() -> list[str]:
    tasks = TASK_FILE.read_text().splitlines()
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("The reference inventory must contain 50 unique tasks")
    return tasks


def _parse_csv_or_sequence(value: Any, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return list(default)
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = [item.strip().strip("\"'") for item in text.split(",")]
    else:
        values = [str(item).strip() for item in value]
    return list(dict.fromkeys(item for item in values if item))


def _parse_phases(value: Any) -> list[str]:
    phases = _parse_csv_or_sequence(value, default=["clean", "random"])
    invalid = [phase for phase in phases if phase not in PHASE_TO_TASK_CONFIG]
    if invalid or not phases:
        raise ValueError(f"Invalid MULTIRUN.phases={phases}; invalid={invalid}")
    return phases


def _parse_gpu_ids(value: Any, *, num_gpus: int) -> list[int]:
    raw = _parse_csv_or_sequence(value, default=[str(index) for index in range(num_gpus)])
    gpu_ids = [int(item) for item in raw]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise ValueError(f"Invalid MULTIRUN.gpu_ids={gpu_ids}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible != [""] and value is None:
        if len(visible) < num_gpus:
            raise ValueError(f"Requested {num_gpus} GPUs but only {len(visible)} are visible")
        return visible[:num_gpus]
    return gpu_ids


def _result_filename(phase: str) -> str:
    return "_result_clean.txt" if phase == "clean" else "_result_random.txt"


def _progress_filename(phase: str) -> str:
    return "_progress_clean.json" if phase == "clean" else "_progress_random.json"


def _heartbeat_filename(phase: str) -> str:
    return "_heartbeat_clean" if phase == "clean" else "_heartbeat_random"


def _parse_success_rate(result_file: Path) -> float:
    last_value = None
    for line in result_file.read_text(encoding="utf-8").splitlines():
        try:
            last_value = float(line.strip())
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"No success rate in {result_file}")
    return float(last_value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
        "EVALUATION.policy_server_socket",
        "EVALUATION.policy_client_id",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


@dataclass
class RunningState:
    task_name: str
    phase: str
    gpu_id: int
    slot_id: int
    group_id: int
    attempt: int
    process: subprocess.Popen[str]
    started_at: float
    heartbeat_path: Path
    heartbeat_baseline_ns: int | None
    heartbeat_seen: bool = False
    last_activity_at: float = 0.0


@dataclass
class ServerState:
    gpu_id: int
    group_id: int
    attempt: int
    process: subprocess.Popen[str]
    socket_path: Path
    health_path: Path
    started_at: float


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None")
    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not SINGLE_ENTRY.is_file() or not TASK_FILE.is_file():
        raise FileNotFoundError("RoboTwin evaluation entrypoints/task config are incomplete")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    gpu_ids = _parse_gpu_ids(cfg.MULTIRUN.get("gpu_ids"), num_gpus=num_gpus)
    phases = _parse_phases(cfg.MULTIRUN.get("phases"))
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("MULTIRUN.max_tasks_per_gpu must be positive")
    policy_server_mode = str(
        cfg.MULTIRUN.get("policy_server_mode", "dedicated")
    ).strip()
    if policy_server_mode not in {"dedicated", "shared_per_gpu"}:
        raise ValueError(
            "MULTIRUN.policy_server_mode must be dedicated or shared_per_gpu, "
            f"got {policy_server_mode!r}"
        )
    shared_mode = policy_server_mode == "shared_per_gpu"
    shared_servers_per_gpu = int(
        cfg.MULTIRUN.get("shared_servers_per_gpu", 1) or 1
    )
    raw_shared_clients_per_server = cfg.MULTIRUN.get("shared_clients_per_server")
    shared_clients_per_server = (
        max_tasks_per_gpu
        if raw_shared_clients_per_server is None
        else int(raw_shared_clients_per_server)
    )
    if shared_servers_per_gpu <= 0:
        raise ValueError("MULTIRUN.shared_servers_per_gpu must be positive")
    if shared_clients_per_server <= 0:
        raise ValueError("MULTIRUN.shared_clients_per_server must be positive")
    if shared_mode and not 1 <= shared_clients_per_server <= 8:
        raise ValueError(
            "shared_per_gpu requires 1 <= MULTIRUN.shared_clients_per_server <= 8"
        )
    servers_per_gpu = shared_servers_per_gpu if shared_mode else max_tasks_per_gpu
    clients_per_server = shared_clients_per_server if shared_mode else 1
    client_slots_per_gpu = (
        shared_servers_per_gpu * shared_clients_per_server
        if shared_mode
        else max_tasks_per_gpu
    )
    shared_request_timeout = float(
        cfg.MULTIRUN.get("shared_server_request_timeout_sec", 300)
    )
    startup_timeout = float(cfg.MULTIRUN.worker_startup_timeout_sec)
    stall_timeout = float(cfg.MULTIRUN.worker_stall_timeout_sec)
    max_restarts = int(cfg.MULTIRUN.max_worker_restarts)

    raw_tasks = cfg.EVALUATION.task_name
    tasks = (
        _load_all_tasks()
        if raw_tasks is None or not str(raw_tasks).strip()
        else _parse_csv_or_sequence(raw_tasks, default=[])
    )
    if not tasks or set(tasks) - set(_load_all_tasks()):
        raise ValueError("Select task names from eval/robotwin/tasks.txt")

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir = output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "checkpoint": {"path": str(ckpt_path), "bytes": ckpt_path.stat().st_size,
                       "mtime_ns": ckpt_path.stat().st_mtime_ns},
        "seed": int(cfg.seed),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": OmegaConf.to_container(cfg.data.train, resolve=True),
        "evaluation": {k: v for k, v in OmegaConf.to_container(cfg.EVALUATION, resolve=True).items()
                       if k not in {"output_dir", "resume", "policy_server_socket", "policy_client_id",
                                    "eval_video_log", "timing_enabled"}},
        "phases": phases,
    }
    stats = _resolve_path(str(cfg.EVALUATION.dataset_stats_path), base=PROJECT_ROOT)
    contract["stats_sha256"] = hashlib.sha256(stats.read_bytes()).hexdigest()
    contract_path = run_output_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Output directory contains a different evaluation configuration; choose a new directory")
        if not cfg.EVALUATION.resume:
            raise ValueError("Output directory already exists; enable resume or choose a new directory")
    elif (run_output_dir / "summary.json").exists():
        raise ValueError("Existing results lack a run contract; choose a new output directory")
    _atomic_write_json(contract_path, contract)
    OmegaConf.save(cfg, run_output_dir / "config.yaml", resolve=True)
    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"
    extra_overrides = _collect_worker_overrides()
    server_action_horizon = cfg.EVALUATION.get("action_horizon")
    if server_action_horizon is None:
        server_action_horizon = int(cfg.data.train.num_frames) - 1
    server_action_hz = float(cfg.EVALUATION.action_hz)
    sim_task_choice = str(HydraConfig.get().runtime.choices.task)
    dataset_stats_path: Path | None = None
    vlm_path: Path | None = None
    if shared_mode:
        dataset_stats_value = cfg.EVALUATION.get("dataset_stats_path")
        if dataset_stats_value is None:
            raise ValueError("EVALUATION.dataset_stats_path is required in shared_per_gpu mode")
        dataset_stats_path = _resolve_path(str(dataset_stats_value), base=PROJECT_ROOT)
        if not dataset_stats_path.is_file():
            raise FileNotFoundError(dataset_stats_path)
        vlm_value = cfg.model.understanding.get("vlm_model_path")
        if vlm_value is None:
            raise ValueError("model.understanding.vlm_model_path is required in shared mode")
        vlm_path = _resolve_path(str(vlm_value), base=PROJECT_ROOT)
        if not vlm_path.exists():
            raise FileNotFoundError(vlm_path)

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    for task in tasks:
        for phase in phases:
            result_file = run_output_dir / task / _result_filename(phase)
            if result_file.is_file():
                task_rates[task][phase] = _parse_success_rate(result_file)

    pending_tasks = deque(
        task for task in tasks if any(task_rates[task][phase] is None for phase in phases)
    )
    running: list[RunningState] = []
    servers: dict[tuple[int, int], ServerState] = {}
    failed_records: list[dict[str, Any]] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()

    def write_outputs() -> None:
        def mean(phase: str) -> float | None:
            values = [task_rates[task][phase] for task in tasks]
            valid = [value for value in values if value is not None]
            return None if not valid else float(sum(valid) / len(valid))

        with summary_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow([task, task_rates[task]["clean"], task_rates[task]["random"]])
            writer.writerow(["__overall__", mean("clean"), mean("random")])

        payload = {
            "phases": phases,
            "requested_tasks": len(tasks),
            "completed_tasks": {phase: sum(task_rates[t][phase] is not None for t in tasks) for phase in phases},
            "complete": all(task_rates[t][phase] is not None for t in tasks for phase in phases),
            "failed_task_phases": len(failed_records),
            "checkpoint": str(ckpt_path),
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": task_rates[task]["clean"],
                    "random_success_rate": task_rates[task]["random"],
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": mean("clean"),
                "random_mean_success_rate": mean("random"),
            },
        }
        _atomic_write_json(summary_json, payload)
        with failed_tasks_file.open("w", encoding="utf-8") as file:
            for record in failed_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def first_incomplete_phase(task_name: str) -> str | None:
        return next(
            (phase for phase in phases if task_rates[task_name][phase] is None),
            None,
        )

    def server_group_for_slot(slot_id: int) -> int:
        return slot_id // clients_per_server

    def server_socket_path(gpu_id: int, group_id: int) -> Path:
        tag = hashlib.sha256(f"{gpu_id}:{group_id}".encode()).hexdigest()[:12]
        return Path(f"/tmp/internw0-rw-{os.getpid()}-{tag}.sock")

    def server_health_path(gpu_id: int, group_id: int) -> Path:
        return (
            run_output_dir
            / "shared_servers"
            / f"server_gpu{gpu_id}_group{group_id}.health.json"
        )

    def build_cmd(
        task_name: str,
        phase: str,
        gpu_id: int,
        slot_id: int,
        attempt: int,
    ) -> list[str]:
        group_id = server_group_for_slot(slot_id)
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={PHASE_TO_TASK_CONFIG[phase]}",
            f"EVALUATION.output_dir={run_output_dir}",
            *extra_overrides,
        ]
        if shared_mode:
            client_id = (
                f"rw-gpu{gpu_id}-group{group_id}-slot{slot_id}-{task_name}-{phase}-a{attempt}-"
                f"{time.time_ns()}"
            )
            cmd.extend(
                [
                    f"EVALUATION.policy_server_socket={server_socket_path(gpu_id, group_id)}",
                    f"EVALUATION.policy_client_id={client_id}",
                ]
            )
        return cmd

    def launch(
        task_name: str,
        phase: str,
        gpu_id: int,
        slot_id: int,
        attempt: int,
    ) -> RunningState:
        heartbeat_path = run_output_dir / task_name / _heartbeat_filename(phase)
        baseline = heartbeat_path.stat().st_mtime_ns if heartbeat_path.is_file() else None
        cmd = build_cmd(task_name, phase, gpu_id, slot_id, attempt)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"group={server_group_for_slot(slot_id)} slot={slot_id} "
            f"attempt={attempt} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            start_new_session=True,
        )
        now = time.monotonic()
        return RunningState(
            task_name=task_name,
            phase=phase,
            gpu_id=gpu_id,
            slot_id=slot_id,
            group_id=server_group_for_slot(slot_id),
            attempt=attempt,
            process=process,
            started_at=now,
            heartbeat_path=heartbeat_path,
            heartbeat_baseline_ns=baseline,
            last_activity_at=now,
        )

    def launch_server(gpu_id: int, group_id: int, attempt: int) -> ServerState:
        assert dataset_stats_path is not None and vlm_path is not None
        socket_path = server_socket_path(gpu_id, group_id)
        health_path = server_health_path(gpu_id, group_id)
        socket_path.unlink(missing_ok=True)
        health_path.unlink(missing_ok=True)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        server_id = f"robotwin-gpu{gpu_id}-group{group_id}-a{attempt}-{os.getpid()}"
        cmd = [
            sys.executable,
            "-u",
            str(MODEL_SERVER_ENTRY),
            "--resolved-config",
            str(run_output_dir / "config.yaml"),
            "--socket",
            str(socket_path),
            "--checkpoint",
            str(ckpt_path),
            "--dataset-stats",
            str(dataset_stats_path),
            "--vlm-path",
            str(vlm_path),
            "--sim-task",
            sim_task_choice,
            "--action-horizon",
            str(int(server_action_horizon)),
            "--action-hz",
            str(server_action_hz),
            "--replan-steps",
            str(int(cfg.EVALUATION.replan_steps)),
            "--num-inference-steps",
            str(int(cfg.EVALUATION.num_inference_steps)),
            "--seed",
            str(int(cfg.seed)),
            "--max-clients",
            str(clients_per_server),
            "--persistent",
            "--health-path",
            str(health_path),
            "--request-timeout-sec",
            str(shared_request_timeout),
            "--server-id",
            server_id,
            "--gpu-id",
            str(gpu_id),
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "PYTHONUNBUFFERED": "1",
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]),
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WAM_PROFILE_DIR": str(
                    run_output_dir
                    / "shared_servers"
                    / f"server_gpu{gpu_id}_group{group_id}"
                    / f"attempt_{attempt}"
                    / "profiler"
                ),
            }
        )
        log_path = run_output_dir / "shared_servers" / f"server_gpu{gpu_id}_group{group_id}.log"
        log(
            f"launch shared server gpu={gpu_id} group={group_id} attempt={attempt} "
            f"cmd={' '.join(cmd)} log={log_path}"
        )
        with log_path.open("a", encoding="utf-8") as output:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return ServerState(
            gpu_id=gpu_id,
            group_id=group_id,
            attempt=attempt,
            process=process,
            socket_path=socket_path,
            health_path=health_path,
            started_at=time.monotonic(),
        )

    def wait_server_ready(state: ServerState) -> None:
        while not state.socket_path.is_socket():
            return_code = state.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"shared server gpu={state.gpu_id} group={state.group_id} "
                    f"exited during startup: {return_code}"
                )
            if time.monotonic() - state.started_at > startup_timeout:
                raise TimeoutError(
                    f"shared server gpu={state.gpu_id} group={state.group_id} startup timed out"
                )
            time.sleep(1)
        log(
            f"shared server ready gpu={state.gpu_id} group={state.group_id} "
            f"pid={state.process.pid} "
            f"socket={state.socket_path}"
        )

    def terminate(state: RunningState) -> None:
        if state.process.poll() is not None:
            return
        try:
            os.killpg(state.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            state.process.wait(timeout=TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(state.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            state.process.wait()

    def terminate_server(state: ServerState) -> None:
        if state.process.poll() is None:
            try:
                os.killpg(state.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                state.process.wait(timeout=TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(state.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                state.process.wait()
        state.socket_path.unlink(missing_ok=True)

    def advance_stalled_seed(state: RunningState) -> None:
        progress_path = run_output_dir / state.task_name / _progress_filename(state.phase)
        if not progress_path.is_file():
            return
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            current_seed = payload.get("current_seed")
            if current_seed is None:
                return
            current_seed = int(current_seed)
            skipped = payload.get("skipped_seeds", [])
            if not isinstance(skipped, list):
                skipped = []
            skipped.append(
                {
                    "seed": current_seed,
                    "reason": "manager_worker_stall",
                    "time": datetime.now().isoformat(),
                }
            )
            payload.update(
                {
                    "next_seed": current_seed + 1,
                    "current_seed": None,
                    "skipped_seeds": skipped,
                    "status": "manager_skipped_stall",
                    "updated_at": datetime.now().isoformat(),
                }
            )
            _atomic_write_json(progress_path, payload)
            completed = int(payload.get("completed_episodes", 0))
            partial_video = progress_path.parent / f"episode{completed}.mp4"
            if partial_video.is_file():
                partial_video.unlink()
            log(
                f"advanced stalled seed task={state.task_name} phase={state.phase} "
                f"seed={current_seed}->{current_seed + 1}"
            )
        except Exception as exc:
            log(f"could not advance stalled progress {progress_path}: {exc!r}")

    def gpu_running_count(gpu_id: int) -> int:
        return sum(
            state.gpu_id == gpu_id and state.process.poll() is None for state in running
        )

    def launch_pending(gpu_id: int) -> None:
        while pending_tasks and gpu_running_count(gpu_id) < client_slots_per_gpu:
            used_slots = {
                state.slot_id
                for state in running
                if state.gpu_id == gpu_id and state.process.poll() is None
            }
            slot_id = next(
                slot for slot in range(client_slots_per_gpu) if slot not in used_slots
            )
            task_name = pending_tasks.popleft()
            phase = first_incomplete_phase(task_name)
            if phase is not None:
                running.append(
                    launch(task_name, phase, gpu_id, slot_id=slot_id, attempt=0)
                )

    def retry_or_fail(state: RunningState, reason: str, *, advance_seed: bool) -> None:
        if advance_seed:
            advance_stalled_seed(state)
        if state.attempt < max_restarts:
            running.append(
                launch(
                    state.task_name,
                    state.phase,
                    state.gpu_id,
                    state.slot_id,
                    attempt=state.attempt + 1,
                )
            )
            log(
                f"retry task={state.task_name} phase={state.phase} gpu={state.gpu_id} "
                f"group={state.group_id} slot={state.slot_id} "
                f"attempt={state.attempt + 1}/{max_restarts} reason={reason}"
            )
            return
        failed_records.append(
            {
                "task_name": state.task_name,
                "phase": state.phase,
                "gpu_id": state.gpu_id,
                "group_id": state.group_id,
                "attempt": state.attempt,
                "reason": reason,
            }
        )
        log(
            f"failed permanently task={state.task_name} phase={state.phase} "
            f"gpu={state.gpu_id} group={state.group_id} reason={reason}"
        )
        launch_pending(state.gpu_id)

    def shared_server_failure_reason(state: ServerState) -> tuple[str, bool] | None:
        return_code = state.process.poll()
        if return_code is not None:
            return f"shared_server_exit_{return_code}", False
        if not state.health_path.is_file():
            return None
        try:
            payload = json.loads(state.health_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        fatal_error = payload.get("fatal_error")
        if fatal_error:
            return f"shared_server_fatal:{fatal_error}", False
        inflight_request = payload.get("inflight_request_id")
        request_started_at = payload.get("last_request_started_at")
        if inflight_request is not None and request_started_at is not None:
            elapsed = time.time() - float(request_started_at)
            if elapsed > shared_request_timeout:
                return (
                    f"shared_server_request_stall:{inflight_request}:{elapsed:.1f}s",
                    True,
                )
        return None

    def recover_shared_server_group(
        server: ServerState,
        reason: str,
        *,
        advance_seed: bool,
    ) -> None:
        affected = [
            state
            for state in running
            if state.gpu_id == server.gpu_id and state.group_id == server.group_id
        ]
        log(
            f"recover shared server group gpu={server.gpu_id} group={server.group_id} "
            f"attempt={server.attempt} clients={len(affected)} reason={reason}"
        )
        for state in affected:
            running.remove(state)
            terminate(state)
            if advance_seed:
                advance_stalled_seed(state)
        terminate_server(server)
        if server.attempt >= max_restarts:
            for state in affected:
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": state.gpu_id,
                        "group_id": state.group_id,
                        "attempt": state.attempt,
                        "reason": f"server_restart_budget_exhausted:{reason}",
                    }
                )
            write_outputs()
            raise RuntimeError(
                f"shared server gpu={server.gpu_id} group={server.group_id} "
                f"exhausted {max_restarts} restarts"
            )
        replacement = launch_server(
            server.gpu_id,
            server.group_id,
            attempt=server.attempt + 1,
        )
        servers[(server.gpu_id, server.group_id)] = replacement
        wait_server_ready(replacement)
        for state in affected:
            retry_or_fail(
                state,
                reason,
                advance_seed=False,
            )

    log(
        f"manager start tasks={len(tasks)} phases={phases} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} startup_timeout={startup_timeout}s "
        f"stall_timeout={stall_timeout}s output_dir={run_output_dir} "
        f"policy_server_mode={policy_server_mode} "
        f"servers_per_gpu={servers_per_gpu} "
        f"clients_per_server={clients_per_server} "
        f"client_slots_per_gpu={client_slots_per_gpu} "
        f"total_servers={len(gpu_ids) * servers_per_gpu} "
        f"total_clients={len(gpu_ids) * client_slots_per_gpu}"
    )
    write_outputs()
    if not pending_tasks:
        log("All requested tasks are already complete")
        return
    if shared_mode:
        for gpu_id in gpu_ids:
            for group_id in range(servers_per_gpu):
                servers[(gpu_id, group_id)] = launch_server(gpu_id, group_id, attempt=0)
        try:
            for server in servers.values():
                wait_server_ready(server)
        except BaseException:
            for server in servers.values():
                terminate_server(server)
            raise
    for gpu_id in gpu_ids:
        launch_pending(gpu_id)

    try:
        while running:
            progressed = False
            now = time.monotonic()
            if shared_mode:
                for server in list(servers.values()):
                    failure = shared_server_failure_reason(server)
                    if failure is None:
                        continue
                    reason, advance_seed = failure
                    recover_shared_server_group(
                        server,
                        reason,
                        advance_seed=advance_seed,
                    )
                    progressed = True
            for state in list(running):
                return_code = state.process.poll()
                if return_code is None:
                    if state.heartbeat_path.is_file():
                        heartbeat_ns = state.heartbeat_path.stat().st_mtime_ns
                        if (
                            state.heartbeat_baseline_ns is None
                            or heartbeat_ns > state.heartbeat_baseline_ns
                        ):
                            state.heartbeat_baseline_ns = heartbeat_ns
                            state.heartbeat_seen = True
                            state.last_activity_at = now
                    timeout = stall_timeout if state.heartbeat_seen else startup_timeout
                    if now - state.last_activity_at <= timeout:
                        continue
                    progressed = True
                    running.remove(state)
                    reason = "worker_stall" if state.heartbeat_seen else "worker_startup_timeout"
                    log(
                        f"watchdog timeout task={state.task_name} phase={state.phase} "
                        f"gpu={state.gpu_id} group={state.group_id} reason={reason} "
                        f"idle={now - state.last_activity_at:.1f}s"
                    )
                    terminate(state)
                    retry_or_fail(state, reason, advance_seed=state.heartbeat_seen)
                    continue

                progressed = True
                running.remove(state)
                if return_code != 0:
                    retry_or_fail(
                        state,
                        f"process_exit_{return_code}",
                        advance_seed=False,
                    )
                    continue

                result_file = run_output_dir / state.task_name / _result_filename(state.phase)
                try:
                    task_rates[state.task_name][state.phase] = _parse_success_rate(result_file)
                except Exception as exc:
                    retry_or_fail(
                        state,
                        f"result_parse_error:{exc!r}",
                        advance_seed=False,
                    )
                    continue
                log(
                    f"done task={state.task_name} phase={state.phase} gpu={state.gpu_id} "
                    f"group={state.group_id} "
                    f"success_rate={task_rates[state.task_name][state.phase]:.4f}"
                )
                write_outputs()
                next_phase = first_incomplete_phase(state.task_name)
                if next_phase is not None:
                    running.append(
                        launch(
                            state.task_name,
                            next_phase,
                            state.gpu_id,
                            state.slot_id,
                            attempt=0,
                        )
                    )
                else:
                    launch_pending(state.gpu_id)

            if not progressed:
                time.sleep(POLL_INTERVAL_SEC)
    except BaseException:
        for state in list(running):
            terminate(state)
        for server in servers.values():
            terminate_server(server)
        raise
    finally:
        for server in servers.values():
            terminate_server(server)
        write_outputs()

    log(f"manager finished failed={len(failed_records)} summary={summary_csv}")
    if failed_records:
        raise RuntimeError(f"{len(failed_records)} RoboTwin task phases exhausted retries")


if __name__ == "__main__":
    main()
