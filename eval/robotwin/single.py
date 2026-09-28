from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVER = Path(__file__).with_name("server.py")
EVAL_CLIENT = Path(__file__).with_name("client.py")
DEPLOY_CONFIG = Path(__file__).with_name("policy.yaml")
AGGREGATE_PROFILER = PROJECT_ROOT / "eval" / "aggregate_profiler.py"


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _library_path(*paths: Path) -> str:
    entries = [str(path) for path in paths if path.is_dir()]
    inherited = os.environ.get("LD_LIBRARY_PATH")
    if inherited:
        entries.append(inherited)
    return os.pathsep.join(entries)


def _client_pythonpath(project_root: Path, robotwin_root: Path) -> str:
    entries = [str(project_root), str(robotwin_root)]
    curobo = os.environ.get("WAM_RW_CUROBO_ROOT")
    if curobo:
        source = Path(curobo) / "src"
        if not (source / "curobo/__init__.py").is_file():
            raise FileNotFoundError(source / "curobo/__init__.py")
        entries.append(str(source))
    return os.pathsep.join(entries)


def _dataset_stats_path(cfg: DictConfig, checkpoint: Path) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit is not None and str(explicit).strip().lower() not in {"", "none", "null"}:
        path = _resolve_path(str(explicit), base=PROJECT_ROOT)
        if path.is_file():
            return path
    for parent in list(checkpoint.parents)[:4]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("dataset_stats.json was not found")


def _override(items: list[str], key: str, value) -> None:
    if isinstance(value, bool):
        encoded = "True" if value else "False"
    elif isinstance(value, (int, float)):
        encoded = str(value)
    else:
        encoded = repr(str(value))
    items.extend([f"--{key}", encoded])


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _aggregate_profiler(profile_dir: Path) -> None:
    if not (profile_dir / "rank_000" / "summary.json").is_file():
        return
    subprocess.run(
        [sys.executable, str(AGGREGATE_PROFILER), str(profile_dir)],
        cwd=PROJECT_ROOT,
        check=True,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None or cfg.EVALUATION.task_name is None:
        raise ValueError("ckpt and EVALUATION.task_name are required")
    checkpoint = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stats_path = _dataset_stats_path(cfg, checkpoint)
    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    simulator_python = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(str(cfg.EVALUATION.simulator_python)))))
    robotwin_env = simulator_python.parent.parent
    vlm_path = _resolve_path(str(cfg.model.understanding.vlm_model_path), base=PROJECT_ROOT)
    for required in (
        robotwin_root / "task_config" / "demo_clean.yml",
        simulator_python,
        vlm_path,
        MODEL_SERVER,
        EVAL_CLIENT,
        DEPLOY_CONFIG,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    task_name = str(cfg.EVALUATION.task_name)
    task_config = str(cfg.EVALUATION.task_config)
    sim_task_choice = str(HydraConfig.get().runtime.choices.task)
    action_horizon_value = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_value is None
        else int(action_horizon_value)
    )
    action_hz = float(cfg.EVALUATION.action_hz)
    run_output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    task_dir = run_output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    profile_root = task_dir / "profiler"
    external_socket_value = cfg.EVALUATION.get("policy_server_socket")
    uses_external_server = external_socket_value is not None and str(
        external_socket_value
    ).strip().lower() not in {"", "none", "null"}
    socket_path = (
        Path(str(external_socket_value)).expanduser().resolve()
        if uses_external_server
        else Path(f"/tmp/wro-{os.getpid()}-{cfg.gpu_id}.sock")
    )
    if not uses_external_server and (socket_path.exists() or socket_path.is_socket()):
        socket_path.unlink()
    client_id_value = cfg.EVALUATION.get("policy_client_id")
    client_id = (
        str(client_id_value)
        if client_id_value is not None
        and str(client_id_value).strip().lower() not in {"", "none", "null"}
        else f"robotwin-gpu{cfg.gpu_id}-{task_name}-{os.getpid()}"
    )

    common_env = os.environ.copy()
    common_env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    common_env["PYTHONUNBUFFERED"] = "1"
    common_env["WORLD_SIZE"] = "1"
    common_env["RANK"] = "0"
    common_env["LOCAL_RANK"] = "0"
    model_env = common_env.copy()
    model_env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + common_env.get("PATH", "")
    model_env["WAM_PROFILE_DIR"] = str(profile_root / "server")
    model_env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"


    resolved_config = task_dir / "policy_config.yaml"
    OmegaConf.save(cfg, resolved_config, resolve=True)
    server_cmd = [
        sys.executable,
        "-u",
        str(MODEL_SERVER),
        "--resolved-config",
        str(resolved_config),
        "--socket",
        str(socket_path),
        "--checkpoint",
        str(checkpoint),
        "--dataset-stats",
        str(stats_path),
        "--vlm-path",
        str(vlm_path),
        "--sim-task",
        sim_task_choice,
        "--action-horizon",
        str(action_horizon),
        "--action-hz",
        str(action_hz),
        "--replan-steps",
        str(int(cfg.EVALUATION.replan_steps)),
        "--num-inference-steps",
        str(int(cfg.EVALUATION.num_inference_steps)),
        "--seed",
        str(int(cfg.seed)),
        "--max-clients",
        "1",
    ]

    overrides: list[str] = []
    for key, value in (
        ("task_name", task_name),
        ("task_config", task_config),
        ("ckpt_setting", checkpoint),
        ("seed", int(cfg.seed)),
        ("policy_name", "wam_policy"),
        ("instruction_type", cfg.EVALUATION.instruction_type),
        ("eval_num_episodes", int(cfg.EVALUATION.eval_num_episodes)),
        ("eval_output_dir", task_dir),
        ("eval_video_log", bool(cfg.EVALUATION.eval_video_log)),
        ("action_horizon", action_horizon),
        ("replan_steps", int(cfg.EVALUATION.replan_steps)),
        ("timing_enabled", True),
        ("skip_get_obs_within_replan", False),
        ("resume", bool(cfg.EVALUATION.resume)),
        ("expert_timeout_sec", int(cfg.EVALUATION.expert_timeout_sec)),
        ("rollout_setup_timeout_sec", int(cfg.EVALUATION.rollout_setup_timeout_sec)),
        ("episode_timeout_sec", int(cfg.EVALUATION.episode_timeout_sec)),
        ("heartbeat_interval_sec", float(cfg.EVALUATION.heartbeat_interval_sec)),
        ("max_seed_attempts", int(cfg.EVALUATION.max_seed_attempts)),
    ):
        _override(overrides, key, value)

    client_cmd = [
        str(simulator_python),
        "-u",
        str(EVAL_CLIENT),
        "--socket",
        str(socket_path),
        "--config",
        str(DEPLOY_CONFIG),
        "--client-id",
        client_id,
        "--task-label",
        task_name,
        "--phase",
        task_config,
        "--overrides",
        *overrides,
    ]
    client_env = common_env.copy()
    client_env["PATH"] = str(simulator_python.parent) + os.pathsep + common_env.get("PATH", "")
    client_env["WAM_PROFILE_DIR"] = str(profile_root / "client")
    client_env["PYTHONPATH"] = _client_pythonpath(PROJECT_ROOT, robotwin_root)
    client_env["LD_LIBRARY_PATH"] = _library_path(
        robotwin_env / "lib",
    )
    client_env["ROBOTWIN_ROOT"] = str(robotwin_root)
    client_env["WAM_RW_CLIENT_CACHE_ROOT"] = str(task_dir / "cache")
    client_env["SAPIEN_RENDER_DEVICE"] = "0"
    client_env["WAM_RW_ACTION_HORIZON"] = str(action_horizon)
    client_env["WAM_RW_ACTION_HZ"] = str(action_hz)
    client_env["WAM_RW_REPLAN_STEPS"] = str(int(cfg.EVALUATION.replan_steps))

    server: subprocess.Popen | None = None
    client: subprocess.Popen | None = None
    try:
        print(
            "OFFICIAL_WORKER_TOPOLOGY "
            f"mode={'shared_per_gpu' if uses_external_server else 'dedicated'} "
            f"gpu={cfg.gpu_id} client_id={client_id} socket={socket_path}",
            flush=True,
        )
        if not uses_external_server:
            print(f"OFFICIAL_WORKER_SERVER_CMD {' '.join(server_cmd)}", flush=True)
            server = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT, env=model_env)
        started = time.monotonic()
        while not socket_path.is_socket():
            if server is not None:
                return_code = server.poll()
                if return_code is not None:
                    raise RuntimeError(f"model server exited during startup: {return_code}")
            if time.monotonic() - started > float(cfg.MULTIRUN.worker_startup_timeout_sec):
                raise TimeoutError("model server startup timed out")
            time.sleep(1)

        print(f"OFFICIAL_WORKER_CLIENT_CMD {' '.join(client_cmd)}", flush=True)
        client = subprocess.Popen(client_cmd, cwd=robotwin_root, env=client_env)
        client_return_code = client.wait()
        if client_return_code != 0:
            raise RuntimeError(f"official simulator client failed: {client_return_code}")
        if server is not None:
            server_return_code = server.wait(timeout=60)
            if server_return_code != 0:
                raise RuntimeError(f"model server failed: {server_return_code}")
    finally:
        _terminate(client)
        _terminate(server)
        if not uses_external_server and (socket_path.exists() or socket_path.is_socket()):
            socket_path.unlink()

    result_name = "_result_clean.txt" if task_config == "demo_clean" else "_result_random.txt"
    if not (task_dir / result_name).is_file():
        raise FileNotFoundError(task_dir / result_name)
    OmegaConf.save(cfg, task_dir / f"eval_config_{task_config}.yaml")
    if not uses_external_server:
        _aggregate_profiler(profile_root / "server")
    _aggregate_profiler(profile_root / "client")


if __name__ == "__main__":
    main()
