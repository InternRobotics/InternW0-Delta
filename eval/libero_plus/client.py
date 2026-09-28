import fcntl
import json
import logging
import os
import sys
import time
import uuid
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ.setdefault("WAM_LIBERO_REPO", "LIBERO-plus")
libero_plus_root = project_root / "third_party" / "LIBERO-plus"
if libero_plus_root.exists() and str(libero_plus_root) not in sys.path:
    sys.path.insert(0, str(libero_plus_root))

from eval.libero.eval_config import apply_training_config_defaults_from_checkpoint
from eval.libero.eval_libero_single import (
    NumpyEncoder,
    _get_max_steps,
    benchmark,
    set_global_seed,
)
from eval.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from eval.libero_plus.profiling import build_eval_profiler
from eval.shared_policy_server import PROTOCOL_VERSION
from eval.profiler import Profiler


def _select_path(cfg: DictConfig, key: str, *, required: bool = True) -> Optional[Path]:
    value = OmegaConf.select(cfg, key)
    if value is None:
        if required:
            raise ValueError(f"{key} must be set.")
        return None
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


def _read_task_file(path: Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            suite, raw_task_id = line.split(",", 1)
            tasks.append((suite, int(raw_task_id)))
    return tasks


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line.rstrip("\n") + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(payload, cls=NumpyEncoder) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class PolicyClient:
    def __init__(
        self,
        socket_path: Path,
        timeout_s: float = 300.0,
        profiler: Optional[Profiler] = None,
        client_id: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> None:
        self.socket_path = socket_path
        self.timeout_s = float(timeout_s)
        self.profiler = profiler
        self.client_id = str(client_id or f"libero-client-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.metadata = dict(metadata or {})
        self.conn = None
        self.server_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self._request_sequence = 0

    def _section(self, name: str):
        if self.profiler is None:
            from contextlib import nullcontext

            return nullcontext()
        return self.profiler.section(name)

    def connect(self) -> None:
        deadline = time.time() + self.timeout_s
        last_error: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                self.conn = Client(str(self.socket_path), family="AF_UNIX")
                self.conn.send(
                    {
                        "type": "hello",
                        "protocol_version": PROTOCOL_VERSION,
                        "client_id": self.client_id,
                        "metadata": {"benchmark": "libero_plus", **self.metadata},
                    }
                )
                hello = self.conn.recv()
                if not isinstance(hello, dict) or not bool(hello.get("ok", False)):
                    raise RuntimeError(f"Policy server rejected session: {hello!r}")
                if int(hello.get("protocol_version", -1)) != PROTOCOL_VERSION:
                    raise RuntimeError(f"Policy protocol mismatch: {hello!r}")
                self.server_id = str(hello.get("server_id", ""))
                self.session_id = str(hello.get("session_id", ""))
                self.call("ping", {})
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError, EOFError) as exc:
                last_error = exc
                if self.conn is not None:
                    self.conn.close()
                    self.conn = None
                time.sleep(0.5)
        raise RuntimeError(f"Timed out connecting to policy server {self.socket_path}: {last_error}") from last_error

    def call(self, command: str, payload: Any) -> Any:
        if self.conn is None:
            raise RuntimeError("PolicyClient is not connected.")
        self._request_sequence += 1
        request_id = f"{self.client_id}:{self._request_sequence}"
        request = {
            "type": "request",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "command": str(command),
            "payload": payload,
        }
        with self._section(f"rpc/{command}"):
            with self._section("ipc_send"):
                self.conn.send(request)
            with self._section("ipc_wait_response"):
                response = self.conn.recv()
        if not isinstance(response, dict):
            raise RuntimeError(f"Malformed policy response for {command!r}: {response!r}")
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"Policy response id mismatch: expected {request_id!r}, got {response!r}"
            )
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                f"Policy server command {command!r} failed "
                f"(fatal={bool(response.get('fatal', False))}, "
                f"type={response.get('error_type')}):\n{response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        result = response.get("result")
        if isinstance(result, dict):
            result = dict(result)
            server_timing = dict(result.get("server_timing") or {})
            server_timing["scheduler_queue_wait_s"] = float(
                response.get("queue_wait_s", 0.0) or 0.0
            )
            server_timing["scheduler_handler_s"] = float(
                response.get("handler_s", 0.0) or 0.0
            )
            result["server_timing"] = server_timing
        return result

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def _prepare_initial_states(task_suite, task_id: int, num_trials: int):
    initial_states = list(task_suite.get_task_init_states(task_id))
    while len(initial_states) < num_trials:
        initial_states.extend(initial_states[: (num_trials - len(initial_states))])
    return initial_states


def _add_elapsed(totals: dict[str, float], key: str, started: float) -> None:
    totals[key] = totals.get(key, 0.0) + (time.perf_counter() - started)


def _accumulate_server_timing(
    totals: dict[str, float],
    counters: dict[str, int],
    payload: Any,
) -> None:
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if isinstance(value, bool):
            if value:
                counters[key] = counters.get(key, 0) + 1
        elif isinstance(value, (int, float)):
            totals[str(key)] = totals.get(str(key), 0.0) + float(value)


def _run_episode_with_server(
    *,
    env,
    policy: PolicyClient,
    initial_state,
    task_suite_name: str,
    task_id: int,
    task_description: str,
    cfg: DictConfig,
    episode_idx: int,
    profiler: Profiler,
    save_rollout_videos: bool = True,
) -> tuple[bool, list, Optional[float], dict[str, Any]]:
    episode_start = time.perf_counter()
    client_seconds: dict[str, float] = {}
    server_seconds: dict[str, float] = {}
    server_counters: dict[str, int] = {}
    max_steps = _get_max_steps(task_suite_name)
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    if visualize_future_video:
        raise ValueError("Pair client/server eval does not support visualize_future_video.")

    stage_start = time.perf_counter()
    with profiler.section("session_reset_rpc"):
        policy.call(
            "reset_session",
            {
                "task_suite_name": task_suite_name,
                "task_id": int(task_id),
                "task_description": task_description,
            },
        )
    _add_elapsed(client_seconds, "session_reset_rpc_s", stage_start)

    stage_start = time.perf_counter()
    with profiler.section("env_reset"):
        env.reset()
    _add_elapsed(client_seconds, "env_reset_s", stage_start)
    stage_start = time.perf_counter()
    with profiler.section("env_set_init_state"):
        obs = env.set_init_state(initial_state)
    _add_elapsed(client_seconds, "env_set_init_state_s", stage_start)
    replay_images = []
    future_psnrs: list[float] = []

    t = 0
    done = False
    pending_actions_remaining = 0
    warmup_steps = 0
    control_steps = 0
    replan_calls = 0
    cached_action_calls = 0
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"{task_suite_name}:{task_id} ep {episode_idx + 1}")
    try:
        while t < max_steps + num_steps_wait:
            pbar.update(1)
            if t < num_steps_wait:
                stage_start = time.perf_counter()
                with profiler.section("warmup_env_step"):
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                _add_elapsed(client_seconds, "warmup_env_step_s", stage_start)
                warmup_steps += 1
                t += 1
                continue

            if save_rollout_videos:
                stage_start = time.perf_counter()
                with profiler.section("rollout_frame_capture"):
                    replay_images.append(get_libero_image(obs))
                _add_elapsed(client_seconds, "rollout_frame_capture_s", stage_start)

            expected_replan = pending_actions_remaining <= 0
            act_obs = obs if expected_replan else None
            act_section = "act_rpc_replan" if expected_replan else "act_rpc_cached"
            stage_start = time.perf_counter()
            with profiler.section(act_section):
                action_payload = policy.call(
                    "act", {"obs": act_obs, "timestep": int(t)}
                )
            act_elapsed = time.perf_counter() - stage_start
            client_seconds["act_rpc_total_s"] = (
                client_seconds.get("act_rpc_total_s", 0.0) + act_elapsed
            )
            client_key = f"{act_section}_s"
            client_seconds[client_key] = client_seconds.get(client_key, 0.0) + act_elapsed
            _accumulate_server_timing(
                server_seconds,
                server_counters,
                action_payload.get("server_timing"),
            )
            if bool(action_payload.get("replanned", expected_replan)):
                replan_calls += 1
            else:
                cached_action_calls += 1

            action = action_payload["action"]
            pending_actions_remaining = int(
                action_payload.get("pending_actions_remaining", 0)
            )
            stage_start = time.perf_counter()
            with profiler.section("control_env_step"):
                obs, _, done, _ = env.step(action)
            _add_elapsed(client_seconds, "control_env_step_s", stage_start)

            stage_start = time.perf_counter()
            with profiler.section("observe_rpc"):
                observe_payload = policy.call(
                    "observe", {"obs": obs, "done": bool(done)}
                )
            _add_elapsed(client_seconds, "observe_rpc_s", stage_start)
            _accumulate_server_timing(
                server_seconds,
                server_counters,
                observe_payload.get("server_timing"),
            )
            control_steps += 1
            if done:
                break
            t += 1
    finally:
        pbar.close()

    episode_mean_psnr = float(np.mean(future_psnrs)) if future_psnrs else None
    episode_timing: dict[str, Any] = {
        "episode_index": int(episode_idx),
        "success": bool(done),
        "episode_total_s": time.perf_counter() - episode_start,
        "max_control_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
        "control_steps": int(control_steps),
        "replan_calls": int(replan_calls),
        "cached_action_calls": int(cached_action_calls),
        "client_seconds": client_seconds,
        "server_seconds": server_seconds,
        "server_counters": server_counters,
    }
    return bool(done), replay_images, episode_mean_psnr, episode_timing


def _run_task(
    *,
    suite_name: str,
    task_id: int,
    task_suite,
    policy: PolicyClient,
    cfg: DictConfig,
    run_output_dir: Path,
    gpu_id: str,
    log_file: str,
    profiler: Profiler,
) -> dict[str, Any]:
    task_start = time.perf_counter()
    task_timing: dict[str, float] = {}
    stage_start = time.perf_counter()
    cfg.EVALUATION.task_suite_name = suite_name
    cfg.EVALUATION.task_id = int(task_id)
    task = task_suite.get_task(task_id)
    initial_states = _prepare_initial_states(task_suite, task_id, int(cfg.EVALUATION.num_trials))
    render_gpu_device_id = int(cfg.EVALUATION.get("render_gpu_device_id", 0))
    _add_elapsed(task_timing, "task_setup_s", stage_start)
    logging.info(
        "Render env: CUDA_VISIBLE_DEVICES=%s MUJOCO_GL=%s PYOPENGL_PLATFORM=%s "
        "MUJOCO_EGL_DEVICE_ID=%s render_gpu_device_id=%s",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("MUJOCO_GL"),
        os.environ.get("PYOPENGL_PLATFORM"),
        os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        render_gpu_device_id,
    )

    stage_start = time.perf_counter()
    with profiler.section("env_create"):
        env, task_description = get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            cfg.get("seed"),
            render_gpu_device_id=render_gpu_device_id,
        )
    _add_elapsed(task_timing, "env_create_s", stage_start)
    save_rollout_videos = bool(cfg.EVALUATION.get("save_rollout_videos", True))
    video_dir = run_output_dir / suite_name / "videos"
    if save_rollout_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "task_suite": suite_name,
        "task_id": int(task_id),
        "task_name": getattr(task, "name", ""),
        "bddl_file": getattr(task, "bddl_file", ""),
        "task_description": task_description,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": str(gpu_id),
        "render_gpu_id": render_gpu_device_id,
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
        "log_file": log_file,
        "task_timing": task_timing,
        "episode_timings": [],
    }

    try:
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            with profiler.section("episode"):
                (
                    success,
                    replay_images,
                    episode_mean_psnr,
                    episode_timing,
                ) = _run_episode_with_server(
                    env=env,
                    policy=policy,
                    initial_state=initial_states[trial_idx],
                    task_suite_name=suite_name,
                    task_id=task_id,
                    task_description=task_description,
                    cfg=cfg,
                    episode_idx=trial_idx,
                    profiler=profiler,
                    save_rollout_videos=save_rollout_videos,
                )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
            if save_rollout_videos:
                stage_start = time.perf_counter()
                with profiler.section("rollout_video_write"):
                    save_rollout_video(
                        video_dir,
                        replay_images,
                        f"task{task_id}_trial{trial_idx}_{gpu_id}",
                        success=success,
                        task_description=task_description,
                    )
                episode_timing["rollout_video_write_s"] = (
                    time.perf_counter() - stage_start
                )
            else:
                episode_timing["rollout_video_write_s"] = 0.0
            results["episode_timings"].append(episode_timing)
            if episode_mean_psnr is not None:
                results.setdefault("episode_future_video_psnr", []).append(episode_mean_psnr)
        if "episode_future_video_psnr" in results:
            values = results["episode_future_video_psnr"]
            results["future_video_psnr_mean"] = float(np.mean(values)) if values else None
    finally:
        stage_start = time.perf_counter()
        try:
            with profiler.section("env_close"):
                env.close()
        except Exception as exc:
            logging.debug("Ignoring LIBERO env close error: %s", exc)
        _add_elapsed(task_timing, "env_close_s", stage_start)
    results["duration"] = time.perf_counter() - task_start
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    # The manager supplies the physical EGL device id for multi-GPU runs.
    # Standalone client invocation falls back to the Hydra render device.
    os.environ.setdefault(
        "MUJOCO_EGL_DEVICE_ID",
        str(int(cfg.EVALUATION.get("render_gpu_device_id", 0))),
    )
    profiler = build_eval_profiler(cfg, role="client")
    profiler.start()
    policy: Optional[PolicyClient] = None
    try:
        with profiler.section("setup/config_defaults"):
            apply_training_config_defaults_from_checkpoint(cfg)
            if cfg.get("seed") is not None:
                set_global_seed(int(cfg.seed), get_worker_init_fn=False)

        socket_path = _select_path(cfg, "PAIR.socket_path")
        task_file = _select_path(cfg, "PAIR.task_file")
        run_output_dir = Path(
            os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        gpu_id = str(OmegaConf.select(cfg, "PAIR.gpu_id", default=cfg.gpu_id))
        log_file = str(OmegaConf.select(cfg, "PAIR.log_file", default=""))

        results_jsonl_path = Path(
            os.path.expanduser(
                os.path.expandvars(
                    str(
                        cfg.EVALUATION.get("results_jsonl_path")
                        or run_output_dir / "task_results.jsonl"
                    )
                )
            )
        )
        episode_timings_path = run_output_dir / "episode_timings.jsonl"
        completed_tasks_path = run_output_dir / "completed_tasks.txt"
        failed_tasks_path = run_output_dir / "failed_tasks.txt"

        tasks = _read_task_file(task_file)
        connect_timeout_s = float(
            OmegaConf.select(cfg, "PAIR.connect_timeout_s", default=300.0)
        )
        policy = PolicyClient(
            socket_path,
            timeout_s=connect_timeout_s,
            profiler=profiler,
            client_id=str(
                OmegaConf.select(cfg, "PAIR.client_id", default=None)
                or f"{gpu_id}-{os.getpid()}"
            ),
            metadata={"worker_label": gpu_id},
        )
        with profiler.section("setup/server_connect"):
            policy.connect()
        logging.info("Connected to policy server: %s", socket_path)

        with profiler.section("setup/benchmark_registry"):
            benchmark_dict = benchmark.get_benchmark_dict()
        suite_cache = {}
        for task_index, (suite_name, task_id) in enumerate(tasks, start=1):
            try:
                if suite_name not in suite_cache:
                    with profiler.section("suite_initialize"):
                        suite_cache[suite_name] = benchmark_dict[suite_name]()
                logging.info(
                    "Client gpu=%s running %s task_id=%s (%s/%s)",
                    gpu_id,
                    suite_name,
                    task_id,
                    task_index,
                    len(tasks),
                )
                with profiler.section("task"):
                    result = _run_task(
                        suite_name=suite_name,
                        task_id=task_id,
                        task_suite=suite_cache[suite_name],
                        policy=policy,
                        cfg=cfg,
                        run_output_dir=run_output_dir,
                        gpu_id=gpu_id,
                        log_file=log_file,
                        profiler=profiler,
                    )
                _append_jsonl(results_jsonl_path, result)
                for episode_timing in result.get("episode_timings", []):
                    _append_jsonl(
                        episode_timings_path,
                        {
                            "task_suite": suite_name,
                            "task_id": int(task_id),
                            "gpu_id": gpu_id,
                            **episode_timing,
                        },
                    )
                _append_line(
                    completed_tasks_path,
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{suite_name},{task_id},gpu={gpu_id},log={log_file}",
                )
                print(
                    f"[client gpu={gpu_id}] {suite_name} task {task_id} "
                    f"completed: {result['successes']}/{result['total_episodes']} successes "
                    f"({task_index}/{len(tasks)})",
                    flush=True,
                )
            except Exception:
                _append_line(
                    failed_tasks_path,
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{suite_name},{task_id},gpu={gpu_id},rc=exception,log={log_file}",
                )
                logging.exception("Client gpu=%s failed on %s task_id=%s", gpu_id, suite_name, task_id)
                if not bool(cfg.EVALUATION.get("continue_on_task_error", False)):
                    raise
    finally:
        if policy is not None:
            policy.close()
        summary_path = profiler.finish()
        if summary_path is not None:
            logging.info("Client profiler summary: %s", summary_path)


if __name__ == "__main__":
    main()
