from __future__ import annotations

import argparse
import builtins
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

from eval.robotwin.ipc import UnixModelClient
from eval.profiler import Profiler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = Path(
    os.path.expanduser(
        os.path.expandvars(
            os.environ.get(
                "ROBOTWIN_ROOT",
                str(PROJECT_ROOT / "third_party" / "RoboTwin"),
            )
        )
    )
).resolve()
# Keep the official simulator, task definitions and assets in ROBOTWIN_ROOT,
# but use the repository-owned resilient evaluator adapter. The pinned
# 13c3c47 upstream evaluator hardcodes 100 episodes and its own output path;
# the adapter adds exact episode counts, explicit output/video locations,
# heartbeat timeouts and resumable progress without editing ROBOTWIN_ROOT.
ROBUST_EVAL_PATH = Path(__file__).with_name("rollout.py")


def _model_observation(observation: dict[str, Any] | None):
    """Keep only the arrays consumed by the InternW0-delta RoboTwin policy.

    RoboTwin observations also contain depth/point-cloud payloads. Sending
    those over the local socket adds copies without changing model inputs.
    """
    if observation is None:
        return None
    observation_data = observation["observation"]
    return {
        "observation": {
            camera: {"rgb": observation_data[camera]["rgb"]}
            for camera in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {"vector": observation["joint_action"]["vector"]},
    }


def _load_robust_eval_module():
    spec = importlib.util.spec_from_file_location("wam_robust_eval_bridge", ROBUST_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load robust eval module from {ROBUST_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _suppress_official_step_progress() -> None:
    """Drop RoboTwin's per-action terminal progress line in batch jobs.

    The official Base_Task.take_action implementation prints one carriage-
    return line for every policy action.  With many non-interactive workers,
    those lines become shared-filesystem log traffic without carrying useful
    evaluation state.  Keep every other official message unchanged.
    """
    if os.environ.get("WAM_RW_QUIET_STEP_PROGRESS", "1") != "1":
        return

    import envs._base_task as base_task_module

    def quiet_print(*values: Any, **kwargs: Any) -> None:
        if values and isinstance(values[0], str) and values[0].startswith("step: \033[92m"):
            return
        builtins.print(*values, **kwargs)

    base_task_module.print = quiet_print


def _build_profiler() -> Profiler:
    default_dir = Path("/tmp") / f"wam-robotwin-client-profile-{os.getpid()}"
    profiler = Profiler.from_env(
        Path(os.environ.get("WAM_PROFILE_DIR", str(default_dir))),
        rank=0,
        local_rank=0,
        world_size=1,
    )
    # The simulator process renders through SAPIEN but does not run the InternW0-delta
    # model.  CUDA synchronization/memory queries here would create noise and
    # may initialize an unnecessary PyTorch CUDA context.
    profiler.sync_cuda = False
    profiler.track_cuda_memory = False
    profiler.torch_profile = False
    profiler.start()
    return profiler


def main() -> None:
    # Concurrent CuRobo planners must not overwrite each other's Warp kernels.
    # Only job-owned scratch is writable; never the shared simulator/env tree.
    scratch = os.environ.get("WAM_RW_CLIENT_CACHE_ROOT")
    if scratch:
        client_scratch = Path(scratch) / f"client-{os.getpid()}"
        client_scratch.mkdir(parents=True, exist_ok=True)
        os.environ["WARP_CACHE_PATH"] = str(client_scratch / "warp")
        os.environ["XDG_CACHE_HOME"] = str(client_scratch / "xdg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id")
    parser.add_argument("--task-label")
    parser.add_argument("--phase")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    cli_args = parser.parse_args()
    profiler = _build_profiler()
    model_client: UnixModelClient | None = None
    try:
        with profiler.section("setup/load_official_evaluator"):
            robust = _load_robust_eval_module()
        with profiler.section("setup/simulator_runtime"):
            _suppress_official_step_progress()
            import sapien

            print(
                f"OFFICIAL_SAPIEN_DEVICE CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
                f"{sapien.render.get_device_summary()}",
                flush=True,
            )

        # The current official RoboTwin evaluator records only the head camera.
        # Keep its exact frame geometry even though the InternW0-delta model consumes all
        # three cameras through the observation payload.
        def official_eval_video_size(args):
            camera = robust.get_camera_config(args["camera"]["head_camera_type"])
            return f"{int(camera['w'])}x{int(camera['h'])}"

        robust.get_eval_video_size = official_eval_video_size
        original_argv = sys.argv
        try:
            with profiler.section("setup/parse_eval_config"):
                sys.argv = [original_argv[0], "--config", cli_args.config]
                if cli_args.overrides:
                    sys.argv.extend(["--overrides", *cli_args.overrides])
                usr_args = robust.parse_args_and_config()
        finally:
            sys.argv = original_argv
        action_horizon = int(usr_args["action_horizon"])
        replan_steps = int(usr_args["replan_steps"])
        expected_action_horizon = int(os.environ.get("WAM_RW_ACTION_HORIZON", "32"))
        expected_replan_steps = int(os.environ.get("WAM_RW_REPLAN_STEPS", "10"))
        if (
            action_horizon != expected_action_horizon
            or replan_steps != expected_replan_steps
        ):
            raise ValueError(
                "Unexpected action/replan cadence: "
                f"expected {expected_action_horizon}/{expected_replan_steps}, "
                f"got {action_horizon}/{replan_steps}"
            )

        def bridge_get_model(_usr_args: dict[str, Any]):
            nonlocal model_client
            model_client = UnixModelClient(
                cli_args.socket,
                profiler=profiler,
                client_id=cli_args.client_id,
                metadata={
                    "task_or_split_label": cli_args.task_label or usr_args.get("task_name"),
                    "phase": cli_args.phase or usr_args.get("task_config"),
                },
            )
            print(
                "EVAL_CLIENT_SESSION "
                f"client_id={model_client.client_id} server_id={model_client.server_id} "
                f"session_id={model_client.session_id}",
                flush=True,
            )
            return model_client

        def filtered_observation(model: UnixModelClient, observation):
            started = time.perf_counter()
            with profiler.section("rollout/observation_filter"):
                filtered = _model_observation(observation)
            model.add_client_timing(
                "observation_filter_s",
                time.perf_counter() - started,
                count_name="observation_filter_calls",
            )
            return filtered

        def bridge_eval(task_env, model: UnixModelClient, observation):
            with profiler.section("rollout/request_prepare"):
                request = {
                    "observation": filtered_observation(model, observation),
                    "instruction": task_env.get_instruction(),
                }
            with profiler.section("rollout/get_action_rpc"):
                actions = model.call("get_action", request)
            for action in actions:
                started = time.perf_counter()
                with profiler.section("rollout/take_action"):
                    task_env.take_action(action, action_type="qpos")
                action_elapsed = time.perf_counter() - started
                model.sim_seconds += action_elapsed
                model.add_client_timing(
                    "take_action_s",
                    action_elapsed,
                    count_name="action_steps",
                )
                step_count = int(task_env.take_action_cnt)
                capture_for_video = task_env.eval_video_path is not None
                capture_for_recent_memory = (
                    step_count % replan_steps == (-action_horizon) % replan_steps
                )
                next_observation = None
                if capture_for_video or capture_for_recent_memory:
                    started = time.perf_counter()
                    with profiler.section("rollout/get_obs"):
                        next_observation = task_env.get_obs()
                    model.add_client_timing(
                        "get_obs_s",
                        time.perf_counter() - started,
                        count_name="get_obs_calls",
                    )
                update_payload = filtered_observation(model, next_observation)
                with profiler.section("rollout/update_obs_rpc"):
                    model.call("update_obs", update_payload)
                if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                    break

        def bridge_reset(model: UnixModelClient):
            with profiler.section("rollout/reset_model_rpc"):
                model.reset_model()

        def bridge_decorator(_policy_name: str, method_name: str):
            methods = {
                "get_model": bridge_get_model,
                "eval": bridge_eval,
                "reset_model": bridge_reset,
            }
            return methods[method_name]

        robust.eval_function_decorator = bridge_decorator
        with profiler.section("evaluation/run"):
            robust.main(usr_args)
    finally:
        if model_client is not None:
            model_client.close()
        summary_path = profiler.finish()
        if summary_path is not None:
            print(f"EVAL_CLIENT_PROFILE {summary_path}", flush=True)


if __name__ == "__main__":
    main()
