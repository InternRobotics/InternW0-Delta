from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import uuid
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from eval.robotwin import policy as deploy_policy
from eval.robotwin.ipc import recv_message, send_message
from eval.shared_policy_server import (
    MultiClientSessionServer,
    SerialRequestScheduler,
    SessionFactory,
    SessionRegistry,
)
from eval.profiler import Profiler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", help="Resolved configuration supplied by the launcher")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument(
        "--sim-task",
        default="robotwin",
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-hz", type=float, required=True)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-clients", type=int, default=1)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--health-path")
    parser.add_argument("--health-interval-sec", type=float, default=5.0)
    parser.add_argument("--request-timeout-sec", type=float, default=300.0)
    parser.add_argument("--server-id")
    parser.add_argument("--gpu-id")
    args = parser.parse_args()
    if not 1 <= int(args.max_clients) <= 8:
        parser.error("--max-clients must be between 1 and 8")
    return args


def _load_runtime(
    args: argparse.Namespace,
    profiler: Profiler,
) -> tuple[deploy_policy.RobotWinSharedRuntime, deploy_policy.RobotWinSessionConfig]:
    sim_cfg_path = str((deploy_policy.PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve())
    sim_task = str(args.sim_task)
    with profiler.section("override_config_compose"):
        cfg = (OmegaConf.load(args.resolved_config) if args.resolved_config
               else deploy_policy._compose_sim_cfg(sim_cfg_path, None, sim_task))
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
        cfg.model.understanding.vlm_model_path = str(Path(args.vlm_path).resolve())
        model_overrides = OmegaConf.to_container(cfg.model, resolve=True)
    with profiler.section("policy_initialize"):
        result = deploy_policy.build_robotwin_runtime_and_session_config(
            {
                "resolved_config": cfg,
                "sim_cfg_path": sim_cfg_path,
                "sim_task": sim_task,
                "ckpt_setting": str(Path(args.checkpoint).resolve()),
                "dataset_stats_path": str(Path(args.dataset_stats).resolve()),
                "action_horizon": int(args.action_horizon),
                "action_hz": float(args.action_hz),
                "replan_steps": int(args.replan_steps),
                "num_inference_steps": int(args.num_inference_steps),
                "seed": int(args.seed),
                "device": "cuda:0",
                "mixed_precision": "bf16",
                "timing_enabled": True,
                "model_overrides": model_overrides,
                "profiler": profiler,
            }
        )
        import torch
        payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        runtime, _ = result
        counts = {}
        for name in ("mot", "proprio_encoder", "action_proprio_encoder"):
            module = getattr(runtime.model, name, None)
            saved = payload.get(name, {})
            actual = module.state_dict() if module is not None else {}
            if saved.keys() != actual.keys():
                raise ValueError(f"Incomplete evaluation checkpoint component: {name}")
            if any(saved[k].shape != actual[k].shape for k in saved):
                raise ValueError(f"Evaluation checkpoint shape mismatch: {name}")
            counts[name] = len(saved)
        saved_adapter = payload.get("understanding", {})
        actual_adapter = runtime.model.understanding.adapter_state_dict()
        if saved_adapter.keys() != actual_adapter.keys():
            raise ValueError("Evaluation understanding adapter keys mismatch")
        if any(saved_adapter[k].shape != actual_adapter[k].shape for k in saved_adapter):
            raise ValueError("Evaluation understanding adapter shapes mismatch")
        counts["understanding"] = len(saved_adapter)
        print(f"MODEL_CHECKPOINT_LOADED step={payload['step']} components={counts}", flush=True)
        return result


class _RobotWinSessionFactory(SessionFactory):
    def __init__(
        self,
        runtime: deploy_policy.RobotWinSharedRuntime,
        config: deploy_policy.RobotWinSessionConfig,
        profiler: Profiler,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.profiler = profiler

    def create(
        self,
        client_id: str,
        metadata: dict[str, object],
    ) -> deploy_policy.RobotWinPolicySession:
        print(
            "MODEL_SESSION_CREATE "
            f"client_id={client_id} task={metadata.get('task_or_split_label')} "
            f"phase={metadata.get('phase')}",
            flush=True,
        )
        # Session construction contains no model/profiler work. All methods
        # that use the shared profiler run only on the dispatcher thread.
        return deploy_policy.RobotWinPolicySession(
            runtime=self.runtime,
            config=self.config,
            profiler=self.profiler,
        )


def _build_profiler() -> Profiler:
    default_dir = Path("/tmp") / f"wam-robotwin-server-profile-{os.getpid()}"
    profiler = Profiler.from_env(
        Path(os.environ.get("WAM_PROFILE_DIR", str(default_dir))),
        rank=0,
        local_rank=0,
        world_size=1,
    )
    profiler.start()
    return profiler


def _close_listener(listener: socket.socket | None) -> None:
    if listener is None:
        return
    try:
        listener.close()
    except OSError:
        pass


def main() -> None:
    args = _parse_args()
    profiler = _build_profiler()
    socket_path = Path(args.socket)
    if len(str(socket_path).encode()) >= 100:
        raise ValueError(f"Unix socket path is too long: {socket_path}")
    server_id = str(args.server_id or f"robotwin-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    listener: socket.socket | None = None
    shared_server: MultiClientSessionServer | None = None
    try:
        with profiler.section("setup/socket_path"):
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            if socket_path.exists() or socket_path.is_socket():
                socket_path.unlink()

        print("MODEL_SERVER_LOADING", flush=True)
        with profiler.section("setup/model_load"):
            runtime, session_config = _load_runtime(args, profiler)
        print(
            "MODEL_SERVER_CONFIG "
            f"server_id={server_id} persistent={bool(args.persistent)} "
            f"max_clients={args.max_clients} model_loads=1 "
            f"action_horizon={session_config.action_horizon} "
            f"action_hz={session_config.action_hz} "
            f"replan_steps={session_config.replan_steps} "
            f"action_valid_dims={int((~runtime.action_dim_is_pad).sum().item())} "
            f"action_total_dims={int(runtime.action_dim_is_pad.numel())} "
            f"video_size={session_config.video_size[0]}x{session_config.video_size[1]} "
            f"layout={session_config.video_layout} "
            f"views={session_config.video_view_names}",
            flush=True,
        )
        with profiler.section("setup/model_validate"):
            if session_config.action_horizon != int(args.action_horizon):
                raise RuntimeError(
                    f"Model action horizon drift: expected {args.action_horizon}, "
                    f"got {session_config.action_horizon}"
                )
            if session_config.replan_steps != int(args.replan_steps):
                raise RuntimeError(
                    f"Model replan cadence drift: expected {args.replan_steps}, "
                    f"got {session_config.replan_steps}"
                )
            if session_config.action_hz != float(args.action_hz):
                raise RuntimeError(
                    f"Model action_hz drift: expected {args.action_hz}, "
                    f"got {session_config.action_hz}"
                )

        with profiler.section("setup/listener"):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(socket_path))
            listener.listen(int(args.max_clients))

        factory = _RobotWinSessionFactory(runtime, session_config, profiler)
        registry = SessionRegistry(factory, max_sessions=int(args.max_clients))
        scheduler = SerialRequestScheduler(
            max_pending=int(args.max_clients),
            request_timeout_sec=float(args.request_timeout_sec),
        )
        shared_server = MultiClientSessionServer(
            listener=listener,
            recv_message=recv_message,
            send_message=send_message,
            registry=registry,
            scheduler=scheduler,
            server_id=server_id,
            persistent=bool(args.persistent),
            health_path=None if args.health_path is None else Path(args.health_path),
            health_interval_sec=float(args.health_interval_sec),
            server_metadata={
                "benchmark": "robotwin",
                "gpu_id": args.gpu_id,
                "socket": str(socket_path),
            },
        )

        def request_stop(_signum: int, _frame: Any) -> None:
            assert shared_server is not None
            shared_server.request_stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        print(
            f"MODEL_SERVER_READY socket={socket_path} server_id={server_id} "
            f"max_clients={args.max_clients}",
            flush=True,
        )
        shared_server.serve()
        if scheduler.fatal_error is not None:
            raise RuntimeError(f"MODEL_SERVER_FATAL {scheduler.fatal_error}")
    finally:
        if shared_server is not None:
            shared_server.request_stop()
            print(
                "MODEL_SERVER_METRICS "
                + json.dumps(shared_server.health_snapshot(), sort_keys=True),
                flush=True,
            )
        _close_listener(listener)
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
        summary_path = profiler.finish()
        if summary_path is not None:
            print(f"MODEL_SERVER_PROFILE {summary_path}", flush=True)
        print("MODEL_SERVER_STOPPED", flush=True)


if __name__ == "__main__":
    main()
