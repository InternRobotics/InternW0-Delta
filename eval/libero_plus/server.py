from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ.setdefault("WAM_LIBERO_REPO", "LIBERO-plus")
libero_plus_root = project_root / "third_party" / "LIBERO-plus"
if libero_plus_root.exists() and str(libero_plus_root) not in sys.path:
    sys.path.insert(0, str(libero_plus_root))

from eval.libero.eval_config import (
    apply_eval_understanding_interval,
    apply_training_config_defaults_from_checkpoint,
    resolve_eval_video_metadata,
)
from eval.libero.eval_libero_single import (
    WAMProcessor,
    _static_dimension_is_pad,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    load_dataset_stats_from_json,
    set_global_seed,
)
from eval.libero_plus.online_policy import OnlinePolicySession
from eval.libero_plus.profiling import build_eval_profiler
from eval.shared_policy_server import (
    MultiClientSessionServer,
    SerialRequestScheduler,
    SessionError,
    SessionFactory,
    SessionRegistry,
)
from eval.profiler import Profiler


def _socket_path(cfg: DictConfig) -> Path:
    value = OmegaConf.select(cfg, "PAIR.socket_path")
    if value is None:
        raise ValueError("PAIR.socket_path must be set for server.py.")
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


@dataclass(frozen=True)
class LiberoSharedRuntime:
    """One immutable model runtime shared by all live LIBERO sessions."""

    model: Any
    processor: WAMProcessor
    cfg: DictConfig
    action_horizon: int
    input_w: int
    input_h: int
    model_device: str
    initial_cpu_rng_state: torch.Tensor


def _build_runtime(cfg: DictConfig, profiler: Profiler) -> LiberoSharedRuntime:
    with profiler.section("setup/config_defaults"):
        apply_training_config_defaults_from_checkpoint(cfg)
        apply_eval_understanding_interval(cfg)

    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "Pair client/server eval does not support EVALUATION.visualize_future_video=true."
        )
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_load_start = time.perf_counter()
    logging.info(
        "Initializing InternW0-delta components: model_id=%s skip_dit_pretrain=%s "
        "load_text_encoder=%s device=%s dtype=%s",
        cfg.model.get("model_id"),
        bool(cfg.model.get("skip_dit_load_from_pretrain", False)),
        bool(cfg.model.get("load_text_encoder", True)),
        model_device,
        model_dtype,
    )
    with profiler.section("setup/model_initialize"):
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    model_paths = getattr(model, "model_paths", {})
    logging.info(
        "Initialized InternW0-delta components in %.2fs: video_dit=%s action_dit=%s vae=%s "
        "text_encoder=%s",
        time.perf_counter() - model_load_start,
        model_paths.get("video_dit", "unknown"),
        model_paths.get("action_dit_backbone", "unknown"),
        model_paths.get("vae", "unknown"),
        model_paths.get("text_encoder", "disabled"),
    )
    logging.info("Loading trained InternW0-delta checkpoint: %s", cfg.ckpt)
    checkpoint_load_start = time.perf_counter()
    with profiler.section("setup/checkpoint_load"):
        _load_model_checkpoint(model, str(cfg.ckpt))
        model = model.to(model_device).eval()
    logging.info(
        "Loaded trained InternW0-delta checkpoint in %.2fs: %s",
        time.perf_counter() - checkpoint_load_start,
        cfg.ckpt,
    )

    processor_start = time.perf_counter()
    with profiler.section("setup/processor_initialize"):
        dataset_stats_path = _resolve_dataset_stats_path(cfg)
        OmegaConf.update(
            cfg,
            "EVALUATION.dataset_stats_path",
            str(dataset_stats_path),
            merge=False,
            force_add=True,
        )
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        processor: WAMProcessor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(dataset_stats)
    logging.info(
        "Initialized processor and dataset stats in %.2fs: %s",
        time.perf_counter() - processor_start,
        dataset_stats_path,
    )

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    if action_horizon <= 0:
        raise ValueError(
            f"EVALUATION.action_horizon must be positive, got {action_horizon}"
        )
    video_size = cfg.data.train.get("video_size", [224, 224])
    return LiberoSharedRuntime(
        model=model,
        processor=processor,
        cfg=cfg,
        action_horizon=action_horizon,
        input_h=int(video_size[0]),
        input_w=int(video_size[1]),
        model_device=str(model_device),
        initial_cpu_rng_state=torch.random.get_rng_state().clone(),
    )


def _write_resolved_contract(
    cfg: DictConfig,
    runtime: LiberoSharedRuntime,
    socket_path: Path,
) -> Path:
    """Persist the same right80 inference contract as the dedicated server."""
    output_dir = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
    )
    contract_dir = output_dir / "resolved_contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    action_pad = _static_dimension_is_pad(runtime.processor, "action")
    state_pad = _static_dimension_is_pad(runtime.processor, "state")
    replan_steps = int(
        cfg.EVALUATION.get("replan_steps") or 10
    )
    video_layout, video_names, vlm_names = resolve_eval_video_metadata(cfg, runtime.processor)
    contract = {
        "vae_layout": video_layout,
        "vae_view_names": video_names,
        "vlm_view_names": vlm_names,
        "checkpoint": str(cfg.ckpt),
        "dataset_stats_path": str(cfg.EVALUATION.dataset_stats_path),
        "action_horizon": int(runtime.action_horizon),
        "replan_steps": replan_steps,
        "action_hz": float(cfg.EVALUATION.action_hz),
        "memory_recent_frame_offset": int(
            cfg.data.train.memory_recent_frame_offset
        ),
        "num_frames": int(cfg.data.train.num_frames),
        "action_dim": int(runtime.model.action_expert.action_dim),
        "proprio_dim": int(runtime.model.proprio_dim),
        "action_valid_indices": (~action_pad).nonzero().flatten().tolist(),
        "state_valid_indices": (~state_pad).nonzero().flatten().tolist(),
        "invalid_action_diffusion": "masked_each_step",
        "recent_validity_source": "executed_model_action_steps",
        "future_delta_num_tokens": int(runtime.model.future_delta_num_tokens),
        "future_delta_num_frames": int(runtime.model.future_delta_num_frames),
        "future_delta_action_video_freq_ratio": int(
            runtime.model.future_delta_action_video_freq_ratio
        ),
        "processor": OmegaConf.to_container(
            cfg.data.train.processor, resolve=True
        ),
    }
    output_path = contract_dir / f"{socket_path.stem}.json"
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    logging.info(
        "Resolved evaluation contract: %s",
        json.dumps(contract, sort_keys=True),
    )
    return output_path


class _LiberoPolicySessionAdapter:
    def __init__(self, session: OnlinePolicySession) -> None:
        self.session = session

    @staticmethod
    def _payload_dict(command: str, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise SessionError(f"{command} payload must be a dict")
        return payload

    def handle(self, command: str, payload: object) -> object:
        if command == "ping":
            return "pong"
        if command == "reset_session":
            values = self._payload_dict(command, payload)
            self.session.reset(
                task_suite_name=str(values["task_suite_name"]),
                task_id=int(values["task_id"]),
                task_description=str(values["task_description"]),
            )
            return {"status": "reset"}
        if command == "act":
            values = self._payload_dict(command, payload)
            return self.session.act(
                values.get("obs"),
                timestep=int(values["timestep"]),
            )
        if command == "observe":
            values = self._payload_dict(command, payload)
            observe_timing = self.session.observe(
                values["obs"],
                done=bool(values.get("done", False)),
                anchor_only=bool(values.get("anchor_only", False)),
            )
            return {"status": "observed", "server_timing": observe_timing}
        raise SessionError(f"Unknown LIBERO policy command: {command}")

    def close(self) -> None:
        self.session.close()


class _LiberoSessionFactory(SessionFactory):
    def __init__(self, runtime: LiberoSharedRuntime, profiler: Profiler) -> None:
        self.runtime = runtime
        self.profiler = profiler

    def create(
        self,
        client_id: str,
        metadata: dict[str, object],
    ) -> _LiberoPolicySessionAdapter:
        logging.info(
            "LIBERO_SESSION_CREATE client_id=%s worker=%s",
            client_id,
            metadata.get("worker_label"),
        )
        # OmegaConf is mutable and OnlinePolicySession writes task fields.
        # Every connection therefore gets a deep config copy while model and
        # processor remain shared.
        session_cfg = OmegaConf.create(
            OmegaConf.to_container(self.runtime.cfg, resolve=False)
        )
        session = OnlinePolicySession(
            model=self.runtime.model,
            processor=self.runtime.processor,
            cfg=session_cfg,
            action_horizon=self.runtime.action_horizon,
            input_w=self.runtime.input_w,
            input_h=self.runtime.input_h,
            model_device=self.runtime.model_device,
            profiler=self.profiler,
            torch_rng_state=self.runtime.initial_cpu_rng_state,
        )
        return _LiberoPolicySessionAdapter(session)


class _MultiprocessingListenerAdapter:
    """Expose multiprocessing.Listener through the common listener contract."""

    def __init__(self, listener: Listener) -> None:
        self.listener = listener

    def settimeout(self, timeout: float) -> None:
        socket_listener = getattr(self.listener, "_listener", None)
        listener_socket = getattr(socket_listener, "_socket", None)
        if listener_socket is None:
            raise RuntimeError("Unable to configure multiprocessing listener timeout")
        listener_socket.settimeout(float(timeout))

    def accept(self):
        return self.listener.accept(), None

    def close(self) -> None:
        self.listener.close()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    profiler = build_eval_profiler(cfg, role="server")
    profiler.start()
    socket_path = _socket_path(cfg)
    if len(str(socket_path).encode()) >= 100:
        raise ValueError(f"Unix socket path is too long: {socket_path}")
    max_clients = int(OmegaConf.select(cfg, "PAIR.max_clients", default=1))
    if not 1 <= max_clients <= 4:
        raise ValueError(f"PAIR.max_clients must be between 1 and 4, got {max_clients}")
    persistent = bool(OmegaConf.select(cfg, "PAIR.persistent", default=False))
    request_timeout_s = float(
        OmegaConf.select(cfg, "PAIR.request_timeout_s", default=300.0)
    )
    health_value = OmegaConf.select(cfg, "PAIR.health_path", default=None)
    health_path = None if health_value in (None, "") else Path(str(health_value))
    server_id_value = OmegaConf.select(cfg, "PAIR.server_id", default=None)
    server_id = str(server_id_value or f"libero-{os.getpid()}-{uuid.uuid4().hex[:8]}")

    raw_listener: Listener | None = None
    shared_server: MultiClientSessionServer | None = None
    try:
        with profiler.section("setup/socket"):
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            socket_path.unlink(missing_ok=True)

        logging.info("LIBERO_MODEL_SERVER_LOADING server_id=%s", server_id)
        with profiler.section("setup/model_load"):
            runtime = _build_runtime(cfg, profiler)
        _write_resolved_contract(cfg, runtime, socket_path)
        logging.info(
            "LIBERO_MODEL_SERVER_CONFIG server_id=%s persistent=%s max_clients=%s "
            "model_loads=1 action_horizon=%s replan_steps=%s",
            server_id,
            persistent,
            max_clients,
            runtime.action_horizon,
            int(cfg.EVALUATION.get("replan_steps") or 10),
        )
        with profiler.section("setup/listener"):
            raw_listener = Listener(str(socket_path), family="AF_UNIX")
            listener = _MultiprocessingListenerAdapter(raw_listener)

        factory = _LiberoSessionFactory(runtime, profiler)
        registry = SessionRegistry(factory, max_sessions=max_clients)
        scheduler = SerialRequestScheduler(
            max_pending=max_clients,
            request_timeout_sec=request_timeout_s,
        )
        shared_server = MultiClientSessionServer(
            listener=listener,
            recv_message=lambda connection: connection.recv(),
            send_message=lambda connection, payload: connection.send(payload),
            registry=registry,
            scheduler=scheduler,
            server_id=server_id,
            persistent=persistent,
            health_path=health_path,
            health_interval_sec=float(
                OmegaConf.select(cfg, "PAIR.health_interval_s", default=5.0)
            ),
            server_metadata={
                "benchmark": "libero_plus",
                "gpu_id": str(OmegaConf.select(cfg, "PAIR.gpu_id", default=cfg.gpu_id)),
                "socket": str(socket_path),
                "model_loads": 1,
            },
        )

        def request_stop(_signum: int, _frame: Any) -> None:
            assert shared_server is not None
            shared_server.request_stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        logging.info(
            "LIBERO_MODEL_SERVER_READY socket=%s server_id=%s max_clients=%s",
            socket_path,
            server_id,
            max_clients,
        )
        shared_server.serve()
        if scheduler.fatal_error is not None:
            raise RuntimeError(f"LIBERO_MODEL_SERVER_FATAL {scheduler.fatal_error}")
    finally:
        if shared_server is not None:
            shared_server.request_stop()
            logging.info(
                "LIBERO_MODEL_SERVER_METRICS %s",
                json.dumps(shared_server.health_snapshot(), sort_keys=True),
            )
        if raw_listener is not None:
            try:
                raw_listener.close()
            except OSError:
                pass
        socket_path.unlink(missing_ok=True)
        summary_path = profiler.finish()
        if summary_path is not None:
            logging.info("Server profiler summary: %s", summary_path)
        logging.info("LIBERO_MODEL_SERVER_STOPPED server_id=%s", server_id)


if __name__ == "__main__":
    main()
