"""Profiler construction shared by LIBERO-plus client/server workers."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eval.profiler import Profiler


def build_eval_profiler(cfg: DictConfig, *, role: str) -> Profiler:
    """Build one rank-safe profiler for an eval worker role.

    Client workers deliberately disable CUDA synchronization and memory
    counters: rendering is outside PyTorch, and touching PyTorch CUDA there
    would create an unnecessary CUDA context alongside the model server.
    """
    if role not in {"client", "server"}:
        raise ValueError(f"Unsupported profiler role: {role!r}")
    output_dir = Path(
        os.path.expanduser(
            os.path.expandvars(str(cfg.EVALUATION.output_dir))
        )
    )
    rank = int(OmegaConf.select(cfg, "PAIR.profiler_rank", default=0))
    world_size = int(
        OmegaConf.select(cfg, "PAIR.profiler_world_size", default=1)
    )
    profiler = Profiler.from_env(
        output_dir / "profiler" / role,
        rank=rank,
        local_rank=0,
        world_size=max(world_size, 1),
    )
    if role == "client":
        profiler.sync_cuda = False
        profiler.track_cuda_memory = False
        profiler.torch_profile = False
    return profiler
