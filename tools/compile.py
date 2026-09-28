#!/usr/bin/env python
"""Populate persistent MoT forward/backward compiler artifacts using real batches."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hydra
from omegaconf import DictConfig, open_dict
from wam.runtime import run_training
from wam.utils.config_resolvers import register_default_resolvers
from scripts.train import _strip_launcher_rank_args

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if str(cfg.model.get("mot_compile_mode", "off")) == "off":
        raise ValueError(
            "Select +infra=compile or +infra=cache to prepare compiler artifacts."
        )
    if cfg.video_latent_cache == "generate":
        raise ValueError(
            "Generate encoder artifacts separately before preparing MoT kernels."
        )
    steps = int(cfg.get("precompute", {}).get("compile_steps", 2))
    if steps < 2:
        raise ValueError(
            "precompute.compile_steps must be at least 2 for forward/backward replay."
        )
    with open_dict(cfg):
        cfg.max_steps = steps
        cfg.eval_every = 0
        cfg.save_every = 0
        cfg.save_on_train_end = False
        cfg.save_training_state = False
        cfg.wandb.enabled = False
    run_training(cfg)


if __name__ == "__main__":
    sys.argv = _strip_launcher_rank_args(sys.argv)
    main()
