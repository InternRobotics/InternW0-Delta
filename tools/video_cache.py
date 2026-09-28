#!/usr/bin/env python
"""Offline VAE/VLM artifact entry point.

Preferred usage is the normal training command with
``video_latent_cache=generate``.  All generation, validation and publication
logic belongs to ``CacheManager``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hydra
from omegaconf import DictConfig

from wam.cache import CacheManager, CacheMode
from wam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


def run_precompute(cfg: DictConfig):
    manager = CacheManager.from_config(cfg)
    if manager.mode is not CacheMode.GENERATE:
        raise ValueError(
            "Encoder cache precompute requires video_latent_cache=generate."
        )
    return manager.generate()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_precompute(cfg)


if __name__ == "__main__":
    main()
