#!/usr/bin/env python
"""Precompute LeRobot HF parquet cache and InternW0-delta normalization stats.

Example:
  python tools/posttrain_stats.py task=libero \
    +precompute.stats_path=.cache/internw0/stats/libero.json
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import logging
import os
import threading
import time
from contextlib import contextmanager
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Iterable

import hydra
import datasets as hf_datasets
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from wam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from wam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from wam.utils.config_resolvers import register_default_resolvers
from wam.utils.logging_config import setup_logging, get_logger

register_default_resolvers()
logger = get_logger(__name__)


@contextmanager
def _heartbeat(stage: str, interval_s: int = 0):
    interval_s = int(interval_s)
    stop = threading.Event()
    thread = None

    def run() -> None:
        while not stop.wait(interval_s):
            logger.info("[heartbeat] %s still running", stage)

    logger.info("[stage:start] %s", stage)
    if interval_s > 0:
        thread = threading.Thread(target=run, name=f"heartbeat:{stage}", daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        logger.info("[stage:done] %s", stage)


def _cfg_get(cfg: DictConfig | dict | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    return cfg.get(key, default)


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve()


def _default_stats_path(train_cfg: DictConfig) -> Path:
    configured = train_cfg.get("pretrained_norm_stats")
    if configured:
        return _as_path(configured)
    dataset_dirs = OmegaConf.to_container(train_cfg.dataset_dirs, resolve=True)
    if not dataset_dirs:
        raise ValueError("data.train.dataset_dirs is empty; cannot choose default stats path.")
    return _as_path(Path(dataset_dirs[0]) / "dataset_stats.json")


def _setup_hf_cache(precompute_cfg: DictConfig | None) -> None:
    cache_dir = _cfg_get(precompute_cfg, "hf_cache_dir")
    tmp_dir = _cfg_get(precompute_cfg, "tmp_dir")

    if cache_dir:
        # WAM_HF_DATASETS_CACHE_DIR is rank-sharded by the LeRobot wrapper.
        # For precompute and reuse, we want the standard shared HF_DATASETS_CACHE path.
        old_rank_cache = os.environ.pop("WAM_HF_DATASETS_CACHE_DIR", None)
        if old_rank_cache:
            logger.warning("Unset WAM_HF_DATASETS_CACHE_DIR=%s for shared cache precompute.", old_rank_cache)
        cache_path = _as_path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache_path)
        hf_home = cache_path.parent
        if hf_home.name == "datasets":
            hf_home = hf_home.parent
        os.environ.setdefault("HF_HOME", str(hf_home))
        logger.info("Using HF_DATASETS_CACHE=%s", os.environ["HF_DATASETS_CACHE"])
    else:
        logger.info("Using existing HF cache env. HF_DATASETS_CACHE=%s", os.environ.get("HF_DATASETS_CACHE"))
        if os.environ.get("WAM_HF_DATASETS_CACHE_DIR"):
            logger.warning(
                "WAM_HF_DATASETS_CACHE_DIR is set; cache will be rank-sharded by InternW0-delta. "
                "Unset it or pass +precompute.hf_cache_dir=... for a shared cache."
            )

    if tmp_dir:
        tmp_path = _as_path(tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_path)
        logger.info("Using TMPDIR=%s", os.environ["TMPDIR"])


def _build_base_dataset(ds_cfg: DictConfig) -> BaseLerobotDataset:
    shape_meta = OmegaConf.to_container(ds_cfg.shape_meta, resolve=True)
    dataset_dirs = OmegaConf.to_container(ds_cfg.dataset_dirs, resolve=True)
    return BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        obs_size=int(ds_cfg.num_frames),
        action_size=int(ds_cfg.num_frames) - 1,
        val_set_proportion=float(ds_cfg.get("val_set_proportion", 0.05)),
        is_training_set=bool(ds_cfg.get("is_training_set", False)),
        global_sample_stride=int(ds_cfg.get("global_sample_stride", 1)),
        load_episode_stats=bool(ds_cfg.get("load_episode_stats", True)),
        episode_selection=OmegaConf.to_container(
            ds_cfg.get("episode_selection"), resolve=True
        )
        if ds_cfg.get("episode_selection") is not None
        else None,
    )


class _FieldAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.global_count = 0
        self.stepwise_min = None
        self.stepwise_max = None
        self.stepwise_q01 = None
        self.stepwise_q99 = None
        self.sum_step_mean = None
        self.sum_step_second = None
        self.sum_global_mean = None
        self.sum_global_second = None

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().cpu().float()
        ep_min = x.amin(0)
        ep_max = x.amax(0)
        ep_mean = x.mean(0)
        ep_var = x.var(0, unbiased=False)
        ep_q01 = torch.quantile(x, 0.01, dim=0, keepdim=False)
        ep_q99 = torch.quantile(x, 0.99, dim=0, keepdim=False)
        ep_second = ep_var + ep_mean.square()

        if self.n == 0:
            self.stepwise_min = ep_min.clone()
            self.stepwise_max = ep_max.clone()
            self.stepwise_q01 = ep_q01.clone()
            self.stepwise_q99 = ep_q99.clone()
            self.sum_step_mean = ep_mean.clone()
            self.sum_step_second = ep_second.clone()
            self.sum_global_mean = ep_mean.sum(0)
            self.sum_global_second = ep_second.sum(0)
        else:
            self.stepwise_min = torch.minimum(self.stepwise_min, ep_min)
            self.stepwise_max = torch.maximum(self.stepwise_max, ep_max)
            self.stepwise_q01 = torch.minimum(self.stepwise_q01, ep_q01)
            self.stepwise_q99 = torch.maximum(self.stepwise_q99, ep_q99)
            self.sum_step_mean += ep_mean
            self.sum_step_second += ep_second
            self.sum_global_mean += ep_mean.sum(0)
            self.sum_global_second += ep_second.sum(0)

        self.n += 1
        self.global_count += ep_mean.shape[0]

    def finalize(self) -> dict[str, torch.Tensor]:
        if self.n == 0:
            raise RuntimeError("Cannot finalize empty stats accumulator.")
        stepwise_mean = self.sum_step_mean / self.n
        stepwise_second = self.sum_step_second / self.n
        stepwise_var = (stepwise_second - stepwise_mean.square()).clamp_min(0.0)

        global_mean = self.sum_global_mean / self.global_count
        global_second = self.sum_global_second / self.global_count
        global_var = (global_second - global_mean.square()).clamp_min(0.0)

        return {
            "stepwise_min": self.stepwise_min,
            "stepwise_max": self.stepwise_max,
            "global_min": self.stepwise_min.amin(0),
            "global_max": self.stepwise_max.amax(0),
            "stepwise_q01": self.stepwise_q01,
            "stepwise_q99": self.stepwise_q99,
            "global_q01": self.stepwise_q01.amin(0),
            "global_q99": self.stepwise_q99.amax(0),
            "stepwise_mean": stepwise_mean,
            "stepwise_std": stepwise_var.sqrt(),
            "global_mean": global_mean,
            "global_std": global_var.sqrt(),
        }


def _process_episode(dataset: BaseLerobotDataset, processor: Any, episode_idx: int) -> dict[str, dict[str, torch.Tensor]]:
    batch = dataset._get_episode_data(episode_idx)
    batch = processor.action_state_transform(batch)
    return batch


def _iter_episode_batches(
    dataset: BaseLerobotDataset,
    processor: Any,
    episode_indices: list[int],
    num_workers: int,
    max_in_flight: int,
) -> Iterable[dict[str, dict[str, torch.Tensor]]]:
    if num_workers <= 1:
        for episode_idx in episode_indices:
            yield _process_episode(dataset, processor, episode_idx)
        return

    max_in_flight = max(max_in_flight, num_workers)
    pending = {}
    index_iter = iter(episode_indices)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        def submit_more() -> None:
            while len(pending) < max_in_flight:
                try:
                    idx = next(index_iter)
                except StopIteration:
                    break
                pending[executor.submit(_process_episode, dataset, processor, idx)] = idx

        submit_more()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                idx = pending.pop(future)
                try:
                    yield future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed while processing episode {idx}") from exc
            submit_more()


def compute_wam_stats(
    dataset: BaseLerobotDataset,
    processor: Any,
    num_workers: int = 1,
    max_in_flight: int = 16,
    max_episodes: int = 0,
    episode_offset: int = 0,
) -> dict[str, Any]:
    episodes_num = int(dataset.multi_dataset.num_episodes)
    start = max(0, int(episode_offset))
    if start >= episodes_num:
        raise ValueError(f"episode_offset={start} is past num_episodes={episodes_num}")
    episode_indices = list(range(start, episodes_num))
    if max_episodes and max_episodes > 0:
        episode_indices = episode_indices[: min(max_episodes, len(episode_indices))]
    if start or (max_episodes and max_episodes > 0):
        logger.warning(
            "Computing stats on episodes [%d, %d) of %d.",
            episode_indices[0],
            episode_indices[-1] + 1,
            episodes_num,
        )

    state_acc = defaultdict(_FieldAccumulator)
    action_acc = defaultdict(_FieldAccumulator)
    num_transition = 0
    state_frame_key = dataset.state_meta[0]["key"] if dataset.state_meta else None

    iterator = _iter_episode_batches(dataset, processor, episode_indices, num_workers, max_in_flight)
    for batch in tqdm(iterator, total=len(episode_indices), desc="Computing InternW0-delta normalization stats"):
        if state_frame_key is not None:
            num_transition += int(batch["state"][state_frame_key].shape[0])
        for meta in dataset.state_meta:
            key = meta["key"]
            state_acc[key].update(batch["state"][key])
        for meta in dataset.action_meta:
            key = meta["key"]
            action_acc[key].update(batch["action"][key])

    stats = {
        "state": defaultdict(dict),
        "action": defaultdict(dict),
        "num_episodes": len(episode_indices),
        "num_transition": int(num_transition),
        "episode_offset": start,
    }
    for meta in dataset.state_meta:
        key = meta["key"]
        stats["state"][key] = state_acc[key].finalize()
    for meta in dataset.action_meta:
        key = meta["key"]
        stats["action"][key] = action_acc[key].finalize()
    return stats


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    hf_datasets.utils.logging.set_verbosity_warning()
    hf_datasets.utils.logging.disable_progress_bar()
    logging.getLogger("datasets").setLevel(logging.WARNING)
    precompute_cfg = cfg.get("precompute")
    _setup_hf_cache(precompute_cfg)

    compute_stats_flag = bool(_cfg_get(precompute_cfg, "compute_stats", True))
    warmup_train = bool(_cfg_get(precompute_cfg, "warmup_train_cache", True))
    warmup_val = bool(_cfg_get(precompute_cfg, "warmup_val_cache", True))
    force = bool(_cfg_get(precompute_cfg, "force", False))
    num_workers = int(_cfg_get(precompute_cfg, "num_workers", 1))
    heartbeat_interval = int(_cfg_get(precompute_cfg, "heartbeat_interval", 0))
    max_in_flight = int(_cfg_get(precompute_cfg, "max_in_flight", max(16, num_workers * 4)))
    max_episodes = int(_cfg_get(precompute_cfg, "max_episodes", 0))
    episode_offset = int(_cfg_get(precompute_cfg, "episode_offset", 0))
    stats_path = _as_path(_cfg_get(precompute_cfg, "stats_path", _default_stats_path(cfg.data.train)))

    train_dataset = None
    if warmup_train or compute_stats_flag:
        logger.info("Building train BaseLerobotDataset; this also warms the HF parquet cache.")
        with _heartbeat("build train BaseLerobotDataset / warm HF parquet cache", heartbeat_interval):
            train_dataset = _build_base_dataset(cfg.data.train)
        logger.info(
            "Train dataset ready: episodes=%s frames=%s",
            train_dataset.multi_dataset.num_episodes,
            train_dataset.multi_dataset.num_frames,
        )

    if compute_stats_flag:
        if stats_path.exists() and not force:
            logger.info("Stats already exists, skip compute: %s", stats_path)
        else:
            if train_dataset is None:
                train_dataset = _build_base_dataset(cfg.data.train)
            processor = instantiate(cfg.data.train.processor)
            processor.train()
            logger.info("Computing InternW0-delta normalization stats -> %s", stats_path)
            with _heartbeat("compute InternW0-delta normalization stats", heartbeat_interval):
                stats = compute_wam_stats(
                    train_dataset,
                    processor,
                    num_workers=num_workers,
                    max_in_flight=max_in_flight,
                    max_episodes=max_episodes,
                    episode_offset=episode_offset,
                )
            save_dataset_stats_to_json(stats, str(stats_path))
            logger.info("Wrote stats: %s", stats_path)

    if warmup_val and cfg.data.get("val") is not None:
        logger.info("Building val BaseLerobotDataset; this warms the val HF parquet cache.")
        with _heartbeat("build val BaseLerobotDataset / warm HF parquet cache", heartbeat_interval):
            val_dataset = _build_base_dataset(cfg.data.val)
        logger.info(
            "Val dataset ready: episodes=%s frames=%s",
            val_dataset.multi_dataset.num_episodes,
            val_dataset.multi_dataset.num_frames,
        )

    logger.info("Done. HF_DATASETS_CACHE=%s", os.environ.get("HF_DATASETS_CACHE"))


if __name__ == "__main__":
    main()
