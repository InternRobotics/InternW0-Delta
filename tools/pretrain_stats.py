#!/usr/bin/env python3
"""Precompute grouped canonical action/proprio stats for pretraining datasets.

The script reads the same canonical tensors that PretrainWAMDataset returns:
- action: canonical action tensor in the configured robot canvas
- state/proprio: canonical absolute state tensor in the configured robot canvas

Stats are grouped by DatasetSpec.stats_group, which is inferred to short names
such as egodex, a1/franka, agibot/a2d, or galaxea/r1lite, unless overridden in source YAML.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import heapq
import json
import multiprocessing as mp
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from wam.datasets import pretrain_lerobot_loader as pretrain_loader
from wam.datasets.pretrain_stats import merge_stats, read_stats, split_stats


def _format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(float(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _progress_message(worker_id: int, processed: int, total: int, start_time: float, prefix: str = "processed") -> str:
    elapsed = max(1e-9, time.perf_counter() - start_time)
    rate = float(processed) / elapsed
    if total > 0 and processed > 0:
        pct = 100.0 * float(processed) / float(total)
        eta = (float(total) - float(processed)) / max(rate, 1e-9)
        return (
            f"[worker {worker_id}] {prefix}={processed}/{total} "
            f"({pct:.1f}%) speed={rate:.1f} samples/s elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        )
    return f"[worker {worker_id}] {prefix}={processed} speed={rate:.1f} samples/s elapsed={_format_seconds(elapsed)}"


class MaskedStatsAccumulator:
    def __init__(self, dim: int, *, reservoir_rows: int, seed: int) -> None:
        self.dim = int(dim)
        self.count = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)
        self.reservoir_rows = max(0, int(reservoir_rows))
        self.reservoir = np.empty((self.reservoir_rows, self.dim), dtype=np.float32) if self.reservoir_rows else None
        self.reservoir_count = 0
        self.rows_seen = 0
        self.rng = np.random.default_rng(int(seed))

    def update(
        self,
        value: torch.Tensor,
        *,
        cell_is_valid: torch.Tensor | None = None,
        time_is_pad: torch.Tensor | None = None,
        dim_is_pad: torch.Tensor | None = None,
        row_weights: torch.Tensor | None = None,
    ) -> None:
        arr = value.detach().cpu().to(torch.float32).numpy()
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {arr.shape[-1]}")

        weights = np.ones(arr.shape[0], dtype=np.int64)
        if row_weights is not None:
            weights = row_weights.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            if weights.shape[0] != arr.shape[0]:
                raise ValueError(f"Expected {arr.shape[0]} row weights, got {weights.shape[0]}")
            if np.any(weights < 0):
                raise ValueError("row weights must be non-negative")
        row_valid = np.ones(arr.shape[0], dtype=bool)
        if time_is_pad is not None:
            row_valid &= ~time_is_pad.detach().cpu().numpy().astype(bool).reshape(-1)[: arr.shape[0]]
        row_valid &= weights > 0
        dim_valid = np.ones(self.dim, dtype=bool)
        if dim_is_pad is not None:
            dim_valid &= ~dim_is_pad.detach().cpu().numpy().astype(bool).reshape(-1)[: self.dim]

        finite = np.isfinite(arr)
        cell_valid = np.ones_like(arr, dtype=bool)
        if cell_is_valid is not None:
            cell_valid = (
                cell_is_valid.detach().cpu().numpy().astype(bool).reshape(-1, self.dim)
            )
            if cell_valid.shape != arr.shape:
                raise ValueError(
                    f"Expected cell validity shape {arr.shape}, got {cell_valid.shape}"
                )
        valid = finite & cell_valid & row_valid[:, None] & dim_valid[None, :]
        if not np.any(valid):
            return

        masked = np.where(valid, arr, 0.0).astype(np.float64, copy=False)
        weighted_valid = valid.astype(np.int64, copy=False) * weights[:, None]
        self.count += weighted_valid.sum(axis=0).astype(np.int64)
        self.sum += (masked * weights[:, None]).sum(axis=0)
        self.sumsq += (masked * masked * weights[:, None]).sum(axis=0)

        for dim in np.flatnonzero(valid.any(axis=0)):
            vals = arr[valid[:, dim], dim].astype(np.float64, copy=False)
            self.min[dim] = min(self.min[dim], float(vals.min()))
            self.max[dim] = max(self.max[dim], float(vals.max()))

        if self.reservoir is not None:
            rows = arr[row_valid].astype(np.float32, copy=True)
            if rows.size:
                rows[~(cell_valid[row_valid] & dim_valid[None, :])] = np.nan
                rows[~np.isfinite(rows)] = np.nan
                valid_weights = weights[row_valid]
                if np.all(valid_weights == 1):
                    self._add_reservoir_rows(rows)
                else:
                    self._add_weighted_reservoir_rows(rows, valid_weights)

    def _add_reservoir_rows(self, rows: np.ndarray) -> None:
        """Add a batch with vectorized Algorithm R reservoir sampling.

        Drawing one replacement index per incoming row is exactly the same
        Algorithm R used by the scalar implementation.  Only accepted rows
        are assigned, and when multiple rows target the same reservoir slot
        the last row in stream order wins, as it would in the scalar loop.
        """

        num_rows = int(len(rows))
        if num_rows <= 0:
            return

        row_offset = 0
        free = int(self.reservoir_rows) - int(self.reservoir_count)
        if free > 0:
            take = min(free, num_rows)
            stop = int(self.reservoir_count) + take
            self.reservoir[self.reservoir_count : stop] = rows[:take]
            self.reservoir_count = stop
            self.rows_seen += take
            row_offset = take

        remaining = num_rows - row_offset
        if remaining <= 0:
            return

        # For stream item n, Algorithm R draws uniformly from [0, n).
        high = np.arange(
            int(self.rows_seen) + 1,
            int(self.rows_seen) + remaining + 1,
            dtype=np.int64,
        )
        candidate_slots = self.rng.integers(0, high)
        accepted = np.flatnonzero(candidate_slots < int(self.reservoir_rows))
        if accepted.size:
            slots = candidate_slots[accepted]
            # Advanced assignment with duplicate indices does not promise an
            # ordering.  Select the last occurrence of each slot explicitly.
            _, reverse_positions = np.unique(slots[::-1], return_index=True)
            keep = accepted.size - 1 - reverse_positions
            self.reservoir[slots[keep]] = rows[row_offset + accepted[keep]]
        self.rows_seen += remaining

    def _add_weighted_reservoir_rows(self, rows: np.ndarray, weights: np.ndarray) -> None:
        """Sample a run-length-encoded logical row stream with Algorithm R.

        Canonical EEF projection is the expensive operation and has already
        been deduplicated.  We still generate Algorithm R's one random slot per
        logical row, but map only retained positions back through the run
        lengths.  This is identical to expanding the stream and avoids copying
        every repeated 80-D row.
        """

        weights = np.asarray(weights, dtype=np.int64).reshape(-1)
        if len(rows) != len(weights):
            raise ValueError(f"Expected {len(rows)} weights, got {len(weights)}")
        positive = weights > 0
        if not np.any(positive):
            return
        rows = rows[positive]
        weights = weights[positive]
        cumulative = np.cumsum(weights, dtype=np.int64)
        num_logical_rows = int(cumulative[-1])

        stream_offset = 0
        free = int(self.reservoir_rows) - int(self.reservoir_count)
        if free > 0:
            take = min(free, num_logical_rows)
            logical_positions = np.arange(take, dtype=np.int64)
            source_rows = np.searchsorted(cumulative, logical_positions, side="right")
            stop = int(self.reservoir_count) + take
            self.reservoir[self.reservoir_count : stop] = rows[source_rows]
            self.reservoir_count = stop
            self.rows_seen += take
            stream_offset = take

        remaining = num_logical_rows - stream_offset
        if remaining <= 0:
            return
        high = np.arange(
            int(self.rows_seen) + 1,
            int(self.rows_seen) + remaining + 1,
            dtype=np.int64,
        )
        candidate_slots = self.rng.integers(0, high)
        accepted = np.flatnonzero(candidate_slots < int(self.reservoir_rows))
        if accepted.size:
            slots = candidate_slots[accepted]
            _, reverse_positions = np.unique(slots[::-1], return_index=True)
            keep = accepted.size - 1 - reverse_positions
            logical_positions = stream_offset + accepted[keep]
            source_rows = np.searchsorted(cumulative, logical_positions, side="right")
            self.reservoir[slots[keep]] = rows[source_rows]
        self.rows_seen += remaining

    def merge(self, other: "MaskedStatsAccumulator") -> None:
        if other.dim != self.dim:
            raise ValueError(f"Cannot merge dim={other.dim} into dim={self.dim}")
        other_valid = other.count > 0
        self.count += other.count
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.min = np.where(other_valid, np.minimum(self.min, other.min), self.min)
        self.max = np.where(other_valid, np.maximum(self.max, other.max), self.max)
        if self.reservoir is None or other.reservoir is None or other.rows_seen <= 0:
            return

        # Each worker reservoir is a uniform sample of a stream whose size is
        # ``rows_seen``.  Concatenating equally sized worker reservoirs and
        # running Algorithm R again would incorrectly give a small worker the
        # same quantile weight as a large worker.  A uniform reservoir of the
        # union can instead be formed by drawing how many retained rows come
        # from each stream with the exact hypergeometric law, then uniformly
        # sub-sampling each worker reservoir by that amount.
        self_seen = int(self.rows_seen)
        other_seen = int(other.rows_seen)
        total_seen = self_seen + other_seen
        target_count = min(int(self.reservoir_rows), total_seen)
        if self_seen <= 0:
            merged = other.reservoir[: other.reservoir_count].copy()
            if len(merged) != target_count:
                chosen = self.rng.choice(
                    len(merged), size=target_count, replace=len(merged) < target_count
                )
                merged = merged[chosen]
        else:
            # ``Generator.hypergeometric`` rejects either population count at
            # one billion even though our exact logical state stream can be
            # larger (samples * sequence length).  Uniformly choosing the
            # retained stream positions and counting how many fall in the
            # first stream is exactly the same hypergeometric draw, has no
            # billion-row limit, and allocates only ``target_count`` integers.
            retained_positions = self.rng.choice(
                total_seen,
                size=target_count,
                replace=False,
                shuffle=False,
            )
            take_self = int(np.count_nonzero(retained_positions < self_seen))
            take_other = target_count - take_self
            self_indices = self.rng.choice(
                self.reservoir_count,
                size=take_self,
                replace=take_self > self.reservoir_count,
            )
            other_indices = self.rng.choice(
                other.reservoir_count,
                size=take_other,
                replace=take_other > other.reservoir_count,
            )
            merged = np.concatenate(
                [self.reservoir[self_indices], other.reservoir[other_indices]],
                axis=0,
            )
            self.rng.shuffle(merged, axis=0)
        self.reservoir[:target_count] = merged
        self.reservoir_count = target_count
        self.rows_seen = total_seen

    def finalize(self) -> dict[str, list[float] | int]:
        valid = self.count > 0
        mean = np.zeros(self.dim, dtype=np.float64)
        std = np.ones(self.dim, dtype=np.float64)
        mean[valid] = self.sum[valid] / self.count[valid]
        var = np.zeros(self.dim, dtype=np.float64)
        var[valid] = self.sumsq[valid] / self.count[valid] - mean[valid] * mean[valid]
        std[valid] = np.sqrt(np.maximum(var[valid], 0.0))

        min_v = np.where(valid, self.min, 0.0)
        max_v = np.where(valid, self.max, 0.0)
        q01 = min_v.copy()
        q99 = max_v.copy()
        if self.reservoir is not None and self.reservoir_count > 0:
            sample = self.reservoir[: self.reservoir_count].astype(np.float64, copy=False)
            q01_sample = np.full(self.dim, np.nan, dtype=np.float64)
            q99_sample = np.full(self.dim, np.nan, dtype=np.float64)
            if bool(valid.any()):
                with np.errstate(all="ignore"):
                    q01_sample[valid] = np.nanquantile(
                        sample[:, valid], 0.01, axis=0
                    )
                    q99_sample[valid] = np.nanquantile(
                        sample[:, valid], 0.99, axis=0
                    )
            q01 = np.where(np.isfinite(q01_sample) & valid, q01_sample, q01)
            q99 = np.where(np.isfinite(q99_sample) & valid, q99_sample, q99)

        return {
            "count": self.count.tolist(),
            "global_min": min_v.astype(float).tolist(),
            "global_max": max_v.astype(float).tolist(),
            "global_q01": q01.astype(float).tolist(),
            "global_q99": q99.astype(float).tolist(),
            "global_mean": mean.astype(float).tolist(),
            "global_std": std.astype(float).tolist(),
            "stepwise_min": min_v.astype(float).tolist(),
            "stepwise_max": max_v.astype(float).tolist(),
            "stepwise_q01": q01.astype(float).tolist(),
            "stepwise_q99": q99.astype(float).tolist(),
            "stepwise_mean": mean.astype(float).tolist(),
            "stepwise_std": std.astype(float).tolist(),
        }


class GroupedStats:
    def __init__(
        self,
        *,
        dim: int,
        reservoir_rows: int,
        seed: int,
        passthrough_dims: dict[str, list[int]] | None = None,
    ) -> None:
        self.dim = int(dim)
        self.reservoir_rows = int(reservoir_rows)
        self.seed = int(seed)
        self.passthrough_dims = {
            kind: sorted({int(dim) for dim in dims})
            for kind, dims in (passthrough_dims or {}).items()
        }
        for kind, dims in self.passthrough_dims.items():
            if kind not in {"action", "state"}:
                raise KeyError(f"Unsupported stats passthrough kind {kind!r}")
            if any(dim < 0 or dim >= self.dim for dim in dims):
                raise IndexError(
                    f"stats passthrough dims for {kind} exceed dim={self.dim}: {dims}"
                )
        self.groups: dict[str, dict[str, MaskedStatsAccumulator]] = {}
        self.samples_per_group: dict[str, int] = {}

    def _stats_dim_mask(
        self, item_mask: torch.Tensor | None, *, kind: str
    ) -> torch.Tensor | None:
        dims = self.passthrough_dims.get(kind, ())
        if item_mask is None and not dims:
            return None
        mask = (
            torch.zeros(self.dim, dtype=torch.bool)
            if item_mask is None
            else item_mask.detach().to(dtype=torch.bool).reshape(-1).clone()
        )
        if int(mask.numel()) != self.dim:
            raise ValueError(
                f"Expected {kind} dim mask of length {self.dim}, got {mask.numel()}"
            )
        if dims:
            mask[torch.as_tensor(dims, dtype=torch.long)] = True
        return mask

    def _group(self, name: str) -> dict[str, MaskedStatsAccumulator]:
        if name not in self.groups:
            offset = zlib.adler32(name.encode("utf-8")) % 1_000_000
            self.groups[name] = {
                "action": MaskedStatsAccumulator(self.dim, reservoir_rows=self.reservoir_rows, seed=self.seed + offset + 11),
                "state": MaskedStatsAccumulator(self.dim, reservoir_rows=self.reservoir_rows, seed=self.seed + offset + 23),
            }
            self.samples_per_group[name] = 0
        return self.groups[name]

    def update(self, group: str, item: dict[str, Any]) -> None:
        acc = self._group(group)
        acc["action"].update(
            item["action"],
            cell_is_valid=item.get("action_mask"),
            time_is_pad=item.get("action_is_pad"),
            dim_is_pad=self._stats_dim_mask(
                item.get("action_dim_is_pad"), kind="action"
            ),
            row_weights=item.get("_stats_action_row_weights"),
        )
        acc["state"].update(
            item["observation.state"],
            cell_is_valid=item.get("observation.state_mask"),
            time_is_pad=item.get("observation.state_is_pad"),
            dim_is_pad=self._stats_dim_mask(
                item.get("observation.state_dim_is_pad"), kind="state"
            ),
            row_weights=item.get("_stats_state_row_weights"),
        )
        sample_count = int(item.get("_stats_effective_num_samples", item.get("_stats_num_samples", 1)))
        self.samples_per_group[group] += sample_count

    def merge(self, other: "GroupedStats") -> None:
        for group, acc in other.groups.items():
            target = self._group(group)
            target["action"].merge(acc["action"])
            target["state"].merge(acc["state"])
            self.samples_per_group[group] += int(other.samples_per_group.get(group, 0))

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for group, acc in sorted(self.groups.items()):
            out[group] = {
                "num_samples": int(self.samples_per_group[group]),
                "action": {"default": acc["action"].finalize()},
                "state": {"default": acc["state"].finalize()},
            }
        return out


def _load_config(path: Path, sources=()) -> dict[str, Any]:
    config = pretrain_loader._read_dataset_config(path)
    if sources:
        config["sources"] = [s for s in config.get("sources", []) if s.get("_source_file") in sources]
        if not config["sources"]:
            raise ValueError(f"No sources match {list(sources)}")
    return config


def _select_specs(args: argparse.Namespace, specs: list[Any]) -> list[Any]:
    """Apply explicit stats scope before datasets are constructed."""
    selected = list(specs)
    if bool(getattr(args, "active_only", False)):
        selected = [spec for spec in selected if float(spec.dataset_weight) > 0.0]
    groups = {
        str(group)
        for group in (getattr(args, "include_stats_group", None) or [])
        if str(group).strip()
    }
    if groups:
        selected = [
            spec for spec in selected if str(spec.stats_group or spec.name) in groups
        ]
    if not selected:
        raise ValueError(
            "Stats selection is empty; check --active-only and --include-stats-group."
        )
    return selected


def _stats_dim(config: dict[str, Any]) -> int:
    space = config.get("canonical_action_space")
    if isinstance(space, dict):
        dim = space.get("dim", space.get("target_dim"))
        if dim is not None:
            return int(dim)
    raise ValueError("Dataset config must define canonical_action_space.dim for mix stats.")


def _stats_passthrough_dims(config: dict[str, Any]) -> dict[str, list[int]]:
    stats_cfg = config.get("stats")
    raw = stats_cfg.get("passthrough_dims") if isinstance(stats_cfg, dict) else None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("stats.passthrough_dims must be a mapping")
    parsed: dict[str, list[int]] = {}
    for kind, dims in raw.items():
        if isinstance(dims, int):
            dims = [dims]
        if not isinstance(dims, (list, tuple)):
            raise TypeError(
                f"stats.passthrough_dims.{kind} must be an int/list"
            )
        parsed[str(kind)] = sorted({int(dim) for dim in dims})
    return parsed


def _build_datasets(args: argparse.Namespace, config: dict[str, Any]) -> tuple[list[Any], list[Any], list[float], dict[str, Any]]:
    ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
    specs = _select_specs(args, pretrain_loader._load_specs(ns, config))
    # Stats do not need cached text embeddings; disabling avoids requiring the text cache first.
    specs = [dataclasses.replace(spec, local_text_embedding_cache_dir=None) for spec in specs]
    num_frames = int(args.num_frames or pretrain_loader._resolved_arg(None, config, "sampling", "num_frames", 33))
    action_size_value = args.action_size
    if action_size_value is None:
        action_size_value = pretrain_loader._resolved_arg(None, config, "sampling", "action_size", 32)
    action_size = int(action_size_value)
    global_sample_stride = int(args.global_sample_stride or pretrain_loader._resolved_arg(None, config, "sampling", "global_sample_stride", 1))
    datasets = [
        pretrain_loader.PretrainLeRobotDataset(
            spec,
            num_frames=num_frames,
            action_size=action_size,
            global_sample_stride=global_sample_stride,
        )
        for spec in specs
    ]
    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
    weights = pretrain_loader._mixture_dataset_weights(
        specs,
        datasets,
        mixture_cfg,
        allow_padding_at_end=allow_padding_at_end,
    )
    return specs, datasets, weights, mixture_cfg



def _make_dataset_from_spec(args: argparse.Namespace, config: dict[str, Any], spec: Any) -> Any:
    num_frames = int(args.num_frames or pretrain_loader._resolved_arg(None, config, "sampling", "num_frames", 33))
    action_size_value = args.action_size
    if action_size_value is None:
        action_size_value = pretrain_loader._resolved_arg(None, config, "sampling", "action_size", 32)
    action_size = int(action_size_value)
    global_sample_stride = int(args.global_sample_stride or pretrain_loader._resolved_arg(None, config, "sampling", "global_sample_stride", 1))
    return pretrain_loader.PretrainLeRobotDataset(
        spec,
        num_frames=num_frames,
        action_size=action_size,
        global_sample_stride=global_sample_stride,
    )


def _split_indices(indices: list[int], num_workers: int) -> list[list[int]]:
    workers = max(1, int(num_workers))
    shards = [[] for _ in range(workers)]
    for pos, index in enumerate(indices):
        shards[pos % workers].append(index)
    return [shard for shard in shards if shard]


def _split_plan_items(items: list[dict[str, int]], num_workers: int) -> list[list[dict[str, int]]]:
    workers = max(1, int(num_workers))
    heaps = [(0, idx, []) for idx in range(workers)]
    heapq.heapify(heaps)
    for item in sorted(items, key=lambda x: int(x.get("windows", 0)), reverse=True):
        total, idx, shard_items = heapq.heappop(heaps)
        shard_items.append(item)
        heapq.heappush(heaps, (total + int(item.get("windows", 0)), idx, shard_items))
    return [items for _, _, items in sorted(heaps, key=lambda x: x[1]) if items]


def _sample_plan_items_per_group(
    units: list[dict[str, int]],
    specs: list[Any],
    *,
    max_episodes_per_group: int,
    seed: int,
) -> list[dict[str, int]]:
    cap = int(max_episodes_per_group)
    if cap <= 0:
        return units

    by_group: dict[str, list[dict[str, int]]] = {}
    for item in units:
        spec = specs[int(item["spec_index"])]
        group = str(spec.stats_group or spec.name)
        by_group.setdefault(group, []).append(item)

    sampled: list[dict[str, int]] = []
    for group, items in sorted(by_group.items()):
        if len(items) <= cap:
            sampled.extend({**item, "sample_weight": 1} for item in items)
            continue

        by_spec: dict[int, list[int]] = {}
        for position, item in enumerate(items):
            by_spec.setdefault(int(item["spec_index"]), []).append(position)
        if len(by_spec) > cap:
            raise ValueError(
                f"plan-max-episodes-per-group={cap} is smaller than the {len(by_spec)} "
                f"configured datasets in stats group {group!r}"
            )

        group_seed = int(seed) + int(zlib.adler32(group.encode("utf-8")))
        rng = np.random.default_rng(group_seed)
        mandatory: set[int] = set()
        spec_sizes: dict[int, int] = {}
        for spec_index, positions in sorted(by_spec.items()):
            spec_sizes[spec_index] = len(positions)
            mandatory.add(positions[int(rng.integers(0, len(positions)))])

        remaining = np.asarray(
            [position for position in range(len(items)) if position not in mandatory],
            dtype=np.int64,
        )
        extra_count = cap - len(mandatory)
        if extra_count > 0:
            extra = rng.choice(remaining, size=extra_count, replace=False).tolist()
        else:
            extra = []
        selected_positions = sorted([*mandatory, *map(int, extra)])
        extra_probability = float(extra_count) / float(len(remaining)) if len(remaining) else 0.0
        for position in selected_positions:
            item = items[position]
            spec_size = spec_sizes[int(item["spec_index"])]
            inclusion_probability = (1.0 / float(spec_size)) + (
                1.0 - 1.0 / float(spec_size)
            ) * extra_probability
            sample_weight = max(1, int(round(1.0 / inclusion_probability)))
            sampled.append({**item, "sample_weight": sample_weight})
    return sampled


def _load_shard_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "pretrain_stats_episode_shard_plan_v1":
        raise ValueError(f"Unsupported shard plan format in {path}: {payload.get('format')}")
    return payload


def _write_episode_shard_plan(args: argparse.Namespace, config: dict[str, Any], output: Path) -> None:
    if output.exists() and not args.force:
        raise FileExistsError(f"Shard plan already exists: {output}. Use --force to overwrite.")

    ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
    specs = _select_specs(args, pretrain_loader._load_specs(ns, config))
    specs = [dataclasses.replace(spec, local_text_embedding_cache_dir=None) for spec in specs]
    if int(getattr(args, "plan_max_datasets", 0) or 0) > 0:
        specs = specs[: int(args.plan_max_datasets)]
        print(f"[plan] debug limit datasets={len(specs)}", flush=True)
    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
    num_shards = max(1, int(args.num_shards or 1))

    units: list[dict[str, int]] = []
    total_windows = 0
    start = time.perf_counter()
    for spec_idx, spec in enumerate(specs):
        try:
            ds = _make_dataset_from_spec(args, config, spec)
            counts = ds.valid_start_counts(allow_padding_at_end)
        except Exception as exc:
            if bool(getattr(args, "strict", False)):
                raise
            print(f"[plan] skip dataset {spec_idx}:{getattr(spec, 'name', '<unknown>')}: {type(exc).__name__}: {exc}", flush=True)
            continue
        ds_windows = 0
        for traj_pos, count in enumerate(counts):
            windows = int(count)
            if windows <= 0:
                continue
            episode_index = int(ds.trajectory_ids[int(traj_pos)])
            max_start = int(ds.max_start_for_trajectory_pos(int(traj_pos), allow_padding_at_end))
            units.append(
                {
                    "spec_index": int(spec_idx),
                    "episode_index": episode_index,
                    "max_start": max_start,
                    "windows": windows,
                }
            )
            ds_windows += windows
        total_windows += ds_windows
        if (spec_idx + 1) % max(1, int(args.progress_interval or 100)) == 0 or spec_idx + 1 == len(specs):
            elapsed = time.perf_counter() - start
            print(
                f"[plan] datasets={spec_idx + 1}/{len(specs)} episodes={len(units)} "
                f"windows={total_windows} elapsed={_format_seconds(elapsed)}",
                flush=True,
            )

    full_num_episode_units = len(units)
    full_total_windows = total_windows
    max_episodes_per_group = int(getattr(args, "plan_max_episodes_per_group", 0) or 0)
    units = _sample_plan_items_per_group(
        units,
        specs,
        max_episodes_per_group=max_episodes_per_group,
        seed=int(args.seed),
    )
    total_windows = int(sum(int(item["windows"]) for item in units))
    estimated_total_windows = int(
        sum(int(item["windows"]) * int(item.get("sample_weight", 1)) for item in units)
    )
    shard_lists = _split_plan_items(units, num_shards)
    shards = [{"shard_index": idx, "total_windows": 0, "items": []} for idx in range(num_shards)]
    for idx, items in enumerate(shard_lists):
        if idx >= num_shards:
            break
        shards[idx] = {
            "shard_index": idx,
            "total_windows": int(sum(int(item["windows"]) for item in items)),
            "items": items,
        }

    dataset_to_group = {spec.name: (spec.stats_group or spec.name) for spec in specs}
    group_to_datasets: dict[str, list[str]] = {}
    for name, group in dataset_to_group.items():
        group_to_datasets.setdefault(group, []).append(name)

    payload = {
        "format": "pretrain_stats_episode_shard_plan_v1",
        "dataset_config": str(args.dataset_config),
        "num_shards": int(num_shards),
        "num_specs": int(len(specs)),
        "num_episode_units": int(len(units)),
        "total_windows": int(total_windows),
        "full_num_episode_units": int(full_num_episode_units),
        "full_total_windows": int(full_total_windows),
        "estimated_total_windows": estimated_total_windows,
        "max_episodes_per_group": max_episodes_per_group,
        "dataset_to_group": dataset_to_group,
        "group_to_datasets": {k: sorted(v) for k, v in sorted(group_to_datasets.items())},
        "shards": shards,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    max_windows = max((int(shard["total_windows"]) for shard in shards), default=0)
    min_windows = min((int(shard["total_windows"]) for shard in shards), default=0)
    print(
        f"[OK] wrote shard plan {output} shards={num_shards} episodes={len(units)}/{full_num_episode_units} "
        f"windows={total_windows}/{full_total_windows} estimated={estimated_total_windows} "
        f"min_shard={min_windows} max_shard={max_windows}",
        flush=True,
    )


def _plan_worker(payload: dict[str, Any]) -> tuple[GroupedStats, int, int]:
    args = argparse.Namespace(**payload["args"])
    config = payload.get("config")
    if config is None:
        config = _load_config(Path(payload["dataset_config"]), getattr(args, "source", ()))
    specs_by_index: dict[int, Any] = payload["specs_by_index"]
    plan_items: list[dict[str, int]] = payload["plan_items"]
    worker_id = int(payload["worker_id"])
    strict = bool(getattr(args, "strict", False))
    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
    grouped = GroupedStats(
        dim=_stats_dim(config),
        reservoir_rows=int(args.reservoir_rows),
        seed=int(args.seed) + worker_id * 1009,
        passthrough_dims=_stats_passthrough_dims(config),
    )

    by_spec: dict[int, list[dict[str, int]]] = {}
    for item in plan_items:
        by_spec.setdefault(int(item["spec_index"]), []).append(item)
    planned_total = int(sum(int(item.get("windows", 0)) for item in plan_items))
    max_samples = int(getattr(args, "max_samples", 0) or 0)
    progress_total = min(planned_total, max_samples) if max_samples > 0 else planned_total
    print(
        f"[worker {worker_id}] prepared plan specs={len(by_spec)} episodes={len(plan_items)} total_samples={planned_total}",
        flush=True,
    )

    processed = 0
    skipped = 0
    worker_start_time = time.perf_counter()
    last_progress_time = worker_start_time
    progress_interval = int(getattr(args, "progress_interval", 0) or 0)
    progress_seconds = float(getattr(args, "progress_seconds", 30.0) or 0.0)
    episode_batch_size = max(1, int(getattr(args, "episode_window_batch_size", 4096) or 4096))

    for spec_idx, items in sorted(by_spec.items()):
        spec = specs_by_index[int(spec_idx)]
        try:
            ds = _make_dataset_from_spec(args, config, spec)
            group = ds.spec.stats_group or ds.name
        except Exception as exc:
            if strict:
                raise
            spec_windows = int(sum(int(item.get("windows", 0)) for item in items))
            skipped += spec_windows
            print(f"[worker {worker_id}] skip spec={spec_idx}: {type(exc).__name__}: {exc}", flush=True)
            continue

        ds_processed = 0
        ds_start_time = time.perf_counter()
        for plan_item in items:
            episode_index = int(plan_item["episode_index"])
            max_start = int(plan_item["max_start"])
            sample_weight = int(plan_item.get("sample_weight", 1))
            try:
                if max_samples > 0:
                    remaining = max_samples - processed
                    if remaining <= 0:
                        print(f"[worker {worker_id}] hit max_samples={max_samples}", flush=True)
                        return grouped, processed, progress_total
                    batch_size = min(episode_batch_size, remaining)
                else:
                    batch_size = episode_batch_size
                for item in ds.iter_episode_stat_batches(
                    episode_index,
                    allow_padding_at_end=allow_padding_at_end,
                    max_windows_per_batch=batch_size,
                    max_start=max_start,
                ):
                    batch_count = int(item.get("_stats_num_samples", 1))
                    if sample_weight != 1:
                        for key in ("_stats_action_row_weights", "_stats_state_row_weights"):
                            weights = item.get(key)
                            if weights is not None:
                                item[key] = weights * sample_weight
                        item["_stats_effective_num_samples"] = batch_count * sample_weight
                    grouped.update(group, item)
                    processed += batch_count
                    ds_processed += batch_count
                    now = time.perf_counter()
                    interval_due = progress_interval > 0 and processed // progress_interval != (processed - batch_count) // progress_interval
                    time_due = progress_seconds > 0 and now - last_progress_time >= progress_seconds
                    if interval_due or time_due:
                        print(_progress_message(worker_id, processed, progress_total, worker_start_time), flush=True)
                        last_progress_time = now
                    if max_samples > 0 and processed >= max_samples:
                        print(f"[worker {worker_id}] hit max_samples={max_samples}", flush=True)
                        return grouped, processed, progress_total
            except Exception as exc:
                if strict:
                    raise
                skipped += int(plan_item.get("windows", 0))
                print(
                    f"[worker {worker_id}] skip {ds.name} episode={episode_index}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        ds_elapsed = max(1e-9, time.perf_counter() - ds_start_time)
        print(
            f"[worker {worker_id}] {ds.name} group={group} samples={ds_processed} "
            f"speed={float(ds_processed)/ds_elapsed:.1f} samples/s elapsed={_format_seconds(ds_elapsed)}",
            flush=True,
        )
    if skipped:
        print(f"[worker {worker_id}] skipped_samples={skipped}", flush=True)
    return grouped, processed, progress_total


def _full_worker(payload: dict[str, Any]) -> tuple[GroupedStats, int, int]:
    args = argparse.Namespace(**payload["args"])
    config = payload.get("config")
    if config is None:
        config = _load_config(Path(payload["dataset_config"]), getattr(args, "source", ()))
    specs = payload.get("specs")
    if specs is None:
        ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
        all_specs = _select_specs(args, pretrain_loader._load_specs(ns, config))
        all_specs = [dataclasses.replace(spec, local_text_embedding_cache_dir=None) for spec in all_specs]
        specs = [all_specs[int(spec_index)] for spec_index in payload["spec_indices"]]
    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
    worker_id = int(payload["worker_id"])
    strict = bool(getattr(args, "strict", False))
    grouped = GroupedStats(
        dim=_stats_dim(config),
        reservoir_rows=int(args.reservoir_rows),
        seed=int(args.seed) + worker_id * 1009,
        passthrough_dims=_stats_passthrough_dims(config),
    )

    max_samples = int(getattr(args, "max_samples", 0) or 0)
    if max_samples > 0:
        spec_totals = [(spec, 0) for spec in specs]
        worker_total = max_samples
        progress_total = max_samples
    else:
        spec_totals: list[tuple[Any, int]] = []
        worker_total = 0
        for spec in specs:
            try:
                ds = _make_dataset_from_spec(args, config, spec)
                ds_total = int(ds.valid_start_counts(allow_padding_at_end).sum())
            except Exception as exc:
                if strict:
                    raise
                spec_name = getattr(spec, "name", "<unknown>")
                print(
                    f"[worker {worker_id}] skip dataset during count {spec_name}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                ds_total = 0
            spec_totals.append((spec, ds_total))
            worker_total += ds_total
        progress_total = worker_total
    print(
        f"[worker {worker_id}] prepared datasets={len(spec_totals)} total_samples={worker_total}",
        flush=True,
    )

    processed = 0
    worker_start_time = time.perf_counter()
    last_progress_time = worker_start_time
    progress_interval = int(getattr(args, "progress_interval", 0) or 0)
    progress_seconds = float(getattr(args, "progress_seconds", 30.0) or 0.0)
    for spec, ds_total in spec_totals:
        try:
            ds = _make_dataset_from_spec(args, config, spec)
            group = ds.spec.stats_group or ds.name
            valid_counts = ds.valid_start_counts(allow_padding_at_end)
        except Exception as exc:
            if strict:
                raise
            spec_name = getattr(spec, "name", "<unknown>")
            print(
                f"[worker {worker_id}] skip dataset during read {spec_name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        ds_processed = 0
        ds_skipped = 0
        ds_start_time = time.perf_counter()
        episode_batch_size = max(1, int(getattr(args, "episode_window_batch_size", 4096) or 4096))
        for traj_pos, count in enumerate(valid_counts):
            if int(count) <= 0:
                continue
            episode_index = int(ds.trajectory_ids[int(traj_pos)])
            max_start = ds.max_start_for_trajectory_pos(int(traj_pos), allow_padding_at_end)
            try:
                if max_samples > 0:
                    remaining = max_samples - processed
                    if remaining <= 0:
                        print(f"[worker {worker_id}] hit max_samples={max_samples}", flush=True)
                        return grouped, processed, progress_total
                    batch_size = min(episode_batch_size, remaining)
                else:
                    batch_size = episode_batch_size
                for item in ds.iter_episode_stat_batches(
                    episode_index,
                    allow_padding_at_end=allow_padding_at_end,
                    max_windows_per_batch=batch_size,
                    max_start=max_start,
                ):
                    batch_count = int(item.get("_stats_num_samples", 1))
                    grouped.update(group, item)
                    processed += batch_count
                    ds_processed += batch_count
                    now = time.perf_counter()
                    interval_due = progress_interval > 0 and processed // progress_interval != (processed - batch_count) // progress_interval
                    time_due = progress_seconds > 0 and now - last_progress_time >= progress_seconds
                    if interval_due or time_due:
                        print(_progress_message(worker_id, processed, progress_total, worker_start_time), flush=True)
                        last_progress_time = now
                    if max_samples > 0 and processed >= max_samples:
                        print(f"[worker {worker_id}] hit max_samples={max_samples}", flush=True)
                        return grouped, processed, progress_total
            except Exception as exc:
                if strict:
                    raise
                ds_skipped += int(count)
                print(
                    f"[worker {worker_id}] skip {ds.name} episode={episode_index}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
        ds_elapsed = max(1e-9, time.perf_counter() - ds_start_time)
        ds_rate = float(ds_processed) / ds_elapsed
        print(
            f"[worker {worker_id}] {ds.name} group={group} samples={ds_processed}/{ds_total} "
            f"skipped={ds_skipped} speed={ds_rate:.1f} samples/s elapsed={_format_seconds(ds_elapsed)}",
            flush=True,
        )
    return grouped, processed, progress_total

def _default_output(config: dict[str, Any]) -> Path:
    cache_root = config.get("cache_root") or ".cache/internw0"
    return Path(cache_root).expanduser() / "stats" / "pretrain"


def _output_targets(output: Path, args, config) -> list[Path]:
    if output.suffix in {".json", ".pkl", ".pickle"} or args.output_format == "pickle":
        return [output]
    ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
    specs = _select_specs(args, pretrain_loader._load_specs(ns, config))
    if any(not spec.normalization_stats for spec in specs):
        raise ValueError("Split statistics output requires normalization_stats in each source config")
    return sorted({output / Path(spec.normalization_stats).name for spec in specs})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/pretrain/dataset.yaml"))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for per-source JSON files (default: <cache_root>/stats/pretrain); .json writes a combined file; .pkl writes a shard.")
    parser.add_argument("--sample-mode", choices=["full", "mixed"], default="full")
    parser.add_argument("--num-samples", type=int, default=0, help="Only used by --sample-mode mixed; 0 uses mixture.epoch_length or 100000.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap for full mode; 0 means no cap.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes for full mode.")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--action-size", type=int, default=None)
    parser.add_argument("--global-sample-stride", type=int, default=None)
    parser.add_argument("--reservoir-rows", type=int, default=200000)
    parser.add_argument("--norm-mode", default="q01/q99")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only construct specs whose resolved dataset_weight is positive.",
    )
    parser.add_argument(
        "--include-stats-group",
        action="append",
        default=[],
        help="Only compute this stats group; repeat for multiple groups.",
    )
    parser.add_argument(
        "--base-stats",
        type=Path,
        default=None,
        help="Copy untouched groups from this statistics directory or JSON and replace newly computed groups.",
    )
    parser.add_argument(
        "--replace-stats-kind",
        choices=["all", "action", "state"],
        default="all",
        help="With --base-stats, replace all stats or only one feature kind.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--progress-seconds", type=float, default=30.0, help="Also print worker progress after this many seconds; 0 disables time-based progress.")
    parser.add_argument(
        "--episode-window-batch-size",
        type=int,
        default=4096,
        help="Number of training windows to expand per episode stats batch.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on unreadable episodes or invalid values instead of skipping them.")
    parser.add_argument("--num-shards", type=int, default=1, help="Distributed stats: total number of external shards/jobs.")
    parser.add_argument("--shard-index", type=int, default=0, help="Distributed stats: this shard/job index, 0-based.")
    parser.add_argument("--output-format", choices=["auto", "json", "pickle"], default="auto")
    parser.add_argument("--merge-partials", nargs="*", default=None, help="Merge pickle partial files/globs and write final JSON stats.")
    parser.add_argument("--write-shard-plan", type=Path, default=None, help="Write an episode-level load-balanced shard plan and exit.")
    parser.add_argument("--shard-plan", type=Path, default=None, help="Use an episode-level shard plan generated by --write-shard-plan.")
    parser.add_argument("--plan-max-datasets", type=int, default=0, help="Debug only: limit datasets while writing shard plan.")
    parser.add_argument(
        "--plan-max-episodes-per-group",
        type=int,
        default=0,
        help=(
            "While writing a shard plan, sample at most this many episodes per stats group. "
            "Every configured dataset is covered once and inverse-probability weights preserve "
            "the full window distribution. 0 keeps every episode."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _full_total_steps(datasets: list[Any], *, allow_padding_at_end: bool) -> int:
    return int(sum(int(ds.valid_start_counts(allow_padding_at_end).sum()) for ds in datasets))


def _iter_full_items(datasets: list[Any], *, allow_padding_at_end: bool):
    for ds_pos, ds in enumerate(datasets):
        valid_counts = ds.valid_start_counts(allow_padding_at_end)
        for traj_pos, count in enumerate(valid_counts):
            if int(count) <= 0:
                continue
            episode_index = int(ds.trajectory_ids[int(traj_pos)])
            for valid_rank in range(int(count)):
                frame_idx = ds.valid_start_for_trajectory_rank(
                    int(traj_pos),
                    valid_rank,
                    allow_padding_at_end,
                )
                item = ds.get_step_item(episode_index, int(frame_idx))
                item["dataset_index"] = torch.tensor(ds_pos, dtype=torch.long)
                yield item


def _iter_mixed_items(args: argparse.Namespace, datasets: list[Any], weights: list[float], mixture_cfg: dict[str, Any], *, allow_padding_at_end: bool):
    mixed = pretrain_loader.PretrainLeRobotMixture(
        datasets,
        dataset_weights=weights,
        training=True,
        balance_dataset_weights=bool(mixture_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(mixture_cfg.get("balance_trajectory_weights", True)),
        seed=int(mixture_cfg.get("seed", args.seed)),
        allow_padding_at_end=allow_padding_at_end,
        epoch_length=None,
        eval_exclude_from_training=None,
        training_block_size=int(mixture_cfg.get("training_block_size", 4)),
    )
    mixed.set_epoch(int(args.epoch))
    total = int(args.num_samples or mixture_cfg.get("epoch_length") or 100000)
    for idx in range(total):
        yield mixed[idx]


def _write_json_stats(
    output: Path,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    grouped: GroupedStats,
    processed: int,
    dataset_to_group: dict[str, str],
    group_to_datasets: dict[str, list[str]],
) -> None:
    refreshed_groups = grouped.finalize()
    refreshed_num_samples = sum(
        int(group.get("num_samples", 0)) for group in refreshed_groups.values()
    )
    payload = {
        "format": "pretrain_grouped_canonical_stats_v1",
        "dataset_config": str(args.dataset_config),
        "sample_mode": str(args.sample_mode),
        "norm_mode": str(args.norm_mode),
        "num_samples": int(refreshed_num_samples),
        "action_dim": int(grouped.dim),
        "state_dim": int(grouped.dim),
        "passthrough_dims": grouped.passthrough_dims,
        "dataset_to_group": dataset_to_group,
        "group_to_datasets": {k: sorted(v) for k, v in sorted(group_to_datasets.items())},
        "groups": refreshed_groups,
    }
    if args.base_stats is not None:
        base_path = Path(args.base_stats).expanduser()
        base = (merge_stats((path, read_stats(path)) for path in sorted(base_path.glob("*.json")))
                if base_path.is_dir() else read_stats(base_path))
        if not isinstance(base.get("groups"), dict):
            raise ValueError(f"Base stats has no groups mapping: {base_path}")
        base_dim = int(base.get("action_dim", grouped.dim))
        if base_dim != int(grouped.dim):
            raise ValueError(
                f"Base stats action_dim={base_dim} does not match computed dim={grouped.dim}"
            )
        old_groups = dict(base["groups"])
        replace_kind = str(args.replace_stats_kind)
        if replace_kind == "all":
            old_groups.update(refreshed_groups)
        else:
            for group, refreshed in refreshed_groups.items():
                if group not in old_groups:
                    raise KeyError(
                        f"Cannot replace only {replace_kind} for missing base group {group!r}"
                    )
                merged_group = dict(old_groups[group])
                merged_group[replace_kind] = refreshed[replace_kind]
                merged_group["num_samples"] = int(refreshed["num_samples"])
                old_groups[group] = merged_group
        merged_dataset_to_group = dict(base.get("dataset_to_group") or {})
        merged_dataset_to_group.update(dataset_to_group)
        merged_group_to_datasets = dict(base.get("group_to_datasets") or {})
        merged_group_to_datasets.update(
            {key: sorted(value) for key, value in group_to_datasets.items()}
        )
        base.update(
            {
                "format": "pretrain_grouped_canonical_stats_v1",
                "dataset_config": str(args.dataset_config),
                "norm_mode": str(args.norm_mode),
                "num_samples": sum(
                    int(group.get("num_samples", 0)) for group in old_groups.values()
                ),
                "action_dim": int(grouped.dim),
                "state_dim": int(grouped.dim),
                "passthrough_dims": grouped.passthrough_dims,
                "dataset_to_group": merged_dataset_to_group,
                "group_to_datasets": merged_group_to_datasets,
                "groups": old_groups,
                "refreshed_groups": sorted(refreshed_groups),
                "refreshed_stats_kind": replace_kind,
                "base_stats": str(base_path),
            }
        )
        payload = base
    if output.suffix == ".json":
        outputs = {output: payload}
    else:
        ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
        specs = pretrain_loader._load_specs(ns, config)
        selected_files = {spec.normalization_stats for spec in _select_specs(args, specs)}
        specs = [spec for spec in specs if spec.normalization_stats in selected_files]
        if args.active_only:
            specs = [spec for spec in specs if spec.dataset_weight > 0]
        outputs = {output / name: part for name, part in split_stats(payload, specs).items()}
    # Check the entire selection before writing any files.
    for path in outputs:
        if path.exists() and not args.force:
            raise FileExistsError(f"Stats file already exists: {path}. Use --force to overwrite.")
    for path, part in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(part, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        print(f"[OK] wrote {path}", flush=True)


def _write_pickle_partial(output: Path, *, grouped: GroupedStats, processed: int, dataset_to_group: dict[str, str], args: argparse.Namespace) -> None:
    payload = {
        "format": "pretrain_grouped_canonical_stats_partial_v1",
        "processed": int(processed),
        "dataset_to_group": dataset_to_group,
        "grouped": grouped,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(output)
    print(f"[OK] wrote partial {output}", flush=True)


def _merge_partial_files(args: argparse.Namespace, config: dict[str, Any], output: Path) -> None:
    patterns = args.merge_partials or []
    partial_paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            partial_paths.extend(Path(path) for path in matches)
        else:
            path = Path(pattern)
            if path.exists():
                partial_paths.append(path)
    if not partial_paths:
        raise FileNotFoundError(f"No partial files matched: {patterns}")

    ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
    specs = _select_specs(args, pretrain_loader._load_specs(ns, config))
    specs = [dataclasses.replace(spec, local_text_embedding_cache_dir=None) for spec in specs]
    dataset_to_group = {spec.name: (spec.stats_group or spec.name) for spec in specs}
    group_to_datasets: dict[str, list[str]] = {}
    for name, group in dataset_to_group.items():
        group_to_datasets.setdefault(group, []).append(name)

    grouped = GroupedStats(
        dim=_stats_dim(config),
        reservoir_rows=int(args.reservoir_rows),
        seed=int(args.seed),
        passthrough_dims=_stats_passthrough_dims(config),
    )
    processed = 0
    start = time.perf_counter()
    for idx, path in enumerate(partial_paths, 1):
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("format") != "pretrain_grouped_canonical_stats_partial_v1":
            raise ValueError(f"Unsupported partial format in {path}: {payload.get('format')}")
        grouped.merge(payload["grouped"])
        processed += int(payload.get("processed", 0))
        if idx % max(1, int(args.progress_interval or 1000)) == 0 or idx == len(partial_paths):
            elapsed = time.perf_counter() - start
            print(
                f"[merge] files={idx}/{len(partial_paths)} processed_samples={processed} "
                f"elapsed={_format_seconds(elapsed)}",
                flush=True,
            )
    _write_json_stats(
        output,
        args=args,
        config=config,
        grouped=grouped,
        processed=processed,
        dataset_to_group=dataset_to_group,
        group_to_datasets=group_to_datasets,
    )


def main() -> None:
    args = parse_args()
    config = _load_config(args.dataset_config, args.source)
    if args.write_shard_plan is not None:
        print(f"[stats] writing episode shard plan to {args.write_shard_plan}", flush=True)
        _write_episode_shard_plan(args, config, args.write_shard_plan)
        return

    output = args.output or _default_output(config)
    for target in _output_targets(output, args, config):
        if target.exists() and not args.force:
            raise FileExistsError(f"Stats file already exists: {target}. Use --force to overwrite.")

    if args.merge_partials:
        print(f"[stats] merging partials into {output}", flush=True)
        _merge_partial_files(args, config, output)
        return

    print(f"[stats] loading specs from {args.dataset_config}", flush=True)
    ns = argparse.Namespace(dataset_specs_json=None, remote_root=None, name="dataset")
    specs = _select_specs(args, pretrain_loader._load_specs(ns, config))
    specs = [dataclasses.replace(spec, local_text_embedding_cache_dir=None) for spec in specs]
    all_specs = specs
    all_spec_count = len(all_specs)
    num_shards = max(1, int(args.num_shards or 1))
    shard_index = int(args.shard_index or 0)
    plan_items: list[dict[str, int]] | None = None
    plan_payload: dict[str, Any] | None = None
    if args.shard_plan is not None:
        plan_payload = _load_shard_plan(args.shard_plan)
        num_shards = int(plan_payload["num_shards"])
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
        shard_payload = plan_payload["shards"][shard_index]
        plan_items = list(shard_payload.get("items", []))
        selected_spec_indices = sorted({int(item["spec_index"]) for item in plan_items})
        specs = [all_specs[idx] for idx in selected_spec_indices]
        print(
            f"[stats] episode shard {shard_index}/{num_shards}: datasets={len(specs)}/{all_spec_count} "
            f"episodes={len(plan_items)} samples={int(shard_payload.get('total_windows', 0))}",
            flush=True,
        )
    else:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
        if num_shards > 1:
            specs = [spec for idx, spec in enumerate(specs) if idx % num_shards == shard_index]
            print(
                f"[stats] dataset shard {shard_index}/{num_shards}: datasets={len(specs)}/{all_spec_count}",
                flush=True,
            )
    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    dim = _stats_dim(config)
    grouped = GroupedStats(
        dim=dim,
        reservoir_rows=int(args.reservoir_rows),
        seed=int(args.seed),
        passthrough_dims=_stats_passthrough_dims(config),
    )

    dataset_to_group = {spec.name: (spec.stats_group or spec.name) for spec in specs}
    group_to_datasets: dict[str, list[str]] = {}
    for name, group in dataset_to_group.items():
        group_to_datasets.setdefault(group, []).append(name)

    if args.sample_mode == "full":
        worker_count = max(1, int(args.num_workers))
        max_samples = int(args.max_samples or 0)
        if max_samples > 0:
            # The debug cap is global; keep that path single-process so the cap is exact.
            worker_count = 1
        print(
            f"[stats] mode=full datasets={len(specs)} groups={len(group_to_datasets)} "
            f"workers={worker_count} samples=all",
            flush=True,
        )
        processed = 0
        args_payload = {
            "num_frames": args.num_frames,
            "action_size": args.action_size,
            "global_sample_stride": args.global_sample_stride,
            "reservoir_rows": args.reservoir_rows,
            "seed": args.seed,
            "max_samples": max_samples,
            "strict": bool(args.strict),
            "episode_window_batch_size": args.episode_window_batch_size,
            "progress_seconds": args.progress_seconds,
        }
        if plan_items is not None:
            local_shards = _split_plan_items(plan_items, worker_count)
            if not local_shards:
                local_shards = [[]]
            needed_spec_indices = sorted({int(item["spec_index"]) for item in plan_items})
            specs_by_index = {idx: all_specs[idx] for idx in needed_spec_indices}
            payloads = [
                {
                    "worker_id": worker_id,
                    "dataset_config": str(args.dataset_config),
                    "config": config,
                    "args": args_payload,
                    "specs_by_index": specs_by_index,
                    "plan_items": shard,
                }
                for worker_id, shard in enumerate(local_shards)
            ]
            worker_fn = _plan_worker
        else:
            shards = _split_indices(list(range(len(specs))), worker_count)
            payloads = [
                {
                    "worker_id": worker_id,
                    "dataset_config": str(args.dataset_config),
                    "config": config,
                    "args": args_payload,
                    "specs": [specs[int(index)] for index in shard],
                    "spec_indices": shard,
                }
                for worker_id, shard in enumerate(shards)
            ]
            worker_fn = _full_worker

        if worker_count > 1 and len(payloads) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(payloads)) as pool:
                # Merge in worker-id order so seeded reservoir quantiles are
                # reproducible regardless of worker completion timing.
                for partial, count, total_count in pool.imap(worker_fn, payloads):
                    grouped.merge(partial)
                    processed += int(count)
                    print(f"  merged worker result: processed={processed}, finished_worker_total={int(total_count)}", flush=True)
        else:
            partial, count, total_count = worker_fn(payloads[0])
            grouped.merge(partial)
            processed += int(count)
            print(f"  worker result: processed={processed}/{int(total_count)}", flush=True)
    else:
        specs, datasets, weights, mixture_cfg = _build_datasets(args, config)
        allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
        dataset_to_group = {ds.name: (ds.spec.stats_group or ds.name) for ds in datasets}
        group_to_datasets = {}
        for name, group in dataset_to_group.items():
            group_to_datasets.setdefault(group, []).append(name)
        total = int(args.num_samples or mixture_cfg.get("epoch_length") or 100000)
        print(f"[stats] mode=mixed datasets={len(datasets)} groups={len(group_to_datasets)} samples={total}", flush=True)
        processed = 0
        iterator = _iter_mixed_items(args, datasets, weights, mixture_cfg, allow_padding_at_end=allow_padding_at_end)
        for item in iterator:
            group = str(item.get("stats_group") or dataset_to_group[str(item.get("dataset_name"))])
            grouped.update(group, item)
            processed += 1
            if args.progress_interval > 0 and processed % int(args.progress_interval) == 0:
                print(f"  processed {processed}/{total}", flush=True)
            if processed >= total:
                break

    output_format = str(args.output_format)
    if output_format == "auto":
        output_format = "pickle" if output.suffix in {".pkl", ".pickle"} else "json"
    if output_format == "pickle":
        _write_pickle_partial(output, grouped=grouped, processed=processed, dataset_to_group=dataset_to_group, args=args)
    else:
        _write_json_stats(
            output,
            args=args,
            config=config,
            grouped=grouped,
            processed=processed,
            dataset_to_group=dataset_to_group,
            group_to_datasets=group_to_datasets,
        )


if __name__ == "__main__":
    main()
