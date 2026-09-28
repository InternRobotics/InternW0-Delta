#!/usr/bin/env python
"""Precompute Wan text embeddings for a pretraining LeRobot config.

This is the pretraining counterpart of tools/text_cache.py.  It
reads configs/pretrain/dataset.yaml directly, expands every source into
DatasetSpec entries, reads task metadata from local dataset roots, and writes
one cached context file per unique prompt into each spec's
configured local_text_embedding_cache_dir.

Example:
  python tools/pretrain_text_cache.py \\
    --dataset-config configs/pretrain/dataset.yaml \\
    --overwrite false
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from wam.model.backbones.wan22.helpers.loader import _load_registered_model, _resolve_configs
from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from wam.utils.config_resolvers import register_default_resolvers
from wam.utils.logging_config import get_logger, setup_logging

from wam.datasets import pretrain_lerobot_loader as pretrain_loader
from tools.data_access import load_specs
from tools.text_cache import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_MODEL_ID,
    _atomic_torch_save,
    _model_id_to_enc_id,
)

register_default_resolvers()
logger = get_logger(__name__)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _make_specs(args: argparse.Namespace) -> list[pretrain_loader.DatasetSpec]:
    _, specs = load_specs(args.dataset_config, args.source, args.dataset)
    root_contains = str(args.root_contains).strip()
    if root_contains:
        specs = [spec for spec in specs if root_contains in spec.remote_root]
    if args.max_datasets > 0:
        specs = specs[: args.max_datasets]
    return specs


def _cache_path(cache_dir: Path, prompt: str, context_len: int, enc_id: str) -> Path:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{hashed}.t5_len{int(context_len)}.{enc_id}.pt"


def _collect_prompt_targets(
    specs: list[pretrain_loader.DatasetSpec],
    args: argparse.Namespace,
) -> dict[int, dict[str, list[tuple[Path, str]]]]:
    """Return {context_len: {prompt: [(cache_dir, enc_id), ...]}}."""

    out: dict[int, dict[str, list[tuple[Path, str]]]] = defaultdict(lambda: defaultdict(list))
    total_tasks = 0
    for idx, spec in enumerate(tqdm(specs, desc="Reading mix task metadata", unit="dataset")):
        if not spec.local_text_embedding_cache_dir:
            raise ValueError(f"Spec {idx} ({spec.name}) has no local_text_embedding_cache_dir.")
        ds = pretrain_loader.PretrainLeRobotDataset(
            spec,
            num_frames=int(args.num_frames),
            action_size=int(args.action_size),
            global_sample_stride=int(args.global_sample_stride),
        )
        cache_dir = Path(spec.local_text_embedding_cache_dir).expanduser()
        context_len = int(args.context_len or spec.context_len)
        enc_id = str(args.text_encoder_id or spec.text_encoder_id)
        tasks = ds.iter_text_tasks_for_cache() if hasattr(ds, "iter_text_tasks_for_cache") else list(ds.tasks.values())
        for task in tasks:
            prompt = spec.prompt_template.format(task=str(task))
            target = (cache_dir, enc_id)
            if target not in out[context_len][prompt]:
                out[context_len][prompt].append(target)
            total_tasks += 1
        logger.info(
            "spec[%d] %s: tasks=%d cache=%s context_len=%d enc_id=%s",
            idx,
            spec.name,
            len(tasks),
            cache_dir,
            context_len,
            enc_id,
        )
    unique_prompts = sum(len(v) for v in out.values())
    unique_targets = sum(len(targets) for by_prompt in out.values() for targets in by_prompt.values())
    logger.info(
        "Collected %d task rows from %d specs -> %d unique prompts and %d prompt/cache targets.",
        total_tasks,
        len(specs),
        unique_prompts,
        unique_targets,
    )
    return out


def _filter_pending_targets(
    prompt_targets: dict[int, dict[str, list[tuple[Path, str]]]],
    overwrite: bool,
) -> tuple[dict[int, dict[str, list[tuple[Path, str]]]], dict[str, dict[str, int]]]:
    pending: dict[int, dict[str, list[tuple[Path, str]]]] = defaultdict(dict)
    counters: dict[str, dict[str, int]] = defaultdict(lambda: {"new": 0, "overwrite": 0, "skip": 0})
    for context_len, by_prompt in prompt_targets.items():
        for prompt, targets in by_prompt.items():
            kept: list[tuple[Path, str]] = []
            for cache_dir, enc_id in targets:
                path = _cache_path(cache_dir, prompt, context_len, enc_id)
                key = str(cache_dir)
                if path.exists() and not overwrite:
                    counters[key]["skip"] += 1
                    continue
                if path.exists():
                    counters[key]["overwrite"] += 1
                else:
                    counters[key]["new"] += 1
                kept.append((cache_dir, enc_id))
            if kept:
                pending[context_len][prompt] = kept
    # Plain dictionaries can be broadcast to worker processes.
    return (
        {length: dict(prompts) for length, prompts in pending.items()},
        {directory: dict(counts) for directory, counts in counters.items()},
    )


def _load_text_encoder(args: argparse.Namespace, device: str) -> tuple[Any, str]:
    model_id = str(args.model_id)
    tokenizer_model_id = str(args.tokenizer_model_id)
    redirect_common_files = bool(args.redirect_common_files)
    logger.info(
        "Loading text encoder model_id=%s tokenizer_model_id=%s device=%s",
        model_id,
        tokenizer_model_id,
        device,
    )
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    return text_encoder, tokenizer_config.path


def _encode_pending(
    pending: dict[int, dict[str, list[tuple[Path, str]]]],
    counters: dict[str, dict[str, int]],
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    if not pending:
        logger.info("All requested mix text embeddings are already cached.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    else:
        device = "cpu"
    text_encoder, tokenizer_path = _load_text_encoder(args, device)

    for context_len in sorted(pending):
        prompts = sorted(pending[context_len])
        prompts_for_rank = prompts[rank::world_size]
        if not prompts_for_rank:
            continue
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=int(context_len),
            clean="whitespace",
        )
        over_length = 0
        with tqdm(
            total=len(prompts_for_rank),
            desc=f"Encoding len={context_len} rank={rank}/{world_size}",
            unit="prompt",
            dynamic_ncols=True,
            disable=world_size > 1 and rank != 0,
        ) as pbar:
            with torch.no_grad():
                for start in range(0, len(prompts_for_rank), int(args.batch_size)):
                    batch_prompts = prompts_for_rank[start : start + int(args.batch_size)]
                    ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                    ids = ids.to(device)
                    mask = mask.to(device=device, dtype=torch.bool)
                    over_length += int(mask.all(dim=1).sum().item())
                    context = text_encoder(ids, mask)
                    for item_idx, prompt in enumerate(batch_prompts):
                        payload = {
                            "context": context[item_idx].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                            "mask": mask[item_idx].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                        }
                        for cache_dir, enc_id in pending[context_len][prompt]:
                            path = _cache_path(cache_dir, prompt, context_len, enc_id)
                            _atomic_torch_save(payload, path)
                    pbar.update(len(batch_prompts))
        if over_length:
            logger.warning(
                "context_len=%d rank=%d has %d prompts that filled the whole token mask.",
                context_len,
                rank,
                over_length,
            )

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        logger.info("Text embedding cache writes:")
        for cache_dir, stat in sorted(counters.items()):
            logger.info(
                "  %s: new=%d overwrite=%d skip=%d",
                cache_dir,
                stat["new"],
                stat["overwrite"],
                stat["skip"],
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/pretrain/dataset.yaml")
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--action-size", type=int, default=32)
    parser.add_argument("--global-sample-stride", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=0, help="0 uses each spec.context_len.")
    parser.add_argument("--text-encoder-id", default="", help="Empty uses each spec.text_encoder_id for filenames.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--redirect-common-files", type=_str_to_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", type=_str_to_bool, default=False)
    parser.add_argument(
        "--root-contains",
        default="",
        help="Only process specs whose local root contains this substring.",
    )
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-datasets", type=int, default=0, help="Limit the number of datasets; 0 selects all.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_datasets < 0:
        parser.error("--batch-size must be positive and --max-datasets nonnegative")
    return args


def main() -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    logging.getLogger("pretrain_loader").setLevel(logging.WARNING)
    args = parse_args()
    args.context_len = int(args.context_len) if int(args.context_len) > 0 else None
    args.text_encoder_id = str(args.text_encoder_id).strip() or None
    if args.text_encoder_id is None:
        model_enc_id = _model_id_to_enc_id(args.model_id)
        logger.info("Using per-spec text_encoder_id for filenames; default model enc_id would be %s.", model_enc_id)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed text precompute enabled: world_size=%d", world_size)

    specs = _make_specs(args)
    if rank == 0:
        logger.info("Expanded %d mix dataset specs.", len(specs))
    # All ranks partition the same snapshot of pending prompts. Reading cache
    # existence independently could change the partition as other ranks write.
    targets = [None]
    if rank == 0:
        prompt_targets = _collect_prompt_targets(specs, args)
        targets[0] = _filter_pending_targets(prompt_targets, overwrite=bool(args.overwrite))
    if is_distributed:
        dist.broadcast_object_list(targets, src=0)
    pending, counters = targets[0]

    pending_prompts = sum(len(by_prompt) for by_prompt in pending.values())
    pending_targets = sum(len(targets) for by_prompt in pending.values() for targets in by_prompt.values())
    if rank == 0:
        logger.info("Pending text cache: prompts=%d targets=%d overwrite=%s", pending_prompts, pending_targets, args.overwrite)
    if args.dry_run:
        if rank == 0:
            for cache_dir, stat in sorted(counters.items()):
                logger.info(
                    "[dry-run] %s: new=%d overwrite=%d skip=%d",
                    cache_dir,
                    stat["new"],
                    stat["overwrite"],
                    stat["skip"],
                )
        if is_distributed:
            dist.destroy_process_group()
        return

    _encode_pending(
        pending,
        counters,
        args,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
