import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


AUTO_EVAL_CONFIG_ROOTS = (
    "data.train.shape_meta",
    "data.train.num_frames",
    "data.train.memory_recent_frame_offset",
    "data.train.action_video_freq_ratio",
    "data.train.action_hz",
    "data.train.video_size",
    "data.train.concat_multi_camera",
    "data.train.single_canvas",
    "data.train.processor",
    "data.libero_state_8to7",
    "model.model_id",
    "model.tokenizer_model_id",
    "model.tokenizer_max_len",
    "model.memory",
    "model.understanding",
    "model.future_delta",
    "model.video_dit_config",
    "model.action_dit_config",
    "model.proprio_dim",
    "model.redirect_common_files",
    "model.mot_checkpoint_mixed_attn",
    "model.mot_attention_backend",
    "model.mot_flex_block_size",
    "model.video_scheduler",
    "model.action_scheduler",
    "EVALUATION.action_horizon",
    "EVALUATION.action_hz",
    "EVALUATION.replan_steps",
)


def resolve_eval_video_metadata(cfg: DictConfig, processor: Any) -> tuple[str, str, str]:
    """Resolve VAE/DiT and VLM metadata from the training canvas contract."""
    view_names = "|".join(
        str(item.get("key", f"view_{idx}"))
        for idx, item in enumerate(processor.shape_meta.get("images", []))
        if isinstance(item, Mapping)
    )
    if bool(cfg.data.train.get("single_canvas", False)):
        return "single", "canvas", view_names
    return (
        str(cfg.data.train.get("concat_multi_camera", "single")),
        view_names,
        view_names,
    )


def resolve_eval_device(value: Any = None) -> str:
    """Return an explicit CUDA device accepted by CUDA-backed VLM helpers."""
    resolved = str(value) if value is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if resolved == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("EVALUATION.device=cuda but CUDA is unavailable.")
        return f"cuda:{torch.cuda.current_device()}"
    return resolved


def find_training_config_for_checkpoint(ckpt: str | os.PathLike[str]) -> Optional[Path]:
    ckpt_path = Path(os.path.expanduser(os.path.expandvars(str(ckpt))))
    candidates = [parent / "config.yaml" for parent in list(ckpt_path.parents)[:6]]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _hydra_override_keys() -> set[str]:
    try:
        raw_overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        return set()
    out: set[str] = set()
    for raw in raw_overrides:
        key = str(raw).split("=", 1)[0].lstrip("+~")
        if key:
            out.add(key)
    return out


def _path_is_explicitly_overridden(path: str, override_keys: Iterable[str]) -> bool:
    for key in override_keys:
        key = str(key)
        if key == path or path.startswith(key + "."):
            return True
    return False


def _iter_leaf_updates(root: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_leaf_updates(f"{root}.{key}", child)
    else:
        yield root, value


def apply_training_config_defaults_from_checkpoint(cfg: DictConfig) -> Optional[Path]:
    if not bool(OmegaConf.select(cfg, "EVALUATION.auto_load_train_config", default=True)):
        return None
    ckpt = cfg.get("ckpt")
    if ckpt is None:
        return None
    config_path = find_training_config_for_checkpoint(str(ckpt))
    if config_path is None:
        return None

    train_cfg = OmegaConf.load(config_path)
    explicit_keys = _hydra_override_keys()
    applied: list[str] = []
    for root in AUTO_EVAL_CONFIG_ROOTS:
        value = OmegaConf.select(train_cfg, root, default=None)
        if value is None:
            continue
        plain = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for path, leaf_value in _iter_leaf_updates(root, plain):
            if _path_is_explicitly_overridden(path, explicit_keys):
                continue
            OmegaConf.update(cfg, path, leaf_value, merge=False, force_add=True)
            applied.append(path)

    if applied:
        logging.info("Loaded %d eval config defaults from training config: %s", len(applied), config_path)
    return config_path


def apply_eval_understanding_interval(cfg: DictConfig) -> None:
    value = OmegaConf.select(cfg, "EVALUATION.understanding_recompute_interval_chunks", default="model")
    if value is None or str(value).strip().lower() == "model":
        return
    interval = max(1, int(value))
    OmegaConf.update(
        cfg,
        "model.understanding.interval_chunks",
        interval,
        merge=False,
        force_add=True,
    )
    logging.info("Using eval understanding recompute interval: every %d chunk(s).", interval)
