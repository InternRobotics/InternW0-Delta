import json
import logging
import math
import os
import subprocess
import sys
import time
import uuid
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.libero.bootstrap import setup_libero_paths

setup_libero_paths()

from eval.libero.eval_config import (
    apply_training_config_defaults_from_checkpoint,
    resolve_eval_device,
    resolve_eval_video_metadata,
)
from eval.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from wam.datasets.lerobot.processors.wam_processor import WAMProcessor
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wam.training.checkpoint import load_wam_checkpoint
from wam.inference.online_action_policy import infer_online_action_chunk
from wam.memory_history import resolve_recent_history_index
from wam.model.modules.conditioning.video_rope import (
    sequence_start_video_rope_time_ids,
)
from wam.utils.pytorch_utils import set_global_seed
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from eval.libero.action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SubprocessLiberoEnv:
    def __init__(
        self,
        *,
        task_suite_name: str,
        task_id: int,
        seed,
        render_gpu_device_id: int,
        env_worker_cuda_visible_devices: Optional[str] = "",
    ) -> None:
        self._address = f"/tmp/wam_libero_env_{os.getpid()}_{uuid.uuid4().hex}.sock"
        worker_script = project_root / "eval" / "libero" / "libero_env_worker_process.py"
        env = os.environ.copy()
        if env_worker_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(env_worker_cuda_visible_devices)
        cmd = [
            sys.executable,
            str(worker_script),
            "--address",
            self._address,
            "--task-suite-name",
            str(task_suite_name),
            "--task-id",
            str(int(task_id)),
            "--seed",
            "None" if seed is None else str(seed),
            "--render-gpu-device-id",
            str(int(render_gpu_device_id)),
            "--resolution",
            str(int(LIBERO_ENV_RESOLUTION)),
        ]
        self._process = subprocess.Popen(cmd, env=env)
        self._conn = None
        deadline = time.time() + 60.0
        last_error: Optional[BaseException] = None
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"LIBERO external env worker exited during startup, exitcode={self._process.returncode}."
                )
            try:
                self._conn = Client(self._address, family="AF_UNIX")
                break
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_error = exc
                time.sleep(0.05)
        if self._conn is None:
            self._process.terminate()
            raise RuntimeError(f"Timed out connecting to LIBERO external env worker: {last_error}") from last_error
        status, payload = self._recv_raw()
        if status != "ready":
            raise RuntimeError(f"LIBERO external env worker failed during startup:\n{payload}")
        self.task_description = payload

    def _worker_exit_message(self) -> str:
        return (
            "LIBERO external env worker exited unexpectedly, "
            f"exitcode={self._process.poll()}, address={self._address}."
        )

    def _recv_raw(self):
        try:
            return self._conn.recv()
        except EOFError as exc:
            raise RuntimeError(self._worker_exit_message()) from exc

    def _call(self, command: str, payload=None):
        self._conn.send((command, payload))
        status, result = self._recv_raw()
        if status != "ok":
            raise RuntimeError(f"LIBERO external env worker command {command!r} failed:\n{result}")
        return result

    def reset(self):
        return self._call("reset")

    def set_init_state(self, initial_state):
        return self._call("set_init_state", initial_state)

    def step(self, action):
        return self._call("step", action)

    def close(self) -> None:
        if getattr(self, "_conn", None) is None:
            return
        try:
            if self._process.poll() is None:
                self._call("close")
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        try:
            os.unlink(self._address)
        except OSError:
            pass
        self._conn = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    return resolve_eval_device(cfg.EVALUATION.get("device"))


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    load_wam_checkpoint(model, ckpt)
    logging.info("Loaded checkpoint: %s", ckpt)


def _preprocess_eval_camera_images(
    imgs: dict[str, np.ndarray],
    processor: WAMProcessor,
) -> list[torch.Tensor]:
    """Apply the exact validation image transforms used by post-training."""
    raw_images: dict[str, torch.Tensor] = {}
    for meta in processor.shape_meta["images"]:
        key = str(meta["key"])
        if key not in imgs:
            raise KeyError(f"Missing simulator image for camera {key!r}.")
        image = np.ascontiguousarray(imgs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Expected HWC RGB image for camera {key!r}, got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 image for camera {key!r}, got {image.dtype}."
            )
        raw_images[key] = (
            torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0)
        )

    processed = processor._transform_images({"images": raw_images})
    return [image[0] for image in processed]


def _normalize_proprio(
    proprio: dict[str, np.ndarray],
    processor: WAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    expected_keys = [str(meta["key"]) for meta in state_meta]
    if set(proprio) != set(expected_keys):
        raise ValueError(
            f"Simulator state keys mismatch: got {sorted(proprio)}, "
            f"expected {sorted(expected_keys)}."
        )
    state_batch = {
        "state": {
            key: torch.as_tensor(proprio[key], dtype=torch.float32).unsqueeze(0)
            for key in expected_keys
        }
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    state_batch = processor.action_state_merger.forward(state_batch)
    return state_batch["state"]


def _static_dimension_is_pad(processor: WAMProcessor, field: str) -> torch.Tensor:
    if field not in {"action", "state"}:
        raise ValueError(f"Unsupported merged field: {field}")
    merger = processor.action_state_merger
    meta = processor.shape_meta[field]
    source_dim = sum(int(item["shape"]) for item in meta)
    target_dim = getattr(merger, f"{field}_target_dim")
    target_dim = source_dim if target_dim is None else int(target_dim)
    target_slices = getattr(merger, f"{field}_target_slices")
    mask = torch.ones(target_dim, dtype=torch.bool)
    if target_slices is None:
        mask[:source_dim] = False
    else:
        for entry in target_slices:
            start, end = (int(value) for value in entry["target_slice"])
            mask[start:end] = False
    return mask




def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: WAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    processed_images = _preprocess_eval_camera_images(imgs, processor)
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = processed_images[0]
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary, wrist = processed_images
        if concatenation == "latent_horizontal":
            if (primary_h, primary_w) != (wrist_h, wrist_w):
                raise ValueError(
                    "`concat_multi_camera='latent_horizontal'` requires matching per-camera shapes, "
                    f"got primary={(primary_h, primary_w)} and wrist={(wrist_h, wrist_w)}."
                )
            actual_h, actual_w = primary_h, primary_w
            expected_h, expected_w = int(height), int(width)
            image_shapes = [meta["shape"] for meta in image_meta]
            assert actual_h == expected_h and actual_w == expected_w, (
                "Input per-camera image size mismatch after resize: "
                f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
                f"from data.train.video_size={[expected_h, expected_w]}; "
                f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
            )
            x = torch.stack(
                [primary, wrist],
                dim=0,
            )
            x = (x * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=dtype)
            proprio = _normalize_proprio(
                _extract_sim_state_for_processor(obs, processor), processor
            )
            return x, proprio, imgs
        if concatenation == "horizontal":
            rgb = torch.cat([primary, wrist], dim=-1)
        elif concatenation == "vertical":
            rgb = torch.cat([primary, wrist], dim=-2)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[-2]), int(rgb.shape[-1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = (rgb * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=dtype)

    proprio = _normalize_proprio(
        _extract_sim_state_for_processor(obs, processor), processor
    )

    return x, proprio, imgs


def _vlm_current_images(imgs: dict, processor: WAMProcessor) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(imgs[meta["key"]])).permute(2, 0, 1)
            for meta in processor.shape_meta["images"]
        ],
        dim=0,
    )


def _extract_sim_state(obs: dict) -> dict[str, np.ndarray]:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    return {
        "joint": np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
        "ee": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]))
        ).astype(np.float32),
        "gripper": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    }


def _extract_sim_state_for_processor(
    obs: dict,
    processor: WAMProcessor,
) -> dict[str, np.ndarray]:
    expected = [str(meta["key"]) for meta in processor.shape_meta["state"]]
    split = _extract_sim_state(obs)
    if expected == ["default"]:
        return {
            "default": np.concatenate([split["ee"], split["gripper"]]).astype(
                np.float32
            )
        }
    missing = [key for key in expected if key not in split]
    if missing:
        raise ValueError(f"Unsupported LIBERO simulator state keys: {missing}")
    return {key: split[key] for key in expected}


def _denormalize_action(action: torch.Tensor, processor: WAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action = action.to(dtype=torch.float32, device="cpu")
    merger = processor.action_state_merger
    dim_is_pad = _static_dimension_is_pad(processor, "action")
    if int(action.shape[-1]) != int(dim_is_pad.numel()):
        raise ValueError(
            f"Model action dim {action.shape[-1]} does not match processor output "
            f"dim {dim_is_pad.numel()}."
        )
    action = action.masked_fill(dim_is_pad.view(1, 1, -1), 0.0)

    action_meta = processor.shape_meta["action"]
    source_dim = sum(int(item["shape"]) for item in action_meta)
    target_slices = merger.action_target_slices
    if target_slices is None:
        merged = action[..., :source_dim]
    else:
        merged = action.new_zeros((*action.shape[:-1], source_dim))
        for entry in target_slices:
            src_start, src_end = (int(value) for value in entry["source_slice"])
            tgt_start, tgt_end = (int(value) for value in entry["target_slice"])
            merged[..., src_start:src_end] = action[..., tgt_start:tgt_end]

    normalized: dict[str, torch.Tensor] = {}
    offset = 0
    for meta in action_meta:
        width = int(meta["shape"])
        normalized[str(meta["key"])] = merged[..., offset : offset + width]
        offset += width
    denormalized = {
        key: processor.normalizer.normalizers["action"][key].backward(value)
        for key, value in normalized.items()
    }
    denorm = torch.cat(
        [denormalized[str(meta["key"])] for meta in action_meta], dim=-1
    )
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _cfg_memory_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "model.memory.enabled", default=False))


def _resolve_replan_steps(cfg: DictConfig, action_horizon: int) -> int:
    value = cfg.EVALUATION.get("replan_steps", None)
    if value is None:
        return 10
    return int(value)


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(
    pred_video: list[Image.Image],
    cfg: DictConfig,
    *,
    replan_steps: int,
) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("Predicted future video is empty.")

    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig, *, replan_steps: int) -> list[int]:
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _uses_ar_history(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "ar_history_enabled", False))


def _uses_memory(model: torch.nn.Module) -> bool:
    return bool(
        int(getattr(model, "memory_video_anchor_frames", 0) or 0) > 0
        or int(getattr(model, "memory_video_recent_frames", 0) or 0) > 0
    )


def _ar_history_flags(model: torch.nn.Module) -> tuple[bool, bool, int, int]:
    if not _uses_ar_history(model):
        return False, False, 0, 0
    cfg = dict(getattr(model, "ar_history_cfg", {}) or {})
    use_action = bool(getattr(model, "use_ar_action_history", cfg.get("use_action_history", True)))
    use_visual = bool(getattr(model, "use_ar_visual_history", cfg.get("use_visual_history", False)))
    num_actions = int(getattr(model, "max_history_actions", cfg.get("num_history_actions", 20)))
    num_frames = int(getattr(model, "max_history_frames", cfg.get("num_history_frames", 2)))
    return use_action, use_visual, num_actions, num_frames


def _build_action_history_tensors(
    history: list[torch.Tensor],
    *,
    target_len: int,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"AR action history target_len must be positive, got {target_len}")
    actions = torch.zeros((target_len, action_dim), dtype=torch.float32)
    is_pad = torch.ones((target_len,), dtype=torch.bool)
    take = history[-target_len:]
    if take:
        stacked = torch.stack([x.to(dtype=torch.float32, device="cpu") for x in take], dim=0)
        if stacked.shape[-1] != action_dim:
            raise ValueError(
                f"AR action history dim mismatch: got {stacked.shape[-1]}, expected {action_dim}"
            )
        actions[-len(take) :] = stacked
        is_pad[-len(take) :] = False
    return actions, is_pad


def _build_visual_history_tensors(
    history: list[torch.Tensor],
    *,
    target_len: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"AR visual history target_len must be positive, got {target_len}")
    is_pad = torch.ones((target_len,), dtype=torch.bool)
    take = history[-target_len:]
    if take and take[-1].ndim == 4:
        num_cameras = int(take[-1].shape[0])
        frames = torch.zeros((num_cameras, 3, target_len, height, width), dtype=torch.float32)
    else:
        frames = torch.zeros((3, target_len, height, width), dtype=torch.float32)
    if take:
        stacked = torch.stack([x.to(dtype=torch.float32, device="cpu") for x in take], dim=0)
        if stacked.ndim == 5:
            expected_shape = (int(stacked.shape[1]), 3, height, width)
            if stacked.shape[1:] != expected_shape:
                raise ValueError(
                    "AR visual history shape mismatch: "
                    f"got {tuple(stacked.shape[1:])}, expected {expected_shape}"
                )
            frames[:, :, -len(take) :] = stacked.permute(1, 2, 0, 3, 4).contiguous()
        elif stacked.shape[1:] != (3, height, width):
            raise ValueError(
                "AR visual history shape mismatch: "
                f"got {tuple(stacked.shape[1:])}, expected {(3, height, width)}"
            )
        else:
            frames[:, -len(take) :] = stacked.permute(1, 0, 2, 3).contiguous()
        is_pad[-len(take) :] = False
    return frames, is_pad


def _build_memory_video_tensor(
    history: list[torch.Tensor],
    *,
    target_len: int,
    height: int,
    width: int,
    from_start: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_len = int(target_len)
    is_pad = torch.ones((target_len,), dtype=torch.bool)
    if target_len <= 0:
        frames = torch.zeros((3, target_len, height, width), dtype=torch.float32)
        return frames, is_pad
    take = history[:target_len] if from_start else history[-target_len:]
    if take and take[-1].ndim == 4:
        num_cameras = int(take[-1].shape[0])
        frames = torch.zeros((num_cameras, 3, target_len, height, width), dtype=torch.float32)
    else:
        frames = torch.zeros((3, target_len, height, width), dtype=torch.float32)
    if not take:
        return frames, is_pad
    stacked = torch.stack([x.to(dtype=torch.float32, device="cpu") for x in take], dim=0)
    if stacked.ndim == 5:
        expected_shape = (int(stacked.shape[1]), 3, height, width)
        if stacked.shape[1:] != expected_shape:
            raise ValueError(
                "Memory video shape mismatch: "
                f"got {tuple(stacked.shape[1:])}, expected {expected_shape}"
            )
        if from_start:
            frames[:, :, : len(take)] = stacked.permute(1, 2, 0, 3, 4).contiguous()
            is_pad[: len(take)] = False
        else:
            frames[:, :, -len(take) :] = stacked.permute(1, 2, 0, 3, 4).contiguous()
            is_pad[-len(take) :] = False
        return frames, is_pad
    if stacked.shape[1:] != (3, height, width):
        raise ValueError(
            "Memory video shape mismatch: "
            f"got {tuple(stacked.shape[1:])}, expected {(3, height, width)}"
        )
    if from_start:
        frames[:, : len(take)] = stacked.permute(1, 0, 2, 3).contiguous()
        is_pad[: len(take)] = False
    else:
        frames[:, -len(take) :] = stacked.permute(1, 0, 2, 3).contiguous()
        is_pad[-len(take) :] = False
    return frames, is_pad


def _build_memory_proprio_tensor(
    history: list[torch.Tensor],
    *,
    target_len: int,
    proprio_dim: int,
    from_start: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_len = int(target_len)
    proprio_dim = int(proprio_dim)
    states = torch.zeros((target_len, proprio_dim), dtype=torch.float32)
    is_pad = torch.ones((target_len,), dtype=torch.bool)
    if target_len <= 0 or proprio_dim <= 0:
        return states, is_pad
    take = history[:target_len] if from_start else history[-target_len:]
    if not take:
        return states, is_pad
    stacked = torch.stack([x.to(dtype=torch.float32, device="cpu") for x in take], dim=0)
    if stacked.shape[-1] != proprio_dim:
        raise ValueError(f"Memory proprio dim mismatch: got {stacked.shape[-1]}, expected {proprio_dim}")
    if from_start:
        states[: len(take)] = stacked
        is_pad[: len(take)] = False
    else:
        states[-len(take) :] = stacked
        is_pad[-len(take) :] = False
    return states, is_pad


def _capture_model_frame(
    obs: dict,
    *,
    cfg: DictConfig,
    processor: WAMProcessor,
    width: int,
    height: int,
    model_device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    image, _, _ = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=width,
        height=height,
        device=model_device,
        dtype=dtype,
    )
    return image[0].detach().to(device="cpu", dtype=torch.float32)


def _capture_model_frame_and_proprio(
    obs: dict,
    *,
    cfg: DictConfig,
    processor: WAMProcessor,
    width: int,
    height: int,
    model_device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, proprio, _ = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=width,
        height=height,
        device=model_device,
        dtype=dtype,
    )
    return (
        image[0].detach().to(device="cpu", dtype=torch.float32),
        proprio[0].detach().to(device="cpu", dtype=torch.float32),
    )


def _build_memory_inputs_for_eval(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    action_history: list[torch.Tensor],
    visual_history: list[torch.Tensor],
    proprio_history: list[torch.Tensor],
    action_horizon: int,
    action_dim: int,
    height: int,
    width: int,
    memory_chunk_index: int = 0,
    current_memory_chunks: Optional[int] = None,
    include_current_recent: bool = False,
    episode_step: Optional[int] = None,
) -> Optional[dict[str, torch.Tensor]]:
    if not _uses_memory(model):
        return None
    del current_memory_chunks
    memory_chunk_index = max(0, int(memory_chunk_index))

    video_anchor_frames = int(getattr(model, "memory_video_anchor_frames", 0) or 0)
    video_recent_frames = int(getattr(model, "memory_video_recent_frames", 0) or 0)
    proprio_dim = int(getattr(model, "proprio_dim", 0) or 0)
    memory_inputs: dict[str, torch.Tensor] = {}

    if video_anchor_frames > 0:
        sampled_anchor = []
        sampled_anchor_proprio = []
        for anchor_idx in range(video_anchor_frames):
            raw_idx = anchor_idx
            if raw_idx < len(visual_history):
                sampled_anchor.append(visual_history[raw_idx])
                if raw_idx < len(proprio_history):
                    sampled_anchor_proprio.append(proprio_history[raw_idx])
            elif sampled_anchor:
                sampled_anchor.append(sampled_anchor[-1])
                if sampled_anchor_proprio:
                    sampled_anchor_proprio.append(sampled_anchor_proprio[-1])
            elif visual_history:
                sampled_anchor.append(visual_history[0])
                if proprio_history:
                    sampled_anchor_proprio.append(proprio_history[0])
        frames, is_pad = _build_memory_video_tensor(
            sampled_anchor,
            target_len=video_anchor_frames,
            height=height,
            width=width,
            from_start=True,
        )
        memory_inputs["memory_video_anchor"] = frames
        memory_inputs["memory_video_anchor_is_pad"] = is_pad
        memory_inputs["memory_video_anchor_frame_ids"] = (
            sequence_start_video_rope_time_ids(
                1,
                video_anchor_frames,
                device=torch.device("cpu"),
            )[0]
        )
        if proprio_dim > 0:
            states, state_is_pad = _build_memory_proprio_tensor(
                sampled_anchor_proprio,
                target_len=video_anchor_frames,
                proprio_dim=proprio_dim,
                from_start=True,
            )
            memory_inputs["memory_video_anchor_proprio"] = states
            memory_inputs["memory_video_anchor_proprio_is_pad"] = state_is_pad
    if (
        include_current_recent
        and video_recent_frames > 0
        and visual_history
        and "memory_video_recent" not in memory_inputs
    ):
        recent_idx, recent_is_real = resolve_recent_history_index(
            len(visual_history),
            int(action_horizon),
            episode_step=episode_step,
        )
        recent_sampled_history = [visual_history[recent_idx]]
        recent_sampled_proprio = [
            proprio_history[min(recent_idx, len(proprio_history) - 1)]
        ] if proprio_history else []
        recent_frames, recent_is_pad = _build_memory_video_tensor(
            recent_sampled_history,
            target_len=video_recent_frames,
            height=height,
            width=width,
            from_start=False,
        )
        if not recent_is_real:
            recent_is_pad = torch.ones_like(recent_is_pad, dtype=torch.bool)
        memory_inputs["memory_video_recent"] = recent_frames
        memory_inputs["memory_video_recent_is_pad"] = recent_is_pad
        latent_frames_per_chunk = int(
            getattr(model, "memory_video_latents_per_chunk", 1) or 1
        )
        vae_temporal_factor = int(
            getattr(model, "memory_video_vae_temporal_factor", 1) or 1
        )
        recent_latent_start = (
            max(0, memory_chunk_index - 1) * latent_frames_per_chunk
        )
        memory_inputs["memory_video_recent_frame_ids"] = torch.arange(
            recent_latent_start,
            recent_latent_start + latent_frames_per_chunk,
            dtype=torch.long,
        ).repeat_interleave(vae_temporal_factor)
        if proprio_dim > 0:
            recent_states, recent_state_is_pad = _build_memory_proprio_tensor(
                recent_sampled_proprio,
                target_len=video_recent_frames,
                proprio_dim=proprio_dim,
                from_start=False,
            )
            if not recent_is_real:
                recent_state_is_pad = torch.ones_like(
                    recent_state_is_pad, dtype=torch.bool
                )
            memory_inputs["memory_video_recent_proprio"] = recent_states
            memory_inputs["memory_video_recent_proprio_is_pad"] = recent_state_is_pad
    return memory_inputs


def _trim_memory_history(
    history: list[torch.Tensor],
    *,
    anchor_len: int,
    recent_len: int,
) -> list[torch.Tensor]:
    anchor_len = max(0, int(anchor_len))
    recent_len = max(0, int(recent_len))
    if len(history) <= anchor_len + recent_len:
        return history
    return history[:anchor_len] + history[-recent_len:]



def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: WAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    replan_steps: int,
    history_actions: Optional[torch.Tensor] = None,
    history_action_is_pad: Optional[torch.Tensor] = None,
    history_video: Optional[torch.Tensor] = None,
    history_image_is_pad: Optional[torch.Tensor] = None,
    memory_inputs: Optional[dict[str, torch.Tensor]] = None,
    memory_chunk_index: int = 0,
    online_action_history: Optional[list[torch.Tensor]] = None,
    online_visual_history: Optional[list[torch.Tensor]] = None,
    online_proprio_history: Optional[list[torch.Tensor]] = None,
    online_anchor_visual_history: Optional[list[torch.Tensor]] = None,
    online_anchor_proprio_history: Optional[list[torch.Tensor]] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], torch.Tensor, torch.Tensor]:
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    current_frame = image[0].detach().to(device="cpu", dtype=torch.float32)
    if not _uses_memory(model):
        raise ValueError("Online eval now requires memory AR inference.")

    online_action_history = list(online_action_history or [])
    effective_memory_chunk_index = max(
        int(memory_chunk_index), len(online_action_history) // max(1, int(replan_steps)))
    if memory_inputs is None:
        memory_inputs = _build_memory_inputs_for_eval(
            model=model,
            cfg=cfg,
            action_history=online_action_history,
            visual_history=list(online_visual_history or []),
            proprio_history=list(online_proprio_history or []),
            action_horizon=int(action_horizon),
            action_dim=int(getattr(model.action_expert, "action_dim")),
            height=input_h,
            width=input_w,
            memory_chunk_index=effective_memory_chunk_index,
            include_current_recent=True,
            episode_step=len(online_action_history),
        )
        memory_chunk_index = effective_memory_chunk_index

    t5_prompt = DEFAULT_PROMPT.format(task=task_description)
    num_inference_steps = int(
        cfg.EVALUATION.get(
            "num_inference_steps", cfg.get("eval_num_inference_steps", 20)
        )
    )
    sigma_shift_cfg = cfg.EVALUATION.get("sigma_shift", None)
    sigma_shift = None if sigma_shift_cfg is None else float(sigma_shift_cfg)
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "Frame-level online inference is action-only; "
            "set EVALUATION.visualize_future_video=false."
        )
    video_layout, video_view_names, vlm_view_names = resolve_eval_video_metadata(
        cfg, processor
    )
    pred = infer_online_action_chunk(
        model,
        prompt=t5_prompt,
        input_image=image,
        vlm_current_images=_vlm_current_images(imgs, processor).to(model.device),
        vlm_view_names=vlm_view_names,
        action_horizon=int(action_horizon),
        proprio=proprio,
        action_dim_is_pad=_static_dimension_is_pad(processor, "action"),
        memory_inputs=memory_inputs,
        understanding_prompt=t5_prompt,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        rand_device="cpu",
        memory_chunk_index=int(memory_chunk_index),
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    model_action_chunk = pred["action"].detach().to(device="cpu", dtype=torch.float32)
    action_chunk = _denormalize_action(model_action_chunk, processor=processor)[0]
    predicted_future_frames = None
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        video = pred.get("video")
        if isinstance(video, list):
            predicted_future_frames = video
        elif isinstance(video, torch.Tensor):
            video_cpu = video[0].detach().to(device="cpu", dtype=torch.float32).clamp(-1, 1)
            video_cpu = ((video_cpu + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            predicted_future_frames = [
                Image.fromarray(video_cpu[:, idx].permute(1, 2, 0).numpy())
                for idx in range(int(video_cpu.shape[1]))
            ]
    return action_chunk, imgs, predicted_future_frames, model_action_chunk, current_frame

def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: WAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = _resolve_replan_steps(cfg, action_horizon)
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg, replan_steps=replan_steps)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    use_ar_action_history, use_ar_visual_history, ar_num_actions, ar_num_frames = _ar_history_flags(model)
    use_memory = _uses_memory(model)
    ar_effective_num_actions = ar_num_actions
    if use_ar_action_history:
        ar_num_chunks = int(getattr(model, "num_history_chunks", 0) or 0)
        if ar_num_chunks > 0:
            ar_effective_num_actions = max(1, min(ar_num_actions, ar_num_chunks * replan_steps))
    action_history: list[torch.Tensor] = []
    visual_history: list[torch.Tensor] = []
    proprio_history: list[torch.Tensor] = []
    anchor_visual_history: list[torch.Tensor] = []
    anchor_proprio_history: list[torch.Tensor] = []
    pending_model_actions: list[torch.Tensor] = []
    action_dim = int(getattr(getattr(model, "action_expert", None), "action_dim", 0))
    if (use_ar_action_history or use_memory) and action_dim <= 0:
        raise ValueError("Action history is enabled, but model.action_expert.action_dim is unavailable.")
    def append_memory_observation(
        current_obs: dict,
        *,
        to_rollout: bool = True,
        to_anchor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not use_memory:
            raise RuntimeError("append_memory_observation should only be called when memory is enabled.")
        frame, proprio = _capture_model_frame_and_proprio(
            current_obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
            model_device=model_device,
            dtype=model.torch_dtype,
        )
        if to_rollout:
            visual_history.append(frame)
            proprio_history.append(proprio)
        if to_anchor:
            anchor_len = int(
                getattr(model, "memory_video_anchor_frames", 1) or 1
            )
            recent_len = int(action_horizon) + 1
            visual_history[:] = _trim_memory_history(
                visual_history, anchor_len=anchor_len, recent_len=recent_len
            )
            proprio_history[:] = _trim_memory_history(
                proprio_history, anchor_len=anchor_len, recent_len=recent_len
            )
            anchor_visual_history.append(frame)
            anchor_proprio_history.append(proprio)
        return frame, proprio

    def ensure_first_rollout_observation(current_obs: dict) -> None:
        if not use_memory:
            return
        if len(visual_history) == 0:
            frame, proprio = append_memory_observation(current_obs, to_rollout=True, to_anchor=False)
            if len(anchor_visual_history) == 0:
                anchor_visual_history.append(frame)
                anchor_proprio_history.append(proprio)

    t = 0
    replan_count = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            history_kwargs: dict[str, torch.Tensor] = {}
            if use_ar_action_history:
                history_actions, history_action_is_pad = _build_action_history_tensors(
                    action_history,
                    target_len=ar_effective_num_actions,
                    action_dim=action_dim,
                )
                history_kwargs["history_actions"] = history_actions
                history_kwargs["history_action_is_pad"] = history_action_is_pad
            if use_ar_visual_history:
                history_video, history_image_is_pad = _build_visual_history_tensors(
                    visual_history,
                    target_len=ar_num_frames,
                    height=input_h,
                    width=input_w,
                )
                history_kwargs["history_video"] = history_video
                history_kwargs["history_image_is_pad"] = history_image_is_pad

            if use_memory and len(visual_history) == 0:
                ensure_first_rollout_observation(obs)

            action_chunk, imgs, predicted_future_frames, model_action_chunk, current_model_image = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                replan_steps=replan_steps,
                memory_chunk_index=replan_count,
                online_action_history=action_history,
                online_visual_history=visual_history,
                online_proprio_history=proprio_history,
                online_anchor_visual_history=anchor_visual_history,
                online_anchor_proprio_history=anchor_proprio_history,
                **history_kwargs,
            )
            if use_ar_visual_history:
                visual_history.append(current_model_image)
                visual_history = visual_history[-ar_num_frames:]
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            replan_count += 1
            n_exec = replan_steps
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + n_exec)]
            else:
                pending_actions = action_chunk[:n_exec].tolist()
            if use_ar_action_history or use_memory:
                num_pending = len(pending_actions)
                pending_model_actions = [
                    model_action_chunk[min(step_idx, model_action_chunk.shape[0] - 1)].clone()
                    for step_idx in range(num_pending)
                ]
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        next_action = pending_actions.pop(0)
        next_model_action = pending_model_actions.pop(0) if pending_model_actions else None
        obs, _, done, _ = env.step(next_action)
        if (use_ar_action_history or use_memory) and next_model_action is not None:
            action_history.append(next_model_action.detach().to(device="cpu", dtype=torch.float32))
            if use_ar_action_history and not use_memory:
                action_history = action_history[-ar_num_actions:]
        if use_memory and not done:
            append_memory_observation(obs, to_rollout=True, to_anchor=False)
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: WAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    render_gpu_device_id = int(cfg.EVALUATION.get("render_gpu_device_id", -1))
    if bool(cfg.EVALUATION.get("subprocess_env", True)):
        env = SubprocessLiberoEnv(
            task_suite_name=str(cfg.EVALUATION.task_suite_name),
            task_id=int(cfg.EVALUATION.task_id),
            seed=cfg.get("seed"),
            render_gpu_device_id=render_gpu_device_id,
            env_worker_cuda_visible_devices=cfg.EVALUATION.get("env_worker_cuda_visible_devices", ""),
        )
        task_description = env.task_description
    else:
        env, task_description = get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            cfg.get("seed"),
            render_gpu_device_id=render_gpu_device_id,
        )
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    try:
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
            if visualize_future_video:
                results["episode_future_video_psnr"].append(episode_mean_psnr)

            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
            if visualize_future_video:
                if len(predicted_future_video_clips) == 0:
                    logging.warning(
                        "No predicted future frames collected for task %s trial %s.",
                        cfg.EVALUATION.task_id,
                        trial_idx,
                    )
                else:
                    all_gt_frames = []
                    all_pred_frames = []
                    for clip in predicted_future_video_clips:
                        all_gt_frames.extend(clip["gt_frames"])
                        all_pred_frames.extend(clip["pred_frames"])
                        save_prediction_video(
                            predicted_video_dir,
                            clip["gt_frames"],
                            clip["pred_frames"],
                            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                            clip["replan_idx"],
                            success=success,
                            task_description=task_description,
                        )
                    save_prediction_video(
                        predicted_video_dir,
                        all_gt_frames,
                        all_pred_frames,
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        "all",
                        success=success,
                        task_description=task_description,
                    )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    apply_training_config_defaults_from_checkpoint(cfg)
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: WAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
