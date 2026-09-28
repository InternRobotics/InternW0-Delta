"""Stateful RoboDojo policy using the common InternW0-delta inference runtime."""

from __future__ import annotations

from typing import Any

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from eval.robodojo.bootstrap import setup_policy_paths

setup_policy_paths()

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def instruction(obs: dict[str, Any], fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    return str(value).strip() if value is not None and str(value).strip() else fallback


def rgb(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _camera_tensor(image: Any) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float32)
        .div(255.0)
    )


def build_training_exact_canvas(
    head: Any,
    left: Any,
    right: Any,
    *,
    processor_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> torch.Tensor:
    """Reproduce training's resize-only three-camera T canvas, without crop."""

    processor_h, processor_w = map(int, processor_size)
    canvas_h, canvas_w = map(int, canvas_size)
    cameras = [
        transforms_F.resize(
            _camera_tensor(image),
            size=[processor_h, processor_w],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        for image in (head, left, right)
    ]
    top_h = (canvas_h * 2) // 3
    bottom_h = canvas_h - top_h
    left_w = canvas_w // 2
    targets = ((top_h, canvas_w), (bottom_h, left_w), (bottom_h, canvas_w - left_w))
    packed = [
        transforms_F.resize(
            camera,
            size=list(size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        for camera, size in zip(cameras, targets, strict=True)
    ]
    canvas = torch.cat([packed[0], torch.cat(packed[1:], dim=-1)], dim=-2)
    if tuple(canvas.shape) != (1, 3, canvas_h, canvas_w):
        raise RuntimeError(f"Unexpected canvas shape: {tuple(canvas.shape)}")
    return canvas.mul(2.0).sub(1.0)


def validate_checkpoint(model, path) -> None:
    """Require every evaluation component and tensor shape to match the model."""
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    for name in ("mot", "proprio_encoder", "action_proprio_encoder", "understanding"):
        module = getattr(model, name, None)
        actual = (module.adapter_state_dict() if name == "understanding"
                  else module.state_dict()) if module is not None else {}
        saved = payload.get(name, {})
        if saved.keys() != actual.keys():
            raise ValueError(f"Checkpoint component keys do not match: {name}")
        if any(saved[key].shape != actual[key].shape for key in actual):
            raise ValueError(f"Checkpoint component shapes do not match: {name}")
    print(f"Checkpoint loaded: step={payload.get('step')}", flush=True)


class Model(ModelTemplate):
    """Single-environment InternW0-delta adapter; subclasses provide model loading."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint")
        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type") or "arx_x5")
        if self.action_type != "joint":
            raise ValueError("This checkpoint was trained for joint actions")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if list(self.robot_action_dim_info.get("arm_dim", [])) != [6, 6]:
            raise ValueError(f"Checkpoint requires dual ARX-X5 arms: {self.robot_action_dim_info}")
        if list(self.robot_action_dim_info.get("ee_dim", [])) != [1, 1]:
            raise ValueError(f"Checkpoint requires one gripper scalar per arm: {self.robot_action_dim_info}")

        self.default_instruction = str(
            self.model_cfg.get("default_instruction") or "follow the instruction"
        )
        self.last_obs: dict[str, Any] | None = None
        self.last_instruction = self.default_instruction
        self.session = None
        self.runtime = None
        self.action_horizon = int(self.model_cfg.get("action_horizon") or 32)
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 10)
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                f"replan_steps must be in [1, {self.action_horizon}], got {self.replan_steps}"
            )
        self._load_real_policy()

    def _load_real_policy(self) -> None:
        cfg = self.model_cfg["config"]
        if not torch.cuda.is_available():
            raise RuntimeError("RoboDojo policy evaluation requires CUDA")
        from eval.robotwin.policy import (
            RobotWinPolicySession, build_robotwin_runtime_and_session_config,
        )
        self.runtime, session_cfg = build_robotwin_runtime_and_session_config({
            "resolved_config": cfg,
            "ckpt_setting": str(Path(cfg.ckpt).expanduser().resolve()),
            "dataset_stats_path": str(Path(cfg.EVALUATION.dataset_stats_path).expanduser().resolve()),
            "seed": int(cfg.seed),
            "device": "cuda:0",
            "mixed_precision": str(cfg.mixed_precision),
        })
        validate_checkpoint(self.runtime.model, cfg.ckpt)
        image_meta = list(cfg.data.train.processor.shape_meta.images)
        if [str(item.key) for item in image_meta] != [
            "cam_high", "cam_left_wrist", "cam_right_wrist"
        ]:
            raise ValueError("Expected head, left wrist, right wrist camera order")
        sizes = [tuple(map(int, item.shape[-2:])) for item in image_meta]
        if len(set(sizes)) != 1:
            raise ValueError("Camera processor sizes must agree")
        processor_size = sizes[0]

        class RoboDojoSession(RobotWinPolicySession):
            def _build_robotwin_image_tensor(self, observation):
                images = observation["observation"]
                canvas = build_training_exact_canvas(
                    images["head_camera"]["rgb"],
                    images["left_camera"]["rgb"],
                    images["right_camera"]["rgb"],
                    processor_size=processor_size,
                    canvas_size=(self.video_height, self.video_width),
                )
                return canvas.to(device=self.model.device, dtype=self.model.torch_dtype)

        self.session = RoboDojoSession(self.runtime, session_cfg)
        print(f"RoboDojo policy ready: horizon={self.action_horizon}, "
              f"replan={self.replan_steps}, denoise={session_cfg.num_inference_steps}", flush=True)

    def _adapt_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        vision = obs["vision"]
        state = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Expected a finite 14D RoboDojo state, got {state.shape}")
        return {
            "observation": {
                "head_camera": {"rgb": rgb(vision["cam_head"]["color"], "cam_head")},
                "left_camera": {
                    "rgb": rgb(vision["cam_left_wrist"]["color"], "cam_left_wrist")
                },
                "right_camera": {
                    "rgb": rgb(vision["cam_right_wrist"]["color"], "cam_right_wrist")
                },
            },
            "joint_action": {"vector": state},
        }

    def update_obs(self, obs):
        adapted = self._adapt_obs(obs)
        self.last_obs = adapted
        self.last_instruction = instruction(obs, self.default_instruction)
        if self.session is not None and self.session.pending_model_actions:
            self.session.update_obs(adapted)

    def update_obs_batch(self, obs_list):
        del obs_list
        raise NotImplementedError("Use eval_batch=false for stateful InternW0-delta inference")

    def get_action(self):
        if self.last_obs is None:
            raise ValueError("Call update_obs before get_action")
        if self.session.pending_model_actions:
            raise RuntimeError("Previous actions have not all been acknowledged")
        packed = np.asarray(
            self.session.get_action(
                {"observation": self.last_obs, "instruction": self.last_instruction}
            ),
            dtype=np.float32,
        )
        if packed.ndim != 2 or packed.shape[1] != 14 or not np.isfinite(packed).all():
            raise ValueError(f"WAM returned an invalid action chunk: {packed.shape}")
        packed[:, 6] = np.clip(packed[:, 6], 0.0, 1.0)
        packed[:, 13] = np.clip(packed[:, 13], 0.0, 1.0)
        return unpack_robot_state(
            packed, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def get_action_batch(self, env_idx_list=None):
        del env_idx_list
        raise NotImplementedError("Batched stateful InternW0-delta inference is disabled")

    def get_timing_rollout(self):
        return {} if self.session is None else self.session.get_timing_rollout()

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        if self.session is not None:
            self.session.reset_model()
