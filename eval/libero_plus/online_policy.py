import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ.setdefault("WAM_LIBERO_REPO", "LIBERO-plus")
libero_plus_root = project_root / "third_party" / "LIBERO-plus"
if libero_plus_root.exists() and str(libero_plus_root) not in sys.path:
    sys.path.insert(0, str(libero_plus_root))

from eval.libero.action_ensembler import ActionEnsembler
from eval.libero.eval_config import resolve_eval_video_metadata
from eval.libero.eval_libero_single import (
    WAMProcessor,
    _ar_history_flags,
    _build_action_history_tensors,
    _build_memory_inputs_for_eval,
    _build_visual_history_tensors,
    _capture_model_frame_and_proprio,
    _denormalize_action,
    _obs_to_model_input,
    _resolve_replan_steps,
    _static_dimension_is_pad,
    _trim_memory_history,
    _uses_memory,
    _vlm_current_images,
)
from eval.libero.libero_utils import invert_gripper_action
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.inference.online_action_policy import infer_online_action_chunk


def _validate_frame_policy_cadence(
    *,
    cfg: DictConfig,
    action_horizon: int,
    replan_steps: int,
) -> None:
    action_horizon = int(action_horizon)
    replan_steps = int(replan_steps)
    if action_horizon <= 0 or replan_steps <= 0 or replan_steps > action_horizon:
        raise ValueError(
            "Frame policy requires 0 < replan_steps <= action_horizon; "
            f"got {action_horizon=} {replan_steps=}."
        )
    train_num_frames = int(cfg.data.train.num_frames)
    train_recent_offset = int(cfg.data.train.memory_recent_frame_offset)
    if train_num_frames != action_horizon + 1:
        raise ValueError(
            "Evaluation horizon must match training num_frames - 1: "
            f"num_frames={train_num_frames}, action_horizon={action_horizon}."
        )
    if train_recent_offset != action_horizon:
        raise ValueError(
            "Evaluation recent-memory offset must match training: "
            f"training_offset={train_recent_offset}, action_horizon={action_horizon}."
        )


def predict_online_action_chunk(
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
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    online_action_history: Optional[list[torch.Tensor]] = None,
    online_visual_history: Optional[list[torch.Tensor]] = None,
    online_proprio_history: Optional[list[torch.Tensor]] = None,
    online_anchor_visual_history: Optional[list[torch.Tensor]] = None,
    online_anchor_proprio_history: Optional[list[torch.Tensor]] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], torch.Tensor, torch.Tensor]:
    del (
        history_actions,
        history_action_is_pad,
        history_video,
        history_image_is_pad,
        online_action_history,
        online_visual_history,
        online_proprio_history,
        online_anchor_visual_history,
        online_anchor_proprio_history,
    )

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
        raise ValueError("LIBERO-plus online eval requires memory AR inference.")

    t5_prompt = DEFAULT_PROMPT.format(task=task_description)
    infer_prompt = None if context is not None and context_mask is not None else t5_prompt
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 20)))
    sigma_shift_cfg = cfg.EVALUATION.get("sigma_shift", None)
    sigma_shift = None if sigma_shift_cfg is None else float(sigma_shift_cfg)
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError("Online cached LIBERO-plus eval is action-only and does not denoise future video.")
    else:
        video_layout, video_view_names, vlm_view_names = resolve_eval_video_metadata(
            cfg, processor
        )
        pred = infer_online_action_chunk(
            model,
            prompt=infer_prompt,
            input_image=image,
            vlm_current_images=_vlm_current_images(imgs, processor).to(model.device),
            vlm_view_names=vlm_view_names,
            action_horizon=int(action_horizon),
            proprio=proprio,
            action_dim_is_pad=_static_dimension_is_pad(processor, "action"),
            memory_inputs=memory_inputs,
            context=context,
            context_mask=context_mask,
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
    action_chunk[..., -1] = action_chunk[..., -1] * 2 - 1
    action_chunk = invert_gripper_action(action_chunk)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action_chunk[..., -1] = np.sign(action_chunk[..., -1])
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


class OnlinePolicySession:
    """Stateful online AR policy used by the LIBERO-plus client/server eval."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: WAMProcessor,
        cfg: DictConfig,
        action_horizon: int,
        input_w: int,
        input_h: int,
        model_device: str,
        profiler: Any = None,
        torch_rng_state: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.action_horizon = int(action_horizon)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.model_device = str(model_device)
        # The shared server serializes model calls, but each client must retain
        # the same CPU diffusion RNG stream it would have in a dedicated process.
        self.profiler = profiler
        self._torch_rng_state = (
            torch.random.get_rng_state().clone()
            if torch_rng_state is None
            else torch_rng_state.clone()
        )
        self.reset(task_suite_name="", task_id=-1, task_description="")

    def reset(self, *, task_suite_name: str, task_id: int, task_description: str) -> None:
        if task_suite_name:
            self.cfg.EVALUATION.task_suite_name = task_suite_name
        if int(task_id) >= 0:
            self.cfg.EVALUATION.task_id = int(task_id)
        self.task_description = str(task_description)

        self.replan_steps = _resolve_replan_steps(self.cfg, self.action_horizon)
        self.use_action_ensembler = bool(self.cfg.EVALUATION.get("use_action_ensembler", False))
        self.ensembler = ActionEnsembler() if self.use_action_ensembler else None
        if self.ensembler is not None:
            self.ensembler.reset()

        self.pending_actions: list[list[float]] = []
        self.pending_model_actions: list[torch.Tensor] = []
        self.last_model_action: Optional[torch.Tensor] = None
        self.prompt_context: Optional[torch.Tensor] = None
        self.prompt_context_mask: Optional[torch.Tensor] = None

        self.use_ar_action_history, self.use_ar_visual_history, self.ar_num_actions, self.ar_num_frames = (
            _ar_history_flags(self.model)
        )
        self.use_memory = _uses_memory(self.model)
        self.ar_effective_num_actions = self.ar_num_actions
        if self.use_ar_action_history:
            ar_num_chunks = int(getattr(self.model, "num_history_chunks", 0) or 0)
            if ar_num_chunks > 0:
                self.ar_effective_num_actions = max(
                    1,
                    min(self.ar_num_actions, ar_num_chunks * self.replan_steps),
                )

        self.action_history: list[torch.Tensor] = []
        self.visual_history: list[torch.Tensor] = []
        self.proprio_history: list[torch.Tensor] = []
        self.anchor_visual_history: list[torch.Tensor] = []
        self.anchor_proprio_history: list[torch.Tensor] = []
        self.action_dim = int(getattr(getattr(self.model, "action_expert", None), "action_dim", 0))
        if (self.use_ar_action_history or self.use_memory) and self.action_dim <= 0:
            raise ValueError("History is enabled, but model.action_expert.action_dim is unavailable.")

        self.memory_video_anchor_frames = int(getattr(self.model, "memory_video_anchor_frames", 0) or 0)
        self.memory_video_recent_frames = int(getattr(self.model, "memory_video_recent_frames", 0) or 0)
        self.memory_video_anchor_raw_steps = (
            self.memory_video_anchor_frames if self.memory_video_anchor_frames > 0 else 0
        )
        if self.use_memory:
            _validate_frame_policy_cadence(
                cfg=self.cfg,
                action_horizon=self.action_horizon,
                replan_steps=self.replan_steps,
            )
        self.last_execution_steps = self.replan_steps
        self.replan_count = 0
        self.memory_executed_steps = 0

    def _get_prompt_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prompt_context is None or self.prompt_context_mask is None:
            t5_prompt = DEFAULT_PROMPT.format(task=self.task_description)
            self.prompt_context, self.prompt_context_mask = self.model.encode_prompt(t5_prompt)
        return self.prompt_context, self.prompt_context_mask

    def act(self, obs: Optional[dict], *, timestep: int) -> dict[str, Any]:
        host_rng_state = torch.random.get_rng_state()
        torch.random.set_rng_state(self._torch_rng_state)
        try:
            return self._act_with_session_rng(obs, timestep=timestep)
        finally:
            self._torch_rng_state = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(host_rng_state)

    def _act_with_session_rng(
        self,
        obs: Optional[dict],
        *,
        timestep: int,
    ) -> dict[str, Any]:
        replanned = False
        if len(self.pending_actions) == 0:
            if obs is None:
                raise ValueError("OnlinePolicySession.act received obs=None but a replan is required.")
            self._replan(obs, timestep=int(timestep))
            replanned = True

        next_action = self.pending_actions.pop(0)
        self.last_model_action = self.pending_model_actions.pop(0) if self.pending_model_actions else None
        return {
            "action": [float(x) for x in next_action],
            "replanned": replanned,
            "pending_actions_remaining": len(self.pending_actions),
            "execution_steps": self.last_execution_steps,
        }

    def observe(
        self,
        obs: dict,
        *,
        done: bool,
        anchor_only: bool = False,
    ) -> dict[str, float]:
        if self.use_memory and bool(anchor_only) and not bool(done):
            anchor_frame, anchor_proprio = _capture_model_frame_and_proprio(
                obs,
                cfg=self.cfg,
                processor=self.processor,
                width=self.input_w,
                height=self.input_h,
                model_device=self.model_device,
                dtype=self.model.torch_dtype,
            )
            self.anchor_visual_history.append(anchor_frame)
            self.anchor_proprio_history.append(anchor_proprio)
            if self.memory_video_anchor_raw_steps > 0:
                self.anchor_visual_history = self.anchor_visual_history[-self.memory_video_anchor_raw_steps :]
                self.anchor_proprio_history = self.anchor_proprio_history[-self.memory_video_anchor_raw_steps :]
            return {}

        if (self.use_ar_action_history or self.use_memory) and self.last_model_action is not None:
            self.action_history.append(self.last_model_action.detach().to(device="cpu", dtype=torch.float32))
            if self.use_memory:
                self.memory_executed_steps += 1
            if self.use_ar_action_history and not self.use_memory:
                self.action_history = self.action_history[-self.ar_num_actions :]
            elif self.use_memory:
                self.action_history = _trim_memory_history(
                    self.action_history,
                    anchor_len=0,
                    recent_len=self.replan_steps,
                )

        if self.use_memory and not bool(done):
            next_frame, next_proprio = _capture_model_frame_and_proprio(
                obs,
                cfg=self.cfg,
                processor=self.processor,
                width=self.input_w,
                height=self.input_h,
                model_device=self.model_device,
                dtype=self.model.torch_dtype,
            )
            self.visual_history.append(next_frame)
            self.proprio_history.append(next_proprio)
            recent_len = self.action_horizon + 1
            self.visual_history = _trim_memory_history(
                self.visual_history,
                anchor_len=self.memory_video_anchor_raw_steps,
                recent_len=recent_len,
            )
            self.proprio_history = _trim_memory_history(
                self.proprio_history,
                anchor_len=self.memory_video_anchor_raw_steps,
                recent_len=recent_len,
            )
        self.last_model_action = None
        return {}

    def close(self) -> None:
        """Release only per-client state; the shared model remains resident."""
        self.pending_actions.clear()
        self.pending_model_actions.clear()
        self.action_history.clear()
        self.visual_history.clear()
        self.proprio_history.clear()
        self.anchor_visual_history.clear()
        self.anchor_proprio_history.clear()
        self.last_model_action = None
        self.prompt_context = None
        self.prompt_context_mask = None

    def _prepare_memory_inputs(self) -> Optional[dict[str, torch.Tensor]]:
        memory_inputs = _build_memory_inputs_for_eval(
            model=self.model,
            cfg=self.cfg,
            action_history=self.action_history,
            visual_history=self.visual_history,
            proprio_history=self.proprio_history,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            height=self.input_h,
            width=self.input_w,
            memory_chunk_index=self.replan_count,
            current_memory_chunks=self.replan_count,
            include_current_recent=True,
            episode_step=self.memory_executed_steps,
        )
        if memory_inputs is None:
            return None
        return memory_inputs


    def _replan(self, obs: dict, *, timestep: int) -> None:
        history_kwargs: dict[str, torch.Tensor] = {}
        if self.use_ar_action_history:
            history_actions, history_action_is_pad = _build_action_history_tensors(
                self.action_history,
                target_len=self.ar_effective_num_actions,
                action_dim=self.action_dim,
            )
            history_kwargs["history_actions"] = history_actions
            history_kwargs["history_action_is_pad"] = history_action_is_pad
        if self.use_ar_visual_history:
            history_video, history_image_is_pad = _build_visual_history_tensors(
                self.visual_history,
                target_len=self.ar_num_frames,
                height=self.input_h,
                width=self.input_w,
            )
            history_kwargs["history_video"] = history_video
            history_kwargs["history_image_is_pad"] = history_image_is_pad

        if self.use_memory and len(self.visual_history) == 0:
            current_frame, current_proprio = _capture_model_frame_and_proprio(
                obs,
                cfg=self.cfg,
                processor=self.processor,
                width=self.input_w,
                height=self.input_h,
                model_device=self.model_device,
                dtype=self.model.torch_dtype,
            )
            self.visual_history.append(current_frame)
            self.proprio_history.append(current_proprio)

        memory_inputs = None
        if self.use_memory:
            memory_inputs = self._prepare_memory_inputs()

        prompt_context, prompt_context_mask = self._get_prompt_context()
        action_chunk, _, _, model_action_chunk, current_model_image = predict_online_action_chunk(
            obs=obs,
            task_description=self.task_description,
            model=self.model,
            processor=self.processor,
            cfg=self.cfg,
            action_horizon=self.action_horizon,
            input_w=self.input_w,
            input_h=self.input_h,
            model_device=self.model_device,
            replan_steps=self.replan_steps,
            memory_inputs=memory_inputs,
            memory_chunk_index=self.replan_count,
            context=prompt_context,
            context_mask=prompt_context_mask,
            online_action_history=self.action_history,
            online_visual_history=self.visual_history,
            online_proprio_history=self.proprio_history,
            online_anchor_visual_history=self.anchor_visual_history,
            online_anchor_proprio_history=self.anchor_proprio_history,
            **history_kwargs,
        )

        if self.use_ar_visual_history and not self.use_memory:
            self.visual_history.append(current_model_image)
            self.visual_history = self.visual_history[-self.ar_num_frames :]

        n_exec = self.replan_steps
        self.last_execution_steps = n_exec
        if self.use_action_ensembler:
            assert self.ensembler is not None
            self.ensembler.add_actions(action_chunk, timestep)
            self.pending_actions = [
                self.ensembler.get_action(ts).tolist()
                for ts in range(timestep, timestep + n_exec)
            ]
        else:
            self.pending_actions = action_chunk[:n_exec].tolist()

        if self.use_ar_action_history or self.use_memory:
            num_pending = len(self.pending_actions)
            self.pending_model_actions = [
                model_action_chunk[min(step_idx, model_action_chunk.shape[0] - 1)].clone()
                for step_idx in range(num_pending)
            ]
        self.replan_count += 1
