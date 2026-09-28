"""Saved-run loading and camera/action preprocessing for real robot inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.transforms import functional as F

from eval.robotwin.policy import RobotWinPolicySession, RobotWinSharedRuntime, RobotWinSessionConfig, _static_dimension_is_pad
from wam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wam.inference.online_action_policy import RTCActionPrefixCondition, infer_online_action_chunk
from wam.model.modules.codecs import video_latent_codec
from wam.utils.config_resolvers import register_default_resolvers

VALID_DIMS = list(range(6)) + [16] + list(range(40, 46)) + [56]


def checkpoint_path(run, selector="latest"):
    run = Path(run)
    if selector != "latest":
        path = Path(selector).expanduser()
        if not path.is_file():
            path = run / "checkpoints/weights" / selector
        entry = {}
    else:
        folder = run / "checkpoints/weights"
        manifest_path = next((folder / n for n in ("weights_manifest.json", "manifest.json") if (folder / n).is_file()), None)
        if manifest_path is None:
            raise FileNotFoundError("No checkpoint manifest; select a completed .pt explicitly with --checkpoint")
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get("named", {}).get("latest") or manifest.get("latest")
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError("Manifest has no latest checkpoint path")
        path = folder / entry["path"]
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Manifest checkpoint must reside inside checkpoints/weights")
    if not path.is_file() or path.suffix != ".pt" or ".partial" in path.name:
        raise FileNotFoundError("Select a completed .pt checkpoint")
    return path, entry.get("step")


def read_run(args):
    register_default_resolvers()
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    if args.model_root:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_root.resolve())
    if args.vlm_path:
        cfg.model.understanding.vlm_model_path = str(args.vlm_path.resolve())
    cfg.model.load_text_encoder = True
    cfg.model.mot_attention_backend = "sdpa"
    cfg.model.mot_checkpoint_mixed_attn = False
    cfg.model.action_dit_pretrained_path = None
    cfg.model.skip_dit_load_from_pretrain = True
    path, expected_step = checkpoint_path(args.run_dir, args.checkpoint)
    stats = args.stats or args.run_dir / "dataset_stats.json"
    if not stats.is_file():
        raise FileNotFoundError(f"Provide matching normalization with --stats: {stats}")
    train = cfg.data.train
    if (int(train.num_frames), int(train.global_sample_stride), float(train.action_hz)) != (33, 1, 30.):
        raise ValueError("Deployment expects 32 action steps at 30 Hz and sample stride 1")
    if int(train.memory_recent_frame_offset) != 32 or not train.get("single_canvas", False):
        raise ValueError("Deployment requires a single camera canvas and recent offset 32")
    cameras = tuple(str(x.key) for x in train.shape_meta.images)
    if (cameras, str(train.concat_multi_camera)) not in {
        (("head", "hand_left", "hand_right"), "robotwin"),
        (("hand_left", "hand_right"), "vertical"),
    }:
        raise ValueError("Expected ordered head/hand_left/hand_right or hand_left/hand_right cameras")
    processor = instantiate(train.processor).eval()
    if processor.action_state_transforms is not None:
        raise ValueError("The deployment action contract requires absolute joints")
    for field in ("action", "state"):
        meta = processor.shape_meta[field]
        if len(meta) != 1 or int(meta[0]["shape"]) != 14:
            raise ValueError("Expected one 14-D left-arm/gripper, right-arm/gripper field")
        mask = _static_dimension_is_pad(processor, field)
        if len(mask) != 80 or (~mask).nonzero().flatten().tolist() != VALID_DIMS:
            raise ValueError("Processor must map both arms and grippers into the canonical 80-D layout")
        slices = getattr(processor.action_state_merger, field + "_target_slices")
        if [(list(x["source_slice"]), list(x["target_slice"])) for x in slices] != [
            ([0, 6], [0, 6]), ([6, 7], [16, 17]), ([7, 13], [40, 46]), ([13, 14], [56, 57])]:
            raise ValueError("Canonical source/target joint ordering differs from the robot contract")
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    for field in ("action", "state"):
        for norm in processor.normalizer.normalizers[field].values():
            if tuple(norm.scale.shape) != (14,) or tuple(norm.offset.shape) != (14,):
                raise ValueError(f"Deployment requires global 14-D {field} normalization")
            if not torch.isfinite(norm.scale).all() or not torch.isfinite(norm.offset).all() or (norm.scale <= 0).any():
                raise ValueError(f"Invalid {field} normalization: expected finite, positive scales")
    if args.mode == "rtc" and args.method == "condition":
        rtc = cfg.model.action_dit_config.get("training_time_rtc", {})
        delay = rtc.get("max_delay_steps", 0)
        if rtc.get("enabled") is not True or type(delay) is not int or not 16 <= delay < 32:
            raise ValueError("RTC condition requires a trained checkpoint with max_delay_steps in [16,31]")
    if bool(cfg.model.understanding.enabled) and not Path(str(cfg.model.understanding.vlm_model_path)).is_dir():
        raise FileNotFoundError("Set --vlm-path to the local RynnBrain model directory")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    step = payload.get("step")
    if type(step) is not int or step < 0 or (expected_step is not None and step != int(expected_step)):
        raise ValueError("Checkpoint step is missing or differs from the manifest")
    mot = payload["mot"]
    for key, dim in [("mixtures.action.action_encoder.weight", -1), ("mixtures.action.head.weight", 0)]:
        if mot[key].shape[dim] != 80:
            raise ValueError("Checkpoint does not have 80 action dimensions")
    report = dict(checkpoint=str(path), step=step, stats=str(stats), cameras=cameras,
                  action_horizon=32, action_hz=30, mode=args.mode, method=args.method,
                  training_time_rtc=OmegaConf.to_container(cfg.model.action_dit_config.get("training_time_rtc", OmegaConf.create({}))))
    return cfg, processor, path, stats, payload, report


def load_policy(args, prepared):
    cfg, processor, path, stats, payload, _ = prepared
    precision = args.precision or str(cfg.get("mixed_precision", "bf16"))
    dtype = {"no": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device).to(args.device).eval()
    model.mot.load_state_dict(payload["mot"], strict=True)
    for name in ("proprio_encoder", "action_proprio_encoder"):
        module = getattr(model, name, None)
        if module is not None:
            module.load_state_dict(payload[name], strict=True)
    if model.understanding is not None:
        model.understanding.load_adapter_state_dict(payload["understanding"], strict=True)
    model.requires_grad_(False)
    if args.compile_action:
        model.mot.forward_action_with_video_cache = torch.compile(model.mot.forward_action_with_video_cache, dynamic=False)
    runtime = RobotWinSharedRuntime(model, processor, str(path), stats, args.device, dtype,
                                   _static_dimension_is_pad(processor, "action"))
    config = RobotWinSessionConfig(32, 30., 32, args.denoise_steps or int(cfg.get("eval_num_inference_steps", 10)),
                                  None, args.seed, "cpu", args.tiled, False, 9,
                                  tuple(cfg.data.train.video_size), "single", "canvas")
    return RobotPolicy(runtime, config, cfg.data.train, args)


class RobotPolicy(RobotWinPolicySession):
    def __init__(self, runtime, config, train, options):
        super().__init__(runtime=runtime, config=config)
        self.camera_keys = tuple(str(x.key) for x in train.shape_meta.images)
        self.image_shapes = {str(x.key): tuple(x.shape) for x in train.shape_meta.images}
        self.canvas_layout = str(train.concat_multi_camera)
        self.options = options
        self.latents = OrderedDict()
        self.text = {}
        size = {"img_w": self.video_width, "img_h": self.video_height}
        self.canvas_transforms = [ResizeSmallestSideAspectPreserving(args=size), CenterCrop(args=size), Normalize(args={"mean": .5, "std": .5})]

    def gripper_transform(self, values, *, state):
        values = np.asarray(values, dtype=np.float32).copy()
        if not self.options.gripper_offset_enabled:
            return values
        indices = {"left": [6], "right": [13], "both": [6, 13]}[self.options.gripper_target]
        if state and self.options.skip_gripper_state_offset:
            return values
        selected = values[..., indices]
        eligible = (selected >= self.options.gripper_lower) & (selected <= self.options.gripper_upper)
        selected = selected + eligible * self.options.gripper_offset * (-1 if state else 1)
        if state:
            selected = np.clip(selected, self.options.gripper_state_min, self.options.gripper_state_max)
        values[..., indices] = selected
        return values

    def _normalize_state(self, state):
        return super()._normalize_state(self.gripper_transform(state, state=True))

    def _build_robotwin_image_tensor(self, observation):
        frames = []
        for key in self.camera_keys:
            tensor = torch.from_numpy(observation["images"][key].copy()).permute(2, 0, 1).unsqueeze(0)
            transforms = self.processor.val_transforms
            if isinstance(transforms, dict):
                transforms = transforms[key]
            for transform in transforms:
                tensor = transform(tensor)
            if tuple(tensor.shape) != (1, *self.image_shapes[key]):
                raise ValueError(f"Camera transform shape does not match training: {key}")
            frames.append(tensor[0])
        if self.canvas_layout == "vertical":
            canvas = torch.cat(frames, dim=-2)
        else:
            h, w = self.video_height, self.video_width
            top = h * 2 // 3
            resize = lambda x, size: F.resize(x, size, antialias=True)
            canvas = torch.cat([resize(frames[0], [top, w]), torch.cat([
                resize(frames[1], [h-top, w//2]), resize(frames[2], [h-top, w-w//2])], dim=-1)], dim=-2)
        canvas = canvas.unsqueeze(0)
        for transform in self.canvas_transforms:
            canvas = transform(canvas)
        return canvas.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _build_robotwin_vlm_images(self, observation):
        return torch.stack([torch.from_numpy(observation["images"][k].copy()).permute(2, 0, 1)
                            for k in self.camera_keys]).to(self.model.device)

    @torch.no_grad()
    def predict(self, observation, instruction, step, guidance=None):
        self.step_count = step
        observation = dict(observation, joint_action={"vector": observation["state"]})
        self._record_memory_observation(observation)
        image = self._build_robotwin_image_tensor(observation)
        memory = self._prepare_memory_for_replan(image_tensor=image)
        for key, source_step, frame in [("video_latents", step, image),
                                       ("memory_video_anchor_latents", 0, memory["memory_video_anchor"][:, 0].unsqueeze(0)),
                                       ("memory_video_recent_latents", max(0, step-32), memory["memory_video_recent"][:, 0].unsqueeze(0))]:
            if source_step not in self.latents:
                self.latents[source_step] = video_latent_codec.encode_input_image_latents_tensor(
                    self.model, input_image=frame.to(device=self.model.device, dtype=self.model.torch_dtype),
                    tiled=self.tiled, video_layout="single", video_view_names="canvas")
            memory[key] = self.latents[source_step]
        for old in list(self.latents):
            if old and old < step - 32:
                del self.latents[old]
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self.text:
            self.text = {prompt: self.model.encode_prompt(prompt)}
        context, mask = self.text[prompt]
        options = dict(prompt=None, context=context, context_mask=mask, input_image=image,
                       action_horizon=32, proprio=self._normalize_state(observation["state"]),
                       action_dim_is_pad=self.runtime.action_dim_is_pad, memory_inputs=memory,
                       vlm_current_images=self._build_robotwin_vlm_images(observation),
                       vlm_view_names="|".join(self.camera_keys), understanding_prompt=prompt,
                       num_inference_steps=self.num_inference_steps, seed=self.seed, tiled=self.tiled,
                       video_layout="single", video_view_names="canvas")
        if guidance is not None and self.options.method == "condition":
            options["rtc_prefix_condition"] = RTCActionPrefixCondition(guidance.prev_action_chunk, guidance.inference_delay)
        elif guidance is not None and self.options.method == "vjp":
            options["rtc_guidance"] = guidance
        sampler = infer_online_action_chunk
        if guidance is not None and self.options.method == "hard-prefix":
            from deploy.hard_prefix import infer_hard_prefix
            sampler = infer_hard_prefix
            options["rtc_guidance"] = guidance
        model_action = sampler(self.model, **options)["action"]
        action = self.gripper_transform(self._denormalize_action(model_action)[0], state=False)
        if not np.isfinite(action).all():
            raise ValueError("Prediction contains non-finite robot commands")
        return action, model_action

    def reset(self):
        super().reset()
        self.latents.clear()
