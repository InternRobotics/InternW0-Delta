"""RTC projection along a fixed noise-to-action flow path."""
from __future__ import annotations
import math
from typing import Any, Optional
import torch
from wam.inference.online_action_policy import RTCActionGuidance, infer_online_action_chunk


def validate_prefix_steps(value: int) -> int:
    steps = int(value)
    if not 2 <= steps <= 16:
        raise ValueError(
            "RTC hard-prefix steps must be in "
            f"[{2}, {16}], "
            f"got {steps}."
        )
    return steps


class HardPrefixScheduler:
    """Project a configurable RTC prefix onto one fixed flow-matching path.

    InternW0-delta uses ``x_sigma = (1 - sigma) * clean + sigma * noise``.  The proxy
    captures the new chunk's initially sampled noise, replaces the selected
    action velocities with ``noise - aligned_old_plan``, and projects the
    scheduler result back onto that same path after every denoising step.  The
    final sigma-zero prefix is therefore exactly the aligned old-plan suffix in
    sampler dtype, while the unconstrained tail continues to use model output.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        clean_action_target: torch.Tensor,
        prefix_steps: int,
        action_dim_is_pad: Optional[torch.Tensor],
        expected_inference_steps: int,
    ) -> None:
        self._scheduler = scheduler
        self.prefix_steps = int(prefix_steps)
        self.expected_inference_steps = int(expected_inference_steps)
        self.prefix_steps = validate_prefix_steps(self.prefix_steps)
        if self.expected_inference_steps <= 0:
            raise ValueError("Hard-prefix inference steps must be positive.")

        target = torch.as_tensor(clean_action_target).detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        expected_shape = (1, 32, 80)
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                "Hard-prefix target must be the aligned canonical action tensor with "
                f"shape {expected_shape}, got {tuple(target.shape)}."
            )
        if not bool(torch.isfinite(target).all().item()):
            raise ValueError("Hard-prefix target must contain only finite values.")

        pad_mask: Optional[torch.Tensor] = None
        if action_dim_is_pad is not None:
            pad_mask = torch.as_tensor(action_dim_is_pad).detach().to(
                device="cpu",
                dtype=torch.bool,
            )
            if tuple(pad_mask.shape) != (80,):
                raise ValueError(
                    "Hard-prefix action_dim_is_pad must have shape (80,), got "
                    f"{tuple(pad_mask.shape)}."
                )
            target = target.masked_fill(pad_mask.view(1, 1, -1), 0.0)

        self._clean_action_target = target.contiguous()
        self._action_dim_is_pad = pad_mask
        self._runtime_clean_action_target: Optional[torch.Tensor] = None
        self._runtime_action_dim_is_pad: Optional[torch.Tensor] = None
        self._timesteps: Optional[torch.Tensor] = None
        self._deltas: Optional[torch.Tensor] = None
        self._prefix_noise: Optional[torch.Tensor] = None
        self._observed_deltas: list[torch.Tensor] = []
        self._step_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)

    def build_inference_schedule(self, *args: Any, **kwargs: Any):
        if self._timesteps is not None:
            raise RuntimeError(
                "Hard-prefix scheduler received more than one inference schedule."
            )
        timesteps, deltas = self._scheduler.build_inference_schedule(
            *args,
            **kwargs,
        )
        if timesteps.ndim != 1 or deltas.ndim != 1:
            raise RuntimeError(
                "Hard-prefix scheduler requires one-dimensional timesteps and "
                f"deltas, got {tuple(timesteps.shape)} and {tuple(deltas.shape)}."
            )
        if len(timesteps) != self.expected_inference_steps or len(deltas) != len(
            timesteps
        ):
            raise RuntimeError(
                "Hard-prefix scheduler step count mismatch: "
                f"timesteps={len(timesteps)}, deltas={len(deltas)}, "
                f"expected={self.expected_inference_steps}."
            )
        self._timesteps = timesteps
        self._deltas = deltas
        return timesteps, deltas

    def step(
        self,
        model_output: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self._timesteps is None or self._deltas is None:
            raise RuntimeError(
                "Hard-prefix scheduler step called before schedule construction."
            )
        if self._step_index >= self.expected_inference_steps:
            raise RuntimeError("Hard-prefix scheduler received too many steps.")
        if tuple(sample.shape) != tuple(self._clean_action_target.shape):
            raise RuntimeError(
                "Hard-prefix latent shape changed during sampling: "
                f"{tuple(sample.shape)} vs {tuple(self._clean_action_target.shape)}."
            )
        if tuple(model_output.shape) != tuple(sample.shape):
            raise RuntimeError(
                "Hard-prefix velocity shape mismatch: "
                f"{tuple(model_output.shape)} vs {tuple(sample.shape)}."
            )
        if delta.numel() != 1:
            raise RuntimeError(
                "Hard-prefix projection requires one scalar scheduler delta, "
                f"got {tuple(delta.shape)}."
            )
        self._observed_deltas.append(delta.detach().reshape(()).clone())

        if self._runtime_clean_action_target is None:
            self._runtime_clean_action_target = self._clean_action_target.to(
                device=sample.device,
                dtype=sample.dtype,
                non_blocking=True,
            )
            if self._action_dim_is_pad is not None:
                self._runtime_action_dim_is_pad = self._action_dim_is_pad.to(
                    device=sample.device,
                    non_blocking=True,
                )
        clean_target = self._runtime_clean_action_target
        pad_mask = self._runtime_action_dim_is_pad
        if clean_target is None:
            raise RuntimeError("Hard-prefix runtime target was not initialized.")
        if self._prefix_noise is None:
            prefix_noise = sample.detach().clone()
            if pad_mask is not None:
                prefix_noise = prefix_noise.masked_fill(
                    pad_mask.view(1, 1, -1),
                    0.0,
                )
            self._prefix_noise = prefix_noise
        prefix_noise = self._prefix_noise

        projected_velocity = model_output.clone()
        projected_velocity[:, : self.prefix_steps] = (
            prefix_noise[:, : self.prefix_steps]
            - clean_target[:, : self.prefix_steps]
        )
        if pad_mask is not None:
            projected_velocity = projected_velocity.masked_fill(
                pad_mask.view(1, 1, -1),
                0.0,
            )

        next_sample = self._scheduler.step(
            projected_velocity,
            delta,
            sample,
        )
        if self._step_index + 1 == self.expected_inference_steps:
            next_sigma = torch.zeros((), device=sample.device, dtype=sample.dtype)
        else:
            next_timestep = self._timesteps[self._step_index + 1].to(
                device=sample.device,
                dtype=sample.dtype,
            )
            next_sigma = (
                next_timestep / float(self._scheduler.num_train_timesteps)
            ).clamp(min=0.0, max=1.0)
        next_sigma = next_sigma.reshape(1, 1, 1)
        noisy_prefix_target = (
            (1.0 - next_sigma) * clean_target + next_sigma * prefix_noise
        )
        projected_sample = next_sample.clone()
        projected_sample[:, : self.prefix_steps] = noisy_prefix_target[
            :, : self.prefix_steps
        ]
        if pad_mask is not None:
            projected_sample = projected_sample.masked_fill(
                pad_mask.view(1, 1, -1),
                0.0,
            )
        self._step_index += 1
        return projected_sample.detach()

    def assert_complete(self) -> None:
        if self._timesteps is None:
            raise RuntimeError("Hard-prefix sampler never built its schedule.")
        if self._deltas is None:
            raise RuntimeError("Hard-prefix sampler never received schedule deltas.")
        if self._step_index != self.expected_inference_steps:
            raise RuntimeError(
                "Hard-prefix sampler did not execute the expected denoising "
                f"steps: {self._step_index}/{self.expected_inference_steps}."
            )
        timesteps = self._timesteps.detach().to(device="cpu", dtype=torch.float32)
        scheduled_deltas = self._deltas.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        observed_deltas = torch.stack(self._observed_deltas).to(
            device="cpu",
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(timesteps).all().item()) or not bool(
            torch.isfinite(scheduled_deltas).all().item()
        ):
            raise RuntimeError("Hard-prefix scheduler produced non-finite values.")
        first_sigma = float(timesteps[0].item()) / float(
            self._scheduler.num_train_timesteps
        )
        if not math.isclose(first_sigma, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "Hard-prefix noise capture requires the first scheduler sigma "
                f"to equal 1, got {first_sigma:.9g}."
            )
        if bool((timesteps[1:] > timesteps[:-1]).any().item()) or bool(
            (scheduled_deltas > 0).any().item()
        ):
            raise RuntimeError(
                "Hard-prefix scheduler requires monotonically decreasing sigma."
            )
        if not torch.equal(observed_deltas, scheduled_deltas):
            raise RuntimeError(
                "Hard-prefix sampler step deltas differ from the built schedule."
            )


def infer_hard_prefix(
    model: Any,
    *,
    rtc_guidance: RTCActionGuidance,
    action_dim_is_pad: Optional[torch.Tensor],
    num_inference_steps: int,
    **inference_kwargs: Any,
) -> dict[str, Any]:
    """Run InternW0-delta sampling with local per-step hard-prefix flow projection."""
    if not isinstance(rtc_guidance, RTCActionGuidance):
        raise TypeError(
            "Hard-prefix sampling requires RTCActionGuidance, got "
            f"{type(rtc_guidance).__name__}."
        )
    prefix_steps = validate_prefix_steps(
        rtc_guidance.prefix_attention_horizon
    )
    original_scheduler = model.infer_action_scheduler
    if isinstance(original_scheduler, HardPrefixScheduler):
        raise RuntimeError("Hard-prefix scheduler proxy is already installed.")
    proxy = HardPrefixScheduler(
        original_scheduler,
        clean_action_target=rtc_guidance.prev_action_chunk,
        prefix_steps=prefix_steps,
        action_dim_is_pad=action_dim_is_pad,
        expected_inference_steps=int(num_inference_steps),
    )
    model.infer_action_scheduler = proxy
    try:
        prediction = infer_online_action_chunk(
            model,
            action_dim_is_pad=action_dim_is_pad,
            num_inference_steps=int(num_inference_steps),
            rtc_guidance=None,
            **inference_kwargs,
        )
        proxy.assert_complete()
        return prediction
    finally:
        model.infer_action_scheduler = original_scheduler
