"""Fixed-cadence double buffering with one inference request in flight."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import time

import numpy as np
import torch
from wam.inference.online_action_policy import RTCActionGuidance


def shift_plan(plan, steps=16):
    plan = torch.as_tensor(plan).detach().cpu()
    if tuple(plan.shape) != (32, 80) or not torch.isfinite(plan).all():
        raise ValueError("Expected a finite normalized [32,80] plan")
    return torch.cat([plan[steps:], plan[-1:].repeat(steps, 1)]).unsqueeze(0)


def resample(actions, count):
    actions = np.asarray(actions, dtype=np.float32)
    if not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Action plans must be nonempty and finite")
    positions = np.linspace(0, len(actions)-1, count)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(actions)-1)
    alpha = (positions-lower).astype(np.float32)[:, None]
    return actions[lower]*(1-alpha) + actions[upper]*alpha


class DeadlineError(RuntimeError):
    pass


class AsyncController:
    def __init__(self, policy, instruction, *, mode="rtc", method="condition", speed=1.,
                 beta=5., prefix_steps=16, delay_buffer=10, clock=time.monotonic):
        if mode not in {"naive", "rtc"} or not .25 <= speed <= 1:
            raise ValueError("Async mode must be naive/rtc and speed in [0.25,1]")
        if method not in {"condition", "hard-prefix", "vjp"} or not 2 <= prefix_steps <= 16 or delay_buffer < 1:
            raise ValueError("Invalid RTC method, prefix length, or delay buffer")
        self.policy, self.instruction = policy, instruction
        self.mode, self.method, self.beta, self.prefix_steps = mode, method, beta, prefix_steps
        self.half = max(1, int(math.floor(16 / speed + .5)))
        self.clock = clock
        self.delays = deque(maxlen=delay_buffer)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")
        self.future = None
        self.tick = self.origin = self.last_submit = 0
        self.pause_seconds = 0.
        self.robot_plan = self.model_plan = None

    @property
    def logical_step(self):
        return (self.tick // self.half)*16 + (self.tick % self.half)*16 // self.half

    def guidance(self, plan):
        if self.mode != "rtc":
            return None
        delay = max(1, min(16, max(self.delays, default=1)*16 // self.half))
        if self.method == "hard-prefix" and delay >= self.prefix_steps:
            raise DeadlineError("Estimated delay exhausts the hard prefix; reduce inference latency or playback speed")
        return RTCActionGuidance(shift_plan(plan), delay,
                                 self.prefix_steps if self.method == "hard-prefix" else 16, self.beta, "exp")

    def infer(self, observation, step, guidance):
        start = self.clock()
        robot, model = self.policy.predict(observation, self.instruction, step, guidance)
        if np.shape(robot) != (32, 14) or tuple(model.shape) != (32, 80):
            raise ValueError("Policy must return [32,14] physical and [32,80] normalized actions")
        if not np.isfinite(robot).all() or not bool(torch.isfinite(model).all()):
            raise ValueError("Policy returned non-finite actions")
        return robot, model, self.clock()-start, self.clock()

    def install(self, robot, model, origin):
        self.robot_plan = np.concatenate([resample(robot[:16], self.half), resample(robot[16:], self.half)])
        self.model_plan, self.origin = model, origin

    def initialize(self, observation, refresh=None):
        # Cold model/compiler work is excluded from the conservative latency seed.
        robot, model, _, _ = self.infer(observation, 0, None)
        for step in (16, 32):
            robot, model, elapsed, _ = self.infer(observation, step, self.guidance(model))
            delay = math.ceil(elapsed * 30) + 1
            if delay > self.half:
                raise DeadlineError("Warm inference exceeds the 16-step overlap; reduce playback speed or inference cost")
            self.delays.append(delay)
        self.policy.reset()
        if refresh is not None:
            observation = refresh()
        robot, model, _, _ = self.infer(observation, 0, None)
        self.install(robot, model, 0)

    def needs_observation(self):
        return self.tick > 0 and self.tick % self.half == 0 and self.last_submit != self.tick

    def submit(self, observation):
        if not self.needs_observation():
            raise RuntimeError("Requests must start at distinct 16-step boundaries")
        self.poll()
        if self.future is not None:
            raise DeadlineError("Inference missed the next request boundary")
        guidance = self.guidance(self.model_plan)
        self.request_tick = self.tick
        self.request_started = self.clock()
        self.request_pause = self.pause_seconds
        self.future = self.worker.submit(self.infer, observation, self.logical_step, guidance)
        self.last_submit = self.tick

    def poll(self):
        if self.future is None:
            return
        elapsed = self.clock()-self.request_started-(self.pause_seconds-self.request_pause)
        if not self.future.done():
            if elapsed > self.half / 30 or self.tick-self.request_tick >= self.half:
                raise DeadlineError("Inference exceeded the available action overlap")
            return
        robot, model, worker_elapsed, _ = self.future.result()
        elapsed = max(elapsed, worker_elapsed)
        executed = self.tick-self.request_tick
        observed = max(executed, math.ceil(elapsed*30))
        logical_executed = executed*16 // self.half
        if observed > self.half or (self.method == "hard-prefix" and self.mode == "rtc" and logical_executed >= self.prefix_steps):
            raise DeadlineError("Completed inference arrived after its playback deadline")
        self.install(robot, model, self.request_tick)
        self.delays.append(observed)
        self.future = None

    def action(self):
        self.poll()
        cursor = self.tick-self.origin
        if self.robot_plan is None or not 0 <= cursor < len(self.robot_plan):
            raise DeadlineError("No unconsumed action remains")
        return self.robot_plan[cursor].copy()

    def commit(self):
        """Advance only after a command was successfully published."""
        self.tick += 1

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.future = None
