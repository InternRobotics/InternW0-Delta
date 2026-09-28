"""Transport-independent robot observation contract."""
from __future__ import annotations

import importlib
import time
import numpy as np


def connect(factory, config):
    module, name = factory.split(":", 1)
    return getattr(importlib.import_module(module), name)(config)


class ObservationValidator:
    def __init__(self, camera_keys, *, max_age=.2, max_skew=.1):
        self.cameras = tuple(camera_keys)
        self.max_age, self.max_skew = max_age, max_skew
        self.last = {}

    def read(self, robot):
        observation = robot.observe()
        now = time.monotonic()
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Robot state must contain 14 finite joint/gripper values")
        stamps, images = [], {}
        for key in self.cameras:
            image = np.asarray(observation["images"][key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key}: expected an RGB uint8 HWC camera image")
            timestamp = float(observation["timestamps"][key])
            if not np.isfinite(timestamp) or not -.001 <= now-timestamp <= self.max_age or timestamp <= self.last.get(key, -float("inf")):
                raise TimeoutError(f"{key}: camera observation is stale or repeated")
            stamps.append(timestamp)
            images[key] = image.copy()
        state_time = float(observation["state_timestamp"])
        if not np.isfinite(state_time) or not -.001 <= now-state_time <= self.max_age:
            raise TimeoutError("Robot state is stale")
        if max(stamps)-min(stamps) > self.max_skew:
            raise TimeoutError("Camera timestamps exceed the allowed skew")
        self.last = dict(zip(self.cameras, stamps))
        return {"state": state.copy(), "images": images}
