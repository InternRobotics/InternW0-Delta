"""ROS2 JointState and RGB transport; topics and joint order belong to the robot config."""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


class Robot:
    def __init__(self, config):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState, Image as ImageMessage, CompressedImage
        self.rclpy, self.JointState = rclpy, JointState
        cfg = yaml.safe_load(Path(config).read_text())
        self.cfg = cfg
        arms = cfg["arms"]
        if list(arms) != ["left", "right"] or any(len(arms[a]["joint_names"]) != 7 for a in arms):
            raise ValueError("Configure left/right arms in order, each with six joints followed by its gripper")
        self.lock = threading.Lock()
        self.states, self.images, self.times, self.sensor_stamps = {}, {}, {}, {}
        self.executor = self.thread = self.node = None
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        try:
            self.node = rclpy.create_node(str(cfg.get("node_name", "internw0_policy")))
            self.publishers = {}
            self.subscriptions = []
            for arm, settings in arms.items():
                self.publishers[arm] = self.node.create_publisher(JointState, settings["command_topic"], 1)
                self.subscriptions.append(self.node.create_subscription(
                    JointState, settings["state_topic"],
                    lambda msg, a=arm: self.on_state(a, msg), qos_profile_sensor_data))
            for key, settings in cfg["cameras"].items():
                compressed = bool(settings.get("compressed", True))
                self.subscriptions.append(self.node.create_subscription(
                    CompressedImage if compressed else ImageMessage, settings["topic"],
                    lambda msg, k=key, c=compressed: self.on_image(k, msg, c), qos_profile_sensor_data))
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def on_state(self, arm, msg):
        stamp = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
        received = time.monotonic()
        stamp_key = ("state", arm)
        names = self.cfg["arms"][arm]["joint_names"]
        try:
            values = np.asarray([msg.position[msg.name.index(k)] for k in names], dtype=np.float32)
            if not np.isfinite(values).all():
                return
        except (ValueError, IndexError):
            return
        with self.lock:
            if stamp <= self.sensor_stamps.get(stamp_key, -1):
                return
            self.states[arm] = (values, received)
            self.sensor_stamps[stamp_key] = stamp

    def on_image(self, key, msg, compressed):
        stamp = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
        received = time.monotonic()
        with self.lock:
            if stamp <= self.sensor_stamps.get(key, -1):
                return
        try:
            if compressed:
                rgb = np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
            else:
                if msg.encoding not in {"rgb8", "bgr8"}:
                    return
                rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
                rgb = rows[:, :msg.width*3].reshape(msg.height, msg.width, 3)
                if msg.encoding == "bgr8":
                    rgb = rgb[..., ::-1]
            rgb = np.ascontiguousarray(rgb)
        except (ValueError, OSError):
            return
        with self.lock:
            self.images[key], self.times[key], self.sensor_stamps[key] = rgb, received, stamp

    def observe(self):
        if not self.rclpy.ok():
            raise RuntimeError("ROS2 is shut down")
        with self.lock:
            if set(self.states) != {"left", "right"} or set(self.images) != set(self.cfg["cameras"]):
                raise TimeoutError("Waiting for joint feedback and camera frames")
            return dict(state=np.concatenate([self.states[a][0] for a in ("left", "right")]),
                        state_timestamp=min(self.states[a][1] for a in self.states),
                        images={k: v.copy() for k, v in self.images.items()}, timestamps=dict(self.times))

    def publish(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Commands must be finite [14] absolute joint/gripper values")
        for offset, arm in [(0, "left"), (7, "right")]:
            msg = self.JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = list(self.cfg["arms"][arm]["joint_names"])
            msg.position = action[offset:offset+7].astype(float).tolist()
            self.publishers[arm].publish(msg)

    def reset(self):
        service = self.cfg.get("reset_service")
        if not service:
            raise RuntimeError("No reset_service configured; reset with your robot controller")
        from std_srvs.srv import Trigger
        client = self.node.create_client(Trigger, service)
        try:
            if not client.wait_for_service(timeout_sec=3.):
                raise TimeoutError("Reset service is unavailable")
            future = client.call_async(Trigger.Request())
            deadline = time.monotonic()+30
            while not future.done() and time.monotonic() < deadline:
                time.sleep(.02)
            if not future.done():
                raise TimeoutError("Robot reset did not complete")
            if not future.result().success:
                raise RuntimeError("Robot controller rejected the reset")
        finally:
            self.node.destroy_client(client)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=3.)
        if self.thread is not None:
            self.thread.join(timeout=3.)
        if self.node is not None:
            self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()
