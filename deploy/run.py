"""Deploy a saved InternW0-delta run through a local robot adapter."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import select
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--stats", type=Path)
    p.add_argument("--model-root", type=Path)
    p.add_argument("--vlm-path", type=Path)
    p.add_argument("--instruction")
    p.add_argument("--mode", choices=["sync", "naive", "rtc"], default="sync")
    p.add_argument("--method", choices=["condition", "vjp", "hard-prefix"], default="condition")
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--robot", default="deploy.ros2:Robot", help="Importable module:factory taking the robot config path")
    p.add_argument("--robot-config", type=Path, default=Path("deploy/robot.yaml"))
    p.add_argument("--execute", action="store_true", help="Publish actions; otherwise observe and infer only")
    p.add_argument("--no-wait", action="store_true", help="Skip the terminal start gate")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["no", "fp16", "bf16"])
    p.add_argument("--denoise-steps", type=int)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--compile-action", action="store_true")
    p.add_argument("--playback-speed", type=float, default=1.)
    p.add_argument("--prefix-steps", type=int, default=16)
    p.add_argument("--guidance-beta", type=float, default=5.)
    p.add_argument("--delay-buffer", type=int, default=10)
    p.add_argument("--smoothing-window", type=int, default=1, help="Causal arm-joint moving average; grippers remain unchanged")
    p.add_argument("--max-steps", type=int, default=0, help="Physical command count; 0 runs until stopped")
    p.add_argument("--camera-max-age", type=float, default=.2)
    p.add_argument("--camera-max-skew", type=float, default=.1)
    p.add_argument("--record-dir", type=Path, help="Optional lossless replan inputs and action traces, outside this repository")
    p.add_argument("--gripper-offset-enabled", action="store_true")
    p.add_argument("--skip-gripper-state-offset", action="store_true")
    p.add_argument("--gripper-offset", type=float, default=.5)
    p.add_argument("--gripper-target", choices=["left", "right", "both"], default="both")
    p.add_argument("--gripper-lower", type=float, default=-3.)
    p.add_argument("--gripper-upper", type=float, default=0.)
    p.add_argument("--gripper-state-min", type=float, default=-3.4)
    p.add_argument("--gripper-state-max", type=float, default=0.)
    return p


def validate_args(args):
    import math
    if not .25 <= args.playback_speed <= (2 if args.mode == "sync" else 1):
        raise ValueError("Playback speed must be [0.25,2] for sync, [0.25,1] for async")
    if not 2 <= args.prefix_steps <= 16 or args.delay_buffer < 1 or args.max_steps < 0:
        raise ValueError("Invalid prefix length, delay buffer, or max steps")
    if args.denoise_steps is not None and args.denoise_steps < 1:
        raise ValueError("Denoising steps must be positive")
    if not 1 <= args.smoothing_window <= 31 or args.smoothing_window % 2 != 1:
        raise ValueError("Smoothing window must be odd and in [1,31]")
    if args.compile_action and args.mode == "rtc" and args.method == "vjp":
        raise ValueError("VJP guidance requires eager action inference")
    if args.camera_max_age <= 0 or args.camera_max_skew < 0 or args.guidance_beta < 0:
        raise ValueError("Invalid camera or guidance thresholds")
    values = [args.camera_max_age, args.camera_max_skew, args.guidance_beta, args.gripper_offset,
              args.gripper_lower, args.gripper_upper, args.gripper_state_min, args.gripper_state_max]
    if not all(math.isfinite(v) for v in values) or args.gripper_lower > args.gripper_upper or args.gripper_state_min > args.gripper_state_max:
        raise ValueError("Gripper/camera/guidance values must be finite with ordered bounds")
    if args.record_dir:
        if args.record_dir.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Choose a recording directory outside the code repository")
        if args.record_dir.exists():
            raise FileExistsError("Use a new recording directory for each launch")


class EpisodeCommand(Exception):
    def __init__(self, command):
        self.command = command


def keyboard():
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def main():
    args = parser().parse_args()
    validate_args(args)
    from deploy.policy import read_run, load_policy
    from deploy.controller import AsyncController, resample
    from deploy.robot import connect, ObservationValidator
    import numpy as np
    prepared = read_run(args)
    print(json.dumps(prepared[-1], indent=2))
    if args.check_config:
        return
    if not args.instruction:
        raise ValueError("Supply --instruction for inference")
    policy = load_policy(args, prepared)
    del prepared
    robot = connect(args.robot, args.robot_config)
    validator = ObservationValidator(policy.camera_keys, max_age=args.camera_max_age, max_skew=args.camera_max_skew)
    controller = None
    trace = None
    episode, total = 0, 0
    capture = 0
    def observe(step):
        nonlocal capture
        held = time.monotonic()
        next_message = held
        while True:
            try:
                obs = validator.read(robot)
                break
            except TimeoutError as exc:
                command = keyboard()
                if command in {"e", "r"}:
                    raise EpisodeCommand(command)
                now = time.monotonic()
                if now >= next_message:
                    print(f"Waiting for fresh observations: {exc}", flush=True)
                    next_message = now + 5.
                time.sleep(.02)
        if controller is not None:
            controller.pause_seconds += time.monotonic()-held
        if args.record_dir:
            with (args.record_dir / f"episode_{episode:04d}_step_{step:06d}_capture_{capture:04d}.npz").open("xb") as f:
                np.savez_compressed(f, state=obs["state"], **obs["images"])
            capture += 1
        return obs

    try:
        if args.record_dir:
            args.record_dir.mkdir(parents=True)
            trace = (args.record_dir / "actions.jsonl").open("x")
        while True:
            if not args.no_wait:
                input("Press Enter to start; e + Enter ends, r + Enter resets and restarts: ")
            smoothing = deque(maxlen=args.smoothing_window)
            logical, cursor, actions = 0, 0, None
            restart = False
            try:
                if args.mode != "sync":
                    controller = AsyncController(policy, args.instruction, mode=args.mode, method=args.method,
                        speed=args.playback_speed, beta=args.guidance_beta, prefix_steps=args.prefix_steps, delay_buffer=args.delay_buffer)
                    controller.initialize(observe(0), refresh=lambda: observe(0))
                deadline = time.monotonic()
                while args.max_steps == 0 or total < args.max_steps:
                    command = keyboard()
                    if command in {"e", "r"}:
                        restart = command == "r"
                        break
                    if controller is not None:
                        controller.poll()
                        if controller.needs_observation():
                            controller.submit(observe(controller.logical_step))
                        action = controller.action()
                    else:
                        if actions is None or cursor == len(actions):
                            obs = observe(logical)
                            actions, _ = policy.predict(obs, args.instruction, logical)
                            actions = resample(actions, max(1, int(np.floor(32/args.playback_speed+.5))))
                            logical += 32
                            cursor = 0
                            deadline = time.monotonic()
                        action = actions[cursor].copy()
                    smoothing.append(action.copy())
                    arm_dims = list(range(6))+list(range(7,13))
                    action[arm_dims] = np.mean(np.stack(smoothing)[:, arm_dims], axis=0)
                    if args.execute:
                        robot.publish(action)
                    if trace:
                        trace.write(json.dumps(dict(episode=episode, step=total, action=action.tolist(), published=args.execute)) + "\n")
                        trace.flush()
                    if controller is not None:
                        controller.commit()
                    else:
                        cursor += 1
                    total += 1
                    deadline = max(deadline + 1/30, time.monotonic())
                    time.sleep(max(0., deadline-time.monotonic()))
            except EpisodeCommand as command:
                restart = command.command == "r"
            if controller is not None:
                controller.close()
                controller = None
            if not restart:
                break
            if args.execute:
                robot.reset()
            policy.reset()
            validator.last.clear()
            episode += 1
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            try:
                robot.close()
            finally:
                policy.close()
                if trace:
                    trace.close()
    print(json.dumps(dict(physical_steps=total, published=args.execute)))


if __name__ == "__main__":
    main()
