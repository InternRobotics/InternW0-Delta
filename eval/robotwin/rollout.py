import sys
import ast
import os
import subprocess
import contextlib
import json
import signal
import time

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
import traceback

import yaml
from datetime import datetime
import importlib
import argparse

from generate_episode_instructions import *

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


class EvaluationTimeout(TimeoutError):
    pass


def _model_profile_section(model, name):
    section = getattr(model, "profile_section", None)
    return section(name) if callable(section) else contextlib.nullcontext()


def _record_model_timing(model, name, elapsed, *, count_name=None):
    recorder = getattr(model, "add_client_timing", None)
    if callable(recorder):
        recorder(name, elapsed, count_name=count_name)


@contextlib.contextmanager
def _time_limit(seconds, label):
    seconds = int(seconds or 0)
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise EvaluationTimeout(f"{label} exceeded {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _atomic_write_json(path, payload):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def _load_progress(path, *, enabled):
    path = Path(path)
    if not enabled or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        print(f"Ignoring unreadable progress file {path}: {exc}", flush=True)
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_close_env(task_env, *, clear_cache=False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception as exc:
        print(f"close_env warning: {exc}", flush=True)


def _safe_close_video(task_env):
    try:
        if getattr(task_env, "eval_video_ffmpeg", None) is not None:
            task_env._del_eval_video_ffmpeg()
    except Exception as exc:
        print(f"ffmpeg close warning: {exc}", flush=True)


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    resume = parse_bool(usr_args.get("resume", True))
    expert_timeout_sec = int(usr_args.get("expert_timeout_sec", 300))
    rollout_setup_timeout_sec = int(usr_args.get("rollout_setup_timeout_sec", 300))
    episode_timeout_sec = int(usr_args.get("episode_timeout_sec", 1800))
    heartbeat_interval_sec = float(usr_args.get("heartbeat_interval_sec", 10))
    max_seed_attempts = int(usr_args.get("max_seed_attempts", 20000))
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)
    result_suffix = _result_suffix_from_task_config(task_config)
    progress_path = save_dir / f"_progress_{result_suffix}.json"
    heartbeat_path = save_dir / f"_heartbeat_{result_suffix}"
    progress = _load_progress(progress_path, enabled=resume)
    progress.update(
        {
            "version": 1,
            "task_name": task_name,
            "task_config": task_config,
            "target_episodes": eval_num_episodes,
            "status": "model_loading",
            "updated_at": datetime.now().isoformat(),
        }
    )
    _atomic_write_json(progress_path, progress)
    heartbeat_path.touch()

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + ", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + ", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]
    configured_start_seed = usr_args.get("eval_start_seed")
    st_seed = (
        100000 * (1 + seed)
        if configured_start_seed is None or str(configured_start_seed).strip() == ""
        else int(configured_start_seed)
    )
    if st_seed < 0:
        raise ValueError(f"`eval_start_seed` must be non-negative, got: {st_seed}")
    suc_nums = []
    test_num = eval_num_episodes

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   progress_path=progress_path,
                                   heartbeat_path=heartbeat_path,
                                   resume=resume,
                                   expert_timeout_sec=expert_timeout_sec,
                                   rollout_setup_timeout_sec=rollout_setup_timeout_sec,
                                   episode_timeout_sec=episode_timeout_sec,
                                   heartbeat_interval_sec=heartbeat_interval_sec,
                                   max_seed_attempts=max_seed_attempts)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    final_progress = _load_progress(progress_path, enabled=True)
    final_progress.update(
        {
            "status": "complete",
            "completed_episodes": test_num,
            "successes": int(suc_num),
            "next_seed": int(st_seed),
            "updated_at": datetime.now().isoformat(),
        }
    )
    _atomic_write_json(progress_path, final_progress)
    heartbeat_path.touch()
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                progress_path=None,
                heartbeat_path=None,
                resume=True,
                expert_timeout_sec=300,
                rollout_setup_timeout_sec=300,
                episode_timeout_sec=1800,
                heartbeat_interval_sec=10,
                max_seed_attempts=20000):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    progress_path = Path(progress_path)
    heartbeat_path = Path(heartbeat_path)
    saved = _load_progress(progress_path, enabled=resume)
    completed = max(0, min(int(saved.get("completed_episodes", 0)), int(test_num)))
    successes = max(0, min(int(saved.get("successes", 0)), completed))
    now_seed = max(int(st_seed), int(saved.get("next_seed", st_seed)))
    seed_attempts = max(0, int(saved.get("seed_attempts", 0)))
    skipped_seeds = saved.get("skipped_seeds", [])
    if not isinstance(skipped_seeds, list):
        skipped_seeds = []

    TASK_ENV.suc = successes
    TASK_ENV.test_num = completed
    now_id = completed
    last_heartbeat = 0.0

    def heartbeat(force=False):
        nonlocal last_heartbeat
        now = time.monotonic()
        if force or now - last_heartbeat >= max(1.0, float(heartbeat_interval_sec)):
            heartbeat_path.touch()
            last_heartbeat = now

    def write_progress(status, *, current_seed=None, next_seed=None):
        payload = {
            "version": 1,
            "task_name": task_name,
            "task_config": args["task_config"],
            "target_episodes": int(test_num),
            "start_seed": int(st_seed),
            "completed_episodes": int(completed),
            "successes": int(TASK_ENV.suc),
            "next_seed": int(now_seed if next_seed is None else next_seed),
            "current_seed": None if current_seed is None else int(current_seed),
            "seed_attempts": int(seed_attempts),
            "skipped_seeds": skipped_seeds,
            "status": str(status),
            "updated_at": datetime.now().isoformat(),
        }
        _atomic_write_json(progress_path, payload)
        heartbeat(force=True)

    def skip_seed(reason, exc=None):
        nonlocal now_seed
        record = {
            "seed": int(now_seed),
            "reason": str(reason),
            "time": datetime.now().isoformat(),
        }
        if exc is not None:
            record["error"] = repr(exc)
        skipped_seeds.append(record)
        now_seed += 1
        write_progress("seed_skipped", next_seed=now_seed)

    eval_video_save_dir = args.get("eval_video_save_dir")
    if eval_video_save_dir:
        partial_video = Path(eval_video_save_dir) / f"episode{completed}.mp4"
        if partial_video.is_file():
            partial_video.unlink()
    write_progress("ready", next_seed=now_seed)
    print(
        f"Resume state: completed={completed}/{test_num}, successes={TASK_ENV.suc}, "
        f"next_seed={now_seed}, seed_attempts={seed_attempts}",
        flush=True,
    )

    while completed < test_num:
        if seed_attempts >= int(max_seed_attempts):
            write_progress("max_seed_attempts_exceeded", current_seed=now_seed)
            raise RuntimeError(
                f"Exceeded max_seed_attempts={max_seed_attempts} after "
                f"{completed}/{test_num} completed episodes"
            )
        seed_attempts += 1
        seed_attempt_started = time.monotonic()
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        episode_info = None
        write_progress("expert_check", current_seed=now_seed, next_seed=now_seed)
        expert_started = time.monotonic()
        try:
            with _time_limit(expert_timeout_sec, f"expert seed {now_seed}"):
                with _model_profile_section(model, "expert/setup_demo"):
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                with _model_profile_section(model, "expert/play_once"):
                    episode_info = TASK_ENV.play_once()
            expert_valid = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
        except (UnStableError, EvaluationTimeout) as exc:
            _safe_close_env(TASK_ENV)
            args["render_freq"] = render_freq
            print(f"Skipping expert seed {now_seed}: {exc}", flush=True)
            skip_seed(type(exc).__name__, exc)
            continue
        except Exception as exc:
            print(" -------------")
            print("Error: ", exc)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            _safe_close_env(TASK_ENV)
            args["render_freq"] = render_freq
            skip_seed("expert_exception", exc)
            continue
        _safe_close_env(TASK_ENV)
        expert_elapsed = time.monotonic() - expert_started
        if not expert_valid:
            args["render_freq"] = render_freq
            skip_seed("expert_plan_failed")
            continue

        args["render_freq"] = render_freq
        write_progress("rollout_setup", current_seed=now_seed, next_seed=now_seed)
        try:
            with _time_limit(rollout_setup_timeout_sec, f"rollout setup seed {now_seed}"):
                with _model_profile_section(model, "rollout/setup_demo"):
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except (UnStableError, EvaluationTimeout) as exc:
            _safe_close_env(TASK_ENV)
            print(f"Skipping rollout setup seed {now_seed}: {exc}", flush=True)
            skip_seed(type(exc).__name__, exc)
            continue
        except Exception as exc:
            print(" -------------")
            print("Error: ", exc)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            _safe_close_env(TASK_ENV)
            skip_seed("rollout_setup_exception", exc)
            continue
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        current_video_path = None
        if TASK_ENV.eval_video_path is not None:
            episode_idx = TASK_ENV.test_num
            current_video_path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        with _model_profile_section(model, "rollout/reset_model"):
            reset_func(model)
        write_progress("rollout", current_seed=now_seed, next_seed=now_seed)
        rollout_started = time.monotonic()
        try:
            with _time_limit(episode_timeout_sec, f"rollout seed {now_seed}"):
                while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                    heartbeat()
                    need_obs = True
                    if (
                        skip_get_obs_within_replan
                        and TASK_ENV.eval_video_path is None
                        and hasattr(model, "should_request_observation")
                    ):
                        need_obs = bool(model.should_request_observation())

                    observation = None
                    if need_obs:
                        observation_started = time.perf_counter()
                        with _model_profile_section(model, "rollout/get_obs"):
                            observation = TASK_ENV.get_obs()
                        _record_model_timing(
                            model,
                            "get_obs_s",
                            time.perf_counter() - observation_started,
                            count_name="get_obs_calls",
                        )
                    with _model_profile_section(model, "rollout/eval_chunk"):
                        eval_func(TASK_ENV, model, observation)
                    heartbeat()
                    if TASK_ENV.eval_success:
                        succ = True
                        break
        except EvaluationTimeout as exc:
            _safe_close_video(TASK_ENV)
            if current_video_path is not None and current_video_path.is_file():
                current_video_path.unlink()
            _safe_close_env(TASK_ENV)
            print(f"Skipping timed-out rollout seed {now_seed}: {exc}", flush=True)
            skip_seed("episode_timeout", exc)
            continue
        except Exception as exc:
            _safe_close_video(TASK_ENV)
            if current_video_path is not None and current_video_path.is_file():
                current_video_path.unlink()
            _safe_close_env(TASK_ENV)
            print(" -------------")
            print("Rollout Error: ", exc)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            write_progress("rollout_exception", current_seed=now_seed, next_seed=now_seed)
            raise
        rollout_elapsed = time.monotonic() - rollout_started

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
            if current_video_path is None or not current_video_path.exists():
                raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
            is_randomized = "randomized" in str(args["task_config"]).lower()
            renamed_video_path = (
                Path(TASK_ENV.eval_video_path)
                / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4"
            )
            os.replace(current_video_path, renamed_video_path)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        completed += 1
        now_id = completed
        TASK_ENV.test_num += 1
        now_seed += 1
        write_progress("episode_complete", next_seed=now_seed)
        _safe_close_env(TASK_ENV, clear_cache=(completed % clear_cache_freq == 0))

        if TASK_ENV.render_freq and getattr(TASK_ENV, "viewer", None) is not None:
            TASK_ENV.viewer.close()

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, next seed: \033[90m{now_seed}\033[0m\n"
        )
        model_timing = (
            model.get_timing_rollout()
            if hasattr(model, "get_timing_rollout")
            else {}
        )
        print(
            "Episode timing: "
            f"expert_s={expert_elapsed:.2f} rollout_s={rollout_elapsed:.2f} "
            f"total_s={time.monotonic() - seed_attempt_started:.2f} "
            f"model={model_timing}",
            flush=True,
        )

    write_progress("complete", next_seed=now_seed)
    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = ast.literal_eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
