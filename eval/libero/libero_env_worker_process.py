import argparse
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.libero.bootstrap import setup_libero_paths

setup_libero_paths()

from eval.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=LIBERO_ENV_RESOLUTION)
    args = parser.parse_args()

    seed = None if args.seed in {None, "", "None"} else int(args.seed)
    listener = Listener(args.address, family="AF_UNIX")
    conn = None
    env = None
    try:
        conn = listener.accept()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        task = task_suite.get_task(args.task_id)
        env, task_description = get_libero_env(
            task,
            int(args.resolution),
            seed,
            render_gpu_device_id=int(args.render_gpu_device_id),
        )
        conn.send(("ready", task_description))
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(("ok", env.reset()))
            elif command == "set_init_state":
                conn.send(("ok", env.set_init_state(payload)))
            elif command == "step":
                conn.send(("ok", env.step(payload)))
            elif command == "close":
                env.close()
                env = None
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown command: {command!r}")
    except BaseException:
        if conn is not None:
            try:
                conn.send(("error", traceback.format_exc()))
            except Exception:
                pass
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        listener.close()
        try:
            os.unlink(args.address)
        except OSError:
            pass


if __name__ == "__main__":
    main()
