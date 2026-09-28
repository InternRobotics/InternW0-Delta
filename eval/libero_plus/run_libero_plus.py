import atexit
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_PLUS_ROOT = Path(
    os.path.expanduser(
        os.path.expandvars(
            os.environ.get(
                "WAM_LIBERO_REPO_ROOT",
                str(PROJECT_ROOT / "third_party" / "LIBERO-plus"),
            )
        )
    )
)
LIBERO_PLUS_CONFIG_PATH = Path(
    os.path.expanduser(
        os.path.expandvars(
            os.environ.get(
                "LIBERO_CONFIG_PATH",
                str(PROJECT_ROOT / ".cache" / "libero_plus"),
            )
        )
    )
)
NUMBA_CACHE_DIR = Path(os.path.expandvars(os.path.expanduser(
    os.environ.get("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "numba"))
)))
MPLCONFIGDIR = Path(os.path.expandvars(os.path.expanduser(
    os.environ.get("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
)))
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ["WAM_LIBERO_REPO"] = "LIBERO-plus"
if LIBERO_PLUS_ROOT.exists():
    sys.path.insert(0, str(LIBERO_PLUS_ROOT))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_PLUS_CONFIG_PATH))

from eval.libero.bootstrap import setup_libero_paths

setup_libero_paths()


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.device",
        "EVALUATION.num_trials",
        "EVALUATION.output_dir",
        "EVALUATION.results_jsonl_path",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.render_gpu_device_id",
        "EVALUATION.env_worker_cuda_visible_devices",
        "PAIR.socket_path",
        "PAIR.task_file",
        "PAIR.gpu_id",
        "PAIR.worker_label",
        "PAIR.profiler_rank",
        "PAIR.profiler_world_size",
        "PAIR.log_file",
        "PAIR.client_id",
        "PAIR.server_id",
        "PAIR.max_clients",
        "PAIR.persistent",
        "PAIR.health_path",
        "PAIR.health_interval_s",
        "PAIR.request_timeout_s",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError("Hydra task choice is empty. Please pass task=libero.")
    return str(task_choice)


def _csv_or_empty(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, ListConfig)):
        return ",".join(str(item) for item in value)
    return str(value)


def _detect_gpus() -> list[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x != ""]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _resolve_gpu_list(manager: DictConfig) -> list[str]:
    explicit = _csv_or_empty(manager.get("model_gpus", None))
    if explicit:
        return [x for x in explicit.split(",") if x != ""]

    available = _detect_gpus()
    num_gpus = manager.get("num_gpus", "auto")
    if num_gpus == "auto":
        if not available:
            raise RuntimeError("Could not detect GPUs. Set CUDA_VISIBLE_DEVICES or MULTIRUN.model_gpus.")
        return available
    n = int(num_gpus)
    if n <= 0:
        raise ValueError(f"MULTIRUN.num_gpus must be positive or auto, got {num_gpus}")
    if available:
        if len(available) < n:
            raise ValueError(f"Requested {n} GPUs, but only {len(available)} are visible: {available}")
        return available[:n]
    return [str(i) for i in range(n)]


def _resolve_workers_per_gpu(manager: DictConfig) -> int:
    workers_per_gpu = int(manager.get("workers_per_gpu", 1) or 1)
    if workers_per_gpu <= 0:
        raise ValueError(f"MULTIRUN.workers_per_gpu must be positive, got {workers_per_gpu}")
    return workers_per_gpu


def _resolve_policy_topology(manager: DictConfig) -> tuple[str, int, int, int]:
    """Return (mode, worker groups/GPU, server groups/GPU, client slots/GPU)."""
    mode = str(manager.get("policy_server_mode", "dedicated"))
    if mode not in {"dedicated", "shared_per_gpu"}:
        raise ValueError(
            "MULTIRUN.policy_server_mode must be dedicated or shared_per_gpu, "
            f"got {mode!r}"
        )
    workers_per_gpu = _resolve_workers_per_gpu(manager)
    if mode == "dedicated":
        return mode, workers_per_gpu, workers_per_gpu, workers_per_gpu

    clients_per_server = int(
        manager.get("shared_clients_per_server", 2)
        or 2
    )
    if not 1 <= clients_per_server <= 4:
        raise ValueError(
            "shared_per_gpu requires 1 <= MULTIRUN.shared_clients_per_server <= 4"
        )
    return mode, workers_per_gpu, workers_per_gpu, workers_per_gpu * clients_per_server


def _iter_workers(gpu_ids: list[str], workers_per_gpu: int) -> list[tuple[str, int]]:
    return [(gpu_id, worker_idx) for worker_idx in range(workers_per_gpu) for gpu_id in gpu_ids]


def _worker_label(gpu_id: str, worker_idx: int) -> str:
    return f"gpu{gpu_id}_w{worker_idx}"


def _shared_worker_and_client(client_slot: int, clients_per_server: int) -> tuple[int, int]:
    return divmod(int(client_slot), int(clients_per_server))


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    expected_counts = json.loads(Path(__file__).with_name("suites.json").read_text())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            benchmark_stdout = StringIO()
            with redirect_stdout(benchmark_stdout):
                task_suite = benchmark_dict[suite_name]()
            for line in benchmark_stdout.getvalue().splitlines():
                if "using task orders" in line:
                    print(f"[info] {suite_name}: task order list omitted ({int(task_suite.n_tasks)} tasks)")
                elif line.strip():
                    print(line)
            n_tasks = int(task_suite.n_tasks)
            if suite_name in expected_counts and n_tasks != expected_counts[suite_name]:
                raise ValueError(
                    f"{suite_name}: expected {expected_counts[suite_name]} tasks, got {n_tasks}. "
                    "Run bash eval/libero_plus/setup.sh to install the supported benchmark."
                )
            print(f"{suite_name}: {n_tasks} tasks")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1
    print(f"Task list created: {output_file} ({total_tasks} tasks)")
    return output_file


def _read_tasks(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"Task file contains duplicate tasks: {path}")
    for task in tasks:
        fields = task.split(",")
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ValueError(f"Expected suite,task_id in {path}, got {task!r}")
    return tasks


def _socket_dir(output_dir: Path) -> Path:
    override = os.environ.get("WAM_LP_SOCKET_DIR")
    socket_dir = (
        Path(os.path.expanduser(os.path.expandvars(override)))
        if override
        else output_dir / "sockets"
    )
    if not override and len(os.fsencode(socket_dir.resolve())) > 70:
        socket_dir = Path(tempfile.mkdtemp(prefix="internw0-"))
        atexit.register(shutil.rmtree, socket_dir, ignore_errors=True)
    if len(os.fsencode(socket_dir.resolve())) > 80:
        raise ValueError("Socket directory is too long; set WAM_LP_SOCKET_DIR to a short local path.")
    socket_dir.mkdir(parents=True, exist_ok=True)
    return socket_dir


def _socket_path(socket_dir: Path, gpu_id: str, worker_idx: int) -> Path:
    name = hashlib.sha256(f"{gpu_id}:{worker_idx}".encode()).hexdigest()[:16]
    return socket_dir / f"{name}.sock"


def _split_tasks(
    task_file: Path,
    gpu_ids: list[str],
    workers_per_gpu: int,
    output_dir: Path,
) -> dict[tuple[str, int], Path]:
    tasks = _read_tasks(task_file)
    if not tasks:
        raise ValueError(f"Task file is empty: {task_file}")
    split_dir = output_dir / "task_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    workers = _iter_workers(gpu_ids, workers_per_gpu)
    split_files = {
        (gpu, worker_idx): split_dir / f"gpu{gpu}_worker{worker_idx}.tasks"
        for gpu, worker_idx in workers
    }
    handles = {worker: path.open("w", encoding="utf-8") for worker, path in split_files.items()}
    task_gpu_map = output_dir / "task_gpu_map.txt"
    with task_gpu_map.open("w", encoding="utf-8") as map_f:
        try:
            for idx, task_line in enumerate(tasks):
                gpu, worker_idx = workers[idx % len(workers)]
                handles[(gpu, worker_idx)].write(task_line + "\n")
                map_f.write(f"{task_line}:{_worker_label(gpu, worker_idx)}\n")
        finally:
            for handle in handles.values():
                handle.close()
    return split_files


def _child_env(gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    # Cluster launchers inject distributed rank variables for the outer job.
    # LIBERO Plus children are independent simulator/model processes, not DDP
    # ranks; an empty or nonzero inherited rank changes seeding or can fail
    # integer parsing in set_global_seed.
    for rank_name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "SLURM_PROCID",
    ):
        env.pop(rank_name, None)
    src_path = PROJECT_ROOT / "src"
    parts = [str(src_path), str(PROJECT_ROOT), str(LIBERO_PLUS_ROOT)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    env.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_PLUS_CONFIG_PATH))
    env["WAM_LIBERO_REPO"] = "LIBERO-plus"
    env["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    # robosuite's import-time check and EGL backend both expect the physical
    # EGL device id, even though CUDA_VISIBLE_DEVICES exposes one GPU locally.
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    env.setdefault("KMP_BLOCKTIME", "0")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def _base_child_overrides(
    *,
    config_name: str,
    task_choice: str,
    ckpt: str,
    dataset_stats_path: str | None,
    output_dir: Path,
    extra_overrides: list[str],
) -> list[str]:
    args = [
        "--config-name",
        config_name,
        f"task={task_choice}",
        f"ckpt={ckpt}",
        "gpu_id=0",
        "EVALUATION.device=cuda:0",
        "EVALUATION.render_gpu_device_id=0",
        f"EVALUATION.output_dir={output_dir}",
        f"EVALUATION.results_jsonl_path={output_dir / 'task_results.jsonl'}",
    ]
    if dataset_stats_path:
        args.append(f"EVALUATION.dataset_stats_path={dataset_stats_path}")
    args.extend(extra_overrides)
    return args


def _summarize(output_dir: Path) -> None:
    summary_script = PROJECT_ROOT / "eval" / "libero_plus" / "summarize_results.py"
    subprocess.run(
        [sys.executable, str(summary_script), "--output_dir", str(output_dir)],
        check=True,
        text=True,
    )


def _aggregate_profiles(output_dir: Path) -> None:
    aggregate_script = PROJECT_ROOT / "eval" / "aggregate_profiler.py"
    for role in ("client", "server"):
        profile_dir = output_dir / "profiler" / role
        if not any(profile_dir.glob("rank_*/summary.json")):
            continue
        subprocess.run(
            [sys.executable, str(aggregate_script), str(profile_dir)],
            check=True,
            text=True,
        )


def _build_pair_commands(
    *,
    config_name: str,
    task_choice: str,
    ckpt: str,
    dataset_stats_path: str | None,
    output_dir: Path,
    extra_overrides: list[str],
    socket_path: Path,
    split_file: Path,
    gpu_id: str,
    worker_idx: int,
    profiler_rank: int,
    profiler_world_size: int,
    client_log_path: Path,
) -> tuple[list[str], list[str]]:
    base_args = _base_child_overrides(
        config_name=config_name,
        task_choice=task_choice,
        ckpt=ckpt,
        dataset_stats_path=dataset_stats_path,
        output_dir=output_dir,
        extra_overrides=extra_overrides,
    )
    worker_label = _worker_label(gpu_id, worker_idx)
    profiler_args = [
        f"PAIR.worker_label={worker_label}",
        f"PAIR.profiler_rank={int(profiler_rank)}",
        f"PAIR.profiler_world_size={int(profiler_world_size)}",
    ]
    server_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "eval" / "libero_plus" / "server.py"),
        *base_args,
        f"PAIR.socket_path={socket_path}",
        *profiler_args,
    ]
    client_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "eval" / "libero_plus" / "client.py"),
        *base_args,
        f"PAIR.socket_path={socket_path}",
        f"PAIR.task_file={split_file}",
        f"PAIR.gpu_id={worker_label}",
        f"PAIR.log_file={client_log_path}",
        *profiler_args,
    ]
    return server_cmd, client_cmd


def _build_shared_server_command(
    *,
    config_name: str,
    task_choice: str,
    ckpt: str,
    dataset_stats_path: str | None,
    output_dir: Path,
    extra_overrides: list[str],
    socket_path: Path,
    health_path: Path,
    gpu_id: str,
    worker_idx: int,
    max_clients: int,
    profiler_rank: int,
    profiler_world_size: int,
) -> list[str]:
    base_args = _base_child_overrides(
        config_name=config_name,
        task_choice=task_choice,
        ckpt=ckpt,
        dataset_stats_path=dataset_stats_path,
        output_dir=output_dir,
        extra_overrides=extra_overrides,
    )
    server_label = f"gpu{gpu_id}_w{worker_idx}_shared"
    return [
        sys.executable,
        str(PROJECT_ROOT / "eval" / "libero_plus" / "server.py"),
        *base_args,
        f"PAIR.socket_path={socket_path}",
        f"PAIR.health_path={health_path}",
        f"PAIR.server_id={server_label}",
        f"PAIR.gpu_id={server_label}",
        f"PAIR.worker_label={server_label}",
        f"PAIR.max_clients={int(max_clients)}",
        "PAIR.persistent=true",
        f"PAIR.profiler_rank={int(profiler_rank)}",
        f"PAIR.profiler_world_size={int(profiler_world_size)}",
    ]


def _build_shared_client_command(
    *,
    config_name: str,
    task_choice: str,
    ckpt: str,
    dataset_stats_path: str | None,
    output_dir: Path,
    extra_overrides: list[str],
    socket_path: Path,
    split_file: Path,
    gpu_id: str,
    worker_idx: int,
    client_idx: int,
    profiler_rank: int,
    profiler_world_size: int,
    client_log_path: Path,
) -> list[str]:
    base_args = _base_child_overrides(
        config_name=config_name,
        task_choice=task_choice,
        ckpt=ckpt,
        dataset_stats_path=dataset_stats_path,
        output_dir=output_dir,
        extra_overrides=extra_overrides,
    )
    client_label = f"gpu{gpu_id}_w{worker_idx}_c{client_idx}"
    return [
        sys.executable,
        str(PROJECT_ROOT / "eval" / "libero_plus" / "client.py"),
        *base_args,
        f"PAIR.socket_path={socket_path}",
        f"PAIR.task_file={split_file}",
        f"PAIR.gpu_id={client_label}",
        f"PAIR.worker_label={client_label}",
        f"PAIR.client_id={client_label}",
        f"PAIR.log_file={client_log_path}",
        f"PAIR.profiler_rank={int(profiler_rank)}",
        f"PAIR.profiler_world_size={int(profiler_world_size)}",
    ]


def _read_health(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _wait_for_shared_servers(
    server_procs: dict[tuple[str, int], subprocess.Popen],
    health_paths: dict[tuple[str, int], Path],
    socket_paths: dict[tuple[str, int], Path],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    pending = set(server_procs)
    while pending:
        for gpu_id in list(pending):
            proc = server_procs[gpu_id]
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Shared LIBERO server {gpu_id} exited before ready: rc={proc.returncode}"
                )
            health = _read_health(health_paths[gpu_id])
            if (
                socket_paths[gpu_id].is_socket()
                and health is not None
                and bool(health.get("ready", False))
            ):
                pending.remove(gpu_id)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for shared LIBERO servers: {sorted(pending)}"
            )
        time.sleep(0.5)


def _launch_shared_subprocess_workers(
    *,
    cfg: DictConfig,
    config_name: str,
    task_choice: str,
    gpu_ids: list[str],
    workers_per_gpu: int,
    clients_per_server: int,
    clients_per_gpu: int,
    split_files: dict[tuple[str, int], Path],
    output_dir: Path,
    dataset_stats_path: str | None,
    extra_overrides: list[str],
) -> None:
    started_perf = time.perf_counter()
    server_procs: dict[tuple[str, int], subprocess.Popen] = {}
    client_procs: dict[str, subprocess.Popen] = {}
    log_handles = []
    socket_dir = _socket_dir(output_dir)
    health_dir = output_dir / "server_health"
    health_dir.mkdir(parents=True, exist_ok=True)
    socket_paths: dict[tuple[str, int], Path] = {}
    health_paths: dict[tuple[str, int], Path] = {}
    failures: list[tuple[str, int | str]] = []
    timing: dict[str, object] = {
        "policy_server_mode": "shared_per_gpu",
        "workers_per_gpu": int(workers_per_gpu),
        "servers_per_gpu": int(workers_per_gpu),
        "clients_per_server": int(clients_per_server),
        "clients_per_gpu": int(clients_per_gpu),
        "total_servers": 0,
        "total_clients": 0,
    }
    active_clients = [
        (gpu_id, client_slot)
        for gpu_id, client_slot in _iter_workers(gpu_ids, clients_per_gpu)
        if len(_read_tasks(split_files[(gpu_id, client_slot)])) > 0
    ]
    clients_by_server: dict[tuple[str, int], list[tuple[int, int]]] = {
        (gpu_id, worker_idx): []
        for gpu_id in gpu_ids
        for worker_idx in range(workers_per_gpu)
    }
    for gpu_id, client_slot in active_clients:
        worker_idx, client_idx = _shared_worker_and_client(
            client_slot, clients_per_server
        )
        clients_by_server[(gpu_id, worker_idx)].append((client_slot, client_idx))
    active_servers = [
        server_key for server_key, clients in clients_by_server.items() if clients
    ]
    timing["total_servers"] = len(active_servers)
    timing["total_clients"] = len(active_clients)
    try:
        for profiler_rank, (gpu_id, worker_idx) in enumerate(active_servers):
            socket_path = _socket_path(socket_dir, gpu_id, worker_idx)
            health_path = health_dir / f"gpu{gpu_id}_w{worker_idx}.json"
            socket_path.unlink(missing_ok=True)
            health_path.unlink(missing_ok=True)
            socket_paths[(gpu_id, worker_idx)] = socket_path
            health_paths[(gpu_id, worker_idx)] = health_path
            server_log_path = (
                output_dir / "task_logs" / f"server_gpu{gpu_id}_w{worker_idx}_shared.log"
            )
            server_log = server_log_path.open("w", encoding="utf-8")
            log_handles.append(server_log)
            server_clients = clients_by_server[(gpu_id, worker_idx)]
            server_cmd = _build_shared_server_command(
                config_name=config_name,
                task_choice=task_choice,
                ckpt=str(cfg.ckpt),
                dataset_stats_path=dataset_stats_path,
                output_dir=output_dir,
                extra_overrides=extra_overrides,
                socket_path=socket_path,
                health_path=health_path,
                gpu_id=gpu_id,
                worker_idx=worker_idx,
                max_clients=len(server_clients),
                profiler_rank=profiler_rank,
                profiler_world_size=len(active_servers),
            )
            print(
                f"Launching shared server gpu={gpu_id} worker={worker_idx} "
                f"clients={len(server_clients)}: "
                f"log={server_log.name}",
                flush=True,
            )
            server_procs[(gpu_id, worker_idx)] = subprocess.Popen(
                server_cmd,
                env=_child_env(gpu_id),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        _wait_for_shared_servers(
            server_procs,
            health_paths,
            socket_paths,
            timeout_s=float(cfg.MULTIRUN.get("server_ready_timeout_s", 1800) or 1800),
        )
        ready_perf = time.perf_counter()
        timing["server_ready_s"] = ready_perf - started_perf
        print(
            f"LIBERO_SHARED_ALL_SERVERS_READY startup_seconds={ready_perf - started_perf:.3f} "
            f"servers={len(server_procs)}",
            flush=True,
        )

        for profiler_rank, (gpu_id, client_slot) in enumerate(active_clients):
            worker_idx, client_idx = _shared_worker_and_client(
                client_slot, clients_per_server
            )
            client_label = f"gpu{gpu_id}_w{worker_idx}_c{client_idx}"
            split_file = split_files[(gpu_id, client_slot)]
            client_log_path = output_dir / "task_logs" / f"client_{client_label}.log"
            client_log = client_log_path.open("w", encoding="utf-8")
            log_handles.append(client_log)
            client_cmd = _build_shared_client_command(
                config_name=config_name,
                task_choice=task_choice,
                ckpt=str(cfg.ckpt),
                dataset_stats_path=dataset_stats_path,
                output_dir=output_dir,
                extra_overrides=extra_overrides,
                socket_path=socket_paths[(gpu_id, worker_idx)],
                split_file=split_file,
                gpu_id=gpu_id,
                worker_idx=worker_idx,
                client_idx=client_idx,
                profiler_rank=profiler_rank,
                profiler_world_size=len(active_clients),
                client_log_path=client_log_path,
            )
            print(
                f"Launching shared client {client_label}: "
                f"tasks={len(_read_tasks(split_file))} log={client_log.name}",
                flush=True,
            )
            client_procs[client_label] = subprocess.Popen(
                client_cmd,
                env=_child_env(gpu_id),
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        for client_label, proc in client_procs.items():
            rc = proc.wait()
            if rc != 0:
                failures.append((client_label, rc))
        clients_done_perf = time.perf_counter()
        timing["client_stage_s"] = clients_done_perf - ready_perf
        print(
            f"LIBERO_SHARED_ALL_CLIENTS_DONE client_seconds={clients_done_perf - ready_perf:.3f} "
            f"clients={len(client_procs)}",
            flush=True,
        )

        for server_key, proc in server_procs.items():
            if proc.poll() is None:
                proc.terminate()
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait(timeout=30)
            if rc != 0:
                failures.append((f"server:{server_key}", rc))

        validations: dict[str, object] = {}
        for server_key in active_servers:
            gpu_id, worker_idx = server_key
            health = _read_health(health_paths[server_key])
            expected_clients = len(clients_by_server[server_key])
            problems: list[str] = []
            if health is None:
                problems.append("missing_health")
            else:
                if int(health.get("model_loads", 0) or 0) != 1:
                    problems.append("model_loads_not_one")
                if int(health.get("sessions_created", 0) or 0) != expected_clients:
                    problems.append("session_count_mismatch")
                if int(health.get("sessions_closed", 0) or 0) != expected_clients:
                    problems.append("session_close_mismatch")
                if int(health.get("peak_sessions", 0) or 0) != expected_clients:
                    problems.append("peak_sessions_mismatch")
                if health.get("fatal_error") not in (None, ""):
                    problems.append("fatal_error")
            validation_key = f"gpu{gpu_id}_w{worker_idx}"
            validations[validation_key] = {
                "expected_clients": expected_clients,
                "problems": problems,
                "health": health,
            }
            if problems:
                failures.append((f"validation:{validation_key}", ",".join(problems)))
        (output_dir / "shared_server_validation.json").write_text(
            json.dumps(validations, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        _summarize(output_dir)
        _aggregate_profiles(output_dir)
        if failures:
            raise RuntimeError(f"LIBERO-plus shared eval failed: {failures}")
    finally:
        for proc in client_procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in server_procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in list(client_procs.values()) + list(server_procs.values()):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for handle in log_handles:
            handle.close()
        timing["manager_total_s"] = time.perf_counter() - started_perf
        timing["failures"] = failures
        (output_dir / "topology_timing.json").write_text(
            json.dumps(timing, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _launch_subprocess_workers(
    *,
    cfg: DictConfig,
    config_name: str,
    task_choice: str,
    gpu_ids: list[str],
    workers_per_gpu: int,
    split_files: dict[tuple[str, int], Path],
    output_dir: Path,
    dataset_stats_path: str | None,
    extra_overrides: list[str],
) -> None:
    server_procs: dict[str, subprocess.Popen] = {}
    client_procs: dict[str, subprocess.Popen] = {}
    log_handles = []
    socket_dir = _socket_dir(output_dir)
    try:
        active_workers = [
            (gpu_id, worker_idx)
            for gpu_id, worker_idx in _iter_workers(gpu_ids, workers_per_gpu)
            if len(_read_tasks(split_files[(gpu_id, worker_idx)])) > 0
        ]
        for profiler_rank, (gpu_id, worker_idx) in enumerate(active_workers):
            split_file = split_files[(gpu_id, worker_idx)]
            worker_label = _worker_label(gpu_id, worker_idx)
            socket_path = _socket_path(socket_dir, gpu_id, worker_idx)
            try:
                socket_path.unlink()
            except FileNotFoundError:
                pass
            env = _child_env(gpu_id)
            server_log_path = output_dir / "task_logs" / f"server_gpu{gpu_id}_w{worker_idx}.log"
            client_log_path = output_dir / "task_logs" / f"client_gpu{gpu_id}_w{worker_idx}.log"
            server_log = server_log_path.open("w", encoding="utf-8")
            client_log = client_log_path.open("w", encoding="utf-8")
            log_handles.extend([server_log, client_log])
            server_cmd, client_cmd = _build_pair_commands(
                config_name=config_name,
                task_choice=task_choice,
                ckpt=str(cfg.ckpt),
                dataset_stats_path=dataset_stats_path,
                output_dir=output_dir,
                extra_overrides=extra_overrides,
                socket_path=socket_path,
                split_file=split_file,
                gpu_id=gpu_id,
                worker_idx=worker_idx,
                profiler_rank=profiler_rank,
                profiler_world_size=len(active_workers),
                client_log_path=client_log_path,
            )

            print(f"Launching worker {worker_label} server: log={server_log.name}")
            server_procs[worker_label] = subprocess.Popen(
                server_cmd,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"Launching worker {worker_label} client: log={client_log.name}")
            client_procs[worker_label] = subprocess.Popen(
                client_cmd,
                env=env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        failures = []
        for worker_label, proc in client_procs.items():
            rc = proc.wait()
            if rc != 0:
                failures.append((worker_label, rc))
        for worker_label, proc in server_procs.items():
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                rc = proc.wait(timeout=30)
            if rc != 0:
                failures.append((f"server:{worker_label}", rc))
        _summarize(output_dir)
        _aggregate_profiles(output_dir)
        if failures:
            raise RuntimeError(f"LIBERO-plus eval failed: {failures}")
    finally:
        for proc in list(client_procs.values()) + list(server_procs.values()):
            if proc.poll() is None:
                proc.terminate()
        for handle in log_handles:
            handle.close()


def _tmux_quote_env(env: dict[str, str]) -> str:
    keys = [
        "PYTHONPATH",
        "LIBERO_CONFIG_PATH",
        "WAM_LIBERO_REPO",
        "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR",
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_WAIT_POLICY",
        "KMP_BLOCKTIME",
        "MALLOC_ARENA_MAX",
    ]
    return " ".join(f"export {key}={shlex.quote(str(env[key]))};" for key in keys if key in env)


def _tmux_worker_command(
    *,
    env: dict[str, str],
    server_cmd: list[str],
    client_cmd: list[str],
    server_log_path: Path,
    client_log_path: Path,
    status_file: Path,
    socket_path: Path,
    server_ready_timeout_s: int,
) -> str:
    quoted_socket = shlex.quote(str(socket_path))
    quoted_status_file = shlex.quote(str(status_file))
    return " ".join(
        [
            "set +e;",
            f"cd {shlex.quote(str(PROJECT_ROOT))};",
            _tmux_quote_env(env),
            f"rm -f {quoted_socket};",
            f"mkdir -p {shlex.quote(str(status_file.parent))};",
            f"{shlex.join(server_cmd)} > {shlex.quote(str(server_log_path))} 2>&1 &",
            "server_pid=$!;",
            f"deadline=$(( $(date +%s) + {int(server_ready_timeout_s)} ));",
            f"echo waiting for policy server socket {quoted_socket};",
            f"while [ ! -S {quoted_socket} ]; do",
            "if ! kill -0 $server_pid 2>/dev/null; then",
            "wait $server_pid; server_rc=$?;",
            f"echo FAILED_SERVER\\|$server_rc\\|$(date +%s) > {quoted_status_file};",
            "exit 1;",
            "fi;",
            "if [ $(date +%s) -ge $deadline ]; then",
            "kill $server_pid 2>/dev/null || true;",
            f"echo FAILED_SERVER_TIMEOUT\\|1\\|$(date +%s) > {quoted_status_file};",
            "exit 1;",
            "fi;",
            "sleep 5;",
            "done;",
            f"echo policy server socket ready {quoted_socket};",
            f"{shlex.join(client_cmd)} > {shlex.quote(str(client_log_path))} 2>&1;",
            "client_rc=$?;",
            "if kill -0 $server_pid 2>/dev/null; then kill $server_pid 2>/dev/null || true; fi;",
            "wait $server_pid 2>/dev/null || true;",
            (
                "if [ $client_rc -eq 0 ]; then "
                f"echo SUCCESS\\|$client_rc\\|$(date +%s) > {shlex.quote(str(status_file))}; "
                "else "
                f"echo FAILED\\|$client_rc\\|$(date +%s) > {shlex.quote(str(status_file))}; "
                "fi;"
            ),
            "exit $client_rc",
        ]
    )


def _tmux_socket_path(session_name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_name)
    return f"/tmp/wam_tmux_{os.getuid()}_{safe_name}.sock"


def _tmux_cmd(socket_path: str, *args: str) -> list[str]:
    return ["tmux", "-S", socket_path, *args]


def _ensure_tmux_panes(session_name: str, pane_count: int, socket_path: str) -> None:
    subprocess.run(_tmux_cmd(socket_path, "new-session", "-d", "-s", session_name), check=True)
    for _ in range(1, pane_count):
        subprocess.run(_tmux_cmd(socket_path, "split-window", "-t", f"{session_name}:0"), check=True)
        subprocess.run(_tmux_cmd(socket_path, "select-layout", "-t", f"{session_name}:0", "tiled"), check=False)


def _launch_tmux_workers(
    *,
    cfg: DictConfig,
    config_name: str,
    task_choice: str,
    gpu_ids: list[str],
    workers_per_gpu: int,
    split_files: dict[tuple[str, int], Path],
    output_dir: Path,
    dataset_stats_path: str | None,
    extra_overrides: list[str],
) -> None:
    status_dir = output_dir / "worker_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    tmux_session = str(cfg.MULTIRUN.get("tmux_session", f"libero_plus_{output_dir.name}"))
    tmux_socket_path = _tmux_socket_path(tmux_session)
    if bool(cfg.MULTIRUN.get("tmux_kill_existing", True)):
        subprocess.run(
            _tmux_cmd(tmux_socket_path, "kill-session", "-t", tmux_session),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    workers_to_launch: list[tuple[str, int]] = []
    for gpu_id, worker_idx in _iter_workers(gpu_ids, workers_per_gpu):
        if len(_read_tasks(split_files[(gpu_id, worker_idx)])) > 0:
            workers_to_launch.append((gpu_id, worker_idx))
    if not workers_to_launch:
        print("No workers launched; task file may be empty.")
        return

    _ensure_tmux_panes(tmux_session, len(workers_to_launch), tmux_socket_path)
    print(f"tmux session: {tmux_session}")
    print(f"tmux socket: {tmux_socket_path}")
    print(f"attach with: tmux -S {tmux_socket_path} attach -t {tmux_session}")
    server_ready_timeout_s = int(cfg.MULTIRUN.get("server_ready_timeout_s", 1800) or 1800)
    socket_dir = _socket_dir(output_dir)

    for pane_idx, (gpu_id, worker_idx) in enumerate(workers_to_launch):
        worker_label = _worker_label(gpu_id, worker_idx)
        split_file = split_files[(gpu_id, worker_idx)]
        socket_path = _socket_path(socket_dir, gpu_id, worker_idx)
        server_log_path = output_dir / "task_logs" / f"server_gpu{gpu_id}_w{worker_idx}.log"
        client_log_path = output_dir / "task_logs" / f"client_gpu{gpu_id}_w{worker_idx}.log"
        status_file = status_dir / f"gpu{gpu_id}_w{worker_idx}.status"
        try:
            status_file.unlink()
        except FileNotFoundError:
            pass
        server_cmd, client_cmd = _build_pair_commands(
            config_name=config_name,
            task_choice=task_choice,
            ckpt=str(cfg.ckpt),
            dataset_stats_path=dataset_stats_path,
            output_dir=output_dir,
            extra_overrides=extra_overrides,
            socket_path=socket_path,
            split_file=split_file,
            gpu_id=gpu_id,
            worker_idx=worker_idx,
            profiler_rank=pane_idx,
            profiler_world_size=len(workers_to_launch),
            client_log_path=client_log_path,
        )
        command = _tmux_worker_command(
            env=_child_env(gpu_id),
            server_cmd=server_cmd,
            client_cmd=client_cmd,
            server_log_path=server_log_path,
            client_log_path=client_log_path,
            status_file=status_file,
            socket_path=socket_path,
            server_ready_timeout_s=server_ready_timeout_s,
        )
        print(f"Launching tmux pane {pane_idx}: worker={worker_label}, tasks={len(_read_tasks(split_file))}")
        subprocess.run(
            _tmux_cmd(tmux_socket_path, "send-keys", "-t", f"{tmux_session}:0.{pane_idx}", "clear", "C-m"),
            check=False,
        )
        subprocess.run(
            _tmux_cmd(tmux_socket_path, "send-keys", "-t", f"{tmux_session}:0.{pane_idx}", command, "C-m"),
            check=True,
        )

    monitor_interval = int(cfg.MULTIRUN.get("monitor_interval", 10) or 10)
    status_interval = int(cfg.MULTIRUN.get("status_interval", 60) or 60)
    last_status_time = 0
    failures: list[str] = []
    while True:
        finished = 0
        failures = []
        for gpu_id, worker_idx in workers_to_launch:
            status_file = status_dir / f"gpu{gpu_id}_w{worker_idx}.status"
            if not status_file.exists():
                continue
            finished += 1
            status = status_file.read_text(encoding="utf-8").strip().split("|", 1)[0]
            if status != "SUCCESS":
                failures.append(f"{_worker_label(gpu_id, worker_idx)}:{status}")
        now = int(time.time())
        if now - last_status_time >= status_interval:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] workers finished: {finished}/{len(workers_to_launch)}")
            last_status_time = now
        if finished == len(workers_to_launch):
            break
        time.sleep(monitor_interval)

    _summarize(output_dir)
    _aggregate_profiles(output_dir)
    if failures:
        raise RuntimeError(f"LIBERO-plus eval failed: {failures}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    task_choice = _resolve_task_choice()
    config_name = str(HydraConfig.get().job.config_name or "sim_libero_plus")
    manager = cfg.MULTIRUN
    gpu_ids = ["0"] if bool(manager.get("create_only", False)) else _resolve_gpu_list(manager)
    (
        policy_server_mode,
        workers_per_gpu,
        servers_per_gpu,
        clients_per_gpu,
    ) = _resolve_policy_topology(manager)
    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "sockets").mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
        if not task_file.exists():
            raise FileNotFoundError(f"MULTIRUN.task_file does not exist: {task_file}")
    else:
        task_file = create_task_file(output_dir / "tasks.txt", list(manager.task_suite_names))
    if task_file.resolve() != (output_dir / "tasks.txt").resolve():
        (output_dir / "tasks.txt").write_text(task_file.read_text(encoding="utf-8"), encoding="utf-8")
        task_file = output_dir / "tasks.txt"

    split_files = _split_tasks(task_file, gpu_ids, clients_per_gpu, output_dir)
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))
    if bool(manager.get("create_only", False)):
        print("create_only=True, only created task files.")
        return

    dataset_stats_path = cfg.EVALUATION.get("dataset_stats_path", None)
    extra_overrides = collect_worker_overrides()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Starting LIBERO-plus eval")
    print(f"task={task_choice}")
    print(f"config={config_name}")
    print(f"ckpt={cfg.ckpt}")
    print(f"dataset_stats={dataset_stats_path}")
    print(f"output_dir={output_dir}")
    print(f"gpu_ids={gpu_ids}")
    print(f"policy_server_mode={policy_server_mode}")
    print(f"workers_per_gpu={workers_per_gpu}")
    print(f"servers_per_gpu={servers_per_gpu}")
    print(f"clients_per_gpu={clients_per_gpu}")
    print(f"total_servers={len(gpu_ids) * servers_per_gpu}")
    print(f"total_clients={len(gpu_ids) * clients_per_gpu}")
    print(f"run_id={run_id}")
    if extra_overrides:
        print(f"forwarded_overrides={extra_overrides}")

    launch_mode = str(manager.get("launch_mode", "subprocess"))
    common_launch_kwargs = dict(
        cfg=cfg,
        config_name=config_name,
        task_choice=task_choice,
        gpu_ids=gpu_ids,
        split_files=split_files,
        output_dir=output_dir,
        dataset_stats_path=None if dataset_stats_path is None else str(dataset_stats_path),
        extra_overrides=extra_overrides,
    )
    if policy_server_mode == "shared_per_gpu":
        if launch_mode != "subprocess":
            raise ValueError(
                "shared_per_gpu currently supports MULTIRUN.launch_mode=subprocess only"
            )
        _launch_shared_subprocess_workers(
            workers_per_gpu=workers_per_gpu,
            clients_per_server=int(manager.get("shared_clients_per_server", 2) or 2),
            clients_per_gpu=clients_per_gpu,
            **common_launch_kwargs,
        )
    elif launch_mode == "subprocess":
        _launch_subprocess_workers(
            workers_per_gpu=workers_per_gpu,
            **common_launch_kwargs,
        )
    elif launch_mode == "tmux":
        _launch_tmux_workers(
            workers_per_gpu=workers_per_gpu,
            **common_launch_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported MULTIRUN.launch_mode={launch_mode!r}. Use subprocess or tmux."
        )


if __name__ == "__main__":
    main()
