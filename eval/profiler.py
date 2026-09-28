"""Low-overhead, rank-aware profiling helpers for InternW0-delta training.

The profiler is intentionally opt-in.  ``Profiler(enabled=False)`` is a
no-op, while ``torch_profile=True`` enables a bounded ``torch.profiler``
window when the caller uses :meth:`begin_torch_section` and
:meth:`step_torch_section`.
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:  # Keep the wall-clock profiler importable in lightweight tooling.
    import torch
except ImportError:  # pragma: no cover - the training environment has torch.
    torch = None  # type: ignore[assignment]


@dataclass
class _Section:
    name: str
    elapsed: float = 0.0
    calls: int = 0
    children: list["_Section"] = field(default_factory=list)
    _children_by_name: dict[str, "_Section"] = field(
        default_factory=dict,
        repr=False,
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Profiler:
    """A rank-aware wall-clock and optional torch profiler.

    Wall-clock sections are aggregated by nesting path, so repeatedly timing
    ``forward`` over a long run produces one row with total time, call count,
    and mean time rather than thousands of rows.  Torch traces are written
    below ``profile_dir/rank_<rank>`` to make multi-process launches safe.
    """

    def __init__(
        self,
        enabled: bool = True,
        torch_profile: bool = False,
        profile_dir: str | os.PathLike[str] = "profiler_traces",
        *,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        sync_cuda: bool = False,
        track_cuda_memory: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.rank = (
            int(rank)
            if rank is not None
            else _env_int("RANK", _env_int("LOCAL_RANK", 0))
        )
        self.local_rank = (
            int(local_rank)
            if local_rank is not None
            else _env_int("LOCAL_RANK", self.rank)
        )
        self.world_size = (
            int(world_size)
            if world_size is not None
            else max(_env_int("WORLD_SIZE", 1), 1)
        )

        base_dir = Path(profile_dir).expanduser().absolute()
        self.base_profile_dir = str(base_dir)
        self.profile_dir = str(base_dir / f"rank_{self.rank:03d}")

        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)
        self.sync_cuda = bool(sync_cuda)
        self.track_cuda_memory = bool(track_cuda_memory)
        self.torch_profile = bool(
            torch_profile and self.enabled and torch is not None
        )

        self._root_sections: list[_Section] = []
        self._root_by_name: dict[str, _Section] = {}
        self._stack: list[_Section] = []
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._trace_index = 0
        self._finished = False

        self._torch_context: Any | None = None
        self._torch_step: Any | None = None
        self._torch_step_count = 0
        self._torch_step_limit: int | None = None

        if self.enabled:
            os.makedirs(self.profile_dir, exist_ok=True)
            # This also preserves a summary when an uncaught training error
            # terminates the process before the trainer's normal shutdown.
            atexit.register(self._atexit_finish)

    @classmethod
    def from_env(
        cls,
        profile_dir: str | os.PathLike[str],
        *,
        rank: int,
        local_rank: int | None = None,
        world_size: int = 1,
    ) -> "Profiler":
        """Build a profiler from the InternW0-delta profiling environment variables.

        Keeping this construction in one place is important because the
        runtime creates the profiler before model/dataset construction while
        the trainer can still be instantiated directly by tests or tooling.
        """
        torch_requested = _env_flag("WAM_TORCH_PROFILE")
        return cls(
            enabled=_env_flag("WAM_PROFILE") or torch_requested,
            torch_profile=torch_requested,
            profile_dir=profile_dir,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            record_shapes=_env_flag("WAM_PROFILE_RECORD_SHAPES", True),
            profile_memory=_env_flag("WAM_PROFILE_MEMORY", True),
            with_stack=_env_flag("WAM_PROFILE_WITH_STACK", False),
            sync_cuda=_env_flag("WAM_PROFILE_SYNC_CUDA", False),
            track_cuda_memory=_env_flag("WAM_PROFILE_CUDA_METRICS", True),
        )

    def start(self) -> None:
        """Start (or restart) the wall-clock session and reset peak memory."""
        # A caller may reuse one profiler for multiple short runs.  Close a
        # previous bounded trace before clearing the timing tree so its
        # exporter can still see the completed events.
        self.end_torch_section()
        self._root_sections.clear()
        self._root_by_name.clear()
        self._stack.clear()
        self._end_time = None
        self._finished = False
        self._start_time = time.perf_counter()
        if (
            self.enabled
            and self.track_cuda_memory
            and torch is not None
            and torch.cuda.is_available()
        ):
            try:
                torch.cuda.reset_peak_memory_stats()
            except (RuntimeError, AssertionError):
                # CUDA may not be initialized yet in lightweight callers.
                pass

    @property
    def started(self) -> bool:
        """Whether a wall-clock session has been started."""
        return self._start_time is not None

    def _push(self, name: str) -> _Section:
        name = str(name)
        if self._stack:
            parent = self._stack[-1]
            node = parent._children_by_name.get(name)
            if node is None:
                node = _Section(name=name)
                parent._children_by_name[name] = node
                parent.children.append(node)
        else:
            node = self._root_by_name.get(name)
            if node is None:
                node = _Section(name=name)
                self._root_by_name[name] = node
                self._root_sections.append(node)
        node.calls += 1
        self._stack.append(node)
        return node

    def _pop(self, node: _Section, elapsed: float) -> None:
        if not self._stack or self._stack[-1] is not node:
            # Do not leave a corrupt stack behind if a caller manually
            # mis-nests a context.  Normal context-manager use never enters
            # this branch.
            if node in self._stack:
                self._stack.remove(node)
        else:
            self._stack.pop()
        node.elapsed += max(float(elapsed), 0.0)

    def _cuda_sync(self) -> None:
        if self.sync_cuda and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Record one aggregated wall-clock section."""
        if not self.enabled:
            yield
            return

        self._cuda_sync()
        node = self._push(name)
        t0 = time.perf_counter()
        # ``record_function`` gives the bounded Torch trace the same semantic
        # stage names as the wall-clock tree.  It is intentionally created for
        # every section only when torch profiling is requested; normal runs
        # retain the original low-overhead path.
        record_ctx = (
            torch.profiler.record_function(str(name))
            if self.torch_profile and torch is not None
            else nullcontext()
        )
        try:
            with record_ctx:
                yield
        finally:
            self._cuda_sync()
            self._pop(node, time.perf_counter() - t0)

    @contextmanager
    def torch_section(
        self,
        name: str,
        *,
        warmup: int = 1,
        active: int = 3,
    ) -> Iterator[Any]:
        """Profile a loop with a finite torch.profiler schedule.

        The yielded callable must be invoked once per loop iteration.  The
        context itself may remain open after the schedule finishes, but
        collection stops after ``warmup + active`` calls.  The trainer uses
        :meth:`begin_torch_section` to close the context immediately after
        that bounded window.
        """
        if not self.enabled:
            yield _noop_step
            return

        warmup = max(int(warmup), 0)
        active = max(int(active), 1)
        node = self._push(name)
        t0 = time.perf_counter()
        tp = None
        trace_path: str | None = None
        try:
            if not self.torch_profile:
                yield _noop_step
                return

            if torch is None:  # pragma: no cover - guarded in __init__.
                yield _noop_step
                return

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            schedule = torch.profiler.schedule(
                wait=0,
                warmup=warmup,
                active=active,
                repeat=1,
            )

            safe_name = (
                str(name).strip().replace(" ", "_").replace("/", "-")
            )
            self._trace_index += 1
            trace_path = os.path.join(
                self.profile_dir,
                f"{safe_name}_{self._trace_index:03d}.json",
            )

            with torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
            ) as tp:
                # Keep the enclosing trace visible in Perfetto while the
                # yielded context spans several training iterations.
                with torch.profiler.record_function(str(name)):
                    yield tp.step
        finally:
            self._pop(node, time.perf_counter() - t0)
            if tp is not None:
                self._export_torch_trace(tp, trace_path, name, warmup, active)

    def begin_torch_section(
        self,
        name: str,
        *,
        warmup: int = 1,
        active: int = 3,
    ) -> None:
        """Start a bounded torch trace without indenting the train loop."""
        self.end_torch_section()
        if not self.enabled or not self.torch_profile:
            return

        self._torch_context = self.torch_section(
            name,
            warmup=warmup,
            active=active,
        )
        try:
            self._torch_step = self._torch_context.__enter__()
        except Exception as exc:  # Profiling must never stop a training run.
            print(
                f"[torch.profiler] rank={self.rank} start failed: {exc}",
                file=sys.stderr,
            )
            self._torch_context = None
            self._torch_step = None
            return
        self._torch_step_count = 0
        self._torch_step_limit = max(int(warmup), 0) + max(int(active), 1)

    def step_torch_section(self) -> None:
        """Advance the active torch trace by one training iteration."""
        if self._torch_step is None:
            return
        try:
            self._torch_step()
        except Exception as exc:  # Profiling must never stop a training run.
            print(
                f"[torch.profiler] rank={self.rank} step failed: {exc}",
                file=sys.stderr,
            )
            self.end_torch_section()
            return
        self._torch_step_count += 1
        if (
            self._torch_step_limit is not None
            and self._torch_step_count >= self._torch_step_limit
        ):
            self.end_torch_section()

    def end_torch_section(self) -> None:
        """Close an active bounded torch trace, if any."""
        context = self._torch_context
        self._torch_context = None
        self._torch_step = None
        self._torch_step_count = 0
        self._torch_step_limit = None
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover - backend-specific.
                print(
                    f"[torch.profiler] rank={self.rank} close failed: {exc}",
                    file=sys.stderr,
                )

    def _export_torch_trace(
        self,
        profiler: Any,
        trace_path: str | None,
        name: str,
        warmup: int,
        active: int,
    ) -> None:
        if profiler is None or trace_path is None:
            return
        try:
            profiler.export_chrome_trace(trace_path)
            sort_key = (
                "cuda_time_total"
                if torch is not None and torch.cuda.is_available()
                else "cpu_time_total"
            )
            table = profiler.key_averages().table(
                sort_by=sort_key,
                row_limit=20,
            )
            Path(trace_path).with_suffix(".txt").write_text(
                "Profiler: "
                f"{name} rank={self.rank} warmup={warmup} active={active}\n\n"
                + table,
                encoding="utf-8",
            )
            print(
                f"[torch.profiler] rank={self.rank} trace saved: {trace_path}"
            )
        except Exception as exc:  # Do not mask a training failure.
            print(
                f"[torch.profiler] rank={self.rank} export failed: {exc}",
                file=sys.stderr,
            )

    def summary(self) -> str:
        total = self._wall_total()
        rows: list[tuple[str, float, float, int, float]] = []
        self._flatten(self._root_sections, total, depth=0, rows=rows)

        top_level = sum(s.elapsed for s in self._root_sections)
        max_name = max((len(row[0]) for row in rows), default=0)
        lines = [
            "",
            "=" * 76,
            "  Profiler Summary",
            "=" * 76,
            f"  rank={self.rank} local_rank={self.local_rank} world_size={self.world_size}",
        ]
        for display_name, elapsed, pct, calls, exclusive in rows:
            mean = elapsed / calls if calls > 0 else 0.0
            lines.append(
                f"  {display_name:<{max_name}}  {_fmt(elapsed):>12}"
                f"  ({pct:5.1f}%)  calls={calls:<6d} mean={_fmt(mean)}"
                f" self={_fmt(exclusive)}"
            )
        lines.append("-" * 76)
        lines.append(f"  {'Tracked total':<{max_name}}  {_fmt(top_level):>12}")
        lines.append(f"  {'Untracked':<{max_name}}  {_fmt(total - top_level):>12}")
        lines.append(f"  {'Wall total':<{max_name}}  {_fmt(total):>12}")
        gpu_peak_allocated = (
            _gpu_peak_allocated_line() if self.track_cuda_memory else None
        )
        if gpu_peak_allocated is not None:
            lines.append(f"  {'GPU peak allocated':<{max_name}}  {gpu_peak_allocated:>12}")
        gpu_peak_reserved = (
            _gpu_peak_reserved_line() if self.track_cuda_memory else None
        )
        if gpu_peak_reserved is not None:
            lines.append(f"  {'GPU peak reserved':<{max_name}}  {gpu_peak_reserved:>12}")
        if self.torch_profile:
            lines.extend(
                [
                    "=" * 76,
                    f"  Chrome traces saved to: {self.profile_dir}/",
                    "  Open in: chrome://tracing or https://ui.perfetto.dev",
                ]
            )
        lines.append("=" * 76)
        return "\n".join(lines)

    def write_summary(self) -> str | None:
        """Write the current rank summary and return its path."""
        if not self.enabled:
            return None
        os.makedirs(self.profile_dir, exist_ok=True)
        summary_path = os.path.join(self.profile_dir, "summary.txt")
        Path(summary_path).write_text(self.summary() + "\n", encoding="utf-8")
        json_path = os.path.join(self.profile_dir, "summary.json")
        import json

        Path(json_path).write_text(
            json.dumps(self.summary_data(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary_path

    def finish(self) -> str | None:
        """Close active traces and persist the summary once."""
        if self._finished:
            return None
        self.end_torch_section()
        self._end_time = time.perf_counter()
        self._finished = True
        return self.write_summary()

    def _atexit_finish(self) -> None:
        try:
            self.finish()
        except Exception:
            # Never make interpreter shutdown fail because of profiling.
            pass

    def _wall_total(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return max(end - self._start_time, 0.0)

    def summary_data(self) -> dict[str, Any]:
        """Return a machine-readable summary for cross-rank aggregation."""
        wall_total = self._wall_total()
        tracked_total = sum(section.elapsed for section in self._root_sections)
        peak_allocated = (
            _gpu_peak_allocated_bytes() if self.track_cuda_memory else None
        )
        peak_reserved = (
            _gpu_peak_reserved_bytes() if self.track_cuda_memory else None
        )
        return {
            "schema_version": 1,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "wall_total_seconds": wall_total,
            "tracked_total_seconds": tracked_total,
            "untracked_seconds": wall_total - tracked_total,
            "torch_profile": self.torch_profile,
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_stack": self.with_stack,
            "sync_cuda": self.sync_cuda,
            "track_cuda_memory": self.track_cuda_memory,
            "profile_dir": self.profile_dir,
            "gpu_peak_allocated_bytes": peak_allocated,
            "gpu_peak_reserved_bytes": peak_reserved,
            "sections": [
                self._section_data(
                    section,
                    parent_elapsed=wall_total,
                    wall_total=wall_total,
                    path=section.name,
                )
                for section in self._root_sections
            ],
        }

    @classmethod
    def _section_data(
        cls,
        section: _Section,
        *,
        parent_elapsed: float,
        wall_total: float,
        path: str,
    ) -> dict[str, Any]:
        child_total = sum(child.elapsed for child in section.children)
        exclusive = max(section.elapsed - child_total, 0.0)
        return {
            "name": section.name,
            "path": path,
            "elapsed_seconds": section.elapsed,
            "inclusive_seconds": section.elapsed,
            "exclusive_seconds": exclusive,
            "calls": section.calls,
            "mean_seconds": section.elapsed / section.calls if section.calls else 0.0,
            "percentage_of_parent": (
                section.elapsed / parent_elapsed * 100.0
                if parent_elapsed > 0
                else 0.0
            ),
            "percentage_of_wall": (
                section.elapsed / wall_total * 100.0
                if wall_total > 0
                else 0.0
            ),
            "children": [
                cls._section_data(
                    child,
                    parent_elapsed=section.elapsed,
                    wall_total=wall_total,
                    path=f"{path}/{child.name}",
                )
                for child in section.children
            ],
        }

    @staticmethod
    def _flatten(
        sections: list[_Section],
        parent_elapsed: float,
        depth: int,
        rows: list[tuple[str, float, float, int, float]],
    ) -> None:
        indent = "  " * depth
        for sec in sections:
            pct = (
                sec.elapsed / parent_elapsed * 100
                if parent_elapsed > 0
                else 0.0
            )
            child_total = sum(child.elapsed for child in sec.children)
            rows.append(
                (
                    f"{indent}{sec.name}",
                    sec.elapsed,
                    pct,
                    sec.calls,
                    max(sec.elapsed - child_total, 0.0),
                )
            )
            if sec.children:
                Profiler._flatten(sec.children, sec.elapsed, depth + 1, rows)


def _noop_step() -> None:
    return None


def _fmt(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _gpu_peak_allocated_line() -> str | None:
    peak_bytes = _gpu_peak_allocated_bytes()
    if peak_bytes is None:
        return None
    try:
        device_idx = torch.cuda.current_device()
    except (RuntimeError, AssertionError):
        device_idx = "?"
    return f"{_fmt_bytes(peak_bytes)} (cuda:{device_idx})"


def _gpu_peak_reserved_line() -> str | None:
    peak_bytes = _gpu_peak_reserved_bytes()
    if peak_bytes is None:
        return None
    try:
        device_idx = torch.cuda.current_device()
    except (RuntimeError, AssertionError):
        device_idx = "?"
    return f"{_fmt_bytes(peak_bytes)} (cuda:{device_idx})"


def _gpu_peak_allocated_bytes() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated(torch.cuda.current_device()))
    except (RuntimeError, AssertionError):
        return None


def _gpu_peak_reserved_bytes() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_reserved(torch.cuda.current_device()))
    except (RuntimeError, AssertionError):
        return None


def _fmt_bytes(n_bytes: int) -> str:
    gib = n_bytes / (1024**3)
    return f"{gib:.2f}GiB"
