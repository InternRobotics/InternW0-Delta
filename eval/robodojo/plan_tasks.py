#!/usr/bin/env python3
"""Balance RoboDojo task groups using estimated runtime weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def canonical_tasks(root: Path) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts/internal/task_inventory.py"),
        "--only-runnable",
    ]
    return subprocess.check_output(command, text=True).splitlines()


def selected_tasks(args: argparse.Namespace, inventory: list[str]) -> list[str]:
    requested: list[str] = []
    if args.tasks.strip().lower() != "all":
        requested.extend(item.strip() for item in args.tasks.split(",") if item.strip())
    if args.tasks_file:
        for line in args.tasks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                requested.append(line)
    if not requested:
        return inventory
    wanted = set(requested)
    unknown = sorted(wanted - set(inventory))
    if unknown:
        raise SystemExit("unknown task(s): " + ", ".join(unknown))
    return [task for task in inventory if task in wanted]


def score(groups: list[list[dict[str, object]]]) -> tuple[int, int]:
    loads = [sum(int(item["seconds"]) for item in group) for group in groups]
    return max(loads), max(loads) - min(loads)


def partition(tasks: list[dict[str, object]], workers: int) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for task in sorted(tasks, key=lambda item: (-int(item["seconds"]), str(item["task"]))):
        target = min(range(workers), key=lambda index: (loads[index], index))
        groups[target].append(task)
        loads[target] += int(task["seconds"])

    # Match the official runner's inexpensive move/swap improvement pass.
    while True:
        current = score(groups)
        loads = [sum(int(item["seconds"]) for item in group) for group in groups]
        source = max(range(workers), key=lambda index: (loads[index], index))
        best: tuple[tuple[int, int], list[list[dict[str, object]]]] | None = None
        for task_index, task in enumerate(groups[source]):
            for target in range(workers):
                if target == source:
                    continue
                candidate = [list(group) for group in groups]
                candidate[source].pop(task_index)
                candidate[target].append(task)
                candidate_score = score(candidate)
                if candidate_score < current and (best is None or candidate_score < best[0]):
                    best = candidate_score, candidate
        for task_index, task in enumerate(groups[source]):
            for target in range(workers):
                if target == source:
                    continue
                for other_index, other in enumerate(groups[target]):
                    candidate = [list(group) for group in groups]
                    candidate[source][task_index] = other
                    candidate[target][other_index] = task
                    candidate_score = score(candidate)
                    if candidate_score < current and (best is None or candidate_score < best[0]):
                        best = candidate_score, candidate
        if best is None:
            break
        groups = best[1]

    for group in groups:
        group.sort(key=lambda item: (-int(item["seconds"]), str(item["task"])))
    return groups


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    inventory = canonical_tasks(args.robodojo_root)
    chosen = selected_tasks(args, inventory)
    if len(chosen) < args.workers:
        raise SystemExit(f"task count {len(chosen)} is smaller than worker count {args.workers}")

    payload = json.loads(args.weights.read_text(encoding="utf-8"))
    weights = payload.get("weights", {})
    missing = [task for task in chosen if task not in weights]
    if missing:
        raise SystemExit("missing InternW0-delta runtime weight(s): " + ", ".join(missing))
    items = [{"task": task, "seconds": int(weights[task])} for task in chosen]
    groups = partition(items, args.workers)
    result = {
        "weights_source": str(args.weights.resolve()),
        "workers": args.workers,
        "tasks": len(chosen),
        "groups": [
            {
                "worker": index,
                "total_seconds": sum(int(item["seconds"]) for item in group),
                "tasks": group,
            }
            for index, group in enumerate(groups)
        ],
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, group in enumerate(groups):
            text = "\n".join(str(item["task"]) for item in group)
            (args.output_dir / f"worker_{index}.tasks").write_text(
                text + ("\n" if text else ""), encoding="utf-8"
            )
        (args.output_dir / "assignment.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
