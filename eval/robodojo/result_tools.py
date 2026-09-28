#!/usr/bin/env python3
"""Small JSON helpers used by the persistent-server watchdog."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def progress(payload: dict) -> int:
    if "eval_time" in payload:
        return int(payload.get("eval_time", 0))
    return int(payload.get("success_nums", 0)) + int(payload.get("fail_nums", 0))


def validate_records(payload: dict, expected: int | None = None) -> list[dict]:
    details = payload.get("details", {})
    if not isinstance(details, dict):
        raise ValueError("Result details must be a mapping")
    entries = list(details.values())
    if expected is not None and (progress(payload) != expected or len(entries) != expected):
        raise ValueError(f"Incomplete result: expected {expected} episodes, found {len(entries)}")
    layouts = set()
    for item in entries:
        if not isinstance(item, dict) or type(item.get("success")) is not bool:
            raise ValueError("Each episode must contain a boolean success value")
        layout = item.get("layout_id")
        if type(layout) is not int or layout < 0 or layout in layouts:
            raise ValueError("Episode layout IDs must be unique nonnegative integers")
        if not math.isfinite(float(item["score"])):
            raise ValueError("Episode scores must be finite")
        layouts.add(layout)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    count_parser = sub.add_parser("native-count")
    count_parser.add_argument("task_config", type=Path)
    count_parser.add_argument("task")
    progress_parser = sub.add_parser("progress")
    progress_parser.add_argument("paths", nargs="+", type=Path)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("result", type=Path)
    validate_parser.add_argument("expected", type=int)
    args = parser.parse_args()

    if args.command == "native-count":
        import yaml

        config = yaml.safe_load(args.task_config.read_text(encoding="utf-8")) or {}
        common = config.get("common") or {}
        task_cfg = (config.get("tasks") or {}).get(args.task) or {}
        value = int(task_cfg.get("eval_nums", common.get("eval_nums", 50)))
        if value < 1:
            raise SystemExit(f"invalid native eval count for {args.task}: {value}")
        print(value)
        return 0
    if args.command == "progress":
        print(max((progress(read_json(path)) for path in args.paths), default=0))
        return 0

    payload = read_json(args.result)
    actual = progress(payload)
    entries = validate_records(payload, args.expected)
    success_nums = sum(bool(value.get("success", False)) for value in entries)
    total_score = sum(float(value.get("score", 0.0)) for value in entries)
    print(
        json.dumps(
            {
                "eval_time": actual,
                "success_nums": success_nums,
                "total_score": total_score,
                "result": str(args.result),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
