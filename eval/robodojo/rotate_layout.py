#!/usr/bin/env python3
"""Mark the next pending RoboDojo layout abandoned after a proven simulator stall."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


REPRO_ROOT = Path(__file__).resolve().parents[2]
ROBODOJO_ROOT = Path(
    os.environ.get("ROBODOJO_ROOT", REPRO_ROOT / "third_party/RoboDojo")
).resolve()
RESULT_ROOT = Path(os.environ.get("ROBODOJO_RESULT_ROOT", REPRO_ROOT / "runs/eval/robodojo/results")) / "RoboDojo"
LAYOUT_ROOT = ROBODOJO_ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5"
POLICY_NAME = os.environ.get("ROBODOJO_POLICY_NAME", "internw0")
CHECKPOINT_NAME = os.environ.get("ROBODOJO_CHECKPOINT_NAME", "robodojo")
ADDITIONAL_INFO = f"ckpt_name={CHECKPOINT_NAME},action_type=joint,replan_steps=10"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("seed", type=int, choices=(0, 1, 2))
    parser.add_argument("run_id")
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"JSON payload must be a mapping: {path}")
    return value


def eval_time(payload: dict) -> int:
    if "eval_time" in payload:
        return int(payload.get("eval_time", 0))
    return int(payload.get("success_nums", 0)) + int(payload.get("fail_nums", 0))


def main() -> int:
    args = parse_args()
    if not SAFE_NAME.fullmatch(args.task) or not SAFE_NAME.fullmatch(args.run_id):
        raise SystemExit("task and run_id must contain only safe filename characters")
    if args.expected < 1:
        raise SystemExit("--expected must be positive")

    config_path = ROBODOJO_ROOT / "task/RoboDojo/config" / f"{args.task}.yml"
    if not config_path.is_file():
        raise SystemExit(f"unknown RoboDojo task: {args.task}")

    seed_dir = RESULT_ROOT / args.task / f"{POLICY_NAME}/arx_x5" / f"{args.seed}_{ADDITIONAL_INFO}"
    save_dir = seed_dir / args.run_id
    result_path = save_dir / "_result.json"
    manifest_path = seed_dir / f"_resume_{args.run_id}.json"
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    source = result if eval_time(result) >= eval_time(manifest) else manifest

    details = source.get("details") or {}
    if not isinstance(details, dict):
        raise SystemExit("details must be a mapping")
    completed = {
        int(value["layout_id"])
        for value in details.values()
        if isinstance(value, dict) and "layout_id" in value
    }
    if len(completed) != eval_time(source):
        raise SystemExit(
            f"completed layout count mismatch: layouts={len(completed)}, eval_time={eval_time(source)}"
        )

    abandoned = {int(value) for value in manifest.get("abandoned_layout_ids", [])}
    layout_dir = LAYOUT_ROOT / str(args.seed)
    pattern = re.compile(rf"{re.escape(args.task)}_\d+\.json")
    layout_count = sum(1 for path in layout_dir.iterdir() if pattern.fullmatch(path.name))
    pending = [layout_id for layout_id in range(layout_count) if layout_id not in completed | abandoned]
    if not pending:
        raise SystemExit("no pending layout remains to rotate")
    stalled_layout = pending[0]
    abandoned.add(stalled_layout)

    remaining_needed = args.expected - eval_time(source)
    remaining_available = layout_count - len(completed | abandoned)
    if remaining_needed < 0 or remaining_available < remaining_needed:
        raise SystemExit(
            f"insufficient layouts after rotation: available={remaining_available}, needed={remaining_needed}"
        )

    success_nums = sum(
        bool(value.get("success")) for value in details.values() if isinstance(value, dict)
    )
    total_score = sum(
        float(value.get("score", 0.0)) for value in details.values() if isinstance(value, dict)
    )
    payload = {
        "run_id": args.run_id,
        "save_dir": str(save_dir),
        "task_name": args.task,
        "policy_name": POLICY_NAME,
        "config_name": "arx_x5",
        "eval_seed": args.seed,
        "additional_info": ADDITIONAL_INFO,
        "success_nums": success_nums,
        "fail_nums": eval_time(source) - success_nums,
        "unstable_nums": max(int(manifest.get("unstable_nums", 0)), len(abandoned)),
        "total_score": total_score,
        "completed_layout_ids": sorted(completed),
        "abandoned_layout_ids": sorted(abandoned),
        "details": details,
        "restart_count": int(manifest.get("restart_count", 0)) + 1,
    }
    summary = {
        "task": args.task,
        "seed": args.seed,
        "run_id": args.run_id,
        "eval_time": eval_time(source),
        "expected": args.expected,
        "layout_count": layout_count,
        "stalled_layout": stalled_layout,
        "abandoned_layout_ids": sorted(abandoned),
        "remaining_available": remaining_available,
        "manifest": str(manifest_path),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, manifest_path)
    print("ROBODOJO_ROTATE_STALLED_LAYOUT", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
