#!/usr/bin/env python3
"""Plan the official RoboDojo 42-task, three-seed evaluation suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DIMENSIONS: dict[str, tuple[str, ...]] = {
    "generalization": (
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ),
    "precision": (
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ),
    "long-horizon": (
        "put_bottles_into_dustbin",
        "fill_pen_holder",
        "classify_objects",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ),
    "memory": (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ),
    "open": (
        "align_blocks",
        "general_pickup",
        "stack_blocks_by_language",
        "solve_equation",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ),
}

SEEDS = (0, 1, 2)
POLICY = "internw0"
EMBODIMENT = "arx_x5"
RUN_CONFIG = "ckpt_name=robodojo,action_type=joint,replan_steps=10"


def execution_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for dimension, canonical_tasks in DIMENSIONS.items():
        runtime_tasks: list[tuple[str, int]] = []
        for task in canonical_tasks:
            if dimension == "generalization":
                runtime_tasks.extend(((task, 25), (f"{task}_random", 25)))
            else:
                runtime_tasks.append((task, 50))
        for task, expected in runtime_tasks:
            for seed in SEEDS:
                units.append(
                    {
                        "dimension": dimension,
                        "task": task,
                        "seed": seed,
                        "expected_episodes": expected,
                        "spec": f"{task}:{seed}",
                    }
                )
    return units


def inspect_latest(root: Path, unit: dict[str, Any]) -> dict[str, Any]:
    run_root = (
        root
        / unit["task"]
        / POLICY
        / EMBODIMENT
        / f'{unit["seed"]}_{RUN_CONFIG}'
    )
    results = sorted(run_root.glob("*/_result.json"), key=lambda path: path.parent.name)
    if not results:
        return {"status": "missing", "latest_result": None, "eval_time": 0, "details": 0}

    latest = results[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        eval_time = int(data.get("eval_time", -1))
        details = data.get("details")
        detail_count = len(details) if isinstance(details, dict) else -1
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "latest_result": str(latest),
            "eval_time": -1,
            "details": -1,
        }

    expected = int(unit["expected_episodes"])
    status = "complete" if eval_time >= expected and detail_count >= expected else "partial"
    return {
        "status": status,
        "latest_result": str(latest),
        "eval_time": eval_time,
        "details": detail_count,
    }


def build_plan(root: Path) -> dict[str, Any]:
    units = []
    for base in execution_units():
        units.append({**base, **inspect_latest(root, base)})
    complete = sum(unit["status"] == "complete" for unit in units)
    pending = len(units) - complete
    scoreboard_complete = 0
    for dimension, tasks in DIMENSIONS.items():
        for task in tasks:
            for seed in SEEDS:
                required = [task]
                if dimension == "generalization":
                    required.append(f"{task}_random")
                if all(
                    any(
                        unit["task"] == runtime_task
                        and unit["seed"] == seed
                        and unit["status"] == "complete"
                        for unit in units
                    )
                    for runtime_task in required
                ):
                    scoreboard_complete += 1
    return {
        "eval_root": str(root),
        "canonical_tasks": sum(len(tasks) for tasks in DIMENSIONS.values()),
        "scoreboard_cells": 126,
        "scoreboard_complete": scoreboard_complete,
        "execution_units": len(units),
        "execution_complete": complete,
        "execution_pending": pending,
        "episodes_total": sum(unit["expected_episodes"] for unit in units),
        "episodes_complete": sum(
            unit["expected_episodes"] for unit in units if unit["status"] == "complete"
        ),
        "units": units,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("ROBODOJO_RESULT_ROOT", Path(__file__).resolve().parents[2] / "runs/eval/robodojo/results")) / "RoboDojo",
    )
    parser.add_argument("--format", choices=("json", "specs", "table"), default="table")
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--exclude-specs",
        type=Path,
        help="newline-delimited TASK:SEED specs already submitted and still in flight",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    plan = build_plan(args.eval_root)
    excluded: set[str] = set()
    if args.exclude_specs is not None:
        for line_number, raw in enumerate(
            args.exclude_specs.read_text(encoding="utf-8").splitlines(), start=1
        ):
            spec = raw.strip()
            if not spec or spec.startswith("#"):
                continue
            if spec in excluded:
                parser.error(f"duplicate excluded spec on line {line_number}: {spec}")
            excluded.add(spec)
        known = {unit["spec"] for unit in plan["units"]}
        unknown = sorted(excluded - known)
        if unknown:
            parser.error(f"unknown excluded specs: {', '.join(unknown)}")
        for unit in plan["units"]:
            if unit["status"] != "complete" and unit["spec"] in excluded:
                unit["status"] = "inflight"
        plan["execution_inflight"] = sum(
            unit["status"] == "inflight" for unit in plan["units"]
        )
        plan["execution_unassigned"] = (
            plan["execution_pending"] - plan["execution_inflight"]
        )
    units = plan["units"]
    if args.pending_only:
        units = [
            unit for unit in units if unit["status"] not in ("complete", "inflight")
        ]
    if args.limit is not None:
        units = units[: args.limit]

    if args.format == "json":
        output = {**plan, "units": units, "selected_units": len(units)}
        print(json.dumps(output, indent=2, sort_keys=True))
    elif args.format == "specs":
        for unit in units:
            print(unit["spec"])
    else:
        print(
            "ROBODOJO_FULL_PLAN "
            f'canonical_tasks={plan["canonical_tasks"]} '
            f'scoreboard={plan["scoreboard_complete"]}/{plan["scoreboard_cells"]} '
            f'execution={plan["execution_complete"]}/{plan["execution_units"]} '
            f'episodes={plan["episodes_complete"]}/{plan["episodes_total"]}'
        )
        if excluded:
            print(
                "ROBODOJO_FULL_SCHEDULING "
                f'inflight={plan["execution_inflight"]} '
                f'unassigned={plan["execution_unassigned"]}'
            )
        print("status\tdimension\ttask\tseed\tepisodes\tlatest_result")
        for unit in units:
            print(
                f'{unit["status"]}\t{unit["dimension"]}\t{unit["task"]}\t'
                f'{unit["seed"]}\t{unit["eval_time"]}/{unit["expected_episodes"]}\t'
                f'{unit["latest_result"] or "-"}'
            )


if __name__ == "__main__":
    main()
