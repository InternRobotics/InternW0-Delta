import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATEGORY_COLUMNS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
REPORT_COLUMNS = CATEGORY_COLUMNS + ["Total"]

CATEGORY_MAP = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def format_time(seconds: float | int | None) -> str:
    seconds = round(float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:02d}s"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes:02d}m{seconds % 60:02d}s"
    hours = seconds // 3600
    remaining = seconds % 3600
    return f"{hours:02d}h{remaining // 60:02d}m{remaining % 60:02d}s"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_scheduled_tasks(output_dir: Path) -> list[tuple[str, int]]:
    tasks_path = output_dir / "tasks.txt"
    if not tasks_path.exists():
        return []
    tasks: list[tuple[str, int]] = []
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            suite, task_id = line.split(",", 1)
            tasks.append((suite, int(task_id)))
    return tasks


def _normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    suite = str(row.get("suite") or row.get("task_suite") or "")
    task_id = int(row.get("task_id", -1))
    total_episodes = int(row.get("total_episodes", 0) or 0)
    successes = int(row.get("successes", 0) or 0)
    duration_sec = float(row.get("duration_sec", row.get("duration", 0.0)) or 0.0)
    status = str(row.get("status") or "completed")
    success_rate = float(successes) / float(total_episodes) * 100.0 if total_episodes > 0 else 0.0
    normalized = dict(row)
    normalized.update(
        {
            "suite": suite,
            "task_id": task_id,
            "status": status,
            "successes": successes,
            "total_episodes": total_episodes,
            "success_rate": success_rate,
            "duration_sec": duration_sec,
            "duration": duration_sec,
            "task_name": row.get("task_name", ""),
            "task_description": row.get("task_description", ""),
            "gpu_id": row.get("gpu_id", ""),
            "log_file": row.get("log_file", ""),
            "result_file": row.get("result_file", ""),
            "return_code": row.get("return_code", ""),
        }
    )
    return normalized


def merge_task_rows(output_dir: Path) -> list[dict[str, Any]]:
    scheduled = _read_scheduled_tasks(output_dir)
    result_rows = [_normalize_result_row(row) for row in _read_jsonl(output_dir / "task_results.jsonl")]
    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    extra_rows: list[dict[str, Any]] = []
    for row in result_rows:
        key = (str(row["suite"]), int(row["task_id"]))
        if key[0]:
            results_by_key[key] = row
        else:
            extra_rows.append(row)

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for suite, task_id in scheduled:
        key = (suite, int(task_id))
        seen.add(key)
        row = results_by_key.get(key)
        if row is None:
            row = {
                "suite": suite,
                "task_id": int(task_id),
                "status": "pending",
                "gpu_id": "",
                "successes": 0,
                "total_episodes": 0,
                "success_rate": 0.0,
                "duration_sec": 0.0,
                "duration": 0.0,
                "task_name": "",
                "task_description": "",
                "log_file": "",
                "result_file": "",
                "return_code": "",
            }
        merged.append(row)

    for key, row in results_by_key.items():
        if key not in seen:
            row = dict(row)
            row["status"] = row.get("status") or "unknown"
            merged.append(row)
    merged.extend(extra_rows)
    return merged


def _empty_stats() -> dict[str, Any]:
    return {
        "scheduled_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "running_tasks": 0,
        "pending_tasks": 0,
        "unknown_tasks": 0,
        "successes": 0,
        "total_episodes": 0,
        "total_duration_sec": 0.0,
        "max_duration_sec": 0.0,
        "task_ids": [],
    }


def _update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    status = str(row.get("status") or "unknown").lower()
    stats["scheduled_tasks"] += 1
    task_id = row.get("task_id")
    if task_id is not None:
        stats["task_ids"].append(task_id)
    if status == "completed":
        stats["completed_tasks"] += 1
        stats["successes"] += int(row.get("successes", 0) or 0)
        stats["total_episodes"] += int(row.get("total_episodes", 0) or 0)
        duration = float(row.get("duration_sec", row.get("duration", 0.0)) or 0.0)
        stats["total_duration_sec"] += duration
        stats["max_duration_sec"] = max(float(stats["max_duration_sec"]), duration)
    elif status == "failed":
        stats["failed_tasks"] += 1
    elif status == "running":
        stats["running_tasks"] += 1
    elif status == "pending":
        stats["pending_tasks"] += 1
    else:
        stats["unknown_tasks"] += 1


def _finalize_stats(name: str, stats: dict[str, Any]) -> dict[str, Any]:
    total_episodes = int(stats["total_episodes"])
    completed_tasks = int(stats["completed_tasks"])
    scheduled_tasks = int(stats["scheduled_tasks"])
    return {
        "name": name,
        "scheduled_tasks": scheduled_tasks,
        "completed_tasks": completed_tasks,
        "failed_tasks": int(stats["failed_tasks"]),
        "running_tasks": int(stats["running_tasks"]),
        "pending_tasks": int(stats["pending_tasks"]),
        "unknown_tasks": int(stats["unknown_tasks"]),
        "successes": int(stats["successes"]),
        "total_episodes": total_episodes,
        "success_rate": float(stats["successes"]) / float(total_episodes) * 100.0 if total_episodes > 0 else 0.0,
        "completion_rate": float(completed_tasks) / float(scheduled_tasks) * 100.0 if scheduled_tasks > 0 else 0.0,
        "total_duration_sec": float(stats["total_duration_sec"]),
        "average_duration_sec": float(stats["total_duration_sec"]) / float(completed_tasks) if completed_tasks > 0 else 0.0,
        "max_duration_sec": float(stats["max_duration_sec"]),
        "task_ids": ",".join(str(item) for item in stats["task_ids"]),
    }


def aggregate_by_suite(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats = defaultdict(lambda: _empty_stats())
    for row in rows:
        _update_stats(stats[str(row.get("suite") or "unknown")], row)
    return [_finalize_stats(name, stats[name]) for name in sorted(stats)]


def aggregate_by_gpu(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats = defaultdict(lambda: _empty_stats())
    for row in rows:
        gpu_id = str(row.get("gpu_id") or "unknown")
        _update_stats(stats[gpu_id], row)
    return [_finalize_stats(name, stats[name]) for name in sorted(stats)]


def _classification_path() -> Path:
    return LIBERO_PLUS_ROOT / "libero" / "libero" / "benchmark" / "task_classification.json"


def load_task_classification() -> dict[tuple[str, int], dict[str, Any]]:
    path = _classification_path()
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    for suite, items in raw.items():
        if not isinstance(items, list):
            continue
        for zero_based_idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            enriched["report_column"] = CATEGORY_MAP.get(str(item.get("category", "")), "")
            mapping[(str(suite), zero_based_idx)] = enriched
            if "id" in item:
                try:
                    mapping[(str(suite), int(item["id"]) - 1)] = enriched
                except (TypeError, ValueError):
                    pass
    return mapping


def enrich_rows(rows: list[dict[str, Any]], classification: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        info = classification.get((str(row["suite"]), int(row["task_id"])), {})
        new_row["task_name"] = new_row.get("task_name") or info.get("name", "")
        new_row["libero_plus_category"] = info.get("category", "")
        new_row["libero_plus_column"] = info.get("report_column", "")
        new_row["difficulty_level"] = info.get("difficulty_level", "")
        enriched_rows.append(new_row)
    return enriched_rows


def aggregate_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats = defaultdict(lambda: _empty_stats())
    for row in rows:
        column = str(row.get("libero_plus_column") or "")
        if column in CATEGORY_COLUMNS:
            _update_stats(stats[column], row)
    return [_finalize_stats(column, stats[column]) for column in CATEGORY_COLUMNS]


def _overall_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = _empty_stats()
    for row in rows:
        _update_stats(stats, row)
    return _finalize_stats("Total", stats)


def build_report_row(category_rows: list[dict[str, Any]], overall: dict[str, Any]) -> dict[str, float]:
    rates = {row["name"]: round(float(row["success_rate"]), 2) for row in category_rows}
    report = {column: rates.get(column, 0.0) for column in CATEGORY_COLUMNS}
    report["Total"] = round(float(overall["success_rate"]), 2)
    return report


def write_report_csv(path: Path, report_row: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerow(report_row)


def write_markdown_report(
    path: Path,
    *,
    output_dir: Path,
    report_row: dict[str, float],
    category_rows: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    overall: dict[str, Any],
) -> None:
    lines = [
        "# LIBERO-plus Evaluation Report",
        "",
        f"- Run directory: `{output_dir}`",
        f"- Completed tasks: {overall['completed_tasks']}/{overall['scheduled_tasks']}",
        f"- Failed tasks: {overall['failed_tasks']}",
        f"- Pending tasks: {overall['pending_tasks']}",
        f"- Success rate over completed episodes: {overall['success_rate']:.2f}%",
        "",
        "## Robustness Summary",
        "",
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "| " + " | ".join(["---:"] * len(REPORT_COLUMNS)) + " |",
        "| " + " | ".join(f"{report_row[column]:.2f}" for column in REPORT_COLUMNS) + " |",
        "",
        "## Category Detail",
        "",
        "| Category | Scheduled | Completed | Failed | Pending | Success Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['name']} | {row['scheduled_tasks']} | {row['completed_tasks']} | "
            f"{row['failed_tasks']} | {row['pending_tasks']} | {row['success_rate']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Suite Summary",
            "",
            "| Suite | Scheduled | Completed | Failed | Pending | Success Rate | Avg Duration |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in suite_rows:
        lines.append(
            f"| {row['name']} | {row['scheduled_tasks']} | {row['completed_tasks']} | "
            f"{row['failed_tasks']} | {row['pending_tasks']} | {row['success_rate']:.2f}% | "
            f"{format_time(row['average_duration_sec'])} |"
        )

    lines.extend(
        [
            "",
            "## GPU Summary",
            "",
            "| GPU | Tasks Seen | Completed | Failed | Running | Success Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gpu_rows:
        lines.append(
            f"| {row['name']} | {row['scheduled_tasks']} | {row['completed_tasks']} | "
            f"{row['failed_tasks']} | {row['running_tasks']} | {row['success_rate']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `libero_plus_report.csv`: one-row report with Camera/Robot/Language/Light/Background/Noise/Layout/Total.",
            "- `category_summary.csv`: detailed counts and success rate for each robustness category.",
            "- `task_results.csv`: one row per scheduled task.",
            "- `summary.json`: machine-readable full summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_results(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    classification = load_task_classification()
    rows = enrich_rows(merge_task_rows(output_dir), classification)
    suite_rows = aggregate_by_suite(rows)
    gpu_rows = aggregate_by_gpu(rows)
    category_rows = aggregate_by_category(rows)
    overall = _overall_row(rows)
    report_row = build_report_row(category_rows, overall)

    task_fields = [
        "suite",
        "task_id",
        "task_name",
        "libero_plus_category",
        "libero_plus_column",
        "difficulty_level",
        "status",
        "gpu_id",
        "successes",
        "total_episodes",
        "success_rate",
        "duration_sec",
        "duration",
        "task_description",
        "start_time",
        "result_file",
        "log_file",
        "return_code",
        "future_video_psnr_mean",
        "future_video_psnr_std",
    ]
    summary_fields = [
        "name",
        "scheduled_tasks",
        "completed_tasks",
        "failed_tasks",
        "running_tasks",
        "pending_tasks",
        "unknown_tasks",
        "successes",
        "total_episodes",
        "success_rate",
        "completion_rate",
        "total_duration_sec",
        "average_duration_sec",
        "max_duration_sec",
    ]

    write_csv(output_dir / "task_results.csv", rows, task_fields)
    write_csv(output_dir / "suite_summary.csv", suite_rows, summary_fields)
    write_csv(output_dir / "gpu_task_report.csv", gpu_rows, summary_fields + ["task_ids"])
    write_csv(output_dir / "category_summary.csv", category_rows, summary_fields)
    write_report_csv(output_dir / "libero_plus_report.csv", report_row)

    summary = {
        "run_id": output_dir.name,
        "classification_file": str(_classification_path()),
        "overall": overall,
        "libero_plus_report": report_row,
        "categories": category_rows,
        "suites": suite_rows,
        "gpus": gpu_rows,
        "tasks": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(
        output_dir / "report.md",
        output_dir=output_dir,
        report_row=report_row,
        category_rows=category_rows,
        suite_rows=suite_rows,
        gpu_rows=gpu_rows,
        overall=overall,
    )

    print("\n=== LIBERO-plus Robustness Report ===")
    print(" | ".join(REPORT_COLUMNS))
    print(" | ".join(f"{report_row[column]:.2f}" for column in REPORT_COLUMNS))
    print(f"Run directory: {output_dir}")
    print(f"CSV report: {output_dir / 'libero_plus_report.csv'}")
    print(f"Markdown report: {output_dir / 'report.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Run directory containing evaluation outputs.")
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
