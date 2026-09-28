#!/usr/bin/env python3
"""Aggregate rank-local InternW0-delta profiler summaries.

Usage:
    python scripts/aggregate_profiler.py OUTPUT_DIR/profiler

The input directory is expected to contain ``rank_*/summary.json`` files.  A
JSON report and a compact text table are written next to the rank folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _flatten_sections(
    sections: list[dict[str, Any]],
    output: dict[str, list[dict[str, Any]]],
    rank: int,
) -> None:
    for section in sections:
        path = str(section.get("path") or section.get("name") or "")
        if path:
            output.setdefault(path, []).append({**section, "_rank": rank})
        children = section.get("children") or []
        if isinstance(children, list):
            _flatten_sections(children, output, rank)


def _rank_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read profiler summary {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sections"), list):
        raise RuntimeError(f"Invalid profiler summary schema: {path}")
    return payload


def aggregate(profile_dir: Path) -> dict[str, Any]:
    summaries = [_rank_summary(path) for path in sorted(profile_dir.glob("rank_*/summary.json"))]
    if not summaries:
        raise RuntimeError(f"No rank summaries found below {profile_dir}")

    by_path: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        _flatten_sections(summary["sections"], by_path, int(summary.get("rank", 0)))

    sections: list[dict[str, Any]] = []
    for path, rows in by_path.items():
        inclusive = [float(row.get("inclusive_seconds", row.get("elapsed_seconds", 0.0))) for row in rows]
        exclusive = [float(row.get("exclusive_seconds", 0.0)) for row in rows]
        calls = [int(row.get("calls", 0)) for row in rows]
        max_index = max(range(len(inclusive)), key=inclusive.__getitem__)
        mean_inclusive = sum(inclusive) / len(inclusive)
        sections.append(
            {
                "path": path,
                "rank_count": len(rows),
                "inclusive_mean_seconds": mean_inclusive,
                "inclusive_min_seconds": min(inclusive),
                "inclusive_max_seconds": max(inclusive),
                "inclusive_p50_seconds": _percentile(inclusive, 0.50),
                "inclusive_p95_seconds": _percentile(inclusive, 0.95),
                "exclusive_mean_seconds": sum(exclusive) / len(exclusive),
                "exclusive_max_seconds": max(exclusive),
                "calls_total": sum(calls),
                "calls_mean_per_rank": sum(calls) / len(calls),
                "slowest_rank": int(rows[max_index].get("_rank", max_index)),
                "rank_imbalance": (
                    max(inclusive) / mean_inclusive - 1.0
                    if mean_inclusive > 0
                    else 0.0
                ),
            }
        )
    sections.sort(
        key=lambda row: (
            float(row["exclusive_mean_seconds"]),
            float(row["inclusive_mean_seconds"]),
        ),
        reverse=True,
    )

    wall_totals = [float(summary.get("wall_total_seconds", 0.0)) for summary in summaries]
    tracked_totals = [float(summary.get("tracked_total_seconds", 0.0)) for summary in summaries]
    allocated = [summary.get("gpu_peak_allocated_bytes") for summary in summaries]
    reserved = [summary.get("gpu_peak_reserved_bytes") for summary in summaries]
    return {
        "schema_version": 1,
        "profile_dir": str(profile_dir.absolute()),
        "rank_count": len(summaries),
        "world_size": int(summaries[0].get("world_size", len(summaries))),
        "wall_total_mean_seconds": sum(wall_totals) / len(wall_totals),
        "wall_total_max_seconds": max(wall_totals),
        "tracked_total_mean_seconds": sum(tracked_totals) / len(tracked_totals),
        "gpu_peak_allocated_max_bytes": max(
            (int(value) for value in allocated if value is not None),
            default=None,
        ),
        "gpu_peak_reserved_max_bytes": max(
            (int(value) for value in reserved if value is not None),
            default=None,
        ),
        "sections": sections,
    }


def _format_seconds(value: float) -> str:
    return f"{float(value):9.3f}s"


def write_text_report(report: dict[str, Any]) -> str:
    path_width = min(
        88,
        max(
            40,
            max((len(str(row["path"])) for row in report["sections"]), default=40),
        ),
    )
    lines = [
        "InternW0-delta profiler aggregate",
        f"profile_dir={report['profile_dir']}",
        f"ranks={report['rank_count']}/{report['world_size']} "
        f"wall_mean={_format_seconds(report['wall_total_mean_seconds'])} "
        f"wall_max={_format_seconds(report['wall_total_max_seconds'])}",
        "",
        f"{'path':<{path_width}} mean incl     mean self     max incl   slow rank  imbalance",
        "-" * (path_width + 72),
    ]
    for row in report["sections"]:
        lines.append(
            f"{str(row['path']):<{path_width}.{path_width}s} "
            f"{_format_seconds(row['inclusive_mean_seconds'])} "
            f"{_format_seconds(row['exclusive_mean_seconds'])} "
            f"{_format_seconds(row['inclusive_max_seconds'])} "
            f"{int(row['slowest_rank']):9d} "
            f"{float(row['rank_imbalance']) * 100.0:8.1f}%"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: <profile_dir>/aggregate.json)",
    )
    args = parser.parse_args()

    profile_dir = args.profile_dir.expanduser().resolve()
    report = aggregate(profile_dir)
    json_path = args.output or profile_dir / "aggregate.json"
    text_path = json_path.with_suffix(".txt")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(write_text_report(report), encoding="utf-8")
    print(write_text_report(report), end="")
    print(f"JSON report: {json_path}")
    print(f"Text report: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
