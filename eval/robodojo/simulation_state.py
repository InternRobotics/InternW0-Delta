#!/usr/bin/env python3
"""Report whether the latest Isaac Lab simulation startup is still pending."""

from __future__ import annotations

import argparse
from pathlib import Path


START_MARKER = b"Starting the simulation"
READY_MARKER = b"Time taken for simulation start"
DEFAULT_TAIL_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES)
    return parser.parse_args()


def detect_state(payload: bytes) -> str:
    latest_start = payload.rfind(START_MARKER)
    if latest_start < 0:
        return "not-started"
    latest_ready = payload.rfind(READY_MARKER)
    if latest_ready > latest_start:
        return "ready"
    return "pending"


def read_tail(path: Path, tail_bytes: int) -> bytes:
    if tail_bytes < 1:
        raise ValueError("--tail-bytes must be positive")
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - tail_bytes))
            return stream.read()
    except FileNotFoundError:
        return b""


def main() -> int:
    args = parse_args()
    print(detect_state(read_tail(args.log, args.tail_bytes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
