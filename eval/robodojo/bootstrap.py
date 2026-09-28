"""Locate the bundled policy transport."""
from pathlib import Path
import sys


def setup_policy_paths():
    benchmark = Path(__file__).resolve().parent / "benchmark"
    for directory in (benchmark, benchmark / "XPolicyLab"):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
