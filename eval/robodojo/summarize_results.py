"""Run RoboDojo's official result aggregation."""
from pathlib import Path
import os
import runpy


def main():
    default = Path(os.environ.get("ROBODOJO_OUTPUT_ROOT", "runs/eval/robodojo")) / "results"
    results = Path(os.environ.get("ROBODOJO_RESULT_ROOT", str(default))) / "RoboDojo"
    os.environ.setdefault("ROBODOJO_EVAL_ROOT", str(results.resolve()))
    path = Path(__file__).resolve().parent / "benchmark/scripts/internal/summarize_result.py"
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
