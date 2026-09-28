"""Prepare the pinned external Track4World source tree for feature extraction."""
import argparse
from pathlib import Path
import subprocess

BASE = "1f18a53771769b98fef3a5ffeb835f53c9c66e1c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="External Track4World checkout directory")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    patch = Path(__file__).with_name("teacher.patch").resolve()
    if not root.exists():
        subprocess.run(["git", "clone", "https://github.com/TencentARC/Track4World.git", str(root)], check=True)
        subprocess.run(["git", "checkout", BASE], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != BASE:
        raise SystemExit(f"Expected revision {BASE}; use a separate checkout directory.")
    command = ["git", "apply", "--check", str(patch)]
    result = subprocess.run(command, cwd=root, capture_output=True)
    if result.returncode == 0:
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    else:
        subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=root, check=True)
    dependency = root / "utils3d"
    revision = "2072c024c73f7c0f83e0da23eef5f2d9ac575249"
    if not dependency.exists():
        subprocess.run(["git", "clone", "https://github.com/jiah-cloud/utils3d.git", str(dependency)], check=True)
        subprocess.run(["git", "checkout", revision], cwd=dependency, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dependency, text=True).strip()
    if actual != revision:
        raise SystemExit(f"utils3d must be at {revision}.")
    print(f"Teacher source ready: {root}")


if __name__ == "__main__":
    main()
