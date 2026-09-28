"""Prepare the pinned simulator sources and optional public benchmark assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def local_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def clone_pinned(url: str, revision: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", url], check=True)
    if not (destination / ".git").is_dir():
        raise ValueError(f"Expected a Git checkout at {destination}")
    status = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True)
    if status.strip():
        raise ValueError(f"Dependency checkout has local changes: {destination}")
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"Dependency revision mismatch at {destination}")


def prepare(root: Path):
    """Copy the evaluation runtime, preserving assets and user-generated results."""
    source_root = HERE / "benchmark"
    files = sorted(p for p in source_root.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for source in files:
        dest = root / source.relative_to(source_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    return len(files)


def prepare_assets(root: Path, destination: Path):
    from huggingface_hub import snapshot_download

    lock = json.loads((HERE / "artifacts.json").read_text())["robodojo_assets"]
    snapshot_download(repo_id=lock["dataset"], repo_type="dataset", revision=lock["revision"],
                      allow_patterns=["Assets/**"], local_dir=str(destination))
    assets = destination / "Assets"
    for name in lock["required_subdirectories"]:
        if not (assets / name).is_dir():
            raise FileNotFoundError(assets / name)
    link = root / "Assets"
    if link.exists() or link.is_symlink():
        if link.resolve() != assets.resolve():
            raise ValueError(f"{link} already points to different assets; use a fresh benchmark root")
    else:
        link.symlink_to(os.path.relpath(assets, root), target_is_directory=True)
    # The planner needs absolute filenames at runtime. Generate them locally
    # from the templates after downloading or moving the checkout.
    templates = list((link / "Robots").rglob("*_tmp.yml"))
    if not templates:
        raise FileNotFoundError("No robot configuration templates in Assets/Robots")
    for template in templates:
        content = template.read_text().replace("${ASSETS_PATH}", str(root)).replace("$ASSETS_PATH", str(root))
        template.with_name(template.name.replace("_tmp.yml", ".yml")).write_text(content)
    if not (link / "Robots/x5/curobo.yml").is_file():
        raise FileNotFoundError("Missing generated x5/curobo.yml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("ROBODOJO_ROOT", "third_party/RoboDojo"))
    parser.add_argument("--dependencies", action="store_true", help="Download pinned Isaac Lab and cuRobo sources")
    parser.add_argument("--assets", action="store_true", help="Download the pinned benchmark assets (about 39 GB)")
    parser.add_argument("--asset-cache", default=os.environ.get("ROBODOJO_ASSET_CACHE", "data/robodojo-assets"))
    args = parser.parse_args()
    root = local_path(args.root)
    count = prepare(root)
    if args.dependencies:
        sources = json.loads((HERE / "sources.json").read_text())
        for name, folder in [("isaaclab", "IsaacLab"), ("curobo", "curobo")]:
            item = sources[name]
            clone_pinned(item["upstream"], item["commit"], root / "third_party" / folder)
    if args.assets:
        prepare_assets(root, local_path(args.asset_cache))
    print(f"Prepared {count} benchmark files in {root}")


if __name__ == "__main__":
    main()
