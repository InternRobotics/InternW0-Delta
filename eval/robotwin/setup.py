"""Prepare the pinned RoboTwin checkout with local assets and portable planner paths."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

REVISION = '13c3c47ff4312dd62484bcd51be034af55c062d1'


def prepare_assets(root: Path, assets: Path):
    for name in ('background_texture', 'objects', 'embodiments'):
        source = assets / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = root / 'assets' / name
        if destination.exists() or destination.is_symlink():
            if name != 'embodiments' and destination.resolve() == source.resolve():
                continue
            if name != 'embodiments' or destination.is_symlink() or not destination.is_dir():
                raise FileExistsError(f'{destination}: use a fresh simulator checkout')
        if name != 'embodiments':
            destination.symlink_to(source.resolve(), target_is_directory=True)
            continue
        # Copy small config files. Link immutable meshes/URDFs without changing downloaded assets.
        for path in source.rglob('*'):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.suffix in ('.yml', '.yaml'):
                import yaml
                value = yaml.safe_load(path.read_text())
                def localize(node):
                    if isinstance(node, dict):
                        return {k: localize(v) for k, v in node.items()}
                    if isinstance(node, list):
                        return [localize(v) for v in node]
                    if isinstance(node, str) and '/embodiments/' in node:
                        suffix = node.split('/embodiments/', 1)[1]
                        return str(destination / suffix)
                    return node
                expected = yaml.safe_dump(localize(value), sort_keys=False)
                if target.exists():
                    if target.is_symlink() or target.read_text() != expected:
                        raise FileExistsError(f'{target}: existing configuration differs; use a fresh checkout')
                else:
                    target.write_text(expected)
            else:
                if target.exists() or target.is_symlink():
                    if not target.is_symlink() or target.resolve() != path.resolve():
                        raise FileExistsError(target)
                else:
                    target.symlink_to(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ROBOTWIN_ROOT', 'third_party/RoboTwin')))
    parser.add_argument('--assets', type=Path, required=True, help='Extracted directory containing embodiments/, objects/, background_texture/')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', 'https://github.com/RoboTwin-Platform/RoboTwin.git', str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'checkout', '--detach', REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected RoboTwin {REVISION}, got {revision}')
    prepare_assets(root, args.assets.expanduser().resolve())
    print(f'Prepared RoboTwin at {root}')


if __name__ == '__main__':
    main()
