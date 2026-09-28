"""Build portable episode path indices from LeRobot metadata."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.data_access import dataset_key, load_specs, open_raw_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/pretrain/dataset.yaml")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--output-root", type=Path, help="Override the configured index cache root")
    args = parser.parse_args()
    config, specs = load_specs(args.dataset_config, args.source, args.dataset)
    destinations = set()
    for spec in specs:
        if args.output_root:
            destination = args.output_root / dataset_key(spec) / "path_index.jsonl"
        elif spec.path_index_dir:
            destination = Path(spec.path_index_dir) / "path_index.jsonl"
        else:
            parser.error(f"No index directory configured for {dataset_key(spec)}; set --output-root")
        if destination.resolve() in destinations:
            raise ValueError(f"Multiple datasets share the index destination {destination}")
        destinations.add(destination.resolve())
        dataset = open_raw_dataset(spec, config)
        rows = dataset.export_path_index()
        if not rows:
            raise ValueError(f"No episodes found in {spec.remote_root}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"{dataset_key(spec)}: {len(rows)} episodes → {destination}")


if __name__ == "__main__":
    main()
