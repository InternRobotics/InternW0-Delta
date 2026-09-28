"""Shared local dataset selection for preparation and inspection tools."""
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from wam.datasets import pretrain_lerobot_loader as loader


def dataset_key(spec):
    return f"{spec.source_name}/{spec.name}" if spec.source_name else spec.name


def load_specs(config_path, sources=(), names=()):
    config = loader._read_dataset_config(Path(config_path))
    if sources:
        config["sources"] = [s for s in config["sources"] if s.get("_source_file") in sources]
        if not config["sources"]:
            raise ValueError(f"No configured sources match {list(sources)}")
    specs = loader._load_specs(Namespace(dataset_specs_json=None, remote_root=None, name="dataset"), config)
    specs = [s for s in specs if s.dataset_weight > 0 and (not names or s.name in names or dataset_key(s) in names)]
    if not specs:
        raise ValueError("No active datasets selected")
    return config, specs


def open_raw_dataset(spec, config):
    """Read episode metadata directly without using a path index."""
    sampling = config.get("sampling", {})
    return loader.PretrainLeRobotDataset(
        replace(spec, require_path_index=False, path_index_dir=None),
        num_frames=int(sampling.get("num_frames", 33)),
        action_size=int(sampling.get("action_size", 32)),
        global_sample_stride=int(sampling.get("global_sample_stride", 1)),
        use_path_index_cache=False,
    )
