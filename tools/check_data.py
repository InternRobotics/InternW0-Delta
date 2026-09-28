"""Check local pretraining metadata, canonical fields and statistics."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from tools.data_access import dataset_key, load_specs
from wam.datasets.pretrain_lerobot_loader import PretrainLeRobotDataset
from wam.datasets.pretrain_stats import load_pretrain_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config', default='configs/pretrain/dataset.yaml')
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--dataset', action='append', default=[])
    parser.add_argument('--list', action='store_true', help='List dataset keys and local roots without opening episodes')
    parser.add_argument('--check-text-cache', action='store_true', help='Also require cached embeddings for the first sample')
    parser.add_argument('--limit', type=int, default=0, help='Check at most this many datasets; 0 checks all selected datasets')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    config, specs = load_specs(args.dataset_config, args.source, args.dataset)
    if args.limit:
        specs = specs[:args.limit]
    if args.list:
        for spec in specs:
            print(f'{dataset_key(spec)}\t{spec.remote_root}\t{spec.stats_group}')
        return
    statistics, _ = load_pretrain_stats(specs, config)
    if statistics is None:
        raise ValueError('Configure normalization_stats for each selected source')
    groups = statistics['groups']
    failures = []
    sampling = config.get('sampling', {})
    for spec in specs:
        key = dataset_key(spec)
        try:
            if spec.stats_group not in groups:
                raise KeyError(f'Missing normalization group: {spec.stats_group}')
            checked = spec if args.check_text_cache else replace(spec, local_text_embedding_cache_dir=None)
            ds = PretrainLeRobotDataset(
                checked, num_frames=int(sampling.get('num_frames', 33)),
                action_size=int(sampling.get('action_size', 32)),
                global_sample_stride=int(sampling.get('global_sample_stride', 1)),
            )
            allow_padding = spec.allow_padding_at_end
            if allow_padding is None:
                allow_padding = bool(config.get('mixture', {}).get('allow_padding_at_end', False))
            counts = ds.valid_start_counts(bool(allow_padding))
            candidates = [i for i, count in enumerate(counts) if count > 0]
            if not candidates:
                raise ValueError('No valid training windows; check the episode lengths and sampling configuration')
            trajectory = candidates[0]
            episode = int(ds.trajectory_ids[trajectory])
            frame = ds.valid_start_for_trajectory_rank(trajectory, 0, bool(allow_padding))
            ds.get_step_item(episode, frame)
            for camera in ds.video_keys:
                video = Path(ds._remote_path(ds._video_rel_path(episode, camera)))
                if not video.is_file():
                    raise FileNotFoundError(video)
            print(f'OK {key}: episodes={len(ds.trajectory_ids)} valid_windows={int(counts.sum())}')
        except Exception as exc:
            failures.append(key)
            print(f'FAIL {key}: {type(exc).__name__}: {exc}', file=sys.stderr)
    print(f'Checked {len(specs)} datasets; {len(failures)} failed. Episode payload checks sample one window per dataset.')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
