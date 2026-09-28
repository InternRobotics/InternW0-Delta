"""Render local dataset video with measured/commanded URDF meshes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import av
import numpy as np
import torch
import yaml

from tools.data_access import load_specs
from tools.visualization.layout import ReplayLayout, WIDTH, HEIGHT
from tools.visualization.video import VideoReader
from wam.datasets.pretrain_lerobot_loader import PretrainLeRobotDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config', default='configs/pretrain/dataset.yaml')
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--joint-map', type=Path, required=True)
    parser.add_argument('--package-root', action='append', default=[], metavar='PACKAGE=PATH')
    parser.add_argument('--geometry', choices=['meshes', 'links'], default='meshes')
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--camera', help='Camera key; defaults to the first configured view')
    parser.add_argument('--joint-dim', type=int, help='Canonical joint dimension shown in the signal chart')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=480, help='Maximum source frames, 0 for the whole episode')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--replay-frame', type=int, help='Append two source seconds around this frame at 0.25x')
    args = parser.parse_args()
    if args.start_frame < 0 or args.max_frames < 0 or args.fps <= 0:
        parser.error('start-frame/max-frames must be nonnegative and fps positive')
    for path in [args.output, args.output.with_suffix('.json'), args.output.with_suffix('.jpg')]:
        if path.exists():
            raise FileExistsError(path)
    config, specs = load_specs(args.dataset_config, args.source, [args.dataset])
    if len(specs) != 1:
        parser.error('Select one dataset with --source and source-name/dataset-name')
    spec = replace(specs[0], local_text_embedding_cache_dir=None)
    sampling = config.get('sampling', {})
    ds = PretrainLeRobotDataset(
        spec, num_frames=int(sampling.get('num_frames', 33)), action_size=int(sampling.get('action_size', 32)),
        global_sample_stride=int(sampling.get('global_sample_stride', 1)))
    length = int(ds.episodes_dict[args.episode]['length'])
    end = min(length, args.start_frame + args.max_frames) if args.max_frames else length
    frames = np.arange(args.start_frame, end)
    if not len(frames):
        raise ValueError('Requested interval contains no source frames')
    parquet = ds._load_episode_parquet(args.episode)
    indices = ds.source_frame_indices(args.episode, frames.tolist())
    next_indices = ds.source_frame_indices(args.episode, np.minimum(frames + ds.global_sample_stride, length - 1).tolist())
    with torch.inference_mode():
        states, state_pad, state_valid = ds._canonical_project_with_validity(parquet, indices, 'state')
        actions, action_pad, action_valid = ds._canonical_project_with_validity(parquet, indices, 'action', next_indices)
    states, actions = states.numpy(), actions.numpy()
    mapping = yaml.safe_load(args.joint_map.read_text())
    recorded = mapping.get('recorded_joints', {})
    if recorded:
        from tools.visualization.recorded import read_joints
        if recorded.get('state'):
            states, state_pad = read_joints(ds, args.episode, indices, recorded['state'])
            state_valid = None
        if recorded.get('action'):
            offset = int(recorded.get('action_frame_offset', 0))
            shifted = np.clip(frames + offset, 0, length - 1).tolist()
            action_indices = ds.source_frame_indices(args.episode, shifted)
            actions, action_pad = read_joints(ds, args.episode, action_indices, recorded['action'])
            action_valid = None
    instances = mapping.get('instances') or [{'joints': mapping.get('joints', mapping)}]
    dims = sorted({int(v if isinstance(v, int) else v['dim']) for item in instances for v in item['joints'].values()})
    allowed = set(range(7)) | set(range(40, 47))
    if not dims or any(d not in allowed or bool(state_pad[d]) for d in dims):
        raise ValueError('URDF mappings require valid absolute joint-state slots [0,7) / [40,47)')
    if state_valid is not None and not bool(state_valid[:, dims].all()):
        raise ValueError('Selected interval contains invalid measured joints; choose a valid interval')
    if not np.isfinite(states[:, dims]).all():
        raise ValueError('Selected interval contains non-finite measured joints')
    if (any(bool(action_pad[d]) for d in dims)
            or (action_valid is not None and not bool(action_valid[:, dims].all()))
            or not np.isfinite(actions[:, dims]).all()):
        actions = None
    dim = args.joint_dim if args.joint_dim is not None else dims[0]
    if dim not in dims:
        raise ValueError('--joint-dim must be a mapped joint')
    camera = args.camera or ds.video_keys[0]
    if camera not in ds.video_keys:
        raise ValueError(f'Unknown camera {camera}; available: {ds.video_keys}')
    sample = {'episode_index': torch.tensor(args.episode), 'frame_index': torch.tensor(frames)}
    if ds.timestamp_key in parquet:
        sample['timestamp'] = parquet[ds.timestamp_key][indices]
    if ds.frame_index_key in parquet:
        sample['frame_index'] = parquet[ds.frame_index_key][indices]
    video_times = np.asarray(ds.resolve_video_timestamps(sample, camera), dtype=float)
    times = frames / ds.fps
    reference = str(mapping.get('label', args.urdf.stem)) + ' · ' + ('URDF meshes' if args.geometry == 'meshes' else 'illustrative links')
    layout = ReplayLayout(args.dataset, args.episode, frames, times, states, actions, dim, reference, camera)
    duration = len(frames) / ds.fps
    timeline = times[0] + np.arange(max(1, int(round(duration * args.fps)))) / args.fps
    regular = len(timeline)
    if args.replay_frame is not None:
        if not frames[0] <= args.replay_frame <= frames[-1]:
            raise ValueError('--replay-frame must lie inside the selected source interval')
        span = min(2., duration)
        start = np.clip(args.replay_frame / ds.fps - .8, times[0], times[0] + duration - span)
        timeline = np.r_[timeline, start + np.arange(int(round(span * 4 * args.fps))) / (4 * args.fps)]
    packages = {}
    for item in args.package_root:
        key, value = item.split('=', 1)
        packages[key] = Path(value).expanduser().resolve()
    from tools.visualization.robot import RobotRenderer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.partial.mp4')
    try:
        with ExitStack() as stack:
            rig = RobotRenderer(args.urdf, mapping, states, actions, packages=packages, geometry=args.geometry, backend=args.backend)
            stack.callback(rig.close)
            path = ds._remote_path(ds._video_rel_path(args.episode, camera))
            reader = VideoReader(path, video_times[0])
            stack.callback(lambda: reader.close())
            output = stack.enter_context(av.open(str(temp), 'w', options={'movflags': '+faststart'}))
            stream = output.add_stream('libx264', rate=Fraction(args.fps))
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
            stream.options = {'crf': '20', 'preset': 'fast', 'threads': '2'}
            last = None
            errors = []
            poster = min(regular - 1, regular // 2)
            for k, timestamp in enumerate(timeline):
                index = int(np.clip(np.searchsorted(times, timestamp, side='right') - 1, 0, len(frames) - 1))
                if k == regular:
                    errors.append(reader.max_error)
                    reader.close()
                    reader = VideoReader(path, video_times[index])
                if index != last:
                    measured = rig.render(states[index])
                    command = rig.render(actions[index], True) if actions is not None else None
                    last = index
                image = layout.frame(index, reader.frame_at(video_times[index]), measured, command, timestamp, k >= regular)
                if k == poster:
                    image.save(args.output.with_suffix('.jpg'), quality=93)
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
            errors.append(reader.max_error)
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)
    report = dict(dataset=args.dataset, episode=args.episode, source_interval=[int(frames[0]), int(frames[-1]) + 1],
                  source_fps=ds.fps, output_fps=args.fps, output_frames=len(timeline), duration_s=len(timeline) / args.fps,
                  camera=camera, max_video_timestamp_error_seconds=max(errors), urdf=rig.provenance,
                  recorded_joints=recorded,
                  command_available=actions is not None)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
