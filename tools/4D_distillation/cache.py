"""Select windows, extract offline teachers, and validate a portable 4D cache."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from wam.datasets.distillation_cache import (
    EXTRACTION, FEATURE_DIM, TeacherCache, WindowPlan, atomic_json, dataset_id, sha256_file,
)


def outside_repository(path):
    path = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if path == repository or repository in path.parents:
        raise ValueError("Store generated windows/features outside the code repository.")
    return path


def load_dataset(config):
    from wam.datasets.pretrain_wam_dataset import PretrainWAMDataset
    return PretrainWAMDataset(
        dataset_config=config, num_frames=33, action_size=32, global_sample_stride=1,
        action_video_freq_ratio=4, video_size=(384, 256), is_training_set=False,
        text_context_required=False, skip_bad_videos=False,
    )


def dataset_contract(wrapper, names):
    names = set(names)
    return {
        dataset_id(source.spec): {
            "video_keys": list(source.video_keys),
            "video_concat": wrapper._video_concat_mode(source),
            "fps": float(source.fps),
        }
        for source in wrapper.datasets if dataset_id(source.spec) in names
    }


def allocate(size, weights, capacities):
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if size < 0 or size > capacities[weights > 0].sum():
        raise ValueError("Requested windows exceed the positive-weight available capacity.")
    result = np.zeros_like(capacities)
    while size:
        active = (result < capacities) & (weights > 0)
        ideal = np.where(active, weights, 0)
        ideal = ideal / ideal.sum() * size
        bulk = np.minimum(np.floor(ideal).astype(np.int64), capacities - result)
        if bulk.sum():
            result += bulk
            size -= int(bulk.sum())
        else:
            order = sorted(np.flatnonzero(active), key=lambda i: (-ideal[i], -weights[i], i))
            chosen = order[:size]
            result[chosen] += 1
            size -= len(chosen)
    return result


def publish_windows(root, rows, names, **metadata):
    root = outside_repository(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "windows.json").exists() or (root / "features.json").exists():
        raise FileExistsError(f"Use a new window-plan directory: {root}")
    if rows.shape[0] <= 0:
        raise ValueError("The selection has no windows.")
    np.save(root / "windows.npy", np.asarray(rows, dtype="<u4"), allow_pickle=False)
    atomic_json(root / "windows.json", {
        "format": "internw0_4d_windows_v1", "sample_count": int(rows.shape[0]),
        "datasets": names, "extraction": EXTRACTION,
        "windows_file_sha256": sha256_file(root / "windows.npy"), **metadata,
    })


def select(args):
    if args.size <= 0:
        raise ValueError("--size must be positive.")
    dataset = load_dataset(args.dataset_config)
    counts = [d.valid_start_counts(False).astype(np.int64) for d in dataset.datasets]
    capacities = np.array([c.sum() for c in counts], dtype=np.int64)
    from wam.datasets.pretrain_lerobot_loader import _mixture_dataset_weights
    weights = np.asarray(_mixture_dataset_weights(dataset.specs, dataset.datasets, dataset.config["mixture"]), dtype=np.float64)
    groups = np.array([s.group_id for s in dataset.specs])
    unique_groups = np.unique(groups)
    group_sizes = allocate(args.size, [weights[groups == g].sum() for g in unique_groups], [capacities[groups == g].sum() for g in unique_groups])
    allocations = np.zeros_like(capacities)
    for group, size in zip(unique_groups, group_sizes):
        mask = groups == group
        allocations[mask] = allocate(int(size), weights[mask], capacities[mask])
    selected = np.empty((args.size, 3), dtype="<u4")
    cursor = 0
    for pos, (source, count, capacity) in enumerate(zip(dataset.datasets, allocations, capacities)):
        ranks = np.sort(np.random.default_rng(args.seed + pos).choice(int(capacity), size=int(count), replace=False))
        cumulative = counts[pos].cumsum()
        trajectories = np.searchsorted(cumulative, ranks, side="right")
        for rank, trajectory in zip(ranks, trajectories):
            before = int(cumulative[trajectory - 1]) if trajectory else 0
            episode = int(source.trajectory_ids[trajectory])
            start = source.valid_start_for_trajectory_rank(int(trajectory), int(rank) - before, False)
            if min(episode, start) < 0 or max(episode, start) >= 2**32:
                raise ValueError("Episode/start exceeds the portable uint32 range.")
            selected[cursor] = pos, episode, start
            cursor += 1
    used = np.unique(selected[:, 0])
    remap = np.zeros(len(dataset.specs), dtype=np.uint32)
    remap[used] = np.arange(len(used), dtype=np.uint32)
    selected[:, 0] = remap[selected[:, 0]]
    names = [dataset_id(dataset.specs[int(i)]) for i in used]
    publish_windows(args.output, selected, names, seed=args.seed,
                    dataset_contract=dataset_contract(dataset, names),
                    selection="Weighted source groups, then valid windows without replacement.")
    print(f"Selected {cursor} windows into {args.output}")


def extract(args):
    import torch
    from teacher import decode_canvas, load_teacher

    root = outside_repository(args.cache)
    plan = WindowPlan(root)
    start = args.shard_index * args.shard_size
    end = min(len(plan), start + args.shard_size)
    if start < 0 or start >= end or args.shard_size <= 0:
        raise ValueError("Shard index/size is outside the window plan.")
    compile_motion = False
    part = root / "parts" / f"{start:09d}_{end:09d}"
    part.mkdir(parents=True, exist_ok=True)
    with (part / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = {"windows_sha256": plan.fingerprint, "start": start, "end": end,
                    "teacher_patch_sha256": sha256_file(Path(__file__).with_name("teacher.patch")),
                    "compiled_motion": compile_motion, "teacher_seed": 42,
                    "teacher_weights": sha256_file(Path(args.checkpoint))}
        metadata_path = part / "part.json"
        state = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if state and any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing shard belongs to a different window plan.")
        feature_file = part / "features.f16"
        if state.get("complete"):
            if sha256_file(feature_file) != state["sha256"]:
                raise ValueError("Completed shard checksum mismatch.")
            print(f"Already complete: {part.name}")
            return
        dataset = load_dataset(args.dataset_config)
        positions = {dataset_id(d.spec): d for d in dataset.datasets}
        needed = {plan.datasets[int(i)] for i in np.unique(plan.rows[start:end, 0])}
        missing = needed - positions.keys()
        if missing:
            raise ValueError(f"Window sources absent from dataset_config: {sorted(missing)[:8]}")
        recorded_contract = plan.manifest.get("dataset_contract", {})
        current_contract = dataset_contract(dataset, needed)
        if any(recorded_contract.get(name) != current_contract.get(name) for name in needed):
            raise ValueError("Camera layout or FPS changed since window selection; generate a new plan.")
        adapter = load_teacher(args.teacher_root, args.checkpoint, args.device, compile_motion)
        features = np.memmap(feature_file, dtype="<f2", mode="r+" if state else "w+", shape=(end - start, FEATURE_DIM))
        done = int(state.get("done", 0))
        if not 0 <= done <= end - start:
            raise ValueError("Invalid shard progress.")
        try:
            for row in range(start + done, end):
                name, episode, first = plan.identity(row)
                clip = decode_canvas(dataset, positions[name], episode, first)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = adapter.extract(clip.unsqueeze(0).to(device=args.device, dtype=torch.float32))
                value = result.teacher_feature.detach().float().cpu().numpy()
                if not result.teacher_valid or value.shape != (FEATURE_DIM,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid teacher feature at window row {row}.")
                features[row - start] = value.astype("<f2")
                done = row + 1 - start
                if done % 100 == 0 or row + 1 == end:
                    features.flush()
                    atomic_json(metadata_path, {**expected, "done": done, "complete": False})
                    print(f"{part.name}: {done}/{end-start}", flush=True)
            features.flush()
            atomic_json(metadata_path, {**expected, "done": done, "complete": True, "sha256": sha256_file(feature_file)})
        finally:
            adapter.close()


def seal(args):
    plan = WindowPlan(args.cache)
    root = outside_repository(args.cache)
    shards = []
    teacher_weights = set()
    for path in sorted((root / "parts").glob("*/part.json")):
        meta = json.loads(path.read_text())
        if (not meta.get("complete") or meta.get("windows_sha256") != plan.fingerprint
                or meta.get("teacher_patch_sha256") != sha256_file(Path(__file__).with_name("teacher.patch"))):
            raise ValueError(f"Incomplete or incompatible shard: {path.parent.name}")
        if not meta.get("teacher_weights"):
            raise ValueError(f"Missing teacher identity: {path}")
        teacher_weights.add(meta["teacher_weights"])
        feature = path.parent / "features.f16"
        if sha256_file(feature) != meta["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path.parent.name}")
        shards.append({"start": meta["start"], "end": meta["end"], "file": feature.relative_to(root).as_posix(), "sha256": meta["sha256"]})
    if len(teacher_weights) != 1:
        raise ValueError("Feature shards must use the same teacher weights.")
    cursor = 0
    for shard in shards:
        if shard["start"] != cursor or shard["end"] <= cursor:
            raise ValueError("Feature shards overlap or leave gaps.")
        if (root / shard["file"]).stat().st_size != (shard["end"] - cursor) * FEATURE_DIM * 2:
            raise ValueError("Feature shard size mismatch.")
        cursor = shard["end"]
    if cursor != len(plan):
        raise ValueError(f"Expected {len(plan)} feature rows, found {cursor} contiguous rows.")
    if sha256_file(root / "windows.npy") != plan.manifest["windows_file_sha256"]:
        raise ValueError("Window index checksum mismatch.")
    atomic_json(root / "features.json", {"format": "internw0_4d_features_v1", "windows_sha256": plan.fingerprint, "teacher_weights": next(iter(teacher_weights)), "shards": shards})
    TeacherCache(root)
    print(f"Ready for training: {cursor} rows, {len(shards)} shards")


def check(args):
    plan = WindowPlan(args.cache)
    if sha256_file(plan.root / "windows.npy") != plan.manifest["windows_file_sha256"]:
        raise ValueError("Window index checksum mismatch.")
    for start in range(0, len(plan), 100000):
        if int(plan.rows[start:start+100000, 0].max()) >= len(plan.datasets):
            raise ValueError("Window references an unknown dataset ordinal.")
    if args.windows_only:
        print(f"Window plan valid: {len(plan)} rows, {len(plan.datasets)} datasets")
        return
    cache = TeacherCache(args.cache)
    for shard in cache.shards:
        cache.feature(shard["start"])
        cache.feature(shard["end"] - 1)
        if args.full and sha256_file(cache.root / shard["file"]) != shard["sha256"]:
            raise ValueError(f"Feature checksum mismatch: {shard['file']}")
    print(f"Cache valid: {len(cache)} rows, {len(cache.shards)} shards")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("select", help="Sample a new window plan from local datasets")
    p.add_argument("--dataset-config", default="configs/4D_distillation/teacher_data.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("extract", help="Generate/resume one independent feature shard on a GPU")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--dataset-config", default="configs/4D_distillation/teacher_data.yaml")
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=100000)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p = commands.add_parser("seal", help="Verify completed shards and publish the training manifest")
    p.add_argument("--cache", type=Path, required=True)
    p = commands.add_parser("check", help="Validate indices and feature coverage without loading models")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--windows-only", action="store_true")
    p.add_argument("--full", action="store_true", help="Verify all feature-file checksums")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
