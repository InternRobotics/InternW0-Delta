"""Track4World loading and the RGB preprocessing used by the offline teacher."""
import json
import os
from pathlib import Path
import sys
import subprocess

import torch
import torch.nn.functional as F



def load_teacher(root, checkpoint, device, compile_motion=False, seed=42):
    root = Path(root).resolve()
    subprocess.run(
        ["git", "apply", "--reverse", "--check", str(Path(__file__).with_name("teacher.patch").resolve())],
        cwd=root, check=True, capture_output=True,
    )
    sys.path.insert(0, str(root))
    os.environ["TRACK4WORLD_OFFLINE_BACKBONE"] = "1"
    from track4world.nets.model import Track4World
    from track4world.distillation_adapter import Track4WorldDistillationAdapter, enable_teacher_motion_compile

    config = json.loads((root / "track4world/config/eval/v1.json").read_text())
    torch.manual_seed(seed)
    model = Track4World(**config["model"], seqlen=32, use_3d=True, use_model="depthanythingv3")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_pretrained_with_remap(state)
    del state
    # The temporal sinusoidal buffers are constructed for H32 from their formula.
    missing = {
        key for key in missing
        if key not in {"time_emb", "time_emb3d"}
        and not key.startswith("backbone.model.da3_metric.")
    }
    if missing or unexpected:
        raise ValueError(f"Incomplete teacher weights: missing={sorted(missing)[:8]}, unexpected={list(unexpected)[:8]}")
    model.to(device=device).requires_grad_(False).eval()
    if compile_motion:
        enable_teacher_motion_compile(model, mode="reduce-overhead")
    return Track4WorldDistillationAdapter(model, iters=4, use_fast_path=True, motion_chunk_size=32)


def decode_canvas(wrapper, source, episode, start):
    from wam.datasets.pretrain_lerobot_loader import _decode_mp4_lerobot

    length = int(source.episodes_dict[episode]["length"])
    if not 0 <= start < length:
        raise ValueError(f"Window start {start} is outside episode {episode} (length {length}).")
    indices = torch.arange(start, start + 33).clamp(max=length - 1).tolist()
    source_indices = source.source_frame_indices(episode, indices)
    clips = []
    for key in source.video_keys:
        offset = source._video_timestamp_offset(episode, key)
        timestamps = [offset + index / float(source.fps) for index in source_indices]
        with source._open_episode_video(episode, key) as video:
            clips.append(_decode_mp4_lerobot(video, timestamps, fps=source.fps))
    canvas = wrapper._clips_to_training_tensor(source, clips)
    if canvas.ndim != 4 or int(canvas.shape[1]) != 33:
        raise ValueError("The teacher requires one RGB canvas with 33 frames.")
    frames = canvas.permute(1, 0, 2, 3).add(1.0).mul(127.5)
    frames = F.interpolate(frames, size=(256, 256), mode="bilinear", align_corners=False)
    return frames.clamp(0, 255).round().to(torch.uint8).contiguous()
