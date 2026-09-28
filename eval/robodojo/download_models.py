"""Download the public encoder files used by RoboDojo evaluation."""
from __future__ import annotations

import os
from pathlib import Path


def main():
    from modelscope import snapshot_download

    root = Path(__file__).resolve().parents[2]
    checkpoint_root = os.environ.get("WAM_CHECKPOINT_ROOT", "checkpoints")
    wan = Path(os.environ.get("WAM_WAN_PATH", f"{checkpoint_root}/Wan2.2-TI2V-5B")).expanduser()
    vlm = Path(os.environ.get("WAM_VLM_PATH", f"{checkpoint_root}/RynnBrain1.1-2B")).expanduser()
    wan = wan if wan.is_absolute() else root / wan
    vlm = vlm if vlm.is_absolute() else root / vlm
    snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir=str(wan), allow_file_pattern=[
        "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*",
    ])
    snapshot_download("Alibaba-DAMO-Academy/RynnBrain1.1-2B", local_dir=str(vlm))
    print("Encoder files downloaded.")


if __name__ == "__main__":
    main()
