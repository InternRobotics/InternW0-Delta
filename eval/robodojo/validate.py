"""Check required evaluation inputs and prevent incompatible resumes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from eval.robodojo.server import load_config

ROOT = Path(__file__).resolve().parents[2]


def local_path(value):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate(cfg, run_dir=None):
    checkpoint = local_path(cfg.ckpt)
    stats_path = local_path(cfg.EVALUATION.dataset_stats_path)
    wan = local_path(cfg.model.model_id)
    vlm = local_path(cfg.model.understanding.vlm_model_path)
    files = [checkpoint, stats_path, wan / "Wan2.2_VAE.pth",
             wan / "models_t5_umt5-xxl-enc-bf16.pth",
             vlm / "config.json", vlm / "tokenizer_config.json"]
    vlm_weights = sorted(vlm.glob("*.safetensors")) + sorted(vlm.glob("pytorch_model*.bin"))
    if not vlm_weights:
        raise FileNotFoundError(f"No VLM weights in {vlm}")
    files.extend(vlm_weights)
    for name in ("tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        files.append(wan / "google/umt5-xxl" / name)
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    stats = json.loads(stats_path.read_text())
    for kind in ("action", "state"):
        values = stats[kind]["default"]
        if len(values["global_mean"]) != 14 or len(values["global_std"]) != 14:
            raise ValueError(f"Expected 14D {kind} normalization")
    benchmark = local_path(os.environ.get("ROBODOJO_ROOT", "third_party/RoboDojo"))
    for name in ("Robots", "Object", "Material", "Eval_Layout"):
        if not (benchmark / "Assets" / name).is_dir():
            raise FileNotFoundError(benchmark / "Assets" / name)
    if not (benchmark / "Assets/Robots/x5/curobo.yml").is_file():
        raise FileNotFoundError("Run python -m eval.robodojo.setup --assets first")
    if run_dir is not None:
        settings = {key: OmegaConf.to_container(cfg[key], resolve=True)
                    for key in ("model", "EVALUATION")}
        settings["processor"] = OmegaConf.to_container(cfg.data.train.processor, resolve=True)
        settings["weights"] = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                               for p in files}
        settings["statistics"] = stats
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "inputs.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError("Evaluation inputs changed; use a new output directory")
        path.write_text(json.dumps(settings, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    validate(load_config(args.seed), args.run_dir)


if __name__ == "__main__":
    main()
