# Models and checkpoints

[Home](../README.md) · [Data](data.md) · [Training](../README.md#training)

## Policy weights

Download the policy for your task and the required backbone files below.

| Model | Hugging Face weights | Destination |
| --- | --- | --- |
| Pretrained InternW0-delta | [InternW0-Delta-Base](https://huggingface.co/InternRobotics/InternW0-Delta-Base) | `checkpoints/pretrain.pt` |
| LIBERO | [InternW0-Delta-Libero](https://huggingface.co/InternRobotics/InternW0-Delta-Libero) | `checkpoints/libero.pt` |
| RoboTwin | [InternW0-Delta-RoboTwin](https://huggingface.co/InternRobotics/InternW0-Delta-RoboTwin) | `checkpoints/robotwin.pt` |
| RoboDojo | [InternW0-Delta-RoboDojo](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo) | `checkpoints/robodojo.pt` |

Use `WAM_PRETRAIN_CHECKPOINT` for post-training initialization and `LIBERO_CHECKPOINT`, `ROBOTWIN_CHECKPOINT`, or `ROBODOJO_CHECKPOINT` for evaluation.

## Backbones

Training, LIBERO Plus, RoboTwin, and deployment use these files:

```bash
export WAM_CHECKPOINT_ROOT=checkpoints
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
  --local-dir "$WAM_CHECKPOINT_ROOT/Wan-AI/Wan2.2-TI2V-5B"
hf download DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  --include 'models_t5_umt5-xxl-enc-bf16.safetensors' 'Wan2.2_VAE.safetensors' \
  --local-dir "$WAM_CHECKPOINT_ROOT/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' \
  --local-dir "$WAM_CHECKPOINT_ROOT/Wan-AI/Wan2.1-T2V-1.3B"
```

The VLM requires RynnBrain1.1-2B weights and processor/tokenizer files at `checkpoints/RynnBrain1.1-2B`, configurable with `WAM_VLM_PATH`. Its Hugging Face link is coming soon. It is also available through ModelScope:

```bash
python -m pip install -e '.[modelscope]'
python -c 'import os; from modelscope import snapshot_download; snapshot_download("Alibaba-DAMO-Academy/RynnBrain1.1-2B", local_dir=os.environ.get("WAM_VLM_PATH", "checkpoints/RynnBrain1.1-2B"))'
```

RoboDojo uses the original Wan text/VAE format; `python -m eval.robodojo.download_models` prepares its encoder files. Teacher weights are covered in [4D distillation](../tools/4D_distillation/README.md).

## Pretraining initialization

For pretraining from the Wan backbone, prepare the action expert once:

```bash
python tools/prepare_action_dit.py --model-config configs/model/wam.yaml \
  --output checkpoints/action_dit.pt
```

To continue from InternW0-delta weights, use `resume=checkpoints/pretrain.pt model.skip_dit_load_from_pretrain=true` instead.
