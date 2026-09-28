# RoboTwin

[Home](../../README.md) · [Models](../../docs/models.md) · [Data](../../docs/data.md#post-training)

## Post-training

Download the [initialization weights](https://huggingface.co/InternRobotics/InternW0-Delta-Base) to `checkpoints/pretrain.pt` and prepare the [RoboTwin data](../../docs/data.md#post-training).

```bash
export ROBOTWIN_DATA_ROOT=data/robotwin
python tools/text_cache.py task=robotwin
NPROC_PER_NODE=8 bash run.sh robotwin
```

The training configuration is [robotwin.yaml](../../configs/task/robotwin.yaml), with augmentation and clean-episode selection in [the data configuration](../../configs/data/robotwin.yaml).

## Simulator installation

The model uses the main Python 3.11 environment. Install the simulator separately:

```bash
python3.10 -m venv .venv-robotwin
.venv-robotwin/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv-robotwin/bin/python -m pip install -r eval/robotwin/requirements-sim.txt
.venv-robotwin/bin/python -m pip install --no-build-isolation -r eval/robotwin/requirements-cuda.txt
.venv-robotwin/bin/python eval/robotwin/configure_simulator.py
```

CUDA extensions need a GPU, C++ compiler, and CUDA 12.1 toolkit. The simulator also requires FFmpeg, Vulkan/EGL, and NVIDIA graphics libraries. For NVIDIA containers, enable `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.

Download `background_texture.zip`, `embodiments.zip`, and `objects.zip` from [RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/bf44be5). Extract them into `data/robotwin-assets/`, then run from the main environment:

```bash
python -m eval.robotwin.setup --assets data/robotwin-assets
```

Use `ROBOTWIN_ROOT` and `ROBOTWIN_PYTHON` to change the defaults `third_party/RoboTwin` and `.venv-robotwin/bin/python`.

## Evaluation

Download the [RoboTwin weights](https://huggingface.co/InternRobotics/InternW0-Delta-RoboTwin) to `checkpoints/robotwin.pt`, or set `ROBOTWIN_CHECKPOINT`. Prepare the backbones in the [model guide](../../docs/models.md#backbones).

```bash
python -m eval.robotwin.run_robotwin MULTIRUN.num_gpus=1
```

The default evaluates clean → random on all tasks in [tasks.txt](tasks.txt). Settings are in [sim_robotwin.yaml](../../configs/sim_robotwin.yaml). To select one task:

```bash
python -m eval.robotwin.run_robotwin \
  EVALUATION.task_name=click_bell MULTIRUN.num_gpus=1
```
