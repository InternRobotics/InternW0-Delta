# RoboDojo evaluation

[Home](../../README.md) · [Models](../../docs/models.md) · [中文](README_zh.md)

Download the [RoboDojo weights](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo) to `checkpoints/robodojo.pt`, or set `ROBODOJO_CHECKPOINT`.

## Installation

Use an [Isaac Sim compatible GPU](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html), CUDA 12.8 toolkit, C++ compiler, and Conda/Miniforge. In the main model environment:

```bash
python -m pip install -e '.[modelscope]'
python -m pip install packaging ninja
python -m pip install --no-build-isolation -r eval/robodojo/requirements-policy.txt
export ROBODOJO_POLICY_ENV="$VIRTUAL_ENV"
python -m eval.robodojo.setup --dependencies --assets
bash eval/robodojo/install_simulator.sh
python -m eval.robodojo.download_models
```

For a Conda model environment, set `ROBODOJO_POLICY_ENV="$CONDA_PREFIX"`. The simulator is installed separately in `.venv-robodojo`. 

`ROBODOJO_ROOT` selects the benchmark directory (default `third_party/RoboDojo`); `ROBODOJO_SIM_ENV` selects the simulator environment. `WAM_WAN_PATH` and `WAM_VLM_PATH` override encoder locations.

## Run

```bash
bash eval/robodojo/evaluate.sh
```

To evaluate a single task:

```bash
bash eval/robodojo/run.sh --tasks stack_bowls
```

Policy settings are in [sim_robodojo.yaml](../../configs/sim_robodojo.yaml). See [RoboDojo](https://robodojo-benchmark.com/doc/) for benchmark usage and [third-party licenses](THIRD_PARTY.md).
