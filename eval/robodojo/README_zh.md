# RoboDojo 评测

[首页](../../README_zh.md) · [模型](../../docs/models.md) · [English](README.md)

将 [RoboDojo 权重](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo)下载到 `checkpoints/robodojo.pt`，或使用 `ROBODOJO_CHECKPOINT` 指定路径。

## 安装

需要 [Isaac Sim 支持的 GPU](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)、CUDA 12.8 toolkit、C++ 编译器及 Conda/Miniforge。在主模型环境中运行：

```bash
python -m pip install -e '.[modelscope]'
python -m pip install packaging ninja
python -m pip install --no-build-isolation -r eval/robodojo/requirements-policy.txt
export ROBODOJO_POLICY_ENV="$VIRTUAL_ENV"
python -m eval.robodojo.setup --dependencies --assets
bash eval/robodojo/install_simulator.sh
python -m eval.robodojo.download_models
```

使用 Conda 模型环境时设为 `ROBODOJO_POLICY_ENV="$CONDA_PREFIX"`。仿真环境单独安装到 `.venv-robodojo`。

`ROBODOJO_ROOT` 指定 benchmark 目录，默认 `third_party/RoboDojo`；`ROBODOJO_SIM_ENV` 指定仿真环境。`WAM_WAN_PATH` 与 `WAM_VLM_PATH` 可修改编码器位置。

## 运行

```bash
bash eval/robodojo/evaluate.sh
```

评测单个任务：

```bash
bash eval/robodojo/run.sh --tasks stack_bowls
```

策略参数见 [sim_robodojo.yaml](../../configs/sim_robodojo.yaml)。Benchmark 使用方式见 [RoboDojo 官方文档](https://robodojo-benchmark.com/doc/)，许可证见[第三方说明](THIRD_PARTY.md)。
