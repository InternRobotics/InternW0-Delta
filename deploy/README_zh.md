# 真机部署与 RTC 训练

[首页](../README_zh.md) · [模型](../docs/models.md) · [English](README.md)

## RTC 训练

准备 ACONe LeRobot 数据和[初始化权重](https://huggingface.co/InternRobotics/InternW0-Delta-Base)：

```bash
export ACONE_DATA_ROOT=data/acone
python tools/posttrain_stats.py task=rtc
python tools/text_cache.py task=rtc
NPROC_PER_NODE=8 bash deploy/train.sh
```

训练参数见 [rtc.yaml](../configs/task/rtc.yaml)，数据配置见 [acone.yaml](../configs/data/acone.yaml)。

## 部署

将下方的 `runs/rtc/my-run` 替换为自己的 RTC 训练输出目录，其中包含 `config.yaml`、`dataset_stats.json` 和保存的权重。基础模型在其他位置时设置 `WAM_VLM_PATH` 和 `DIFFSYNTH_MODEL_BASE_PATH`。

按机器人修改 [robot.yaml](robot.yaml)中的话题、关节名称和相机配置。在主模型环境中准备 ROS2 的 `rclpy`、`sensor_msgs`，需要重置服务时安装 `std_srvs`。

```bash
bash deploy/run.sh runs/rtc/my-run --mode rtc --check-config
bash deploy/run.sh runs/rtc/my-run --mode rtc \
  --robot-config deploy/robot.yaml --instruction 'Place the cup on the tray.'
```

第二条命令接收观测并推理。添加 `--execute` 后才发布动作，按 Enter 开始；`e` + Enter 结束当前回合，`r` + Enter 重置，Ctrl+C 退出。机器人控制器需实现关节限位和指令看门狗。

| 模式 | 行为 |
| --- | --- |
| `--mode sync` | 同步执行预测动作块 |
| `--mode naive` | 无前缀引导的异步重规划 |
| `--mode rtc --method condition` | 以已提交动作为条件，需要 RTC 训练权重 |
| `--mode rtc --method hard-prefix` | 在去噪中投影动作前缀 |
| `--mode rtc --method vjp` | 对上一动作计划施加软引导 |

更多部署选项见 `bash deploy/run.sh --help`。

## 机器人接口

[ROS2 示例](ros2.py)使用带关节名称的 `JointState` 及 RGB/BGR 图像。其他控制器可通过 `--robot your_package.adapter:Robot` 接入，接口约定见 [Robot adapters](README.md#robot-adapters)。启用动作发布前需确认数据单位、关节顺序与训练一致。
