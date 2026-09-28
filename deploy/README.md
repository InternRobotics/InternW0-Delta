# Real robot deployment and RTC training

[Home](../README.md) · [Models](../docs/models.md) · [中文](README_zh.md)

## RTC training

Prepare the ACONe LeRobot export and the [initialization weights](https://huggingface.co/InternRobotics/InternW0-Delta-Base):

```bash
export ACONE_DATA_ROOT=data/acone
python tools/posttrain_stats.py task=rtc
python tools/text_cache.py task=rtc
NPROC_PER_NODE=8 bash deploy/train.sh
```

See [rtc.yaml](../configs/task/rtc.yaml) for training settings and [acone.yaml](../configs/data/acone.yaml) for data configuration.

## Deployment

Use your RTC training output directory in place of `runs/rtc/my-run` below. It contains `config.yaml`, `dataset_stats.json`, and saved checkpoints. Set `WAM_VLM_PATH` and `DIFFSYNTH_MODEL_BASE_PATH` if the backbones are stored elsewhere.

Edit [robot.yaml](robot.yaml) with your robot's topics, joint names, and camera configuration. Use the main model environment with ROS2's `rclpy`, `sensor_msgs`, and optional `std_srvs` available.

```bash
bash deploy/run.sh runs/rtc/my-run --mode rtc --check-config
bash deploy/run.sh runs/rtc/my-run --mode rtc \
  --robot-config deploy/robot.yaml --instruction 'Place the cup on the tray.'
```

The second command receives observations and runs inference. Add `--execute` to publish robot commands; press Enter at the start gate. `e` + Enter ends the episode, `r` + Enter resets, and Ctrl+C exits. The robot controller must enforce joint limits and a command watchdog.

| Mode | Behavior |
| --- | --- |
| `--mode sync` | Execute each predicted chunk synchronously |
| `--mode naive` | Replan asynchronously without prefix guidance |
| `--mode rtc --method condition` | Condition on committed actions; requires RTC-trained weights |
| `--mode rtc --method hard-prefix` | Project the committed prefix during denoising |
| `--mode rtc --method vjp` | Apply soft previous-plan guidance |

Additional deployment options are listed in `bash deploy/run.sh --help`.

## Robot adapters

The included [ROS2 adapter](ros2.py) uses named `JointState` messages and RGB/BGR images. To use another controller, pass `--robot your_package.adapter:Robot`. Its factory receives `--robot-config` and implements:

- `observe()`: copied `state: float32[14]`, `images: {camera: RGB uint8[H,W,3]}`, camera `timestamps`, and `state_timestamp` in monotonic reception time.
- `publish(action)`: one absolute 14D joint/gripper command in the configured order.
- `reset()`: explicit episode reset, waiting until complete.
- `close()`: release resources.

Use sensor reception timestamps so stale observations can be detected. Match command units and joint ordering to the training data before enabling execution.
