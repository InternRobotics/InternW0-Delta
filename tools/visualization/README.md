# Video and URDF visualization

[Home](../../README.md) · [Data](../../docs/data.md)

Render source video beside measured/commanded URDF motion and synchronized joint curves.

```bash
pip install -e '.[visualize]'
```

## Robot descriptions

Download the URDF and its meshes from the robot's official repository:

| Robot | Source | Joint map |
| --- | --- | --- |
| YAM | [I2RT](https://github.com/i2rt-robotics/i2rt), `i2rt/robot_models/arm/yam/v1` | `joints/yam.yaml` |
| Kuavo | [Leju Robotics](https://github.com/LejuRobotics/kuavo-ros-opensource), `biped_s42` | `joints/leju.yaml` |
| Unitree G1 | [Unitree ROS](https://github.com/unitreerobotics/unitree_ros), G1 description | `joints/g1.yaml` |
| Piper | [AgileX Piper ROS](https://github.com/agilexrobotics/piper_ros) | `joints/piper.yaml` |

Keep the mesh directory structure. Convert Xacro using the manufacturer's instructions when needed. `--package-root PACKAGE=../robot-assets/PACKAGE` resolves `package://` mesh paths. Robot assets are downloaded separately.

## Render

```bash
python tools/visualize.py --source ABC --dataset put_the_trash_bags_into_the_trash_bin \
  --episode 179 --urdf ../robot-assets/yam/yam.urdf \
  --joint-map tools/visualization/joints/yam.yaml \
  --max-frames 480 --output ../visualizations/yam.mp4
```

Camera, frame selection, and replay options are listed in `--help`.

Match the joint map and robot model to the dataset. The Kuavo S42 example provides approximate geometry; the visualization shows joint-space replay.

CPU rendering uses PyBullet TinyRenderer. Install [nvdiffrast](https://github.com/NVlabs/nvdiffrast) and add `--backend cuda` for CUDA rendering.
