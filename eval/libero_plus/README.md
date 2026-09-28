# LIBERO Plus evaluation

[Home](../../README.md) · [Models](../../docs/models.md) · [Training](../../README.md#training)

Download the [LIBERO weights](https://huggingface.co/InternRobotics/InternW0-Delta-Libero) to `checkpoints/libero.pt`, or set `LIBERO_CHECKPOINT`. Prepare the backbones in the [model guide](../../docs/models.md#backbones).

## Installation

In the main Python 3.11 environment:

```bash
pip install -e '.[eval]'
bash eval/libero_plus/setup.sh
```

Download `assets.zip` from [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and extract it under `third_party/LIBERO-plus/libero/libero/` to create `assets/`.

The simulator requires EGL, NVIDIA graphics libraries, and ImageMagick/MagickWand. On Debian/Ubuntu:

```bash
sudo apt-get install libegl1 libgl1 libglib2.0-0 libexpat1 libfontconfig1 libmagickwand-dev
```

For NVIDIA containers, enable `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.

## Run

```bash
python -m eval.libero_plus.run_libero_plus MULTIRUN.num_gpus=1
```

Set `MULTIRUN.num_gpus` to the number of GPUs to use. Evaluation settings are in [sim_libero_plus.yaml](../../configs/sim_libero_plus.yaml); normalization is in [libero.json](../../assets/stats/libero.json).
