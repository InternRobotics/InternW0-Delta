# Optional training acceleration

[Home](../README.md) · [Models](models.md) · [Training](../README.md#training)

| Profile | Features |
| --- | --- |
| `+infra=online` | Sparse video decoding, pinned input transfer, VLM metadata reuse |
| `+infra=compile` | Online optimizations plus MoT forward/backward compilation and activation checkpointing |
| `+infra=cache` | Compile profile plus VAE and frozen-VLM cache reads |

```bash
NPROC_PER_NODE=8 bash run.sh libero +infra=online
NPROC_PER_NODE=8 bash run.sh robotwin +infra=compile
```

Compilation uses PyTorch Inductor/Triton and needs a C++ compiler. Supported configurations are defined in [configs/infra](../configs/infra/).

## Compiler cache

```bash
export WAM_COMPILE_CACHE=.cache/internw0/torch_compile
WAM_TRAIN_ENTRY=tools/compile.py NPROC_PER_NODE=8 \
  bash run.sh robotwin +infra=compile
NPROC_PER_NODE=8 bash run.sh robotwin +infra=compile
```

Prepare compiler kernels using the same configuration and hardware as training.

## VAE and VLM caches

Static caches require deterministic preprocessing and a frozen VLM. Use `online` or `compile` with random augmentation. The following example uses a custom `configs/data/my_deterministic.yaml` with deterministic preprocessing.

```bash
export WAM_LATENT_CACHE=.cache/internw0/latents/my_dataset
python tools/video_cache.py task=robotwin data=my_deterministic +infra=cache \
  video_latent_cache=generate
bash run.sh robotwin data=my_deterministic +infra=cache
```

To cache all VAE inputs but only part of the VLM inputs, use a fresh cache directory:

```bash
python tools/video_cache.py task=robotwin data=my_deterministic +infra=cache \
  video_latent_cache=generate \
  'video_latent_cache_fields=[current,memory_anchor,memory_recent]'
python tools/video_cache.py task=robotwin data=my_deterministic +infra=cache \
  video_latent_cache=generate 'video_latent_cache_fields=[vlm]' \
  precompute.max_samples=1000
bash run.sh robotwin data=my_deterministic +infra=cache
```

Missing VLM features are computed online. Selected VAE fields require complete coverage of the training dataset; `precompute.max_samples` limits VLM cache generation.

Use matching data and encoder settings for generation and training. Encoder caching supports post-training datasets; teacher features use the [4D workflow](../tools/4D_distillation/README.md).

## A800 performance reference

With deterministic preprocessing and encoder caching, A800 training throughput reached 127.96 samples/s on LIBERO (3.80x) and 30.42 samples/s on RoboTwin (1.56×). Performance depends on the selected configuration.
