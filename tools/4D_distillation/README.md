# 4D distillation

[Home](../../README.md) · [Data](../../docs/data.md) · [Models](../../docs/models.md) · [中文](README_zh.md)

Cache Track4World teacher features for selected data, then continue pretraining with the [4D configuration](../../configs/task/4D_distillation.yaml).

## Prepare the teacher

```bash
python -m pip install -e '.[train,distillation]'
python tools/4D_distillation/prepare_teacher.py --root ../dependencies/Track4World
hf download TencentARC/Track4World track4world_da3.pth \
  --local-dir ../checkpoints/Track4World
```

Teacher code and weights use the upstream academic-use license; see [NOTICE](NOTICE).

## Select data and extract features

Prepare the data, statistics, and text cache using the main [README](../../README.md). Select sources in `configs/4D_distillation/teacher_data.yaml`. The training mixture in `configs/4D_distillation/dataset.yaml` must include these sources.

```bash
export WAM_4D_CACHE=../cache/4D_distillation
python tools/4D_distillation/cache.py select \
  --dataset-config configs/4D_distillation/teacher_data.yaml \
  --size 1000 --output "$WAM_4D_CACHE"
python tools/4D_distillation/cache.py extract \
  --cache "$WAM_4D_CACHE" \
  --dataset-config configs/4D_distillation/teacher_data.yaml \
  --teacher-root ../dependencies/Track4World \
  --checkpoint ../checkpoints/Track4World/track4world_da3.pth \
  --shard-size 1000 --shard-index 0
python tools/4D_distillation/cache.py seal --cache "$WAM_4D_CACHE"
```

Set `--size` to the desired number of windows. For multiple shards, run `extract` for each `--shard-index` before `seal`.

## Train

```bash
export WAM_DISTILL_INIT_CHECKPOINT=checkpoints/pretrain.pt
NPROC_PER_NODE=8 bash run.sh 4D_distillation gradient_accumulation_steps=8
```
