# 4D 蒸馏

[首页](../../README_zh.md) · [数据](../../docs/data.md) · [模型](../../docs/models.md) · [English](README.md)

为所选数据计算 Track4World 教师特征，再使用 [4D 配置](../../configs/task/4D_distillation.yaml)继续预训练。

## 准备教师模型

```bash
python -m pip install -e '.[train,distillation]'
python tools/4D_distillation/prepare_teacher.py --root ../dependencies/Track4World
hf download TencentARC/Track4World track4world_da3.pth \
  --local-dir ../checkpoints/Track4World
```

教师代码与权重遵循上游学术用途许可证，见 [NOTICE](NOTICE)。

## 选择数据并提取特征

按主 [README](../../README_zh.md)准备数据、统计和文本缓存。在 `configs/4D_distillation/teacher_data.yaml` 中选择需要缓存的源；训练使用的 `configs/4D_distillation/dataset.yaml` 应包含这些源。

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

通过 `--size` 设置窗口数量。使用多个分片时，为每个 `--shard-index` 执行 `extract`，全部完成后运行 `seal`。

## 训练

```bash
export WAM_DISTILL_INIT_CHECKPOINT=checkpoints/pretrain.pt
NPROC_PER_NODE=8 bash run.sh 4D_distillation gradient_accumulation_steps=8
```
