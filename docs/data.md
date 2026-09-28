# Data preparation

[Home](../README.md) · [Models](models.md) · [Training](../README.md#training)

## Paths and source selection

Place your LeRobot datasets under `data/`, or choose your own locations:

```bash
export WAM_DATA_ROOT=/path/to/datasets
export WAM_CACHE_ROOT=/path/to/cache
```

Select `source_files` in `configs/pretrain/dataset.yaml`. For example:

```yaml
source_files:
- sources/ABC.yaml
- sources/umi.yaml
```

Preparation commands accept `--source <file-stem>`; training follows the YAML selection.

```bash
python tools/check_data.py --list
python tools/check_data.py --source umi
python tools/pretrain_text_cache.py --source umi
```

Relative data paths resolve from the working directory; YAML includes resolve beside their containing file.

## Dataset layout

```text
data/<collection>/<dataset>/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<camera-key>/episode_000000.mp4
```

Pretraining supports LeRobot v2 and v3, including shared videos with per-episode offsets. Fields and camera names must match the selected source and embodiment YAMLs.

| Sources | Required export |
| --- | --- |
| Intern A1, AgiBotWorld, RoboMIND, RoboMIND2, RoboCOIN, RDT, RH20T, RoboSet, ABC, MolmoAct2 | LeRobot fields and cameras specified by their adapters |
| ActionNet, Dexora, HABIT, RealSource, mobile RoboMIND, RWRL, Galaxea with mobile-base or torso channels | LeRobot exports with the 80D fields in `embodiments/processed.yaml` |
| EgoDex and EgoVerse | Prepared wrist/hand exports matching `embodiments/human.yaml` |
| Human-to-robot sources | Retargeted exports matching `embodiments/human2robot.yaml`, including validity fields |
| UMI | Three-camera, independent-episode LeRobot export at `data/umi` |

Canonical 80D exports contain `derived.state80`, `derived.action80`, `derived.state_mask80`, `derived.action_mask80`, and `derived.contract = internw0.pose80.v1`. See the [80D definition](../README.md#80d-state-and-action-representation). Raw downloads with different fields require conversion to the configured schema.

## Path indices

Generate episode indices for the selected sources:

```bash
python tools/path_index.py
```

To index one source or dataset:

```bash
python tools/path_index.py --source umi
python tools/path_index.py --source umi --dataset hy_umi_episode_v21
```

Each `path_index.jsonl` records episode lengths, data/video paths, and shared-file offsets for LeRobot v3. Paths are relative to the dataset root. Indices are written under `WAM_CACHE_ROOT/path_index/`; training also builds missing indices from metadata automatically. Run the command again after adding or changing episodes.

Use `datasets_glob: "**"` in a source configuration to discover LeRobot datasets beneath its `root`. Explicit `datasets` entries can specify different camera layouts or adapters.

## Text cache and normalization

```bash
python tools/pretrain_text_cache.py
python tools/check_data.py --check-text-cache
```

Text embeddings are written beneath `WAM_CACHE_ROOT/text_embed/`. Training requires these caches when `model.load_text_encoder=false`.

Statistics are selected automatically from `assets/stats/pretrain/<source>.json`; only enabled datasets' groups are read. To recompute a source after changing its data:

```bash
python tools/pretrain_stats.py --source umi --active-only --num-workers 8 \
  --output .cache/internw0/stats/pretrain
```

Set that source's `normalization_stats` to the new JSON, or point `WAM_PRETRAIN_STATS` at a directory containing all selected sources' statistics.

## Post-training

| Task | Default layout |
| --- | --- |
| RoboTwin | `data/robotwin/{meta,data,videos}` |
| LIBERO | `data/libero/{libero_spatial_no_noops_lerobot,libero_object_no_noops_lerobot,libero_goal_no_noops_lerobot,libero_10_no_noops_lerobot}/` |
| RoboDojo | `data/robodojo/{meta,data,videos}` |
| Ebench | Task roots listed in `configs/data/ebench.yaml` |

Dataset paths and episode selection are defined in [configs/data](../configs/data/). If the RoboTwin export has different episode numbering, adjust `episode_selection` to its clean split.

```bash
python tools/text_cache.py task=libero
```

LIBERO, RoboTwin, and RoboDojo statistics are included. For Ebench, first run `python tools/posttrain_stats.py task=ebench` (or `task=ebench_joint_ee`).

To compute statistics for a different export:

```bash
python tools/posttrain_stats.py task=libero \
  +precompute.stats_path=.cache/internw0/stats/libero.json
LIBERO_NORM_STATS=.cache/internw0/stats/libero.json bash run.sh libero
```
