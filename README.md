<p align="center">
  <img src="assets/readme/logo.png" alt="InternW0-Δ" width="760">
</p>

<p align="center"><strong>A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data</strong></p>

<p align="center">
  <b>English</b> · <a href="README_zh.md">中文</a><br>
  <a href="#quick-start">🚀 Quick start</a> ·
  <a href="#training">🔥 Training</a> ·
  <a href="#evaluation-and-deployment">🦾 Evaluation & deployment</a> ·
  <a href="docs/models.md">📦 Models</a>
</p>

<p align="center">
  <a href="https://internrobotics.github.io/InternW0-Delta/"><b>🌐 Project page</b></a> ·
  <a href="https://arxiv.org/abs/2609.31394"><b>📄 Paper / arXiv</b></a> ·
  <b>🤗 Hugging Face</b> (coming soon)
</p>

<p align="center">
  <img src="assets/readme/teaser.png" alt="InternW0-delta overview: open data, predictive dynamics, 4D distillation, and robot manipulation." width="100%">
</p>

<a name="overview"></a>

## ✨ Overview

**InternW0-delta (InternW0-Δ)** is a unified world–action model that learns from robot and human demonstrations. It combines pretrained video dynamics, a frozen vision-language model, and training-only 4D supervision to generate robot actions.

- 🧩 **World–Action MoT.** A pretrained video expert and an action expert exchange information through masked self-attention, with visual memory and task-conditioned scene semantics.
- 🔮 **Causal Imprint.** Compact queries learn action-relevant scene changes from future supervision, making predictive features available to the action expert.
- 🌌 **4D distillation.** A frozen Track4World teacher transfers geometry and motion knowledge during training. The teacher and distillation branch are absent from action inference.

<p align="center">
  <img src="assets/readme/architecture.png" alt="InternW0-delta architecture: sparse visual memory, a video expert and an action expert coupled by directed MoT, with Causal Imprint queries and VLM conditioning." width="100%">
</p>

The repository covers data preparation, pretraining, 4D distillation, post-training, evaluation, and real robot deployment.

<a name="quick-start"></a>

## 🚀 Quick start

### 1. Install

Use Linux, Python 3.11, a CUDA GPU, and FFmpeg. DeepSpeed extensions require a C++ compiler and a matching CUDA toolkit.

```bash
git clone https://github.com/InternRobotics/InternW0-Delta.git
cd InternW0-Delta
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train]'
```

### 2. Prepare models and data

Download the backbones and policy weights following the [model guide](docs/models.md). For post-training, place the initialization weights at `checkpoints/pretrain.pt`.

Prepare the selected dataset using the [data guide](docs/data.md). Run the commands below from the repository root.

### 3. Post-train a policy

For LIBERO:

```bash
export LIBERO_DATA_ROOT=data/libero
python tools/text_cache.py task=libero
NPROC_PER_NODE=8 bash run.sh libero
```

Continue to [LIBERO Plus evaluation](eval/libero_plus/README.md), or choose another task below.

<a name="training"></a>

## 🔥 Training

### Pretraining

Choose datasets through `source_files` in [dataset.yaml](configs/pretrain/dataset.yaml). Training loads each selected source's normalization statistics automatically.

Prepare the [action expert initialization](docs/models.md#pretraining-initialization), then run:

```bash
python tools/path_index.py
python tools/pretrain_text_cache.py
NPROC_PER_NODE=8 bash run.sh pretrain
```

To continue from pretrained weights, add `resume=checkpoints/pretrain.pt model.skip_dit_load_from_pretrain=true`; action expert initialization is then unnecessary. For custom data, see [normalization](docs/data.md#text-cache-and-normalization).

### 4D distillation

Select data, cache Track4World teacher features, and continue training with the [4D distillation guide](tools/4D_distillation/README.md).

### Task configurations

Use the task name with `python tools/text_cache.py task=<task>` and `bash run.sh <task>`. See [post-training data](docs/data.md#post-training) for the required exports.

| Task | Configuration | Evaluation / deployment |
| --- | --- | --- |
| LIBERO | [`libero`](configs/task/libero.yaml) | [LIBERO Plus](eval/libero_plus/README.md) |
| RoboTwin | [`robotwin`](configs/task/robotwin.yaml) | [RoboTwin](eval/robotwin/README.md) |
| RoboDojo | [`robodojo`](configs/task/robodojo.yaml) | [RoboDojo](eval/robodojo/README.md) |
| Ebench | [`ebench`](configs/task/ebench.yaml), [`ebench_joint_ee`](configs/task/ebench_joint_ee.yaml) | — |
| Real robots / RTC | [`rtc`](configs/task/rtc.yaml) | [Deployment](deploy/README.md) |

<a name="evaluation-and-deployment"></a>

## 🦾 Evaluation and deployment

Download the task [weights](docs/models.md), then follow the corresponding environment setup and run instructions:

| Guide | Use |
| --- | --- |
| [LIBERO Plus](eval/libero_plus/README.md) | Evaluate LIBERO policies under task perturbations |
| [RoboTwin](eval/robotwin/README.md) | Post-train on clean demonstrations and evaluate clean → random |
| [RoboDojo](eval/robodojo/README.md) | Run the Isaac Sim benchmark |
| [Real robot deployment](deploy/README.md) | RTC training, robot adapters, and synchronous / asynchronous execution |

<a name="infra"></a>

## ⚡ Infra

Building on training optimization work from the [LiteGen](https://github.com/DeepLink-org/LiteGen), we provide optional acceleration for InternW0-Δ, including sparse video decoding, VLM execution optimizations, VAE/VLM caching, and MoT compilation.

```bash
bash run.sh libero +infra=online
bash run.sh robotwin +infra=compile
```

See the [infra guide](docs/infra.md) for profiles, cache generation, and an A800 performance reference.

<a name="data-and-tools"></a>

## 🗂️ Data and tools

| Guide | Use |
| --- | --- |
| [Data preparation](docs/data.md) | Dataset selection, path indices, text caches, and normalization |
| [Visualization](tools/visualization/README.md) | Video, URDF state/action playback, and synchronized curves |

The default paths are configurable through environment variables or YAML:

| Variable | Default |
| --- | --- |
| `WAM_DATA_ROOT` | `data` |
| `WAM_CACHE_ROOT` | `.cache/internw0` |


<a name="80d-state-and-action-representation"></a>

## 📐 80D state and action representation

Robot data shares a masked 80D layout across embodiments.

<details>
<summary><b>Dimension map and conventions</b></summary>

| Dimensions (zero-based, end-exclusive) | State | Action |
| --- | --- | --- |
| `[0,7)` / `[40,47)` | Left / right arm joints | Joint targets |
| `[7,10)` / `[47,50)` | Left / right end-effector position | Translation delta |
| `[10,16)` / `[50,56)` | End-effector rotation, 6D | Rotation-vector delta in the first 3 dimensions; last 3 masked |
| `[16,17)` / `[56,57)` | Left / right gripper | Gripper command |
| `[17,29)` / `[57,69)` | Left / right hand, 12 control channels | Hand command |
| `[29,34)` | Torso joints, up to 5 | Absolute joint targets |
| `[34,39)` / `[69,74)` | Reserved, masked | Reserved, masked |
| `[39,40)` | Independent lift height | Absolute height target |
| `[74,77)` | Head joints, up to 3 | Absolute joint targets |
| `[77,80)` | Base `(x, y, yaw)` when available | Local `(forward, left, yaw)` displacement, m / m / rad per source row |

Missing channels are padded and masked. Hand channels represent independent finger controls; torso/head ordering and native joint, gripper, and lift units are specified by the source adapter. Base actions are finite displacements, not velocities; unavailable base states are masked. End-effector reference frames and delta conventions follow each adapter. The layout is defined in [robot.yaml](configs/pretrain/projections/robot.yaml).

</details>

<a name="citation"></a>

## 📚 Citation

```bibtex
@misc{miao2026internw0deltaworldactionmodel,
      title={InternW0-$\Delta$: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data},
      author={Xingyu Miao and Zizun Li and Baole Fang and Kaiwen Song and Tenghui Wang and Hanxue Zhang and Yating Wang and Xudong Li and Yuping He and Xueyuan Wei and Chao Gao and Xijie Yang and Yingxiang Xu and Kerui Ren and Wenqi Guo and Jianjun Zhou and Xinzhe Wang and Weiguang Zhao and Ni Yang and Zetao Cai and Yufei Xue and Hengjie Li and Zeyu He and Yuanzhen Zhou and Rong Fu and Jianyang Zhang and Siwei Cui and Fuxian Huang and Yunsong Zhou and Xing Gao and Yifei Yao and Qiaojun Yu and Kailin Li and Ming Zhou and Mu Huang and Xinyue Li and Wenze Cui and Bingqi Jiang and Xueyue Zhu and Junting Dong and Haoyu Guo and Tao Lu and Mulin Yu and Bowen Zhou and Bin Zhao and Tianfan Xue and Weinan Zhang and Chunhua Shen},
      year={2026},
      eprint={2609.31394},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.31394},
}
```

<a name="license"></a>

## 📜 License

The code is released under the [MIT License](LICENSE). Third-party components retain their respective licenses; see [NOTICE](NOTICE).
