# HSGM: Hierarchical Semantic-Geometric Map for Vision-Language Navigation

Official code for the CVPR 2026 paper:

> **Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation**
> Kailing Li, Tianwen Qian, Lijin Yang, Yuqian Fu, Jingyu Gong, Xiaoling Wang, Liang He
> *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 15243-15252.*
>
> [[Paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Bridging_the_2D-3D_Gap_A_Hierarchical_Semantic-Geometric_Map_for_Vision_CVPR_2026_paper.html) · [[Project]](https://github.com/Teacher-Tom/HSGM_public)

---

## Overview

VLN requires an embodied agent to reach a target in an unseen environment by following natural-language
instructions. VLMs excel at language and 2D visual understanding, but struggle with **3D spatial reasoning**
in zero-shot settings.

**HSGM** bridges this gap by transforming online 3D geometry into a structured, VLM-friendly
representation — a **multi-channel top-down map** with three levels:

1. **Geometric level** — navigable regions, obstacles.
2. **Semantic level** — objects and their spatial relations (via YOLOE instance segmentation).
3. **Decision level** — high-level waypoint selection by the VLM planner.

The VLM acts as a **high-level semantic planner**, while **low-level collision-free motion** between
waypoints is executed by a classical **A\*** path planner. Long instructions are further **decomposed
into subtasks** to mitigate forgetting and hallucination in long-horizon navigation.

Our zero-shot framework achieves state-of-the-art on **R2R-CE** and **RxR-CE**, surpassing several
supervised methods.

---

## Repository Structure

```
HSGM/
├── config/                     # YAML configuration files
│   ├── vlnce_test.yaml         # R2R / RxR (VLN-CE, MP3D)
│   └── objnav_test.yaml        # ObjectNav (HM3D)
├── scripts/                    # Test script
│   └── batch_test.sh           # multi-episode evaluation
├── src/
│   ├── run_experiments.py      # main entry point
│   ├── simWrapper.py           # Habitat-Sim wrapper + PolarAction
│   ├── mapper.py               # online HSGM mapping (Instruct_Mapper)
│   ├── agent/
│   │   └── agent.py            # GPTAgent (VLM planner) + PathPlannerAgent + Instruction
│   ├── segmentation/           # YOLOE instance segmentation
│   ├── mapping_utils/          # geometry / projection / transform / A* path planning
│   └── utils.py
├── .env.example                # environment variable template
├── requirements.txt
└── LICENSE                     # Apache-2.0
```

---

## Installation

Python 3.9+, CUDA GPU (`cuda:0` by default).

```bash
# 1. Create environment
conda create -n hsgm python=3.9 -y
conda activate hsgm

# 2. Install Habitat-Sim (must be via conda)
conda install -c conda-forge -c aihabitat habitat-sim withbullet -y

# 3. Install Habitat-Lab
pip install habitat-lab

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### YOLOE checkpoint

Place a YOLOE-seg checkpoint at `ckpt/yoloe-26l-seg.pt`:

```bash
mkdir -p ckpt
# Download YOLOE weights and save as:
#   ckpt/yoloe-26l-seg.pt
```

---

## Datasets

The code reads datasets from `../data/` (relative to `src/`).

```
data/
├── scene_datasets/
│   ├── hm3d_v0.2/val/...                  # HM3D scenes (ObjectNav)
│   └── mp3d/<scene>/<scene>.glb           # MP3D scenes (R2R / RxR)
└── datasets/
    ├── objectnav_hm3d_v2/val/content/*.json.gz  # ObjectNav episodes
    ├── r2r_vlnce/val_unseen.json                # R2R-CE episodes
    └── RxR_VLNCE_v0/val_unseen/                 # RxR-CE episodes (+ *_guide_gt.json)
```

Refer to [VLN-CE](https://github.com/jacobkrantz/VLN-CE) for R2R-CE / RxR-CE episode files, and
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab) for HM3D / MP3D scene data.
Update `scene_path` in the YAML configs accordingly.

---

## VLM Configuration

`GPTAgent` uses an **OpenAI / Azure OpenAI compatible** chat-completion endpoint, configured via
environment variables.

**Standard OpenAI (or any compatible server, such as a local vLLM):**

```bash
export OPENAI_API_KEY=sk-xxxxxxxx          # or "EMPTY" for local vLLM
export OPENAI_BASE_URL=https://api.openai.com/v1   # or http://localhost:8000/v1
export OPENAI_MODEL=gpt-5                  # optional, defaults to gpt-5
```

**Azure OpenAI (takes precedence when endpoint and key are both set):**

```bash
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
export AZURE_OPENAI_API_KEY=<your-azure-key>
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

The proxy at `127.0.0.1:7897` is auto-detected at import time. It is only enabled when you have
**not** already set `http_proxy`/`https_proxy` **and** the proxy is TCP-reachable; otherwise it is
skipped silently.

---

## Quick Start

All commands run from inside `src/`. Alternatively, use the convenience scripts in `scripts/`.

### Using the test script

```bash
# Batch evaluation (10 episodes)
bash scripts/batch_test.sh r2r 0 10 Qwen/Qwen3.6-27B
```

### Direct invocation

```bash
cd src

# R2R-CE
python run_experiments.py --task r2r --config ../config/vlnce_test.yaml \
    --model_name Qwen/Qwen3.6-27B --begin_idx 0 --end_idx -1 --max_steps 100

# RxR-CE
python run_experiments.py --task rxr --config ../config/vlnce_test.yaml

# ObjectNav (HM3D)
python run_experiments.py --task objectnav --config ../config/objnav_test.yaml
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `r2r` | `objectnav` / `r2r` / `rxr` |
| `--config` | `../config/vlnce_test.yaml` | YAML config path |
| `--comment` | `exp` | tag appended to run name |
| `--begin_idx` | `0` | start episode index (inclusive) |
| `--end_idx` | `-1` | end index (exclusive, `-1` for all) |
| `--max_steps` | `100` | max agent steps per episode |
| `--model_name` | `$OPENAI_MODEL` or `gpt-5` | VLM model name |
| `--show` | off | enable live visualization (requires display) |
| `--episode_ids` | — | `"1,2,3"` or path to a file |

---

## Configuration

Example (`config/vlnce_test.yaml`):

```yaml
agent:
  radius: 0.1
  initial_position: [x, y, z]
  initial_rotation: [x, y, z, w]
camera:
  height: 1.2
  fov: 135.0
  pitch: -30.0
  min_depth: 0.5
  max_depth: 50.0
  res_factor: 2
env:
  max_steps: 200
  success_threshold: 3.0
  name: 'r2r'
```

---

## Outputs

- `results/results_<env.name>/episode_<id>_results.json` — per-episode metrics and VLM log
- `results/results_<env.name>/episode_<id>.mp4` — observation video (always generated)
- `results/results_<env.name>/aggregate_results.json` — aggregated SR / SPL / nDTW / sDTW
- VLN metrics computed via `fastdtw` against the reference path

---

## Citation

```bibtex
@InProceedings{Li_2026_CVPR,
    author    = {Li, Kailing and Qian, Tianwen and Yang, Lijin and Fu, Yuqian and Gong, Jingyu and Wang, Xiaoling and He, Liang},
    title     = {Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {15243-15252}
}
```

---

## License

[Apache License 2.0](LICENSE)
