# EPAR — Embedding Predictive Architecture and Refinement for Multi-Modal Industrial Anomaly Detection

EPAR is a two-stage multimodal anomaly detection framework that seperately adapts pretrained RGB (DINO ViT) and 3D point cloud (Point-MAE) encoders via a **Joint-Embedding Predictive Architecture (JEPA)**, followed by a **Noise-Suppression Network (NSN)**.
It supports both **MVTec-3D AD** and **Eyecandies** benchmarks from a single codebase.

---

## Pipeline Overview

```
Stage 1 — train_adapt.py
  Pretrained DINO ViT  ──► RGB Context / Target Encoder  ──► RGB JEPA Loss + VICReg
  Pretrained Point-MAE ──► XYZ Context / Target Encoder  ──► XYZ JEPA Loss + VICReg

Stage 2 — train_nsn.py
  Stage-1 Encoders  ──► Feature Extraction (+ PFA for 3D)  ──► Coreset Sampling
                    ──► Pseudo-anomaly Generation  ──► NSN Training
                    ──► Evaluation (I-AUROC / P-AUROC / AUPRO@30%, AUPRO@1%)
```

---

## Environment

| | |
|---|---|
| Python | 3.10.4 |
| PyTorch | 1.13.1 |
| CUDA | 11.6 (recommended) |
| OS | Ubuntu 18.04 / 20.04 |

---

## Installation

### 1. Create conda environment

```bash
conda create -n epar python=3.10.4
conda activate epar
```

### 2. Install PyTorch 1.13.1

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

> For CUDA 11.7: replace `cu116` with `cu117`

### 3. Install base dependencies

```bash
pip install -r requirements.txt
```

### 4. Install KNN-CUDA

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

If the pre-built wheel does not match your GPU architecture, build from source:

```bash
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
export TORCH_CUDA_ARCH_LIST="8.6"   # adjust to your GPU (e.g. 7.5 for RTX 2080)
pip install -e .
cd ..
```

### 5. Install Pointnet2 Ops

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

If the git install fails, clone and install manually:

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd ../..
```

### 6. Install PyTorch3D

PyTorch3D requires a matching PyTorch + CUDA build. For PyTorch 1.13.1 + CUDA 11.6:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If the above fails, use the pre-built wheel directly:

```bash
# Python 3.10 + CUDA 11.6 + PyTorch 1.13.1
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu116_pyt1131/download.html
```

For other combinations, refer to the official install guide:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

---

## Checkpoints

Download the following pretrained weights and place them in `checkpoints/`:

| Model | Source |
|-------|--------|
| DINO ViT-B/8 | [M3DM](https://github.com/nomewang/M3DM) |
| Point-MAE | [M3DM](https://github.com/nomewang/M3DM) |

> Both weights are originally from [DINO](https://github.com/facebookresearch/dino) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), respectively.
> We refer to the [M3DM repository](https://github.com/nomewang/M3DM) for convenience.
> Please change the filename into `vit_base_patch8_224_dino.pth` and `pointmae_pretrain.pth`, respectively.

```
checkpoints/
├── vit_base_patch8_224_dino.pth
└── pointmae_pretrain.pth
```

---

## Dataset

### MVTec-3D AD

Download from the [official website](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and run preprocessing:

```bash
python utils/preprocess_mvtec3d.py datasets/mvtec/
```

### Eyecandies

Download from the [official website](https://eyecan-ai.github.io/eyecandies/) and run preprocessing:

```bash
python utils/preprocess_eyecandies.py datasets/eyecandies/
```

Expected directory structure after preprocessing:

```
datasets/
├── mvtec/
│   ├── bagel/
│   ├── cable_gland/
│   ├── carrot/
│   └── ...
└── eyecandies/
    ├── CandyCane/
    ├── ChocolateCookie/
    └── ...
```

---

## Usage

### Quick start — single class

```bash
# Stage 1: JEPA Adaptation
python train_adapt.py --config configs/train_adapt_mvtec.yaml --class_name cookie

# Stage 2: NSN Memory Bank
python train_nsn.py --config configs/train_nsn_mvtec.yaml --class_name cookie
```

### Run all classes via shell scripts

```bash
# MVTec-3D AD
./run_adapt.sh mvtec all
./run_nsn.sh   mvtec all

# Eyecandies
./run_adapt.sh eyecandies all
./run_nsn.sh   eyecandies all
```

### Run a specific class

```bash
./run_adapt.sh mvtec cookie
./run_nsn.sh   mvtec cookie combined   # mode: combined | only_jepa | only_memory

./run_adapt.sh eyecandies CandyCane
./run_nsn.sh   eyecandies CandyCane
```

### Override config values from CLI

Any key defined in the YAML config can be overridden at runtime:

```bash
python train_adapt.py \
    --config configs/train_adapt_mvtec.yaml \
    --class_name cookie \
    --num_epochs 3 \
    --lr_xyz 1e-5 \
    --bg_threshold 50
```

---

## Configuration

Per-class hyperparameters (epochs, learning rates, `bg_threshold`, `bank_lr`, etc.) are managed in YAML configs under `configs/`.

| File | Stage | Dataset |
|---|---|---|
| `configs/train_adapt_mvtec.yaml` | 1 | MVTec-3D AD |
| `configs/train_adapt_eyecandies.yaml` | 1 | Eyecandies |
| `configs/train_nsn_mvtec.yaml` | 2 | MVTec-3D AD |
| `configs/train_nsn_eyecandies.yaml` | 2 | Eyecandies |

Each config has a top-level `defaults` section and a `classes` section for per-class overrides:

```yaml
# default (applies to all classes)
bank_lr: 1.0e-5

classes:
  CandyCane:
    bank_lr: 1.0e-4    # override for this class only
  HazelnutTruffle:
    bank_lr: 1.0e-4
```

---

## Output Structure

Checkpoints are saved under `outputs/` after each training run:

```
outputs/
├── 1st_stage_mvtec/
│   ├── cookie_best_jepa_rgb_dino_main_4blocks_full_0.01.pth
│   └── cookie_best_jepa_xyz_pmae_main_4blocks_full_0.01.pth
├── 1st_stage_eyecandies/
│   ├── CandyCane_best_jepa_rgb_dino_new.pth
│   └── CandyCane_best_jepa_xyz_pmae_new.pth
├── 2nd_stage_mvtec/
│   └── ...
└── 2nd_stage_eyecandies/
    └── ...
```

---

## Acknowledgements

This codebase builds upon the following works:

- [M3DM (CVPR 2023)](https://github.com/nomewang/M3DM) 
- [I-JEPA (CVPR 2023)](https://github.com/facebookresearch/ijepa) 
- [Point-JEPA (WACV 2025)](https://github.com/Ayumu-J-S/Point-JEPA) 
- [VIC-Regularization (ICLR 2022)](https://github.com/facebookresearch/vicreg) 
- [PatchCore (CVPR 2022)](https://github.com/amazon-science/patchcore-inspection) 
