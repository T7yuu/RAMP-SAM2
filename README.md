# RAMP-SAM2

Research code for retrieval-augmented, multi-prompt medical image segmentation.
Candidate boxes are generated per image, a prior is retrieved from a medical
knowledge base, and an Accept/Defer routing rule decides whether that prior is
fused into the mask decoder.

This repository contains the main model only. Comparison baselines and ablation
experiments are kept separately and are not part of this tree.

## Repository layout

```
scripts/V9/
  modules/            core components (retriever, engine, utils)
  configs/            dataset and model configs (ISIC / MSD)
  3wd2/               three-way decoupled feature-space pipeline (ISIC / PH2)
  scripts/            knowledge-base and FAISS index builders, threshold table
  msd/                the same pipeline adapted to MSD
  CVC-ClinicDB/       the same pipeline adapted to CVC-ClinicDB
  mysam_v1.py         first-generation single-box implementation
  v5_多框版.py         multi-box implementation
  medsam.py           MedSAM-only inference
  train_3wd_feature_space.py   adapter training (top-level copy)
utils/                prompt-label generation and metric utilities
sam2/                 SAM2 smoke test and config
```

## Requirements

Python 3.10+, PyTorch with CUDA. The reported numbers were produced on an AutoDL
cloud instance with an NVIDIA GeForce RTX 4090 (24 GB), PyTorch 2.5.1+cu124
(CUDA 12.4), Python 3.12, plus:

```
torch transformers open_clip_torch sam2 opencv-python numpy scipy
pillow matplotlib tqdm pymupdf pyyaml
```

## Data

The experiments use ISIC2018, PH2, MSD and CVC-ClinicDB. Datasets, model
weights and retrieved feature caches are **not** included in this repository.
Each script expects a local root directory; most entry points currently read a
hard-coded `ROOT` constant near the top of the file — adjust it to your own
checkout before running.

## Typical workflow

1. Build the knowledge base and the FAISS index
   (`3wd2/build_knowledge_base_v3.py`, `3wd2/build_faiss_index.py`).
2. Train the retrieval adapter / feature space
   (`3wd2/train_3wd_feature_space.py`).
3. Tune thresholds on the validation split (`3wd2/grid_search_val.py`).
4. Run inference (`3wd2/v5_multi_box.py` for ISIC, `3wd2/v5_multi_box_PH2.py`
   for PH2).

## Not tracked

Model weights (`*.pth`, `*.safetensors`, `*.npz`), experiment output
directories, comparison baselines, ablation scripts, LaTeX build artifacts and
local editor state are excluded via `.gitignore`. See that file for the exact
rules.

## MedSAM

The MedSAM comparison is an external dependency and is not vendored here:

```bash
git clone https://github.com/bowang-lab/MedSAM.git
```

Point `scripts/V9/medsam.py` at your local MedSAM checkout before running.
