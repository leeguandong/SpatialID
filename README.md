<div align="center">

# Inject Where It Matters

**Training-Free Spatially-Adaptive Identity Preservation for Text-to-Image Personalization**

[Guandong Li](https://github.com/leeguandong)<sup>1</sup> · Mengxia Ye<sup>2</sup>

<sup>1</sup>iFLYTEK &nbsp;&nbsp; <sup>2</sup>Aegon THTF

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)]()
[![GitHub](https://img.shields.io/github/stars/leeguandong/SpatialID?style=social)](https://github.com/leeguandong/SpatialID)

</div>

<p align="center">
  <img src="paper/figure1.png" width="95%">
</p>

## Overview

Existing tuning-free ID injection methods (PuLID, InstantID, etc.) broadcast identity features **uniformly** across all image patches — including backgrounds, clothing, and scenes. This causes background contamination, style disconnection, and unnatural lighting.

**SpatialID** upgrades the scalar injection weight to a **spatially-adaptive mask**, restricting ID injection to face-relevant regions only:

```
PuLID (uniform):  h ← h + α · CA(Z_id, h)          # same weight everywhere
SpatialID:        h ← h + α · M_t ⊙ CA(Z_id, h)    # face-focused, scene-free
```

Zero training. Zero extra parameters. ~2-3% overhead.

## Qualitative Results

<p align="center">
  <img src="paper/figure2.png" width="95%">
</p>

## Method

<table>
<tr>
<td width="50%">

### Spatial Mask Extractor

Extracts a spatial relevance mask from cross-attention output via L2 norm, then refines with:
- Gaussian smoothing (σ=1.5)
- Soft-hard combination (β=0.7, τ=0.3)
- 3×3 morphological dilation

No external detection model needed — the model "self-perceives" where identity belongs.

</td>
<td width="50%">

### Temporal-Spatial Scheduling

Three-phase strategy adapted to diffusion dynamics:

| Phase | Condition | Strategy |
|-------|-----------|----------|
| Early | `t > 0.7` | Center Gaussian prior |
| Mid | `0.3 < t ≤ 0.7` | Attention-derived mask |
| Late | `t ≤ 0.3` | Mask relaxation |

</td>
</tr>
</table>

## Getting Started

### Installation

```bash
pip install torch torchvision einops safetensors transformers
pip install insightface facexlib onnxruntime-gpu
pip install gradio  # for demo
```

### Usage

```python
from spatialid import (
    TemporalSpatialScheduler,
    replace_pulid_ca_modules,
    spatialid_denoise,
)

# 1. Replace PuLID CA modules with spatial-aware versions
replace_pulid_ca_modules(model)

# 2. Create temporal-spatial scheduler
scheduler = TemporalSpatialScheduler(
    early_threshold=0.7,
    late_threshold=0.3,
    late_floor=0.5,
    center_sigma=0.3,
)

# 3. Run spatially-adaptive denoising
x = spatialid_denoise(
    model, **inp, timesteps=timesteps,
    spatial_scheduler=scheduler,
    h_patches=h_patches, w_patches=w_patches,
)
```

### Gradio Demo

```bash
python app.py --device cuda:0 --port 7860
```

## Project Structure

```
SpatialID/
├── spatialid/                    # Core package
│   ├── core.py                   # Spatial mask, temporal scheduling, denoise loop
│   └── models/
│       ├── flux/                 # FLUX DiT backbone
│       ├── pulid/                # PuLID ID injection pipeline
│       └── eva_clip/             # EVA-CLIP vision encoder
├── app.py                        # Gradio demo
├── test_spatialid_quick.py       # Quick validation
└── ablation_study.py             # Ablation experiments
```

## Model Weights

| Model | Source |
|-------|--------|
| FLUX.1-dev | Auto-download from [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| PuLID | Download from [PuLID repo](https://github.com/ToTheBeginning/PuLID) |

## Citation

```bibtex
@article{li2025spatialid,
  title={Inject Where It Matters: Training-Free Spatially-Adaptive Identity Preservation
         for Text-to-Image Personalization},
  author={Li, Guandong and Ye, Mengxia},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgements

This project builds upon [PuLID](https://github.com/ToTheBeginning/PuLID), [FLUX](https://github.com/black-forest-labs/flux), and [EVA-CLIP](https://github.com/baaivision/EVA).
