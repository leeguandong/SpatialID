# Inject Where It Matters

### Training-Free Spatially-Adaptive Identity Preservation for Text-to-Image Personalization

<p align="center">
  <img src="paper/figure1.png" width="100%">
</p>

> **Guandong Li** (iFLYTEK) · **Mengxia Ye** (Aegon THTF)

## TL;DR

Existing tuning-free ID injection methods (PuLID, InstantID, etc.) broadcast identity features **uniformly** across all image patches — including backgrounds, clothing, and scenes. This causes background contamination, style disconnection, and unnatural lighting.

**SpatialID** fixes this by upgrading the scalar injection weight to a **spatially-adaptive mask**:

```
# PuLID (uniform):  h ← h + α · CA(Z_id, h)
# SpatialID:        h ← h + α · M_t ⊙ CA(Z_id, h)
```

where `M_t ∈ [0,1]^{H×W}` is a time-varying spatial mask that restricts ID injection to face-relevant regions only. **Zero training, zero extra parameters, ~2-3% overhead.**

## Key Results (IBench: 100 IDs × 41 Prompts)

| Method | IQ↑ | CLIP-I↑ | CLIP-T↑ | FaceSim↑ |
|--------|-----|---------|---------|----------|
| PuLID (Krea) | 0.505 | 0.793 | 0.277 | 0.495 |
| Dreamo | 0.510 | 0.805 | 0.266 | 0.398 |
| DVI | 0.515 | 0.804 | 0.269 | 0.557 |
| **SpatialID** | **0.523** | **0.827** | **0.281** | 0.533 |

SpatialID achieves **SOTA** in Image Quality, CLIP-I, and CLIP-T simultaneously.

<p align="center">
  <img src="paper/figure2.png" width="100%">
</p>

## Method

SpatialID consists of two key components:

**1. Spatial Mask Extractor** — Extracts a spatial relevance mask from cross-attention output using L2 norm, followed by Gaussian smoothing + soft-hard combination + morphological dilation. No external detection model needed.

**2. Temporal-Spatial Scheduling** — Three-phase strategy adapted to the diffusion denoising dynamics:
- **Early** (`t > 0.7`): Center Gaussian prior for stable composition
- **Mid** (`0.3 < t ≤ 0.7`): Attention-derived mask for precise face anchoring
- **Late** (`t ≤ 0.3`): Mask relaxation for natural light-shadow fusion

## Quick Start

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

# 3. Use spatialid_denoise instead of the original denoise
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

### IBench Evaluation

```bash
python test_ibench_spatial.py
```

### Ablation Study

```bash
python ablation_study.py
```

## Project Structure

```
SpatialID/
├── spatialid/                    # Core package
│   ├── __init__.py               # Public API
│   ├── core.py                   # Spatial mask, temporal scheduling, denoise loop
│   └── models/
│       ├── flux/                 # FLUX DiT backbone
│       ├── pulid/                # PuLID ID injection pipeline
│       └── eva_clip/             # EVA-CLIP vision encoder
├── app.py                        # Gradio demo
├── test_spatialid_quick.py       # Quick validation (5 IDs × 5 prompts)
├── test_ibench_spatial.py        # Full IBench benchmark (100 IDs × 41 prompts)
├── ablation_study.py             # Ablation experiments
└── config_spatialid.py           # IBench evaluation config
```

## Model Weights

- **FLUX**: Auto-downloaded from HuggingFace (`black-forest-labs/FLUX.1-dev`)
- **PuLID**: Download from [PuLID repo](https://github.com/ToTheBeginning/PuLID) (`pulid_flux_v0.9.1.safetensors`)

## Citation

```bibtex
@article{li2025spatialid,
  title={Inject Where It Matters: Training-Free Spatially-Adaptive Identity Preservation for Text-to-Image Personalization},
  author={Li, Guandong and Ye, Mengxia},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgements

This project builds upon [PuLID](https://github.com/ToTheBeginning/PuLID), [FLUX](https://github.com/black-forest-labs/flux), and [EVA-CLIP](https://github.com/baaivision/EVA). We thank the authors for their excellent work.
