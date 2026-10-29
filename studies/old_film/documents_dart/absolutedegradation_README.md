<div align="center">

# AbsoluteDegradation

### A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark

**NeurIPS 2026 · Evaluations and Datasets Track**

[Mikołaj Jastrzębski](https://www.mikjas.com) · [Dawid Glinkowski](https://www.linkedin.com/in/dawid-glinkowski-444354224/) · [Dawid Zieliński](https://www.linkedin.com/in/dawziel/) · [Daniel Borkowski](https://www.linkedin.com/in/daniel-borkowski-/) · Wojciech Kozłowski · Kamil Adamczewski

Wrocław University of Science and Technology

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13.svg)](https://openreview.net/forum?id=gKunvq63KY)
[![arXiv](https://img.shields.io/badge/arXiv-2607.02131-b31b1b.svg)](https://arxiv.org/abs/2607.02131)
[![Project page](https://img.shields.io/badge/Project-page-d9a441.svg)](https://absolutecinema.absolutedegradation.workers.dev)
[![Datasets](https://img.shields.io/badge/Kaggle-datasets-20beff.svg)](https://www.kaggle.com/absolutedegradation/datasets)
[![DART](https://img.shields.io/badge/Models%20%26%20eval-DART-2b6cb0.svg)](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue.svg)](pyproject.toml)

<img src="assets/teaser.gif" width="100%" alt="REDS input, grayscale ground truth, AbsoluteDegradation output and the artifact mask, side by side">

<sub>Clean REDS input → grayscale ground truth → degraded clip → mask of the spatial artifacts (dust, textures, scratches).</sub>

</div>

---

**TL;DR.** A modular, physics-inspired pipeline that turns clean video into temporally coherent film damage (**7 analog artifact families**), plus a benchmark of **81,576** real archival frames. Models trained on it were preferred by human raters in **64.7%** of 1,395 blinded comparisons, for **every** one of six architectures.

## Contents

- [Highlights](#highlights)
- [The pipeline](#the-pipeline)
- [Quick start](#quick-start)
- [Data preparation](#data-preparation)
- [Using the pipeline in your own training](#using-the-pipeline-in-your-own-training)
- [Configuration and extension](#configuration-and-extension)
- [Datasets](#datasets)
- [Results](#results)
- [Training and evaluation code](#training-and-evaluation-code)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Highlights

- **Signal-dependent blue-noise film grain**, not Gaussian noise
- **Ornstein–Uhlenbeck gate weave** and parametric, flickering scratches
- **5,040 operator orderings** per clip and three severity tiers
- **Deterministic** with a fixed `seed`; CPU only

| Pipeline | Domain | Temporal coherence | Film-specific artifacts | Artifact families |
|---|---|:---:|:---:|:---:|
| Real-ESRGAN | Image | × | × | 3 |
| BSRGAN | Image | × | × | 3 |
| Bringing Old Films | Video | ∼ | ∼ | 4 |
| Ivanova et al. | Image | × | ∼ | 2 |
| **AbsoluteDegradation (ours)** | **Video** | **✓** | **✓** | **7** |

<sub>Table 1 of the paper. Artifact families: optical blur, film grain, parametric scratches, dust and texture overlays, negative-film distortions, camera gate weave, compression and photometric shifts.</sub>

## The pipeline

<p align="center"><img src="assets/pipeline_overview.svg" width="100%" alt="AbsoluteDegradation pipeline: six stages from a clean clip to a degraded / ground-truth pair"></p>

<p align="center"><img src="assets/operator_atlas.png" width="100%" alt="Operator atlas: one panel per degradation operator"></p>
<p align="center"><sub><b>Operator atlas.</b> One operator per panel. The gate-weave panel overlays two time steps in red and cyan. Some panels use exaggerated settings so that a single frame shows the effect clearly.</sub></p>

<details>
<summary><b>Gate weave up close</b></summary>
<br>
<p align="center"><img src="assets/gate_weave.gif" width="60%" alt="A static frame with and without gate weave"></p>
<p align="center"><img src="assets/gate_weave_trajectories.png" width="80%" alt="Sampled horizontal and vertical gate-weave trajectories for each severity tier"></p>
<p align="center"><sub>Integer crop offsets of the first-order (x) and second-order (y) Ornstein–Uhlenbeck processes, one sampled clip per tier.</sub></p>
</details>

## Quick start

```bash
git clone https://github.com/TytanMikJas/AbsoluteDegradation.git
cd AbsoluteDegradation

# 1. Environment (CPU only, no GPU or PyTorch needed). Python 3.11–3.13.
uv sync                                   # exact versions from uv.lock

# 2. Assets: blue-noise basis + Bringing Old Films textures (see "Data preparation")
uv run python scripts/download_blue_noise.py
uv run python scripts/adaptive_background_removal.py   # data/noise_data -> data/textures

# 3. Degrade a folder of clips (REDS layout: <root>/<clip>/<frame>.png)
uv run python scripts/degrade.py input_path=data/REDS/val_sharp output_path=data/out
```

Each clip folder produces `data/out/degraded/<clip>/` and `data/out/gt/<clip>/`, as 256×256 training patches by default.

<details>
<summary>Without <code>uv</code> (plain pip)</summary>

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python scripts/download_blue_noise.py
```

`pip` resolves dependencies from the ranges in `pyproject.toml`. For bit-exact reproduction use `uv sync`, which installs the exact versions pinned in `uv.lock`.
</details>

## Data preparation

Nothing under `data/` is tracked by git. The expected layout is:

```
data/
├── REDS/{train_sharp,val_sharp}/<clip>/<frame>.png   # clean sources
├── noise_data/001 … 012/                              # Bringing Old Films scratch templates (raw)
├── textures/001 … 012/                                # RGBA textures (generated, see below)
└── blue_noise/blue_noise_2048_5ch.npy                 # film-grain basis
```

**REDS.** Download `train_sharp` and `val_sharp` from the [REDS dataset page](https://seungjunnah.github.io/Datasets/reds.html) (CC BY 4.0).

**Film textures.** The dust, stain and scratch templates come from the scanned emulsion library of [*Bringing Old Films Back to Life*](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) (Wan et al., CVPR 2022), as in prior work. **We do not redistribute them.** Download the scratch templates from the [Bringing Old Films repository](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) (section *Usage*), respect the terms stated there, and extract them so that `data/noise_data/001 … 012` exist. Then convert them to transparency-keyed RGBA textures with adaptive Gaussian thresholding and morphological erosion:

```bash
uv run python scripts/adaptive_background_removal.py   # config: config/background_removal_config.yaml
```

Group `008` (pure noise) is skipped. Group `001` holds the moving-line templates.

**Blue-noise basis.** `scripts/download_blue_noise.py` fetches the 2048×2048×5 basis used in the paper (≈40 MB, a [release asset](https://github.com/TytanMikJas/AbsoluteDegradation/releases/tag/v1.0.0)) and checks its SHA-256. Any other zero-mean blue-noise texture of shape H×W or H×W×C also works: point `FilmGrainStep.blue_noise_path` at it.

## Using the pipeline in your own training

The paper trains with **online** degradation inside the data loader. The pipeline crops the 256×256 patch first and degrades only that patch, which is far cheaper than degrading full frames.

```python
import random
from pathlib import Path

import numpy as np
from absolute_degradation.factory import create_pipeline_from_config
from absolute_degradation.degradation_degree import DegradationDegree
from absolute_degradation.utils.io_utils import read_image_grayscale

random.seed(42); np.random.seed(42)
pipeline = create_pipeline_from_config(texture_path="data/textures", seed=42)

clip_dir = Path("data/REDS/train_sharp/000")
frames = [read_image_grayscale(p) for p in sorted(clip_dir.glob("*.png"))]  # H×W float32 in [0, 1]
degraded, gt = pipeline(frames)                          # tier sampled with p = 0.30 / 0.35 / 0.35
degraded, gt = pipeline(frames, degree=DegradationDegree.HEAVY)   # or force a tier
# degraded[t], gt[t]: 256×256 uint8 grayscale, spatially aligned
```

`create_pipeline_from_config(..., include_mask=True)` also returns a per-frame soft mask of where textures and scratches were applied. Inputs and targets are grayscale: the task is restoration, not colourisation.

## Configuration and extension

All parameters live in [Hydra](https://hydra.cc) YAML files and correspond one-to-one to **Table 6** of the paper:

| File | Contents |
|---|---|
| [`config/conf.yaml`](config/conf.yaml) | seed, crop and motion padding, Stage 2 textures, Stage 3 negative-film chain, Stage 5 tail |
| [`config/degradation_pipeline/default.yaml`](config/degradation_pipeline/default.yaml) | severity probabilities, Stage 6 scratches |
| [`config/degradation_pipeline/{light,medium,heavy}_steps/default.yaml`](config/degradation_pipeline) | Stage 4 core for each tier |

Override any value from the command line, for example:

```bash
uv run python scripts/degrade.py seed=123 degradation_pipeline.scratch_generator.config.probability=0.6
```

**Adding an operator.** Subclass [`AbstractDistortStep`](absolute_degradation/distortions/steps/abstract_distort_step.py). Sample per-clip parameters in `rebuild()` and apply them per frame in `distort(image, ground_truth)`. Then add the class to a YAML list. Inside a `ComposeDistortStep` with `permute: true`, it joins the random ordering automatically.

```python
class VignetteStep(AbstractDistortStep):
    def rebuild(self) -> None:                 # once per clip
        self.strength = np.random.uniform(0.1, 0.4)

    def distort(self, image, ground_truth):    # every frame
        h, w = image.shape[:2]
        yy, xx = np.mgrid[-1:1:h * 1j, -1:1:w * 1j]
        return image * (1 - self.strength * (xx**2 + yy**2) / 2), ground_truth
```

## Datasets

All datasets are hosted on [Kaggle](https://www.kaggle.com/absolutedegradation/datasets) with Croissant metadata.

| Dataset | Content |
|---|---|
| [Real Scene-Split Full](https://www.kaggle.com/datasets/absolutedegradation/absolutecinema-real-scene-split-full) | **81,576** frames of single-scene archival clips, for temporally consistent evaluation |
| [Real Scene-Split Eval](https://www.kaggle.com/datasets/absolutedegradation/absolutecinema-real-scene-split-eval) | Compact subset, **13,252** frames (up to 5 clips per film, up to 150 central frames) |
| [Real Fixed-Length Full](https://www.kaggle.com/datasets/absolutedegradation/absolutecinema-real-fixed-length-full) | **121,200** frames in fixed 150-frame clips that keep scene transitions |
| [Real Review Sample](https://www.kaggle.com/datasets/absolutedegradation/absolutecinema-real-scene-split-review-sample) | Small review sample of the scene-split benchmark |
| [Synthetic Degraded Eval](https://www.kaggle.com/datasets/absolutedegradation/absolutecinema-synthetic-degraded-eval) / [Train](https://www.kaggle.com/datasets/absolutedegradation/absolutecinema-synthetic-degraded-train) | Sample degraded / ground-truth pairs from REDS |

The real benchmark comes from 30 public-domain films (1896–1918) from the [Library of Congress](https://www.loc.gov), stored as lossless PNG at 1440×1080 to 1920×1080. Scenes were detected with SuperPoint + LightGlue and every clip was curated by hand (Appendix D of the paper).

## Results

Six architectures were trained with each degradation pipeline under identical protocols and evaluated on real archival footage.

**Blinded human preference** (2AFC, 93 raters, 1,395 trials on 29 clips of our benchmark). The table shows the share of trials in which the AbsoluteDegradation-trained restoration was preferred over the Bringing Old Films-trained one:

| Architecture | Trials | Win rate | 95% CI (Wilson) | p |
|---|---:|---:|:---:|---:|
| BasicVSR++ | 237 | **89.9%** | [85.4, 93.1] | < 10⁻⁴ |
| Shift-Net | 228 | **71.5%** | [65.3, 77.0] | < 10⁻⁴ |
| RVRT | 229 | **59.8%** | [53.4, 66.0] | 0.0036 |
| DeepRemaster | 234 | **56.8%** | [50.4, 63.0] | 0.0425 |
| MambaOFR | 231 | **55.8%** | [49.4, 62.1] | 0.0869 |
| RTN | 236 | **54.2%** | [47.9, 60.5] | 0.2161 |
| **All** | **1395** | **64.7%** | [62.2, 67.2] | < 10⁻⁴ |

<details>
<summary><b>No-reference metrics on our benchmark</b> (Table 3; bold = better pipeline per model)</summary>
<br>

| Model | Training pipeline | BRISQUE ↓ | CLIPIQA+ ↑ | MUSIQ ↑ | MANIQA ↑ |
|---|---|---:|---:|---:|---:|
| BasicVSR++ | Bringing Old Films | 30.62 | 0.313 | 32.55 | 0.170 |
|  | AbsoluteDegradation (ours) | **20.68** | **0.369** | **42.91** | **0.238** |
| Shift-Net | Bringing Old Films | 26.74 | 0.370 | 36.74 | 0.213 |
|  | AbsoluteDegradation (ours) | **16.59** | **0.438** | **46.77** | **0.276** |
| MambaOFR | Bringing Old Films | 32.79 | 0.347 | 38.81 | 0.221 |
|  | AbsoluteDegradation (ours) | **20.28** | **0.419** | **47.24** | **0.278** |
| RTN | Bringing Old Films | 19.89 | 0.366 | 41.46 | 0.220 |
|  | AbsoluteDegradation (ours) | **14.39** | **0.422** | **46.87** | **0.266** |
| DeepRemaster | Bringing Old Films | 35.52 | 0.320 | 39.48 | 0.225 |
|  | AbsoluteDegradation (ours) | **23.33** | **0.357** | **44.75** | **0.254** |
| RVRT† | Bringing Old Films | **38.29** | **0.315** | **40.28** | **0.237** |
|  | AbsoluteDegradation (ours) | 38.55 | 0.300 | 35.55 | 0.200 |
| *Real degraded input* | – | 47.86 | 0.248 | 26.84 | 0.162 |

<sub>† RVRT is not fully trained (4 days on 4×A100, compared with 9.7 days on 8×A100 reported by its authors).</sub>
</details>

<details>
<summary><b>No-reference metrics on SRWOV</b> (Table 3)</summary>
<br>

| Model | Training pipeline | BRISQUE ↓ | CLIPIQA+ ↑ | MUSIQ ↑ | MANIQA ↑ |
|---|---|---:|---:|---:|---:|
| BasicVSR++ | Bringing Old Films | 26.20 | 0.397 | 49.17 | 0.245 |
|  | AbsoluteDegradation (ours) | **24.99** | **0.443** | **58.04** | **0.323** |
| Shift-Net | Bringing Old Films | 28.00 | 0.419 | 50.64 | 0.284 |
|  | AbsoluteDegradation (ours) | **25.59** | **0.458** | **57.93** | **0.334** |
| MambaOFR | Bringing Old Films | 27.39 | 0.398 | 50.89 | 0.271 |
|  | AbsoluteDegradation (ours) | **20.73** | **0.440** | **57.36** | **0.321** |
| RTN | Bringing Old Films | 19.86 | 0.436 | 56.42 | 0.300 |
|  | AbsoluteDegradation (ours) | **19.24** | **0.444** | **58.46** | **0.332** |
| DeepRemaster | Bringing Old Films | 37.04 | **0.399** | **57.04** | **0.312** |
|  | AbsoluteDegradation (ours) | **20.90** | 0.372 | 53.55 | 0.264 |
| RVRT† | Bringing Old Films | **32.98** | 0.402 | 48.83 | **0.272** |
|  | AbsoluteDegradation (ours) | 34.23 | **0.410** | **49.66** | 0.270 |
| *Real degraded input* | – | 43.34 | 0.389 | 42.86 | 0.231 |
</details>

Clip-level DOVER scores, temporal stability, repeated-seed runs and ablations are in the [paper](https://openreview.net/forum?id=gKunvq63KY) (Appendices B and F).

**Scope.** The benchmark and pipeline target early black-and-white film (1896–1918) and grayscale restoration. Operator parameters were tuned by qualitative comparison with archival material. Colour film, VHS and other media would need re-parameterisation, which the modular design is meant to make easy.

## Training and evaluation code

Training and evaluation code lives in the **[DART repository](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer)**, together with DART, our degradation-aware restoration model (ACCV 2026), and the evaluation protocols. DART extends AbsoluteDegradation: it is trained with this pipeline and evaluated on this benchmark, and future researchers can submit official models to its protocols. This repository contains the degradation pipeline and the tools to generate data with it.

## Repository layout

```
AbsoluteDegradation/
├── absolute_degradation/                # the pipeline (Python package)
│   ├── distortions/                     #   degradation steps (Stages 1–5) and the DegradationPipeline
│   ├── scratches/                       #   Stage 6: parametric vertical scratches
│   ├── textures/                        #   Stage 2: texture loading, augmentation and blending
│   └── factory.py                       #   create_pipeline_from_config(): Python entry point
├── config/                              # Hydra configs (= Table 6)
├── scripts/
│   ├── degrade.py                       # CLI: degrade a folder of clips into degraded/GT pairs
│   ├── adaptive_background_removal.py   # Bringing Old Films templates -> RGBA textures
│   └── download_blue_noise.py           # fetch + verify the blue-noise basis
└── assets/                              # README figures
```

## Citation

```bibtex
@inproceedings{jastrzebski2026absolutedegradation,
  title     = {{AbsoluteDegradation}: A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark},
  author    = {Jastrz{\k{e}}bski, Miko{\l}aj and Glinkowski, Dawid and Zieli{\'n}ski, Dawid and Borkowski, Daniel and Koz{\l}owski, Wojciech and Adamczewski, Kamil},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

GitHub's *Cite this repository* button uses [`CITATION.cff`](CITATION.cff).

## Acknowledgements

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2026/019902.

The film grain and dust texture assets used in this pipeline are adopted from [Bringing Old Films Back to Life](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life). Ground-truth sequences follow the [REDS](https://seungjunnah.github.io/Datasets/reds.html) format, and the archival benchmark is built from public-domain films of the [Library of Congress](https://www.loc.gov). We thank the authors and institutions for making their data publicly available.

## License

The code is released under the [MIT License](LICENSE). Third-party assets keep their own terms: REDS (CC BY 4.0), the Bringing Old Films texture templates (as distributed by their repository; not included here) and Library of Congress public-domain films.

Questions and issues: open a [GitHub issue](https://github.com/TytanMikJas/AbsoluteDegradation/issues) or write to contact@mikjas.com.
