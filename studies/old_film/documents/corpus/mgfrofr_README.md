# MgfrOFR

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

Official implementation of **Multi-Granularity Feature Decomposition with Ranking-Aware Regularization for Old Film Restoration**.

MgfrOFR is an old-film restoration framework built on a recurrent video restoration backbone. It introduces multi-granularity feature decomposition for degradation-aware restoration and ranking-aware temporal regularization for more stable video recovery.

> Paper and pretrained weight links will be updated after the final camera-ready/release assets are frozen. The source code, evaluation scripts, and configuration files are already available in this repository.

## News

- **2026-05-10**: Initial public-release codebase prepared with training/testing configs and BasicOFR/BasicSR runtime.

## Highlights

- **Old-film restoration focus**: designed for mixed analog degradations such as scratches, noise, blur, flicker, and local structured artifacts.
- **Multi-granularity feature decomposition**: adds MGFE modules to separate restoration cues at different feature granularities.
- **Ranking-aware temporal regularization**: provides a training loss for temporally consistent restoration behavior.
- **Reproducible configs**: includes full-model and ablation configs for training and testing.
- **Standalone release**: packages the required `mgfrofr`, `basicofr`, and `basicsr` code paths in one repository.

## Method Overview

<p align="center">
  <img src="assets/figures/architecture.png" width="92%" alt="MgfrOFR architecture">
</p>

<p align="center"><b>MgfrOFR architecture.</b> The model extends a recurrent Mamba-based old-film restoration backbone with multi-granularity feature extraction and training-time ranking-aware regularization.</p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/figures/mgfe_module.png" width="100%" alt="MGFE module"><br>
      <b>MGFE.</b> Multi-granularity restoration-token branches summarize coarse-to-fine degradation cues and recalibrate propagated features.
    </td>
    <td align="center" width="50%">
      <img src="assets/figures/ratr_module.png" width="100%" alt="RATR module"><br>
      <b>RATR.</b> Ranking-aware triplet regularization discourages branch collapse and promotes complementary temporal descriptors during training.
    </td>
  </tr>
</table>

## Qualitative Results

<p align="center">
  <img src="assets/figures/qualitative_synthetic_readme.jpg" width="96%" alt="Synthetic REDS qualitative comparison">
</p>

<p align="center"><b>Synthetic REDS-style old-film restoration.</b> MgfrOFR improves structural recovery and suppresses local old-film artifacts compared with the recurrent baseline.</p>

<p align="center">
  <img src="assets/figures/qualitative_realworld_readme.jpg" width="82%" alt="Real-world SRWOV qualitative comparison">
</p>

<p align="center"><b>Real-world archival footage.</b> MgfrOFR produces visually cleaner restorations on real-world SRWOV clips without ground-truth supervision at test time.</p>

## Experimental Analysis

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/figures/ablation_qualitative_readme.jpg" width="100%" alt="Ablation qualitative comparison"><br>
      <b>Ablation.</b> MGFE and RATR provide complementary improvements over the baseline.
    </td>
    <td align="center" width="50%">
      <img src="assets/figures/sensitivity_readme.png" width="100%" alt="Sensitivity analysis"><br>
      <b>Sensitivity.</b> Token/branch choices are stable around the default full-model setting.
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/figures/convergence.png" width="100%" alt="Training convergence"><br>
      <b>Convergence.</b> The full model preserves stable optimization behavior.
    </td>
    <td align="center" width="50%">
      <img src="assets/figures/branch_similarity.png" width="100%" alt="Branch similarity"><br>
      <b>Branch diversity.</b> RATR reduces redundant branch behavior and encourages multi-granularity specialization.
    </td>
  </tr>
</table>

## TODO

- [ ] Add paper link.
- [ ] Add pretrained model download URLs and SHA256 checksums.
- [ ] Add final benchmark tables after metric provenance is frozen.
- [x] Add qualitative examples and method figures.

## Repository Structure

```text
MgfrOFR/
├── mgfrofr/              # MgfrOFR architecture, model wrapper, and RATR loss
├── basicofr/             # Video restoration runtime components
├── basicsr/              # BasicSR-style training/testing infrastructure
├── options/
│   ├── train/            # Training and ablation configs
│   └── test/             # Synthetic and real-world evaluation configs
├── scripts/              # Train/test entry points
├── docs/                 # Dataset, model-zoo, and result notes
├── tests/                # Import/registry/config smoke test
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── setup.py
```

## Installation

### 1. Clone

```bash
git clone https://github.com/wscffaa/MgfrOFR.git
cd MgfrOFR
```

### 2. Create Environment

```bash
conda create -n mgfrofr python=3.10 -y
conda activate mgfrofr
```

Install PyTorch according to your CUDA version from the official PyTorch instructions. For example:

```bash
# Example only. Pick the command matching your CUDA/driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies and this repository:

```bash
pip install -r requirements.txt
pip install -e .
```

`mamba-ssm` and CUDA extensions can be environment-sensitive. If installation fails, install the PyTorch/CUDA-compatible wheel first and then rerun `pip install -r requirements.txt`.

## Quick Check

Run the lightweight smoke test after installation:

```bash
python tests/test_mgfrofr_smoke.py
```

Expected output:

```text
MgfrOFR import, registry, and config smoke test passed
```

This checks package imports, registry entries, and YAML config parsing. It does not require pretrained restoration weights.

## Dataset Preparation

This repository does not redistribute REDS, SRWOV, or third-party datasets. Please download datasets from their official sources and arrange them as follows:

```text
datasets/OldFilmRestoration/
├── train/
│   └── REDS/train_sharp/
├── val/
│   ├── miniREDS/
│   │   ├── GT/val_sharp/
│   │   └── LQ/val_sharp_degraded/
│   ├── REDS/
│   │   ├── GT/val_sharp/
│   │   └── LQ/val_sharp_degraded/
│   └── RealFilm/MambaOFR_SRWOV/
└── noise_data/
```

See [docs/DATASETS.md](docs/DATASETS.md) for more notes.

## Model Zoo

Weights are not committed to git. Put downloaded models under `pretrained/`:

```text
pretrained/
├── mambaofr_reds_16_net_g_best.pth
├── mgfrofr_a3_full.pth
└── mgfrofr_a0_mambaofr.pth
```

| Model | Config | Download | Notes |
| --- | --- | --- | --- |
| MgfrOFR-A3 full | `options/test/mgfrofr_a3_synthetic.yml` | To be released | Full MGFE + RATR model |
| MgfrOFR-A0 baseline | `options/train/mgfrofr_a0_mambaofr.yml` | To be released | Baseline reference |
| MambaOFR warm start | `options/train/mgfrofr_a3_full.yml` | To be released | Used for warm-start training |

See [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md). Add checksums before publishing pretrained weights.

## Evaluation

### Synthetic REDS-style Evaluation

```bash
python scripts/test.py -opt options/test/mgfrofr_a3_synthetic.yml
```

Outputs are saved under `results/`.

### Real-world Old-film Evaluation

```bash
python scripts/test.py -opt options/test/mgfrofr_a3_realworld.yml
```

The real-world config saves restored frames and does not require ground-truth frames.

## Training

Full MgfrOFR training:

```bash
python scripts/train.py -opt options/train/mgfrofr_a3_full.yml
```

Ablation configs:

| Config | Purpose |
| --- | --- |
| `options/train/mgfrofr_a0_mambaofr.yml` | recurrent baseline |
| `options/train/mgfrofr_a1_mgfe.yml` | baseline + MGFE |
| `options/train/mgfrofr_a2_ratr.yml` | baseline + RATR |
| `options/train/mgfrofr_a3_full.yml` | MGFE + RATR full model |

Training logs and checkpoints are written to `experiments/` and `tb_logger/`, both ignored by git.

## Results

Final paper metrics will be added after the evaluation protocol, checkpoint hashes, and metric provenance are frozen. See [docs/RESULTS.md](docs/RESULTS.md) for the intended reporting fields.

When reporting results, please include:

- checkpoint file and hash;
- config file;
- dataset split;
- metric implementation;
- exact command;
- date and environment.

## Citation

If this repository helps your research, please cite the paper. The final BibTeX will be updated after publication.

```bibtex
@article{cai2026mgfrofr,
  title={Multi-Granularity Feature Decomposition with Ranking-Aware Regularization for Old Film Restoration},
  author={Cai, Feifan},
  year={2026}
}
```

A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## Acknowledgements

This repository follows the BasicSR-style restoration pipeline and includes BasicOFR/BasicSR runtime components required by MgfrOFR. MgfrOFR also builds on the old-film restoration setting and recurrent restoration baseline popularized by prior works such as MambaOFR and earlier old-film restoration frameworks. Please cite the relevant upstream projects and datasets when using this code.

## License

This project is released under the [Apache License 2.0](LICENSE).
