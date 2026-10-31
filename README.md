# PWHFL: Performance-Weighted Heterogeneous Federated Learning

[![Python](https://img.shields.io/badge/Python-3.12_tested-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1_tested-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10--C%20%7C%20CIFAR--100-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

This repository contains a research implementation of **PWHFL**, a heterogeneous federated learning pipeline for robust image classification under corrupted private, public, and test distributions. The codebase follows a conference-project layout: dataset utilities, model definitions, pretraining scripts, and collaborative heterogeneous FL training routines are organized as reusable modules.

## Overview

Federated learning systems often need to train clients with different model architectures while preserving robustness to distribution shifts and corrupted data. This project explores performance-weighted collaborative training for heterogeneous clients using:

- **Heterogeneous client models**: ResNet10, ResNet12, ShuffleNet, and MobileNetV2.
- **Corruption-aware private training** on CIFAR-10-C style data.
- **Public-data collaboration** with CIFAR-100 as an auxiliary transfer set.
- **AugMix/JSD regularization** for corruption robustness.
- **Contrastive objectives** through supervised contrastive learning or decoupled contrastive learning.
- **Performance-weighted knowledge transfer** that emphasizes stronger participant models during collaboration.

## Repository Structure

```text
PWHFL/
├── Dataset/                  # Dataset wrappers, CIFAR-C generation, sampling, augmentations
│   ├── cifar.py
│   ├── dataaug.py
│   ├── init_dataset.py
│   ├── make_cifar_c.py
│   ├── sampling.py
│   └── utils.py
├── HHF/                      # Federated collaboration and ablation routines
│   ├── ablation.py
│   ├── loss.py
│   ├── pwhfl.py
│   └── rahfl.py
└── Network/                  # Model definitions and participant pretraining
    ├── Models_Def/
    │   ├── mobilnet_v2.py
    │   ├── resnet.py
    │   └── shufflenet.py
    └── pretrain.py
```

## Method Components

### 1. Client Pretraining

`Network/pretrain.py` trains each participant model on its private data partition and stores checkpoints for later collaborative training.

Default participant models:

| Client | Architecture |
| --- | --- |
| 0 | ResNet10 |
| 1 | ResNet12 |
| 2 | ShuffleNet |
| 3 | MobileNetV2 |

### 2. Robust Local Updates

`HHF/pwhfl.py` supports corruption-aware local updates with label smoothing, adaptive JSD weighting, and optional contrastive learning. These components are intended to improve robustness when private data are corrupted.

### 3. Public-Data Collaboration

Collaborative rounds use public CIFAR-100 samples to exchange softened predictions between participants. Performance weighting favors more reliable models, while rescue-style weighting can help clients whose validation performance declines.

## Setup

### Requirements

From the repository root, install a matching PyTorch/torchvision build for your CPU or CUDA environment, then the core dependencies. The regression suite was run on Python 3.12 with PyTorch 2.5.1 and torchvision 0.20.1 on CPU.

```bash
python -m pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for full experiments.

### Data Preparation

The scripts expect CIFAR-style data directories under `Dataset/`, for example:

```text
Dataset/
├── cifar_10_c/
├── cifar_100/
└── cifar_100_c/
```

CIFAR-10-C style corruption data can be prepared with `Dataset/make_cifar_c.py`. This requires `requirements-generation.txt` and the ImageMagick system library used by Wand (for example, `libmagickwand-dev` on Ubuntu). Clean CIFAR datasets are downloaded by the generator and the clean dataset wrappers. The generated private/public arrays have `train/` and `test/` subdirectories containing `labels.npy`, `clean.npy`, and/or `random_corrupt_<rate>.npy`.

For the default private/public corruption rate of 0.5 and clean evaluation, prepare both clean and corrupted arrays:

```bash
python -m pip install -r requirements-generation.txt
python Dataset/make_cifar_c.py --dataset cifar10
python Dataset/make_cifar_c.py --dataset cifar10 --apply-corruption --corrupt-rate 0.5
python Dataset/make_cifar_c.py --dataset cifar100
python Dataset/make_cifar_c.py --dataset cifar100 --apply-corruption --corrupt-rate 0.5
```

For ablations that evaluate corruption rate 1, also generate `--corrupt-rate 1` for the relevant dataset. Generation defaults retain severities 1–4 and the first five frost textures; `--max-severity 5 --frost-textures 6` explicitly enables the previously unreachable choices. These switches change the generated dataset and should be recorded with an experiment.

Historical `RAHFL.*` imports work through the included compatibility namespace; the checkout can remain named `PWHFL`. Training paths now resolve from the checkout root, so no Colab-specific working directory or repository rename is needed.

## Running Experiments

### 1. Pretrain participant models

```bash
python Network/pretrain.py
```

This creates per-client checkpoints under `Network/Model_Storage_*` according to the settings defined near the top of `Network/pretrain.py`.

### 2. Run PWHFL collaboration

```bash
python HHF/pwhfl.py
```

This loads pretrained participant checkpoints, performs communication rounds on public data, updates clients locally, and logs accuracy/loss metrics.

### 3. Run baseline or ablation code

```bash
python HHF/rahfl.py
python HHF/ablation.py
```

Use these scripts to compare the proposed pipeline with baseline heterogeneous FL behavior or component ablations.

The default ablation tables and rows are retained. To run the previously disabled component table and full-method reference rows explicitly:

```bash
python HHF/ablation.py --tables all --include-reference
```

PWHFL and RAHFL also save their existing accuracy/loss histories as `training_metrics.json` beside the final checkpoints.

## Additional datasets: validation only

The standalone validator evaluates saved checkpoints on **CIFAR-10.1 v6** and the **CIFAR-10.2 test split**, with 2,000 images each. They share CIFAR-10's ten classes, so the existing classification heads require no changes. The downloader never fetches CIFAR-10.2 training data.

```bash
python -m Validation.download
python -m Validation.evaluate \
  --config validation.example.json \
  --checkpoint-dir APEX_Storage_hetero_4client/cifar10_iid_0/random_corrupt_0.5 \
  --device cpu \
  --output validation_results/pwhfl.json
```

Use `--device cuda` when available, or add `--download` to evaluation to fetch missing registered test assets. The checkpoint directory must contain `ResNet10_0.ckpt`, `ResNet12_1.ckpt`, `ShuffleNet_2.ckpt`, and `Mobilenetv2_3.ckpt`; these are the existing training filenames. You must supply trained checkpoints to obtain meaningful model results.

Validation runs in a separate command after the checkpoints are frozen. It performs no training, optimization, teacher updates, hyperparameter tuning, or checkpoint selection. Training loaders do not register these extra datasets. The original in-training evaluation and weighting logic is unchanged.

Results include JSON and CSV metrics, per-class accuracy, confusion matrices, input/checkpoint hashes, preprocessing, and software versions. Every image is evaluated, including the last partial batch. See [validation instructions](docs/VALIDATION.md) for individual checkpoints, official CIFAR-10-C severity blocks, and custom held-out arrays or class folders.

## Cleanup and verification

The [audit resolution](docs/AUDIT_FIXES.md) records removed dead code, consolidated duplicate helpers, repaired optional branches, and features deliberately retained for compatibility. Model parameters, default forward outputs, and the PWHFL objectives are preserved. Broken optional paths such as `nojsd` now execute their intended behavior.

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q Dataset HHF Network RAHFL Validation tests
ruff check --select F821,F401,F841 Dataset HHF Network RAHFL Validation tests
python -m pytest -q
```

The tests do not download datasets or train a full experiment. Generation tests use small fixtures and require the generation dependencies above.

## Key Configuration Options

Most experiment settings are defined as module-level constants near the top of the training scripts:

| Option | Meaning |
| --- | --- |
| `CommunicationEpoch` | Number of collaborative communication rounds |
| `TrainBatchSize`, `TestBatchSize` | Batch sizes for train/test loaders |
| `Private_Corrupt_Rate` | Corruption level for client private data |
| `Public_Corrupt_Rate` | Corruption level for public collaboration data |
| `Test_Corrupt_Rate` | Corruption level for evaluation data |
| `Model_setting` | Homogeneous or heterogeneous client setup |
| `Nets_Name_List` | Ordered list of client architectures |
| `Data_Partition` | IID or non-IID private partitioning |
| `Noniid_Dirichlet_Beta` | Dirichlet concentration for non-IID partitioning |
| `AugMix_Module` | Enables/disables AugMix JSD behavior |
| `CL_Module` | Contrastive loss mode (`supcon`, `dcl`, or disabled) |

## Reproducibility Notes

- Existing seed calls are preserved. In particular, pretraining's former NumPy wildcard import bound `random` to `numpy.random`; its replacement retains that effective binding instead of silently reseeding Python's RNG differently.
- Checkpoint and log directories are created automatically by utility functions.
- Exact results depend on CUDA availability, dataset placement, corruption generation, and the selected hyperparameters.

## Acknowledgements

This project builds on common federated learning, AugMix, contrastive learning, and robust image-classification research workflows. The implementation uses PyTorch and torchvision for model training and dataset handling.
