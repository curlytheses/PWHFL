# PWHFL: Performance-Weighted Heterogeneous Federated Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
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

Install a Python environment with the core dependencies below:

```bash
pip install torch torchvision numpy pillow tqdm
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

CIFAR-10-C style corruption data can be prepared with the utilities in `Dataset/make_cifar_c.py`. Clean CIFAR datasets are downloaded automatically by the dataset wrappers when supported by the configured path.

> **Note:** Some scripts currently assume the project package name `RAHFL` in imports and paths. If running from this repository as `PWHFL`, either place the repository on `PYTHONPATH` with the expected package name or update the imports/paths consistently before launching experiments.

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

- Random seeds are configured in the training scripts for Python, NumPy, and PyTorch.
- Checkpoint and log directories are created automatically by utility functions.
- Exact results depend on CUDA availability, dataset placement, corruption generation, and the selected hyperparameters.

## Acknowledgements

This project builds on common federated learning, AugMix, contrastive learning, and robust image-classification research workflows. The implementation uses PyTorch and torchvision for model training and dataset handling.
