# Lead-Minimal ECG: How Few Leads Do You Really Need?

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-FFCC33?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

> **"A 3-lead ECG configuration (I, II, V2) retains 99.1% of 12-lead diagnostic performance."**

This repository contains the code, experiments, and results for our systematic study on ECG lead reduction for automated cardiac diagnosis. We demonstrate that carefully selected lead subsets can achieve near-baseline performance, with significant implications for wearable ECG devices.

---

## 🎯 Key Findings

| Configuration | Leads | AUROC | Performance Retention |
|--------------|-------|-------|----------------------|
| **12-lead** (baseline) | I, II, III, aVR, aVL, aVF, V1-V6 | **0.913** | 100.0% |
| **3-lead (I, II, V2)** 🏆 | I, II, V2 | **0.905** | **99.1%** |
| 3-lead (II, V2, V5) | II, V2, V5 | 0.897 | 98.3% |
| 6-lead (limb) | I, II, III, aVR, aVL, aVF | 0.889 | 97.4% |
| 2-lead (I, II) | I, II | 0.887 | 97.2% |
| **1-lead (II)** | II | 0.856 | **93.7%** |

### 🔬 Key Insights

1. **3-lead outperforms 6-lead**: The optimal 3-lead configuration (I, II, V2) achieves higher AUROC (0.905) than using all 6 limb leads (0.889).

2. **V2 is critical**: Adding precordial lead V2 to limb leads dramatically improves MI detection (0.922 vs 0.876 with limb leads only).

3. **Lead II alone is remarkably robust**: A single lead achieves 93.7% of full 12-lead performance, making it ideal for continuous monitoring.

4. **Hypertrophy (HYP) is hardest to detect with fewer leads**: This class shows the largest performance drop (0.830 → 0.784) when reducing leads.

---

## 📊 Results Overview

### Main Results Table

| Configuration | N | AUROC | F1 | Brier | LRS |
|--------------|---|-------|-----|-------|-----|
| 12-lead | 12 | **0.913** | **0.688** | **0.084** | **1.000** |
| 6-lead (limb) | 6 | 0.889 | 0.641 | 0.095 | 0.968 |
| 3-lead (I-II-V2) | 3 | 0.905 | 0.667 | 0.088 | 0.988 |
| 3-lead (I-II-III) | 3 | 0.888 | 0.634 | 0.096 | 0.966 |
| 3-lead (II-V2-V5) | 3 | 0.897 | 0.659 | 0.091 | 0.980 |
| 2-lead (I-II) | 2 | 0.887 | 0.630 | 0.097 | 0.965 |
| 2-lead (II-V2) | 2 | 0.884 | 0.619 | 0.097 | 0.961 |
| 1-lead (II) | 1 | 0.856 | 0.566 | 0.109 | 0.926 |
| 1-lead (V5) | 1 | 0.851 | 0.568 | 0.109 | 0.921 |
| 1-lead (I) | 1 | 0.842 | 0.537 | 0.116 | 0.907 |
| 1-lead (V2) | 1 | 0.793 | 0.444 | 0.128 | 0.854 |

*AUROC: Area Under ROC Curve (macro-averaged). LRS: Lead-Robustness Score. Brier: Brier Score (lower is better).*

### Per-Class Performance (AUROC)

| Configuration | NORM | MI | STTC | CD | HYP |
|--------------|------|-----|------|-----|-----|
| 12-lead | 0.953 | 0.929 | 0.936 | 0.917 | 0.830 |
| 3-lead (I-II-V2) | 0.947 | 0.922 | 0.924 | 0.910 | 0.821 |
| 6-lead (limb) | 0.940 | 0.886 | 0.917 | 0.885 | 0.818 |
| 1-lead (II) | 0.917 | 0.842 | 0.876 | 0.860 | 0.784 |

---

## 🏗️ Method

### Model Architecture

We use a **1D ResNet** optimized for ECG signals:
- **Stem**: Conv1d(n_leads, 32, k=15, s=2) → BN → ReLU → MaxPool
- **Residual Blocks**: 4 blocks with spatial dropout and stochastic depth
- **Head**: AdaptiveAvgPool → Dropout(0.3) → Linear(n_classes)
- **Parameters**: ~100K (varies by input leads)

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Batch Size | 128 |
| Epochs | 30 (early stopping, patience=7) |
| Scheduler | CosineAnnealingLR |
| Label Smoothing | 0.1 |
| Mixup Alpha | 0.2 |
| Dropout | 0.3 |
| Stochastic Depth | 0.1 |

### Lead-Robustness Score (LRS)

We propose the **Lead-Robustness Score** to quantify how well a reduced-lead model maintains both discrimination and calibration:

```
LRS = α × (AUROC_subset / AUROC_baseline) + β × (1 - ΔBrier / 0.25)
```

Where α = 0.7, β = 0.3, and ΔBrier is the calibration degradation.

---

## 📁 Project Structure

```
lead-minimal-ecg/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── run_all_experiments.py             # Full experiment suite
├── generate_paper_results.py          # Generate publication tables
│
├── src/
│   ├── train.py                       # Training script with W&B logging
│   ├── model.py                       # 1D ResNet architecture
│   ├── dataset.py                     # PTB-XL dataset loader
│   └── preprocess.py                  # Data preprocessing
│
├── scripts/
│   └── download_data.py               # Download PTB-XL
│
├── outputs/
│   ├── models/                        # Trained model checkpoints
│   ├── experiments/                   # Experiment sweep results
│   │   └── full_sweep_*/
│   │       ├── results_final.json     # All results
│   │       └── paper_tables/          # Publication-ready tables
│   │           ├── table_main.tex     # Main LaTeX table
│   │           ├── table_perclass.tex # Per-class LaTeX table
│   │           ├── RESULTS.md         # Markdown summary
│   │           └── summary.json       # Structured results
│   └── wandb/                         # W&B logs
│
└── paper/
    └── main.tex                       # Paper draft
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n ecg python=3.11 -y
conda activate ecg

# Install dependencies
pip install -r requirements.txt

# Login to W&B (optional, for experiment tracking)
wandb login
```

### 2. Data Preparation

```bash
# Download PTB-XL dataset
python scripts/download_data.py

# Preprocess (generates HDF5 file)
python src/preprocess.py
```

### 3. Run Experiments

```bash
# Run full experiment suite (all 11 lead configurations)
python run_all_experiments.py --epochs 30

# Run single configuration
python run_all_experiments.py --config 3-lead-I-II-V2 --epochs 30

# Quick test (5 epochs)
python run_all_experiments.py --quick
```

### 4. Generate Results

```bash
# Generate publication tables
python generate_paper_results.py

# Results will be in outputs/experiments/full_sweep_*/paper_tables/
```

---

## 📈 Experiment Tracking

All experiments are logged to **Weights & Biases** with:
- Training/validation curves
- Per-class AUROC metrics
- Brier scores for calibration
- Model artifacts

---

## 🔗 Related Work

- [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/) - The ECG dataset used
- [PhysioNet Challenge 2021](https://physionetchallenges.org/2021/) - Multi-lead ECG classification challenge
