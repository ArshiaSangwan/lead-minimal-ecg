# Lead-Minimal ECG: How Few Leads Do You Really Need?

> **"A 3-lead ECG configuration (I, II, V2) retains 99.1% of 12-lead diagnostic performance."**

This repository contains the code, experiments, and results for our systematic study on ECG lead reduction for automated cardiac diagnosis.

---

## 🎯 Key Findings

| Configuration | Leads | AUROC | Performance Retention |
|--------------|-------|-------|----------------------|
| **12-lead** (baseline) | All 12 | **0.913** | 100.0% |
| **3-lead (I, II, V2)** 🏆 | I, II, V2 | **0.905** | **99.1%** |
| 3-lead (II, V2, V5) | II, V2, V5 | 0.897 | 98.3% |
| 6-lead (limb) | Limb leads | 0.889 | 97.4% |
| 2-lead (I, II) | I, II | 0.887 | 97.2% |
| **1-lead (II)** | II | 0.856 | **93.7%** |

### Key Insights

1. **3-lead outperforms 6-lead**: The optimal 3-lead (I, II, V2) achieves higher AUROC than all 6 limb leads.
2. **V2 is critical**: Adding V2 dramatically improves MI detection (0.922 vs 0.876).
3. **Lead II alone is robust**: Single lead achieves 93.7% of full 12-lead performance.

---

## 📊 Full Results

### Main Results

| Config | N | AUROC | F1 | Brier | LRS |
|--------|---|-------|-----|-------|-----|
| 12-lead | 12 | **0.913** | **0.688** | **0.084** | **1.000** |
| 6-lead-limb | 6 | 0.889 | 0.641 | 0.095 | 0.968 |
| 3-lead-I-II-V2 | 3 | 0.905 | 0.667 | 0.088 | 0.988 |
| 3-lead-I-II-III | 3 | 0.888 | 0.634 | 0.096 | 0.966 |
| 3-lead-II-V2-V5 | 3 | 0.897 | 0.659 | 0.091 | 0.980 |
| 2-lead-I-II | 2 | 0.887 | 0.630 | 0.097 | 0.965 |
| 2-lead-II-V2 | 2 | 0.884 | 0.619 | 0.097 | 0.961 |
| 1-lead-II | 1 | 0.856 | 0.566 | 0.109 | 0.926 |
| 1-lead-V5 | 1 | 0.851 | 0.568 | 0.109 | 0.921 |
| 1-lead-I | 1 | 0.842 | 0.537 | 0.116 | 0.907 |
| 1-lead-V2 | 1 | 0.793 | 0.444 | 0.128 | 0.854 |

### Per-Class AUROC

| Config | NORM | MI | STTC | CD | HYP |
|--------|------|-----|------|-----|-----|
| 12-lead | 0.953 | 0.929 | 0.936 | 0.917 | 0.830 |
| 3-lead-I-II-V2 | 0.947 | 0.922 | 0.924 | 0.910 | 0.821 |
| 6-lead-limb | 0.940 | 0.886 | 0.917 | 0.885 | 0.818 |
| 1-lead-II | 0.917 | 0.842 | 0.876 | 0.860 | 0.784 |

---

## 🏗️ Method

### Model: 1D ResNet
- **Stem**: Conv1d → BN → ReLU → MaxPool
- **Blocks**: 4 residual blocks with spatial dropout (0.3) and stochastic depth (0.1)
- **Head**: AdaptiveAvgPool → Dropout → Linear
- **Parameters**: ~100K

### Training
- **Optimizer**: AdamW (lr=0.001, wd=0.01)
- **Epochs**: 30 with early stopping
- **Regularization**: Label smoothing (0.1), Mixup (α=0.2)
- **Hardware**: RTX 4090, mixed precision

### Lead-Robustness Score (LRS)
```
LRS = 0.7 × (AUROC_subset / AUROC_baseline) + 0.3 × (1 - ΔBrier / 0.25)
```

---

## 🚀 Quick Start

```bash
# Setup
conda create -n ecg python=3.11 -y && conda activate ecg
pip install -r requirements.txt

# Download and preprocess PTB-XL
python scripts/download_data.py
python src/preprocess.py

# Run all experiments
python run_all_experiments.py --epochs 30

# Generate paper tables
python generate_paper_results.py
```

---

## 📁 Project Structure

```
lead-minimal-ecg/
├── src/
│   ├── train.py          # Training with W&B logging
│   ├── model.py          # 1D ResNet architecture
│   └── dataset.py        # PTB-XL data loader
├── run_all_experiments.py # Full experiment suite
├── generate_paper_results.py
├── outputs/
│   ├── models/           # Checkpoints
│   └── experiments/      # Results & tables
└── paper/
    └── main.tex          # LaTeX paper
```

---

## 📖 Citation

```bibtex
@article{leadminimal2024,
  title={Lead-Minimal ECG: Systematic Evaluation of Reduced-Lead Configurations},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
