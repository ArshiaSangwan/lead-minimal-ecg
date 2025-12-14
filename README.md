# Lead-Minimal ECG Diagnosis

**"Which leads actually matter?"** — Systematic evaluation of diagnostic performance when reducing 12-lead ECGs to realistic 1-3 lead subsets.

## Key Contribution

1. **Systematic Lead Ablation Study** — Evaluate all clinically-plausible lead subsets on PTB-XL
2. **Lead-Robustness Score (LRS)** — Novel metric combining sensitivity drop and calibration shift
3. **Lead-Robust Model** — Compact model that maintains accuracy with fewer leads

## Why This Matters

Wearable ECGs (Apple Watch, Kardia, Fitbit) use 1-3 leads. Hospitals and device companies need algorithms that maintain accuracy with fewer leads. This work directly answers: **"What's the minimum number of leads needed for reliable diagnosis?"**

## Dataset

- **PTB-XL** (21,837 12-lead ECGs, 10-second recordings)
- 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP
- Uses official stratified 10-fold split

## Quick Start

```bash
# 1. Setup environment
conda create -n ecg python=3.11 -y
conda activate ecg
pip install -r requirements.txt

# 2. Download PTB-XL dataset
python scripts/download_data.py

# 3. Preprocess and generate lead subsets
python src/preprocess.py

# 4. Train baseline (12-lead)
python src/train.py --leads all --epochs 30

# 5. Train lead-subset models
python src/train.py --leads II --epochs 30
python src/train.py --leads I,II --epochs 30
python src/train.py --leads I,II,V2 --epochs 30

# 6. Evaluate and compute LRS
python src/evaluate.py --model_dir outputs/

# 7. Generate figures
python src/visualize.py
```

## Project Structure

```
lead-minimal-ecg/
├── README.md
├── requirements.txt
├── scripts/
│   └── download_data.py
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── visualize.py
├── configs/
│   └── config.yaml
├── notebooks/
│   └── demo.ipynb
├── outputs/
│   ├── models/
│   └── figures/
└── paper/
    └── main.tex
```

## Lead Subsets Evaluated

| Subset | Leads | Clinical Rationale |
|--------|-------|-------------------|
| Single-Lead | II | Standard monitoring lead |
| Single-Lead | V2 | Best for RBBB/LBBB |
| 2-Lead | I, II | Limb leads only |
| 3-Lead | I, II, V2 | Minimal + precordial |
| 6-Lead | Limb leads | Standard limb |
| 12-Lead | All | Full standard ECG |