#  Lead-Minimal ECG Results Summary

**Generated:** 2025-12-13 23:02:27

## Main Results

| Configuration | N | AUROC | Retention | LRS |
|--------------|---|-------|-----------|-----|
| 12-lead | 12 | 0.9130 | 100.0% | 1.000 |
| 6-lead-limb | 6 | 0.8893 | 97.4% | 0.968 |
| 3-lead-I-II-V2 | 3 | 0.9050 | 99.1% | 0.988 |
| 3-lead-I-II-III | 3 | 0.8885 | 97.3% | 0.966 |
| 3-lead-II-V2-V5 | 3 | 0.8972 | 98.3% | 0.980 |
| 2-lead-I-II | 2 | 0.8875 | 97.2% | 0.965 |
| 2-lead-II-V2 | 2 | 0.8837 | 96.8% | 0.961 |
| 1-lead-II | 1 | 0.8558 | 93.7% | 0.926 |
| 1-lead-V2 | 1 | 0.7928 | 86.8% | 0.854 |
| 1-lead-I | 1 | 0.8424 | 92.3% | 0.907 |
| 1-lead-V5 | 1 | 0.8507 | 93.2% | 0.921 |

##  Key Findings

- **Best 3-lead configuration:** 3-lead-I-II-V2 achieves **99.1%** of 12-lead performance (AUROC: 0.9050)
- **Best single-lead:** 1-lead-II achieves **93.7%** of 12-lead performance (AUROC: 0.8558)
- **3-lead outperforms 6-lead:** Best 3-lead (0.9050) > 6-lead limb (0.8893)

## Per-Class Performance (AUROC)

| Configuration | NORM | MI | STTC | CD | HYP |
|--------------|------|-----|------|-----|-----|
| 12-lead | 0.953 | 0.929 | 0.936 | 0.917 | 0.830 |
| 6-lead-limb | 0.940 | 0.886 | 0.917 | 0.885 | 0.818 |
| 3-lead-I-II-V2 | 0.947 | 0.922 | 0.924 | 0.910 | 0.821 |
| 3-lead-I-II-III | 0.939 | 0.888 | 0.915 | 0.885 | 0.814 |
| 3-lead-II-V2-V5 | 0.943 | 0.917 | 0.924 | 0.896 | 0.807 |
| 2-lead-I-II | 0.939 | 0.876 | 0.916 | 0.885 | 0.821 |
| 2-lead-II-V2 | 0.931 | 0.909 | 0.896 | 0.896 | 0.787 |
| 1-lead-II | 0.917 | 0.842 | 0.876 | 0.860 | 0.784 |
| 1-lead-V2 | 0.841 | 0.828 | 0.765 | 0.812 | 0.718 |
| 1-lead-I | 0.902 | 0.801 | 0.876 | 0.822 | 0.811 |
| 1-lead-V5 | 0.913 | 0.817 | 0.910 | 0.820 | 0.794 |
