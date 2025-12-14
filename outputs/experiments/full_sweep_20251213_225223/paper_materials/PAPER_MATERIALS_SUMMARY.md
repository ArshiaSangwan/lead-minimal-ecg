# [RESULTS] Lead-Minimal ECG: Paper Materials Summary

**Generated:** 2025-12-13 23:53:08

## Key Findings

- **Baseline (12-lead):** AUROC = 0.913
- **Best 3-lead (3-lead-I-II-V2):** AUROC = 0.905, Retention = 99.1%
- **Best single-lead (1-lead-II):** AUROC = 0.856, Retention = 93.7%
- ** 3-lead BEATS 6-lead:** 0.905 vs 0.889 (Δ = +0.016)
- **Hardest class:** Hypertrophy (HYP), AUROC = 0.830

## Results Table

| Configuration | N | AUROC | Retention | F1 | Brier |
|--------------|---|-------|-----------|-----|-------|
| 12-lead (Full) | 12 | 0.913 | 100.0% | 0.688 | 0.084 |
| 6-lead (Limb) | 6 | 0.889 | 97.4% | 0.641 | 0.095 |
| 3-lead (I, II, V2) | 3 | 0.905 | 99.1% | 0.667 | 0.088 |
| 3-lead (I, II, III) | 3 | 0.888 | 97.3% | 0.634 | 0.096 |
| 3-lead (II, V2, V5) | 3 | 0.897 | 98.3% | 0.659 | 0.090 |
| 2-lead (I, II) | 2 | 0.887 | 97.2% | 0.630 | 0.097 |
| 2-lead (II, V2) | 2 | 0.884 | 96.8% | 0.619 | 0.097 |
| 1-lead (II) | 1 | 0.856 | 93.7% | 0.566 | 0.109 |
| 1-lead (V2) | 1 | 0.793 | 86.8% | 0.444 | 0.128 |
| 1-lead (I) | 1 | 0.842 | 92.3% | 0.537 | 0.116 |
| 1-lead (V5) | 1 | 0.851 | 93.2% | 0.568 | 0.109 |

## Generated Files

### Tables
- `tables/table_main.tex` - Main results table (LaTeX)
- `tables/table_perclass.tex` - Per-class AUROC table (LaTeX)
- `tables/results_all.csv` - All results (CSV)
- `tables/key_findings.json` - Key findings (JSON)

### Figures
- `figures/fig_performance_comparison.pdf` - Bar chart of all configurations
- `figures/fig_perclass_heatmap.pdf` - Heatmap of per-class performance
- `figures/fig_retention_vs_leads.pdf` - Performance retention vs lead count
- `figures/fig_radar_chart.pdf` - Radar chart comparing key configurations
- `figures/fig_3lead_vs_6lead.pdf` - Detailed 3-lead vs 6-lead comparison

## Abstract Claims (Supported by Data)

1. "A 3-lead configuration (I, II, V2) retains 99.1% of 12-lead performance"
2. "The optimal 3-lead configuration OUTPERFORMS the 6-lead limb configuration"
3. "A single lead (II) achieves 93.7% of baseline performance"
