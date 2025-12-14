#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner f
Runs experiments across multiple random seeds to compute:
- Mean ± standard deviation for all metrics
- 95% confidence intervals (bootstrap)
- Statistical significance tests (paired t-tests, Wilcoxon)
- Effect sizes (Cohen's d)

Usage:
    python run_multiseed_experiments.py                    # Run 5 seeds
    python run_multiseed_experiments.py --seeds 10         # Run 10 seeds
    python run_multiseed_experiments.py --config 3-lead-I-II-V2 --seeds 5
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================================
# LEAD CONFIGURATIONS (from paper)
# ============================================================================
LEAD_CONFIGS = {
    # 12-lead baseline
    "12-lead": "all",
    
    # 6-lead (limb leads only)
    "6-lead-limb": "I,II,III,aVR,aVL,aVF",
    
    # 3-lead configurations
    "3-lead-I-II-V2": "I,II,V2",
    "3-lead-I-II-III": "I,II,III",
    "3-lead-II-V2-V5": "II,V2,V5",
    
    # 2-lead configurations
    "2-lead-I-II": "I,II",
    "2-lead-II-V2": "II,V2",
    
    # Single-lead configurations
    "1-lead-II": "II",
    "1-lead-V2": "V2",
    "1-lead-I": "I",
    "1-lead-V5": "V5",
}

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    bootstrap_means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return lower, upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def run_single_seed(config_name: str, leads_str: str, seed: int, args) -> Dict:
    """Run a single experiment with a specific seed."""
    from train import train, set_seed
    
    print(f"   Seed {seed}...")
    
    # Count leads
    if leads_str == "all":
        n_leads = 12
    else:
        n_leads = len(leads_str.split(","))
    
    # Build tags for W&B
    tags = [
        f"{n_leads}-lead",
        config_name,
        f"seed-{seed}",
        "multi-seed-experiment",
    ]
    
    # Run training
    results = train(
        leads=leads_str,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        output_dir=args.output_dir,
        data_path=args.data_path,
        seed=seed,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=tags,
    )
    
    results['seed'] = seed
    results['config_name'] = config_name
    results['leads_str'] = leads_str
    
    return results


def run_multiseed_config(config_name: str, leads_str: str, seeds: List[int], args) -> Dict:
    """Run multiple seeds for a single configuration."""
    
    print(f"\n{'='*70}")
    print(f" Configuration: {config_name}")
    print(f"   Leads: {leads_str}")
    print(f"   Seeds: {seeds}")
    print(f"{'='*70}")
    
    seed_results = []
    for seed in seeds:
        try:
            result = run_single_seed(config_name, leads_str, seed, args)
            seed_results.append(result)
        except Exception as e:
            print(f"   Seed {seed} failed: {e}")
    
    if not seed_results:
        return {'config_name': config_name, 'error': 'All seeds failed'}
    
    # Aggregate results
    aurocs = np.array([r['test_auroc'] for r in seed_results])
    f1s = np.array([r['test_f1'] for r in seed_results])
    briers = np.array([r['test_brier'] for r in seed_results])
    
    # Per-class AUROC aggregation
    perclass_aurocs = {cls: [] for cls in CLASSES}
    for r in seed_results:
        for cls in CLASSES:
            if cls in r.get('test_auroc_per_class', {}):
                perclass_aurocs[cls].append(r['test_auroc_per_class'][cls])
    
    # Compute statistics
    auroc_ci = bootstrap_ci(aurocs)
    f1_ci = bootstrap_ci(f1s)
    brier_ci = bootstrap_ci(briers)
    
    aggregated = {
        'config_name': config_name,
        'leads_str': leads_str,
        'n_leads': seed_results[0]['n_leads'],
        'n_seeds': len(seed_results),
        'seeds': seeds[:len(seed_results)],
        
        # AUROC statistics
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'auroc_ci_lower': float(auroc_ci[0]),
        'auroc_ci_upper': float(auroc_ci[1]),
        'auroc_min': float(np.min(aurocs)),
        'auroc_max': float(np.max(aurocs)),
        
        # F1 statistics
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'f1_ci_lower': float(f1_ci[0]),
        'f1_ci_upper': float(f1_ci[1]),
        
        # Brier statistics
        'brier_mean': float(np.mean(briers)),
        'brier_std': float(np.std(briers)),
        'brier_ci_lower': float(brier_ci[0]),
        'brier_ci_upper': float(brier_ci[1]),
        
        # Per-class AUROC
        'auroc_per_class_mean': {cls: float(np.mean(vals)) for cls, vals in perclass_aurocs.items() if vals},
        'auroc_per_class_std': {cls: float(np.std(vals)) for cls, vals in perclass_aurocs.items() if vals},
        
        # Raw results for further analysis
        'raw_aurocs': aurocs.tolist(),
        'raw_f1s': f1s.tolist(),
        'raw_briers': briers.tolist(),
        
        # Individual seed results
        'seed_results': seed_results,
    }
    
    return aggregated


def compute_statistical_tests(all_results: List[Dict], baseline_config: str = '12-lead') -> Dict:
    """Compute statistical significance tests comparing all configs to baseline."""
    
    baseline = next((r for r in all_results if r['config_name'] == baseline_config), None)
    if not baseline or 'raw_aurocs' not in baseline:
        return {}
    
    baseline_aurocs = np.array(baseline['raw_aurocs'])
    
    tests = {}
    for result in all_results:
        if result['config_name'] == baseline_config:
            continue
        if 'raw_aurocs' not in result:
            continue
            
        config_aurocs = np.array(result['raw_aurocs'])
        
        # Ensure same number of samples for paired tests
        min_len = min(len(baseline_aurocs), len(config_aurocs))
        if min_len < 3:
            continue
            
        bl = baseline_aurocs[:min_len]
        cfg = config_aurocs[:min_len]
        
        # Paired t-test
        try:
            t_stat, t_pval = ttest_rel(bl, cfg)
        except:
            t_stat, t_pval = np.nan, np.nan
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = wilcoxon(bl, cfg)
        except:
            w_stat, w_pval = np.nan, np.nan
        
        # Effect size (Cohen's d)
        d = cohens_d(bl, cfg)
        
        tests[result['config_name']] = {
            'vs_baseline': baseline_config,
            'paired_ttest': {
                'statistic': float(t_stat) if not np.isnan(t_stat) else None,
                'p_value': float(t_pval) if not np.isnan(t_pval) else None,
                'significant_005': bool(t_pval < 0.05) if not np.isnan(t_pval) else None,
                'significant_001': bool(t_pval < 0.01) if not np.isnan(t_pval) else None,
            },
            'wilcoxon': {
                'statistic': float(w_stat) if not np.isnan(w_stat) else None,
                'p_value': float(w_pval) if not np.isnan(w_pval) else None,
                'significant_005': bool(w_pval < 0.05) if not np.isnan(w_pval) else None,
            },
            'cohens_d': float(d),
            'effect_size': 'large' if abs(d) > 0.8 else ('medium' if abs(d) > 0.5 else 'small'),
            'auroc_difference': float(result['auroc_mean'] - baseline['auroc_mean']),
            'auroc_retention_pct': float(result['auroc_mean'] / baseline['auroc_mean'] * 100),
        }
    
    return tests


def generate_latex_with_ci(all_results: List[Dict], stat_tests: Dict, output_dir: Path):
    """Generate LaTeX table with confidence intervals and significance markers."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Classification Performance by Lead Configuration (Mean ± Std across 5 seeds). 
AUROC and F1 shown with 95\% confidence intervals. 
$^*$: p < 0.05, $^{**}$: p < 0.01 vs 12-lead (paired t-test).}
\label{tab:main_results_ci}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{N} & \textbf{AUROC} & \textbf{F1} & \textbf{Brier↓} & \textbf{LRS} \\
\midrule
"""
    
    # Sort by number of leads descending
    sorted_results = sorted(all_results, key=lambda x: x.get('n_leads', 0), reverse=True)
    
    # Get baseline for LRS calculation
    baseline = next((r for r in sorted_results if r['config_name'] == '12-lead'), None)
    baseline_auroc = baseline['auroc_mean'] if baseline else 0.9
    
    for result in sorted_results:
        if 'auroc_mean' not in result:
            continue
            
        config = result['config_name'].replace('_', '-').replace('-lead-', ': ')
        n_leads = result.get('n_leads', '?')
        
        # AUROC with CI
        auroc_str = f"{result['auroc_mean']:.3f}±{result['auroc_std']:.3f}"
        
        # Add significance marker
        if result['config_name'] in stat_tests:
            test = stat_tests[result['config_name']]
            if test['paired_ttest'].get('significant_001'):
                auroc_str += r"$^{**}$"
            elif test['paired_ttest'].get('significant_005'):
                auroc_str += r"$^{*}$"
        
        f1_str = f"{result['f1_mean']:.3f}±{result['f1_std']:.3f}"
        brier_str = f"{result['brier_mean']:.4f}"
        
        # Compute LRS
        lrs = result['auroc_mean'] / baseline_auroc if baseline_auroc > 0 else 0
        lrs_str = f"{lrs:.3f}"
        
        latex += f"{config} & {n_leads} & {auroc_str} & {f1_str} & {brier_str} & {lrs_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / "table_main_ci.tex", 'w') as f:
        f.write(latex)
    print(f" LaTeX with CI saved: {output_dir / 'table_main_ci.tex'}")
    
    return latex


def generate_significance_table(stat_tests: Dict, output_dir: Path):
    """Generate a table of statistical significance results."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Statistical Comparison with 12-lead Baseline. Effect sizes: small ($|d|<0.5$), medium ($0.5≤|d|<0.8$), large ($|d|≥0.8$).}
\label{tab:significance}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Configuration} & \textbf{ΔAUROC} & \textbf{Retention} & \textbf{p-value} & \textbf{Cohen's d} & \textbf{Effect} \\
\midrule
"""
    
    for config, test in sorted(stat_tests.items(), key=lambda x: x[1]['auroc_retention_pct'], reverse=True):
        config_short = config.replace('-lead-', ': ')
        delta = test['auroc_difference']
        retention = test['auroc_retention_pct']
        p_val = test['paired_ttest']['p_value']
        d = test['cohens_d']
        effect = test['effect_size']
        
        p_str = f"{p_val:.4f}" if p_val else "N/A"
        sig = "*" if p_val and p_val < 0.05 else ""
        if p_val and p_val < 0.01:
            sig = "**"
        
        latex += f"{config_short} & {delta:+.4f} & {retention:.1f}\\% & {p_str}{sig} & {d:.3f} & {effect} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / "table_significance.tex", 'w') as f:
        f.write(latex)
    print(f" Significance table saved: {output_dir / 'table_significance.tex'}")


def run_all_multiseed(args):
    """Run multi-seed experiments for all configurations."""
    
    print("\n" + "" * 30)
    print("   MULTI-SEED EXPERIMENT SUITE")
    print("   Statistical Rigor for Publication")
    print("" * 30)
    
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"\nSeeds to run: {seeds}")
    print(f"Configurations: {len(LEAD_CONFIGS)}")
    print(f"Total experiments: {len(LEAD_CONFIGS) * len(seeds)}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / "experiments" / f"multiseed_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {exp_dir}")
    
    # Run all configurations
    all_results = []
    
    for i, (config_name, leads_str) in enumerate(LEAD_CONFIGS.items()):
        print(f"\n[{i+1}/{len(LEAD_CONFIGS)}] {config_name}")
        
        try:
            result = run_multiseed_config(config_name, leads_str, seeds, args)
            all_results.append(result)
            
            # Save intermediate results
            with open(exp_dir / "results_partial.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
                
            # Print summary
            if 'auroc_mean' in result:
                print(f"   AUROC: {result['auroc_mean']:.4f} ± {result['auroc_std']:.4f}")
                print(f"     95% CI: [{result['auroc_ci_lower']:.4f}, {result['auroc_ci_upper']:.4f}]")
                
        except Exception as e:
            print(f"   Error: {e}")
            all_results.append({'config_name': config_name, 'error': str(e)})
    
    # Compute statistical tests
    print("\n" + "="*70)
    print(" COMPUTING STATISTICAL SIGNIFICANCE")
    print("="*70)
    
    stat_tests = compute_statistical_tests(all_results)
    
    for config, test in stat_tests.items():
        p = test['paired_ttest']['p_value']
        d = test['cohens_d']
        sig = "**" if p and p < 0.01 else ("*" if p and p < 0.05 else "")
        print(f"  {config}: p={p:.4f}{sig}, Cohen's d={d:.3f} ({test['effect_size']})")
    
    # Generate tables
    print("\n" + "="*70)
    print(" GENERATING PUBLICATION TABLES")
    print("="*70)
    
    generate_latex_with_ci(all_results, stat_tests, exp_dir)
    generate_significance_table(stat_tests, exp_dir)
    
    # Generate CSV summary
    summary_rows = []
    for r in all_results:
        if 'auroc_mean' not in r:
            continue
        row = {
            'Config': r['config_name'],
            'N_Leads': r['n_leads'],
            'N_Seeds': r['n_seeds'],
            'AUROC_Mean': r['auroc_mean'],
            'AUROC_Std': r['auroc_std'],
            'AUROC_CI_Lower': r['auroc_ci_lower'],
            'AUROC_CI_Upper': r['auroc_ci_upper'],
            'F1_Mean': r['f1_mean'],
            'F1_Std': r['f1_std'],
            'Brier_Mean': r['brier_mean'],
            'Brier_Std': r['brier_std'],
        }
        # Add per-class
        for cls in CLASSES:
            if cls in r.get('auroc_per_class_mean', {}):
                row[f'AUROC_{cls}_Mean'] = r['auroc_per_class_mean'][cls]
                row[f'AUROC_{cls}_Std'] = r['auroc_per_class_std'][cls]
        summary_rows.append(row)
    
    df = pd.DataFrame(summary_rows)
    df.to_csv(exp_dir / "results_summary.csv", index=False)
    print(f" CSV summary saved: {exp_dir / 'results_summary.csv'}")
    
    # Save final results
    final_output = {
        'experiment_info': {
            'timestamp': timestamp,
            'n_seeds': args.n_seeds,
            'seeds': seeds,
            'epochs': args.epochs,
            'model': args.model,
        },
        'results': all_results,
        'statistical_tests': stat_tests,
    }
    
    with open(exp_dir / "results_final.json", 'w') as f:
        json.dump(final_output, f, indent=2, default=str)
    print(f" Final results saved: {exp_dir / 'results_final.json'}")
    
    # Print summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    baseline = next((r for r in all_results if r['config_name'] == '12-lead'), None)
    if baseline and 'auroc_mean' in baseline:
        print(f"\n12-lead Baseline: {baseline['auroc_mean']:.4f} ± {baseline['auroc_std']:.4f}")
        print(f"                  95% CI: [{baseline['auroc_ci_lower']:.4f}, {baseline['auroc_ci_upper']:.4f}]")
        
        print("\nKey Findings:")
        for r in sorted(all_results, key=lambda x: x.get('n_leads', 0)):
            if 'auroc_mean' not in r or r['config_name'] == '12-lead':
                continue
            retention = r['auroc_mean'] / baseline['auroc_mean'] * 100
            test = stat_tests.get(r['config_name'], {})
            p = test.get('paired_ttest', {}).get('p_value')
            sig = " (p<0.05)" if p and p < 0.05 else ""
            print(f"  {r['config_name']:20s}: {r['auroc_mean']:.4f}±{r['auroc_std']:.4f} ({retention:.1f}% retention){sig}")
    
    print(f"\n All experiments complete! Results in: {exp_dir}")
    
    return all_results, stat_tests


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed experiments for statistical rigor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Seed configuration
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of random seeds to run (default: 5)")
    parser.add_argument("--seed_start", type=int, default=42,
                        help="Starting seed (default: 42)")
    
    # Experiment selection
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config only")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (3 epochs, 2 seeds)")
    
    # Training params
    parser.add_argument("--model", type=str, default="resnet1d")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=7)
    
    # Data and output
    parser.add_argument("--data_path", type=str,
                        default="data/processed/ptbxl_processed.h5")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    
    # W&B
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lead-minimal-ecg-multiseed")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        print(" QUICK MODE: 3 epochs, 2 seeds")
        args.epochs = 3
        args.n_seeds = 2
        args.patience = 2
    
    # Run
    if args.config:
        if args.config not in LEAD_CONFIGS:
            print(f" Unknown config: {args.config}")
            sys.exit(1)
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
        result = run_multiseed_config(args.config, LEAD_CONFIGS[args.config], seeds, args)
        print(f"\n {args.config}: AUROC = {result['auroc_mean']:.4f} ± {result['auroc_std']:.4f}")
    else:
        run_all_multiseed(args)


if __name__ == "__main__":
    main()
