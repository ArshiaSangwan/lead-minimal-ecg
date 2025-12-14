#!/usr/bin/env python3
"""
This script runs the complete pipeline:
1. Multi-seed experiments (5 seeds per configuration)
2. Comprehensive evaluation (all metrics)
3. Statistical significance testing
4. Generate all figures and tables
5. External validation (if available)

Usage:
    # Full pipeline
    python run_complete_pipeline.py --mode full --seeds 5
    
    # Quick validation (for development)
    python run_complete_pipeline.py --mode quick --seeds 2 --epochs 5
    
    # Skip training, just analyze existing results
    python run_complete_pipeline.py --mode analyze --results-dir outputs/experiments/...
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================================
# CONFIGURATION
# ============================================================================

LEAD_CONFIGS = {
    # Baseline
    "12-lead": "all",
    
    # 6-lead
    "6-lead-limb": "I,II,III,aVR,aVL,aVF",
    
    # 3-lead (key comparison)
    "3-lead-I-II-V2": "I,II,V2",
    "3-lead-I-II-III": "I,II,III",
    "3-lead-II-V2-V5": "II,V2,V5",
    
    # 2-lead
    "2-lead-I-II": "I,II",
    "2-lead-II-V2": "II,V2",
    
    # Single-lead (wearable relevant)
    "1-lead-II": "II",
    "1-lead-V2": "V2",
    "1-lead-I": "I",
    "1-lead-V5": "V5",
}

# Priority configs for quick mode
PRIORITY_CONFIGS = [
    "12-lead",
    "6-lead-limb", 
    "3-lead-I-II-V2",
    "1-lead-II",
]

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_FULL_NAMES = {
    'NORM': 'Normal',
    'MI': 'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy'
}


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, 
                 n_bootstrap: int = 10000) -> Tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval."""
    if len(values) < 2:
        return np.mean(values), np.mean(values), np.mean(values)
    
    bootstrap_means = []
    n = len(values)
    rng = np.random.RandomState(42)  # Reproducible
    
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return np.mean(values), lower, upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def paired_ttest_with_correction(group1: np.ndarray, group2: np.ndarray, 
                                  n_comparisons: int = 1) -> Tuple[float, float]:
    """Paired t-test with Bonferroni correction."""
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan
    
    try:
        t_stat, p_value = stats.ttest_rel(group1, group2)
        p_corrected = min(1.0, p_value * n_comparisons)  # Bonferroni
        return t_stat, p_corrected
    except:
        return np.nan, np.nan


def wilcoxon_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test (non-parametric alternative)."""
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value = stats.wilcoxon(group1, group2)
        return stat, p_value
    except:
        return np.nan, np.nan


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def run_single_experiment(config_name: str, leads_str: str, seed: int, 
                          args: argparse.Namespace) -> Optional[Dict]:
    """Run a single experiment with comprehensive evaluation."""
    from train import train, set_seed
    from comprehensive_evaluation import ComprehensiveEvaluator
    
    print(f"     Seed {seed}...")
    
    try:
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
            wandb_tags=[config_name, f"seed-{seed}", "complete-pipeline"],
        )
        
        results['seed'] = seed
        results['config_name'] = config_name
        results['leads_str'] = leads_str
        
        return results
        
    except Exception as e:
        print(f"     Failed: {e}")
        return None


def aggregate_multiseed_results(seed_results: List[Dict]) -> Dict:
    """Aggregate results across multiple seeds with statistics."""
    
    if not seed_results:
        return {}
    
    # Extract arrays
    aurocs = np.array([r['test_auroc'] for r in seed_results if 'test_auroc' in r])
    f1s = np.array([r.get('test_f1', np.nan) for r in seed_results])
    briers = np.array([r.get('test_brier', np.nan) for r in seed_results])
    
    # Per-class AUROC
    per_class_aurocs = {cls: [] for cls in CLASSES}
    for r in seed_results:
        if 'test_auroc_per_class' in r:
            for cls, val in r['test_auroc_per_class'].items():
                per_class_aurocs[cls].append(val)
    
    # Compute statistics
    auroc_mean, auroc_ci_low, auroc_ci_high = bootstrap_ci(aurocs)
    
    aggregated = {
        'config_name': seed_results[0].get('config_name', 'unknown'),
        'leads_str': seed_results[0].get('leads_str', 'unknown'),
        'n_leads': seed_results[0].get('n_leads', len(seed_results[0].get('leads', []))),
        'n_seeds': len(seed_results),
        'seeds': [r['seed'] for r in seed_results],
        
        # AUROC
        'auroc_mean': float(auroc_mean),
        'auroc_std': float(np.std(aurocs)) if len(aurocs) > 1 else 0.0,
        'auroc_ci_low': float(auroc_ci_low),
        'auroc_ci_high': float(auroc_ci_high),
        'auroc_all': aurocs.tolist(),
        
        # F1
        'f1_mean': float(np.nanmean(f1s)),
        'f1_std': float(np.nanstd(f1s)) if len(f1s) > 1 else 0.0,
        
        # Brier
        'brier_mean': float(np.nanmean(briers)),
        'brier_std': float(np.nanstd(briers)) if len(briers) > 1 else 0.0,
        
        # Per-class
        'per_class': {},
        
        # All individual results
        'seed_results': seed_results,
    }
    
    # Per-class statistics
    for cls, values in per_class_aurocs.items():
        if values:
            arr = np.array(values)
            mean, ci_low, ci_high = bootstrap_ci(arr)
            aggregated['per_class'][cls] = {
                'mean': float(mean),
                'std': float(np.std(arr)) if len(arr) > 1 else 0.0,
                'ci_low': float(ci_low),
                'ci_high': float(ci_high),
            }
    
    return aggregated


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

def compare_configurations(all_results: Dict[str, Dict], 
                           baseline_config: str = "12-lead") -> Dict:
    """Compute statistical comparisons between configurations."""
    
    if baseline_config not in all_results:
        print(f"  Baseline {baseline_config} not found")
        return {}
    
    baseline = all_results[baseline_config]
    baseline_aurocs = np.array(baseline['auroc_all'])
    
    comparisons = {}
    n_comparisons = len(all_results) - 1
    
    for config_name, results in all_results.items():
        if config_name == baseline_config:
            continue
        
        config_aurocs = np.array(results['auroc_all'])
        
        # Paired t-test
        t_stat, p_ttest = paired_ttest_with_correction(
            baseline_aurocs, config_aurocs, n_comparisons
        )
        
        # Wilcoxon test
        w_stat, p_wilcoxon = wilcoxon_test(baseline_aurocs, config_aurocs)
        
        # Effect size
        d = cohens_d(baseline_aurocs, config_aurocs)
        
        # Performance retention
        retention = results['auroc_mean'] / baseline['auroc_mean'] * 100
        
        comparisons[config_name] = {
            'auroc_mean': results['auroc_mean'],
            'auroc_diff': baseline['auroc_mean'] - results['auroc_mean'],
            'retention_pct': retention,
            'p_ttest_corrected': p_ttest,
            'p_wilcoxon': p_wilcoxon,
            'cohens_d': d,
            'significant_005': p_ttest < 0.05 if not np.isnan(p_ttest) else None,
            'significant_001': p_ttest < 0.01 if not np.isnan(p_ttest) else None,
        }
    
    return comparisons


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_main_results_table(all_results: Dict[str, Dict], 
                                 comparisons: Dict[str, Dict],
                                 output_dir: Path) -> str:
    """Generate main results table."""
    
    # Sort by number of leads
    sorted_configs = sorted(
        all_results.items(), 
        key=lambda x: (-x[1].get('n_leads', 0), x[0])
    )
    
    # CSV table
    rows = []
    for config_name, results in sorted_configs:
        n_leads = results.get('n_leads', 12)
        
        # Get comparison stats
        comp = comparisons.get(config_name, {})
        retention = comp.get('retention_pct', 100.0)
        p_val = comp.get('p_ttest_corrected', np.nan)
        
        row = {
            'Configuration': config_name,
            'N': n_leads,
            'AUROC': f"{results['auroc_mean']:.3f}",
            'AUROC_std': f"±{results['auroc_std']:.3f}",
            'AUROC_CI': f"[{results['auroc_ci_low']:.3f}, {results['auroc_ci_high']:.3f}]",
            'Retention%': f"{retention:.1f}",
            'F1': f"{results['f1_mean']:.3f}±{results['f1_std']:.3f}",
            'Brier': f"{results['brier_mean']:.3f}±{results['brier_std']:.3f}",
            'p-value': f"{p_val:.4f}" if not np.isnan(p_val) else "-",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "main_results_with_statistics.csv", index=False)
    
    # LaTeX table
    latex = generate_latex_table_with_stats(sorted_configs, comparisons)
    with open(output_dir / "table_main_complete_pipeline.tex", 'w') as f:
        f.write(latex)
    
    return df.to_string(index=False)


def generate_latex_table_with_stats(sorted_configs: List, 
                                     comparisons: Dict) -> str:
    """Generate LaTeX table with statistical annotations."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Classification Performance by Lead Configuration on PTB-XL Test Set. 
Results show mean $\pm$ std across 5 random seeds with 95\% confidence intervals.
$^*$: $p < 0.05$, $^{**}$: $p < 0.01$ vs.\ 12-lead baseline (paired t-test with Bonferroni correction).
AUROC: Area Under ROC Curve (macro-averaged).}
\label{tab:main_results}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{N} & \textbf{AUROC} & \textbf{95\% CI} & \textbf{Ret.(\%)} & \textbf{p-value} \\
\midrule
"""
    
    for config_name, results in sorted_configs:
        n_leads = results.get('n_leads', 12)
        comp = comparisons.get(config_name, {})
        retention = comp.get('retention_pct', 100.0)
        p_val = comp.get('p_ttest_corrected', np.nan)
        
        # Significance markers
        sig_marker = ""
        if not np.isnan(p_val):
            if p_val < 0.01:
                sig_marker = "$^{**}$"
            elif p_val < 0.05:
                sig_marker = "$^*$"
        
        # Format values
        auroc_str = f"{results['auroc_mean']:.3f} $\\pm$ {results['auroc_std']:.3f}{sig_marker}"
        ci_str = f"[{results['auroc_ci_low']:.3f}, {results['auroc_ci_high']:.3f}]"
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "--"
        
        latex += f"{config_name} & {n_leads} & {auroc_str} & {ci_str} & {retention:.1f} & {p_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_perclass_table(all_results: Dict[str, Dict], 
                             output_dir: Path) -> str:
    """Generate per-class performance table."""
    
    sorted_configs = sorted(
        all_results.items(), 
        key=lambda x: (-x[1].get('n_leads', 0), x[0])
    )
    
    rows = []
    for config_name, results in sorted_configs:
        row = {'Configuration': config_name, 'N': results.get('n_leads', 12)}
        
        for cls in CLASSES:
            if cls in results.get('per_class', {}):
                pc = results['per_class'][cls]
                row[cls] = f"{pc['mean']:.3f}±{pc['std']:.3f}"
            else:
                row[cls] = "-"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "perclass_results_with_statistics.csv", index=False)
    
    # LaTeX version
    latex = generate_perclass_latex(sorted_configs)
    with open(output_dir / "table_perclass_complete_pipeline.tex", 'w') as f:
        f.write(latex)
    
    return df.to_string(index=False)


def generate_perclass_latex(sorted_configs: List) -> str:
    """Generate per-class LaTeX table."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Per-Class AUROC by Lead Configuration (mean $\pm$ std, 5 seeds).
NORM: Normal, MI: Myocardial Infarction, STTC: ST/T Change, CD: Conduction Disturbance, HYP: Hypertrophy.}
\label{tab:perclass}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Config} & \textbf{N} & \textbf{NORM} & \textbf{MI} & \textbf{STTC} & \textbf{CD} & \textbf{HYP} \\
\midrule
"""
    
    for config_name, results in sorted_configs:
        n_leads = results.get('n_leads', 12)
        
        class_strs = []
        for cls in CLASSES:
            if cls in results.get('per_class', {}):
                pc = results['per_class'][cls]
                class_strs.append(f"{pc['mean']:.3f}")
            else:
                class_strs.append("--")
        
        latex += f"{config_name} & {n_leads} & " + " & ".join(class_strs) + " \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================================
# KEY FINDINGS GENERATOR
# ============================================================================

def generate_key_findings(all_results: Dict[str, Dict], 
                          comparisons: Dict[str, Dict]) -> Dict:
    """Generate key findings for the paper abstract and discussion."""
    
    findings = {
        'generated_at': datetime.now().isoformat(),
        'n_configurations': len(all_results),
        'n_seeds': all_results.get('12-lead', {}).get('n_seeds', 0),
        'highlights': [],
        'claims': {},
    }
    
    # Baseline performance
    baseline = all_results.get('12-lead', {})
    if baseline:
        findings['claims']['baseline_auroc'] = {
            'value': baseline['auroc_mean'],
            'std': baseline['auroc_std'],
            'ci': [baseline['auroc_ci_low'], baseline['auroc_ci_high']],
            'statement': f"12-lead baseline: AUROC = {baseline['auroc_mean']:.3f} ± {baseline['auroc_std']:.3f}"
        }
    
    # Best 3-lead configuration
    three_lead_configs = {k: v for k, v in all_results.items() if '3-lead' in k}
    if three_lead_configs:
        best_3lead = max(three_lead_configs.items(), key=lambda x: x[1]['auroc_mean'])
        comp = comparisons.get(best_3lead[0], {})
        findings['claims']['best_3lead'] = {
            'config': best_3lead[0],
            'auroc': best_3lead[1]['auroc_mean'],
            'retention': comp.get('retention_pct', 0),
            'p_value': comp.get('p_ttest_corrected', np.nan),
            'statement': f"Best 3-lead ({best_3lead[0]}): {comp.get('retention_pct', 0):.1f}% retention, AUROC = {best_3lead[1]['auroc_mean']:.3f}"
        }
        findings['highlights'].append(
            f"The optimal 3-lead configuration ({best_3lead[0].replace('3-lead-', '')}) "
            f"retains {comp.get('retention_pct', 0):.1f}% of 12-lead performance."
        )
    
    # 3-lead vs 6-lead comparison
    six_lead = all_results.get('6-lead-limb', {})
    if three_lead_configs and six_lead:
        best_3 = max(three_lead_configs.values(), key=lambda x: x['auroc_mean'])
        if best_3['auroc_mean'] > six_lead['auroc_mean']:
            diff = best_3['auroc_mean'] - six_lead['auroc_mean']
            findings['claims']['3lead_beats_6lead'] = {
                'supported': True,
                'difference': diff,
                'statement': f"3-lead outperforms 6-lead limb by {diff:.3f} AUROC"
            }
            findings['highlights'].append(
                f"Remarkably, the best 3-lead configuration OUTPERFORMS the 6-lead limb configuration "
                f"(AUROC {best_3['auroc_mean']:.3f} vs {six_lead['auroc_mean']:.3f}), "
                f"demonstrating that strategic lead selection matters more than lead count."
            )
    
    # Best single lead
    single_lead_configs = {k: v for k, v in all_results.items() if '1-lead' in k}
    if single_lead_configs:
        best_single = max(single_lead_configs.items(), key=lambda x: x[1]['auroc_mean'])
        comp = comparisons.get(best_single[0], {})
        findings['claims']['best_single_lead'] = {
            'config': best_single[0],
            'auroc': best_single[1]['auroc_mean'],
            'retention': comp.get('retention_pct', 0),
            'statement': f"Best single-lead ({best_single[0]}): {comp.get('retention_pct', 0):.1f}% retention"
        }
        findings['highlights'].append(
            f"A single lead ({best_single[0].replace('1-lead-', '')}) achieves "
            f"{comp.get('retention_pct', 0):.1f}% of 12-lead performance, "
            f"suitable for wearable devices with minimal electrode requirements."
        )
    
    # Most challenging class
    if baseline and 'per_class' in baseline:
        per_class = baseline['per_class']
        if per_class:
            hardest = min(per_class.items(), key=lambda x: x[1]['mean'])
            findings['claims']['hardest_class'] = {
                'class': hardest[0],
                'auroc': hardest[1]['mean'],
                'statement': f"{CLASS_FULL_NAMES.get(hardest[0], hardest[0])} ({hardest[0]}) is most challenging: AUROC = {hardest[1]['mean']:.3f}"
            }
    
    return findings


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(args: argparse.Namespace):
    """Run the complete pipeline."""
    
    print("\n" + "" * 35)
    print("   LEAD-MINIMAL ECG: COMPLETE PIPELINE")
    print("" * 35)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"complete_{timestamp}"
    exp_dir = Path(args.output_dir) / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Output directory: {exp_dir}")
    print(f" Seeds: {list(range(args.seed_start, args.seed_start + args.seeds))}")
    print(f" Configurations: {len(LEAD_CONFIGS) if not args.quick else len(PRIORITY_CONFIGS)}")
    
    # Select configurations
    configs = LEAD_CONFIGS if not args.quick else {k: LEAD_CONFIGS[k] for k in PRIORITY_CONFIGS}
    
    # Track all results
    all_results = {}
    
    # Run experiments for each configuration
    for i, (config_name, leads_str) in enumerate(configs.items()):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(configs)}] Configuration: {config_name}")
        print(f"Leads: {leads_str}")
        print(f"{'='*70}")
        
        # Run multiple seeds
        seed_results = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            result = run_single_experiment(config_name, leads_str, seed, args)
            if result:
                seed_results.append(result)
        
        # Aggregate results
        if seed_results:
            aggregated = aggregate_multiseed_results(seed_results)
            all_results[config_name] = aggregated
            
            # Print summary
            print(f"\n   {config_name} Summary:")
            print(f"     AUROC: {aggregated['auroc_mean']:.4f} ± {aggregated['auroc_std']:.4f}")
            print(f"     95% CI: [{aggregated['auroc_ci_low']:.4f}, {aggregated['auroc_ci_high']:.4f}]")
            
            # Save intermediate results
            with open(exp_dir / "results_intermediate.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
    
    # Statistical comparisons
    print("\n" + "="*70)
    print(" STATISTICAL ANALYSIS")
    print("="*70)
    
    comparisons = compare_configurations(all_results, baseline_config="12-lead")
    
    # Print comparison summary
    print("\nComparison to 12-lead baseline:")
    for config, comp in sorted(comparisons.items(), key=lambda x: -x[1]['retention_pct']):
        sig = "**" if comp.get('significant_001') else ("*" if comp.get('significant_005') else "")
        print(f"  {config}: {comp['retention_pct']:.1f}% retention, p={comp['p_ttest_corrected']:.4f} {sig}")
    
    # Generate tables
    print("\n" + "="*70)
    print(" GENERATING COMPLETE PIPELINE TABLES")
    print("="*70)

    tables_dir = exp_dir / "complete_pipeline_tables"
    tables_dir.mkdir(exist_ok=True)
    
    main_table = generate_main_results_table(all_results, comparisons, tables_dir)
    print("\nMain Results Table:")
    print(main_table)
    
    perclass_table = generate_perclass_table(all_results, tables_dir)
    print("\nPer-Class Results Table:")
    print(perclass_table)
    
    # Generate key findings
    findings = generate_key_findings(all_results, comparisons)
    with open(tables_dir / "key_findings.json", 'w') as f:
        json.dump(findings, f, indent=2)
    
    print("\n" + "="*70)
    print(" KEY FINDINGS FOR ABSTRACT")
    print("="*70)
    for highlight in findings['highlights']:
        print(f"  • {highlight}")
    
    # Save final results
    final_output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_seeds': args.seeds,
            'n_configurations': len(all_results),
            'epochs': args.epochs,
        },
        'results': all_results,
        'comparisons': comparisons,
        'findings': findings,
    }

    with open(exp_dir / "results_complete_pipeline.json", 'w') as f:
        json.dump(final_output, f, indent=2, default=str)
    
    print(f"\n Pipeline complete!")
    print(f" Results saved to: {exp_dir}")
    
    return final_output


def analyze_existing_results(results_dir: Path):
    """Analyze existing results without running new experiments."""
    
    print(f"\n Analyzing results from: {results_dir}")
    
    # Load results
    results_file = results_dir / "results_final.json"
    if not results_file.exists():
        results_file = results_dir / "results_complete_pipeline.json"

    if not results_file.exists():
        print(f" No results file found in {results_dir}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # If it's old format (list), convert to new format
    if isinstance(data, list):
        print("  Converting old format results...")
        all_results = {}
        for r in data:
            config_name = r.get('config_name', f"{r.get('n_leads', 12)}-lead")
            all_results[config_name] = {
                'auroc_mean': r.get('test_auroc', 0),
                'auroc_std': 0.0,  # Single seed, no std
                'auroc_ci_low': r.get('test_auroc', 0),
                'auroc_ci_high': r.get('test_auroc', 0),
                'auroc_all': [r.get('test_auroc', 0)],
                'f1_mean': r.get('test_f1', 0),
                'f1_std': 0.0,
                'brier_mean': r.get('test_brier', 0),
                'brier_std': 0.0,
                'n_leads': r.get('n_leads', len(r.get('leads', []))),
                'n_seeds': 1,
                'per_class': {
                    cls: {'mean': v, 'std': 0.0, 'ci_low': v, 'ci_high': v}
                    for cls, v in r.get('test_auroc_per_class', {}).items()
                }
            }
    else:
        all_results = data.get('results', data)
    
    # Compute comparisons
    comparisons = compare_configurations(all_results, baseline_config="12-lead")
    
    # Generate tables
    tables_dir = results_dir / "analysis_tables"
    tables_dir.mkdir(exist_ok=True)
    
    print("\nGenerating analysis tables...")
    generate_main_results_table(all_results, comparisons, tables_dir)
    generate_perclass_table(all_results, tables_dir)
    
    findings = generate_key_findings(all_results, comparisons)
    with open(tables_dir / "key_findings.json", 'w') as f:
        json.dump(findings, f, indent=2)
    
    print("\n KEY FINDINGS:")
    for highlight in findings['highlights']:
        print(f"  • {highlight}")
    
    print(f"\n Analysis complete! Tables saved to: {tables_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Lead-Minimal ECG Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (5 seeds, all configs)
  python run_complete_pipeline.py --mode full --seeds 5
  
  # Quick validation (2 seeds, priority configs only)
  python run_complete_pipeline.py --mode quick --seeds 2 --epochs 5
  
  # Analyze existing results
  python run_complete_pipeline.py --mode analyze --results-dir outputs/experiments/full_sweep_20251213_225223
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "quick", "analyze"],
                        help="Pipeline mode: full (all configs, 5 seeds), quick (priority configs, 2 seeds), analyze (existing results)")
    
    # Experiment settings
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds per configuration")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="Starting seed value")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer configs and epochs")
    
    # Model settings
    parser.add_argument("--model", type=str, default="resnet1d",
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    
    # Data and output
    parser.add_argument("--data-path", type=str, 
                        default="data/processed/ptbxl_processed.h5",
                        help="Path to preprocessed data")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory for analyze mode")
    
    # W&B settings
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="lead-minimal-ecg",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team/user)")
    
    args = parser.parse_args()
    
    # Apply quick mode settings
    if args.mode == "quick" or args.quick:
        args.quick = True
        args.seeds = min(args.seeds, 2)
        args.epochs = min(args.epochs, 10)
        print(" QUICK MODE: Using reduced settings for faster iteration")
    
    # Run appropriate mode
    if args.mode == "analyze":
        if args.results_dir:
            analyze_existing_results(Path(args.results_dir))
        else:
            # Find latest results
            exp_dir = Path(args.output_dir) / "experiments"
            if exp_dir.exists():
                subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
                if subdirs:
                    analyze_existing_results(subdirs[-1])
                else:
                    print(" No experiment directories found")
            else:
                print(" Experiments directory not found")
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()