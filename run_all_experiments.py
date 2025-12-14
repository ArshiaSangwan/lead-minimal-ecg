#!/usr/bin/env python3
"""Lead-Minimal ECG: Full Experiment Suite
==========================================

Runs all lead configurations for the paper with comprehensive W&B logging.
Generates results tables and computes Lead-Robustness Score (LRS).

Lead Configurations (from paper):
- Single-lead: II, V2, I, V5
- 2-lead: (I, II), (II, V2)
- 3-lead: (I, II, V2), (I, II, III), (II, V2, V5)
- 6-lead: Limb leads (I, II, III, aVR, aVL, aVF)
- 12-lead: All leads (baseline)

Usage:
    python run_all_experiments.py                    # Run all configs
    python run_all_experiments.py --config II        # Run single config
    python run_all_experiments.py --quick            # Quick test (5 epochs)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def compute_lrs(test_auroc, baseline_auroc, brier_delta=0.0, alpha=0.7, beta=0.3):
    """
    Compute Lead-Robustness Score (LRS) from the paper.
    
    LRS = α * (AUROC_subset / AUROC_baseline) + β * (1 - ΔBrier/0.25)
    
    Args:
        test_auroc: AUROC of the lead subset model
        baseline_auroc: AUROC of the 12-lead baseline
        brier_delta: Calibration degradation (Brier score difference)
        alpha: Weight for discrimination term (default 0.7)
        beta: Weight for calibration term (default 0.3)
    
    Returns:
        LRS score in [0, 1], higher is better
    """
    discrimination_term = test_auroc / baseline_auroc
    calibration_term = 1 - (brier_delta / 0.25)
    calibration_term = max(0, min(1, calibration_term))  # Clamp to [0, 1]
    
    lrs = alpha * discrimination_term + beta * calibration_term
    return min(1.0, lrs)  # Cap at 1.0


def compute_brier_score(predictions, labels):
    """Compute Brier score for calibration assessment."""
    return np.mean((predictions - labels) ** 2)


def run_single_experiment(config_name, leads_str, args, wandb_group=None):
    """Run a single lead configuration experiment."""
    from train import train, set_seed
    
    print(f"\n{'='*70}")
    print(f"🔬 Running: {config_name}")
    print(f"   Leads: {leads_str}")
    print(f"{'='*70}")
    
    # Count leads
    if leads_str == "all":
        n_leads = 12
    else:
        n_leads = len(leads_str.split(","))
    
    # Build tags for W&B
    tags = [
        f"{n_leads}-lead",
        config_name,
        "paper-experiment",
        args.model,
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
        seed=args.seed,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=tags,
    )
    
    # Add config name to results
    results['config_name'] = config_name
    results['leads_str'] = leads_str
    
    return results


def run_all_experiments(args):
    """Run all lead configuration experiments."""
    
    print("\n" + "🔬" * 30)
    print("   LEAD-MINIMAL ECG: FULL EXPERIMENT SUITE")
    print("🔬" * 30)
    print(f"\nConfigurations to run: {len(LEAD_CONFIGS)}")
    for name, leads in LEAD_CONFIGS.items():
        n = 12 if leads == "all" else len(leads.split(","))
        print(f"  - {name}: {n} leads")
    
    # Create experiment output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / "experiments" / f"full_sweep_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment directory: {exp_dir}")
    
    # Track all results
    all_results = []
    
    # Run each configuration
    for i, (config_name, leads_str) in enumerate(LEAD_CONFIGS.items()):
        print(f"\n[{i+1}/{len(LEAD_CONFIGS)}] ", end="")
        
        try:
            results = run_single_experiment(
                config_name, leads_str, args,
                wandb_group=f"sweep_{timestamp}"
            )
            all_results.append(results)
            
            # Save intermediate results
            with open(exp_dir / "results_partial.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error in {config_name}: {e}")
            all_results.append({
                'config_name': config_name,
                'leads_str': leads_str,
                'error': str(e)
            })
    
    # Compute LRS scores
    print("\n" + "="*70)
    print("📊 COMPUTING LEAD-ROBUSTNESS SCORES")
    print("="*70)
    
    # Find baseline (12-lead) AUROC
    baseline_result = next((r for r in all_results if r.get('config_name') == '12-lead'), None)
    if baseline_result and 'test_auroc' in baseline_result:
        baseline_auroc = baseline_result['test_auroc']
        print(f"\n12-lead baseline AUROC: {baseline_auroc:.4f}")
        
        # Compute LRS for each config
        for result in all_results:
            if 'test_auroc' in result:
                result['lrs'] = compute_lrs(result['test_auroc'], baseline_auroc)
                print(f"  {result['config_name']}: AUROC={result['test_auroc']:.4f}, LRS={result['lrs']:.4f}")
    else:
        print("⚠️ No baseline found, skipping LRS computation")
    
    # Generate results table
    print("\n" + "="*70)
    print("📋 RESULTS TABLE (for paper)")
    print("="*70)
    
    generate_results_table(all_results, exp_dir)
    
    # Save final results
    with open(exp_dir / "results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ All experiments complete!")
    print(f"📁 Results saved to: {exp_dir}")
    
    return all_results


def generate_results_table(results, output_dir):
    """Generate publication-quality LaTeX and CSV tables for the paper."""
    
    # Filter out failed experiments
    valid_results = [r for r in results if 'test_auroc' in r]
    
    if not valid_results:
        print("No valid results to generate table")
        return
    
    # Build DataFrame
    rows = []
    for r in valid_results:
        row = {
            'Config': r['config_name'],
            'N Leads': r['n_leads'],
            'AUROC': r['test_auroc'],
            'LRS': r.get('lrs', 1.0),
            'F1': r.get('test_f1', 0),
            'Brier': r.get('test_brier', 0),
        }
        # Add per-class AUROC
        if 'test_auroc_per_class' in r:
            for cls in CLASSES:
                row[f'AUROC_{cls}'] = r['test_auroc_per_class'].get(cls, 0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('N Leads', ascending=False)
    
    # Print summary table
    print("\n" + "="*80)
    print("📊 MAIN RESULTS TABLE")
    print("="*80)
    summary_cols = ['Config', 'N Leads', 'AUROC', 'LRS', 'F1', 'Brier']
    print(df[summary_cols].to_string(index=False, float_format='%.4f'))
    
    # Print per-class table
    print("\n" + "="*80)
    print("📊 PER-CLASS AUROC")
    print("="*80)
    perclass_cols = ['Config', 'N Leads'] + [f'AUROC_{cls}' for cls in CLASSES if f'AUROC_{cls}' in df.columns]
    print(df[perclass_cols].to_string(index=False, float_format='%.4f'))
    
    # Save CSV
    df.to_csv(output_dir / "results_table.csv", index=False)
    print(f"\n📄 CSV saved: {output_dir / 'results_table.csv'}")
    
    # Generate main LaTeX table
    latex_main = generate_latex_table(df)
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_main)
    print(f"📄 LaTeX (main) saved: {output_dir / 'results_table.tex'}")
    
    # Generate per-class LaTeX table
    latex_perclass = generate_perclass_latex_table(df)
    with open(output_dir / "perclass_table.tex", 'w') as f:
        f.write(latex_perclass)
    print(f"📄 LaTeX (per-class) saved: {output_dir / 'perclass_table.tex'}")
    
    # Generate per-class heatmap data
    perclass_df = df[perclass_cols]
    perclass_df.to_csv(output_dir / "perclass_auroc.csv", index=False)
    print(f"📄 Per-class CSV saved: {output_dir / 'perclass_auroc.csv'}")
    
    # Generate summary statistics
    generate_summary_stats(df, output_dir)
    
    return df


def generate_latex_table(df):
    """Generate publication-quality LaTeX table for the main results."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Classification Performance by Lead Configuration. AUROC: Area Under the ROC Curve (macro-averaged). LRS: Lead-Robustness Score. Brier: Brier Score (lower is better). Best results per metric in \textbf{bold}.}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{N} & \textbf{AUROC} & \textbf{F1} & \textbf{Brier} & \textbf{LRS} \\
\midrule
"""
    
    # Find best values for highlighting
    best_auroc = df['AUROC'].max()
    best_f1 = df['F1'].max()
    best_lrs = df['LRS'].max()
    best_brier = df['Brier'].min()
    
    for _, row in df.iterrows():
        config = row['Config'].replace('_', '-').replace('-lead-', ': ')
        
        # Format with bold for best values
        auroc_str = f"\\textbf{{{row['AUROC']:.3f}}}" if row['AUROC'] == best_auroc else f"{row['AUROC']:.3f}"
        f1_str = f"\\textbf{{{row['F1']:.3f}}}" if row['F1'] == best_f1 else f"{row['F1']:.3f}"
        brier_str = f"\\textbf{{{row['Brier']:.4f}}}" if row['Brier'] == best_brier else f"{row['Brier']:.4f}"
        lrs_str = f"\\textbf{{{row['LRS']:.3f}}}" if row['LRS'] == best_lrs else f"{row['LRS']:.3f}"
        
        latex += f"{config} & {int(row['N Leads'])} & {auroc_str} & {f1_str} & {brier_str} & {lrs_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_perclass_latex_table(df):
    """Generate LaTeX table for per-class AUROC results."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Per-Class AUROC by Lead Configuration. NORM: Normal, MI: Myocardial Infarction, STTC: ST/T Change, CD: Conduction Disturbance, HYP: Hypertrophy.}
\label{tab:perclass_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Configuration} & \textbf{N} & \textbf{NORM} & \textbf{MI} & \textbf{STTC} & \textbf{CD} & \textbf{HYP} \\
\midrule
"""
    
    for _, row in df.iterrows():
        config = row['Config'].replace('_', '-').replace('-lead-', ': ')
        
        aurocs = []
        for cls in CLASSES:
            col = f'AUROC_{cls}'
            if col in row:
                aurocs.append(f"{row[col]:.3f}")
            else:
                aurocs.append("-")
        
        latex += f"{config} & {int(row['N Leads'])} & {' & '.join(aurocs)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_summary_stats(df, output_dir):
    """Generate summary statistics and key findings for the paper."""
    
    print("\n" + "="*80)
    print("📈 KEY FINDINGS FOR PAPER")
    print("="*80)
    
    # Get baseline
    baseline = df[df['Config'] == '12-lead']
    if len(baseline) > 0:
        baseline_auroc = baseline['AUROC'].values[0]
        print(f"\n🎯 12-lead Baseline AUROC: {baseline_auroc:.4f}")
    else:
        baseline_auroc = df['AUROC'].max()
        print(f"\n⚠️  No 12-lead baseline found, using max AUROC: {baseline_auroc:.4f}")
    
    # Performance retention analysis
    print("\n📊 Performance Retention (vs 12-lead):")
    for _, row in df.iterrows():
        retention = (row['AUROC'] / baseline_auroc) * 100
        delta = row['AUROC'] - baseline_auroc
        sign = "+" if delta >= 0 else ""
        print(f"   {row['Config']:20s}: {retention:5.1f}% ({sign}{delta:.4f})")
    
    # Best configurations by lead count
    print("\n🏆 Best Configuration per Lead Count:")
    for n in sorted(df['N Leads'].unique(), reverse=True):
        subset = df[df['N Leads'] == n]
        best = subset.loc[subset['AUROC'].idxmax()]
        print(f"   {n:2d} leads: {best['Config']:20s} (AUROC: {best['AUROC']:.4f}, LRS: {best['LRS']:.4f})")
    
    # Key statistic: 3-lead vs 12-lead
    three_lead = df[df['Config'].str.contains('3-lead')]
    if len(three_lead) > 0:
        best_3lead = three_lead.loc[three_lead['AUROC'].idxmax()]
        retention_3 = (best_3lead['AUROC'] / baseline_auroc) * 100
        print(f"\n🔑 KEY FINDING: Best 3-lead config ({best_3lead['Config']}) retains {retention_3:.1f}% of 12-lead performance")
    
    # Per-class analysis
    print("\n📋 Per-Class Performance Summary:")
    for cls in CLASSES:
        col = f'AUROC_{cls}'
        if col in df.columns:
            best_config = df.loc[df[col].idxmax(), 'Config']
            best_val = df[col].max()
            worst_config = df.loc[df[col].idxmin(), 'Config']
            worst_val = df[col].min()
            print(f"   {cls:5s}: Best={best_val:.4f} ({best_config}), Worst={worst_val:.4f} ({worst_config})")
    
    # Save summary to file
    summary = {
        'baseline_auroc': float(baseline_auroc),
        'best_overall': {
            'config': df.loc[df['AUROC'].idxmax(), 'Config'],
            'auroc': float(df['AUROC'].max()),
        },
        'best_per_lead_count': {},
        'key_findings': []
    }
    
    for n in sorted(df['N Leads'].unique(), reverse=True):
        subset = df[df['N Leads'] == n]
        best = subset.loc[subset['AUROC'].idxmax()]
        summary['best_per_lead_count'][int(n)] = {
            'config': best['Config'],
            'auroc': float(best['AUROC']),
            'lrs': float(best['LRS']),
            'retention': float((best['AUROC'] / baseline_auroc) * 100)
        }
    
    # Add key findings
    if len(three_lead) > 0:
        best_3lead = three_lead.loc[three_lead['AUROC'].idxmax()]
        summary['key_findings'].append(
            f"Best 3-lead configuration ({best_3lead['Config']}) achieves {(best_3lead['AUROC']/baseline_auroc)*100:.1f}% of 12-lead performance"
        )
    
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📄 Summary saved: {output_dir / 'summary_stats.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all lead configuration experiments for the paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py                    # Run all 11 configs
  python run_all_experiments.py --config 12-lead  # Run only 12-lead
  python run_all_experiments.py --quick           # Quick test (5 epochs)
  python run_all_experiments.py --epochs 50       # Train for 50 epochs
        """
    )
    
    # Experiment selection
    parser.add_argument("--config", type=str, default=None,
                        help=f"Run specific config only. Options: {list(LEAD_CONFIGS.keys())}")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5 epochs, no patience)")
    
    # Training params
    parser.add_argument("--model", type=str, default="resnet1d",
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per config")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Data and output
    parser.add_argument("--data_path", type=str, 
                        default="data/processed/ptbxl_processed.h5",
                        help="Path to processed data")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Output directory")
    
    # W&B
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lead-minimal-ecg",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity")
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        print("🚀 QUICK MODE: 5 epochs, patience=2")
        args.epochs = 5
        args.patience = 2
    
    # Run specific config or all
    if args.config:
        if args.config not in LEAD_CONFIGS:
            print(f"❌ Unknown config: {args.config}")
            print(f"   Available: {list(LEAD_CONFIGS.keys())}")
            sys.exit(1)
        
        # Run single config
        results = run_single_experiment(
            args.config,
            LEAD_CONFIGS[args.config],
            args
        )
        print(f"\n✅ Completed: {args.config}")
        print(f"   Test AUROC: {results.get('test_auroc', 'N/A'):.4f}")
    else:
        # Run all experiments
        run_all_experiments(args)


if __name__ == "__main__":
    main()
