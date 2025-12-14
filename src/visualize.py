#!/usr/bin/env python3
"""
Visualization for Lead-Minimal ECG experiments.
Generates publication-quality figures.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_NAMES = {
    'NORM': 'Normal',
    'MI': 'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy'
}

# Color palette
COLORS = {
    '12-lead': '#2ecc71',
    '6-lead': '#3498db',
    '3-lead': '#9b59b6',
    '2-lead': '#e74c3c',
    '1-lead': '#f39c12'
}


def get_color(n_leads: int) -> str:
    """Get color based on number of leads."""
    if n_leads >= 12:
        return COLORS['12-lead']
    elif n_leads >= 6:
        return COLORS['6-lead']
    elif n_leads >= 3:
        return COLORS['3-lead']
    elif n_leads >= 2:
        return COLORS['2-lead']
    else:
        return COLORS['1-lead']


def plot_auroc_comparison(results: Dict, output_path: Path):
    """
    Plot AUROC comparison across lead configurations.
    
    Figure 1 in paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by number of leads
    sorted_results = sorted(results.items(), key=lambda x: x[1]['n_leads'], reverse=True)
    
    # Left: Bar chart of macro AUROC
    ax = axes[0]
    names = []
    aurocs = []
    colors = []
    
    for name, res in sorted_results:
        names.append(name.replace('-lead', '\nlead').replace('(', '\n('))
        aurocs.append(res['test_auroc'])
        colors.append(get_color(res['n_leads']))
    
    bars = ax.bar(range(len(names)), aurocs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Macro AUROC')
    ax.set_title('(A) Macro AUROC by Lead Configuration')
    ax.set_ylim(0.5, 1.0)
    
    # Add value labels
    for bar, auroc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auroc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Right: Per-class AUROC heatmap
    ax = axes[1]
    
    # Create matrix
    matrix = []
    row_labels = []
    for name, res in sorted_results:
        row = [res['test_auroc_per_class'].get(cls, 0) for cls in CLASSES]
        matrix.append(row)
        row_labels.append(name)
    
    matrix = np.array(matrix)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASSES], rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('(B) Per-Class AUROC by Lead Configuration')
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(CLASSES)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if matrix[i, j] < 0.7 else 'black')
    
    plt.colorbar(im, ax=ax, label='AUROC')
    
    plt.tight_layout()
    plt.savefig(output_path / 'fig1_auroc_comparison.png')
    plt.savefig(output_path / 'fig1_auroc_comparison.pdf')
    plt.close()
    
    print(f"  Saved fig1_auroc_comparison")


def plot_lrs_chart(results: Dict, output_path: Path):
    """
    Plot Lead-Robustness Score (LRS) chart.
    
    Figure 2 in paper.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract LRS data
    data = []
    for name, res in results.items():
        if 'lrs' in res:
            data.append({
                'name': name,
                'n_leads': res['n_leads'],
                'lrs': res['lrs'],
                'auroc_retention': res['test_auroc'] / max(r['test_auroc'] for r in results.values() if r['n_leads'] == 12)
            })
    
    if not data:
        print("  No LRS data available")
        return
    
    # Sort by LRS
    data = sorted(data, key=lambda x: x['lrs'], reverse=True)
    
    names = [d['name'] for d in data]
    lrs_values = [d['lrs'] for d in data]
    colors = [get_color(d['n_leads']) for d in data]
    
    bars = ax.barh(range(len(names)), lrs_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Lead-Robustness Score (LRS)')
    ax.set_title('Lead-Robustness Score (LRS) Ranking')
    ax.set_xlim(0, 1.2)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline')
    
    # Add value labels
    for bar, lrs in zip(bars, lrs_values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{lrs:.3f}', va='center', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['12-lead'], label='12-lead'),
        mpatches.Patch(color=COLORS['6-lead'], label='6-lead'),
        mpatches.Patch(color=COLORS['3-lead'], label='3-lead'),
        mpatches.Patch(color=COLORS['2-lead'], label='2-lead'),
        mpatches.Patch(color=COLORS['1-lead'], label='1-lead'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path / 'fig2_lrs_ranking.png')
    plt.savefig(output_path / 'fig2_lrs_ranking.pdf')
    plt.close()
    
    print(f"  Saved fig2_lrs_ranking")


def plot_roc_curves(results: Dict, output_path: Path):
    """
    Plot ROC curves comparing lead configurations.
    
    Figure 3 in paper.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Sort by number of leads for consistent ordering
    sorted_results = sorted(results.items(), key=lambda x: x[1]['n_leads'], reverse=True)
    
    # Plot for each class
    for idx, cls in enumerate(CLASSES):
        ax = axes[idx]
        
        for name, res in sorted_results:
            if 'predictions' not in res or 'labels' not in res:
                continue
            
            class_idx = CLASSES.index(cls)
            y_true = res['labels'][:, class_idx]
            y_pred = res['predictions'][:, class_idx]
            
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auroc = res['test_auroc_per_class'].get(cls, 0)
            
            color = get_color(res['n_leads'])
            ax.plot(fpr, tpr, color=color, linewidth=2, 
                    label=f"{name} (AUC={auroc:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{CLASS_NAMES[cls]}')
        ax.legend(loc='lower right', fontsize=8)
    
    # Hide empty subplot
    axes[5].axis('off')
    
    plt.suptitle('ROC Curves by Diagnostic Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'fig3_roc_curves.png')
    plt.savefig(output_path / 'fig3_roc_curves.pdf')
    plt.close()
    
    print(f"  Saved fig3_roc_curves")


def plot_calibration_curves(results: Dict, output_path: Path):
    """
    Plot calibration curves.
    
    Supplementary figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sort by number of leads
    sorted_results = sorted(results.items(), key=lambda x: x[1]['n_leads'], reverse=True)
    
    # Plot calibration curves (aggregate over all classes)
    ax = axes[0]
    
    for name, res in sorted_results:
        if 'predictions' not in res or 'labels' not in res:
            continue
        
        # Flatten predictions and labels
        y_true = res['labels'].flatten()
        y_pred = res['predictions'].flatten()
        
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
            color = get_color(res['n_leads'])
            ax.plot(prob_pred, prob_true, marker='o', color=color, linewidth=2, label=name)
        except:
            continue
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('(A) Calibration Curves')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Plot Brier scores
    ax = axes[1]
    
    names = []
    briers = []
    colors = []
    
    for name, res in sorted_results:
        if 'test_brier_macro' in res:
            names.append(name)
            briers.append(res['test_brier_macro'])
            colors.append(get_color(res['n_leads']))
    
    if briers:
        bars = ax.bar(range(len(names)), briers, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Brier Score (lower is better)')
        ax.set_title('(B) Calibration Error (Brier Score)')
        
        for bar, brier in zip(bars, briers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{brier:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'fig_supp_calibration.png')
    plt.savefig(output_path / 'fig_supp_calibration.pdf')
    plt.close()
    
    print(f"  Saved fig_supp_calibration")


def plot_lead_importance(output_path: Path):
    """
    Plot lead importance summary.
    
    Shows which leads are most valuable for each diagnostic class.
    """
    # This would require additional experiments (single-lead evaluations)
    print("  Lead importance plot requires single-lead experiments")


def generate_all_figures(output_dir: str = "outputs/"):
    """Generate all publication figures."""
    
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation summary
    summary_path = output_path / "evaluation_summary.json"
    
    if not summary_path.exists():
        print("No evaluation summary found. Run evaluate.py first.")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load predictions for each model
    models_dir = output_path / "models"
    results = {}
    
    for model_name, model_data in summary['models'].items():
        model_dir = Path(model_data['model_dir'])
        pred_path = model_dir / "test_predictions.npz"
        
        results[model_name] = model_data.copy()
        
        if pred_path.exists():
            preds = np.load(pred_path)
            results[model_name]['predictions'] = preds['predictions']
            results[model_name]['labels'] = preds['labels']
    
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Generate figures
    plot_auroc_comparison(results, figures_dir)
    plot_lrs_chart(results, figures_dir)
    
    # Only plot ROC if we have predictions
    has_predictions = any('predictions' in r for r in results.values())
    if has_predictions:
        plot_roc_curves(results, figures_dir)
        plot_calibration_curves(results, figures_dir)
    
    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Directory containing evaluation results")
    
    args = parser.parse_args()
    
    generate_all_figures(args.output_dir)
