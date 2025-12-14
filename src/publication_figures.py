#!/usr/bin/env python3
"""
Publication-Quality Visualization Module
=========================================

Generates all figures required for the paper:
- Lead configuration diagrams
- Performance comparison plots  
- AUROC heatmaps
- Training curves
- Statistical comparison plots

This module addresses the critique of missing/incomplete figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ECG lead positions on the body (approximate)
LEAD_POSITIONS = {
    # Limb leads
    'I': (0.15, 0.5),
    'II': (0.35, 0.3),
    'III': (0.55, 0.3),
    'aVR': (0.25, 0.7),
    'aVL': (0.10, 0.65),
    'aVF': (0.45, 0.15),
    # Chest leads (precordial)
    'V1': (0.60, 0.75),
    'V2': (0.65, 0.70),
    'V3': (0.72, 0.63),
    'V4': (0.80, 0.58),
    'V5': (0.85, 0.52),
    'V6': (0.88, 0.48),
}

LEAD_COLORS = {
    'I': '#1f77b4', 'II': '#ff7f0e', 'III': '#2ca02c',
    'aVR': '#d62728', 'aVL': '#9467bd', 'aVF': '#8c564b',
    'V1': '#e377c2', 'V2': '#7f7f7f', 'V3': '#bcbd22',
    'V4': '#17becf', 'V5': '#aec7e8', 'V6': '#ffbb78',
}


def plot_lead_configuration_diagram(
    leads: List[str],
    title: str = "",
    highlight: bool = True,
    save_path: Optional[Path] = None,
    figsize: Tuple = (8, 6)
) -> plt.Figure:
    """
    Plot a diagram showing which ECG leads are included.
    
    Creates a schematic showing lead positions with included leads highlighted.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw body outline (simplified torso)
    torso = plt.Circle((0.5, 0.5), 0.45, fill=False, color='gray', linewidth=2)
    ax.add_patch(torso)
    
    # Draw all leads
    for lead, (x, y) in LEAD_POSITIONS.items():
        is_selected = lead in leads
        color = LEAD_COLORS[lead] if is_selected else 'lightgray'
        alpha = 1.0 if is_selected else 0.3
        size = 1500 if is_selected else 800
        edgecolor = 'black' if is_selected else 'gray'
        
        ax.scatter([x], [y], s=size, c=color, alpha=alpha, 
                  edgecolors=edgecolor, linewidths=2, zorder=3)
        
        fontweight = 'bold' if is_selected else 'normal'
        fontcolor = 'black' if is_selected else 'gray'
        ax.annotate(lead, (x, y), ha='center', va='center', 
                   fontsize=10 if is_selected else 8,
                   fontweight=fontweight, color=fontcolor, zorder=4)
    
    # Add legend
    included_patch = mpatches.Patch(color='#1f77b4', label=f'Included ({len(leads)} leads)')
    excluded_patch = mpatches.Patch(color='lightgray', label=f'Excluded ({12-len(leads)} leads)')
    ax.legend(handles=[included_patch, excluded_patch], loc='upper left')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title or f"Lead Configuration: {', '.join(leads)}", fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    return fig


def plot_all_lead_configurations(
    configs: Dict[str, List[str]],
    save_path: Optional[Path] = None,
    ncols: int = 4
) -> plt.Figure:
    """Plot all lead configurations in a grid."""
    n_configs = len(configs)
    nrows = (n_configs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes = axes.flatten() if n_configs > 1 else [axes]
    
    for idx, (config_name, leads) in enumerate(configs.items()):
        ax = axes[idx]
        
        # Draw body outline
        torso = plt.Circle((0.5, 0.5), 0.4, fill=False, color='gray', linewidth=1.5)
        ax.add_patch(torso)
        
        # Draw leads
        for lead, (x, y) in LEAD_POSITIONS.items():
            is_selected = lead in leads
            color = LEAD_COLORS[lead] if is_selected else 'lightgray'
            alpha = 1.0 if is_selected else 0.2
            size = 600 if is_selected else 300
            
            ax.scatter([x], [y], s=size, c=color, alpha=alpha, 
                      edgecolors='gray', linewidths=1, zorder=3)
            if is_selected:
                ax.annotate(lead, (x, y), ha='center', va='center',
                           fontsize=7, fontweight='bold', zorder=4)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{config_name}\n({len(leads)} leads)", fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("ECG Lead Configurations", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    return fig


def plot_auroc_comparison_bar(
    results: Dict[str, float],
    baseline: Optional[str] = '12-lead',
    save_path: Optional[Path] = None,
    figsize: Tuple = (12, 6)
) -> plt.Figure:
    """
    Create a bar chart comparing AUROC across configurations.
    Includes error bars if std is provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Parse results
    configs = list(results.keys())
    
    # Check if we have mean±std or just values
    if isinstance(list(results.values())[0], dict):
        aurocs = [r['mean'] for r in results.values()]
        stds = [r.get('std', 0) for r in results.values()]
    else:
        aurocs = list(results.values())
        stds = [0] * len(aurocs)
    
    # Sort by AUROC
    sorted_idx = np.argsort(aurocs)[::-1]
    configs = [configs[i] for i in sorted_idx]
    aurocs = [aurocs[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    
    # Colors based on number of leads
    colors = []
    for cfg in configs:
        if '12' in cfg:
            colors.append('#2ecc71')  # Green for baseline
        elif '6' in cfg:
            colors.append('#3498db')
        elif '3' in cfg:
            colors.append('#9b59b6')
        elif '2' in cfg:
            colors.append('#e74c3c')
        else:
            colors.append('#f39c12')
    
    x = np.arange(len(configs))
    bars = ax.bar(x, aurocs, yerr=stds, color=colors, capsize=3, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, auroc, std in zip(bars, aurocs, stds):
        height = bar.get_height()
        label = f'{auroc:.3f}'
        if std > 0:
            label += f'±{std:.3f}'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height + std + 0.005),
                   ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Baseline reference line
    if baseline and baseline in configs:
        baseline_auroc = aurocs[configs.index(baseline)]
        ax.axhline(y=baseline_auroc, color='green', linestyle='--', 
                  label=f'12-lead baseline ({baseline_auroc:.3f})', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('AUROC (Macro)')
    ax.set_xlabel('Lead Configuration')
    ax.set_title('Classification Performance by Lead Configuration')
    ax.set_ylim(0.5, max(aurocs) + 0.08)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig


def plot_auroc_heatmap(
    results: Dict[str, Dict[str, float]],
    classes: List[str] = ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
    save_path: Optional[Path] = None,
    figsize: Tuple = (10, 8)
) -> plt.Figure:
    """
    Create a heatmap of per-class AUROC across configurations.
    
    Args:
        results: Dict mapping config name to {class: auroc}
    """
    configs = list(results.keys())
    
    # Build matrix
    matrix = []
    for config in configs:
        row = [results[config].get(cls, 0) for cls in classes]
        matrix.append(row)
    matrix = np.array(matrix)
    
    # Sort by macro AUROC
    macro_aurocs = np.mean(matrix, axis=1)
    sorted_idx = np.argsort(macro_aurocs)[::-1]
    matrix = matrix[sorted_idx]
    configs = [configs[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('AUROC', rotation=-90, va='bottom')
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(configs)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if matrix[i, j] < 0.8 else 'black')
    
    ax.set_title('Per-Class AUROC by Lead Configuration', fontsize=14)
    ax.set_xlabel('Diagnostic Class')
    ax.set_ylabel('Lead Configuration')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig


def plot_performance_vs_leads(
    results: List[Dict],
    metric: str = 'auroc_mean',
    save_path: Optional[Path] = None,
    figsize: Tuple = (10, 6)
) -> plt.Figure:
    """
    Scatter plot of performance vs number of leads.
    Shows diminishing returns and optimal configurations.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    n_leads = [r['n_leads'] for r in results]
    metrics = [r[metric] for r in results]
    configs = [r['config_name'] for r in results]
    
    # Add jitter to x for overlapping points
    jitter = np.random.uniform(-0.15, 0.15, len(n_leads))
    x = np.array(n_leads) + jitter
    
    # Color by lead count
    cmap = plt.cm.viridis
    colors = [cmap(n/12) for n in n_leads]
    
    scatter = ax.scatter(x, metrics, c=colors, s=150, edgecolors='black', 
                        linewidths=1, alpha=0.8, zorder=3)
    
    # Add labels
    for xi, yi, cfg in zip(x, metrics, configs):
        # Shorten config name
        short = cfg.replace('-lead-', ':').replace('-lead', '')
        ax.annotate(short, (xi, yi), textcoords='offset points',
                   xytext=(5, 5), fontsize=8, alpha=0.7)
    
    # Trend line (best performance at each lead count)
    unique_leads = sorted(set(n_leads))
    best_at_leads = []
    for n in unique_leads:
        vals = [m for m, nl in zip(metrics, n_leads) if nl == n]
        best_at_leads.append(max(vals))
    ax.plot(unique_leads, best_at_leads, 'r--', linewidth=2, 
           label='Best at each lead count', alpha=0.7)
    
    # Reference lines
    baseline = max(metrics)
    ax.axhline(y=baseline, color='green', linestyle=':', 
              label=f'Best overall ({baseline:.4f})', alpha=0.7)
    ax.axhline(y=0.95 * baseline, color='orange', linestyle=':',
              label=f'95% of best ({0.95*baseline:.4f})', alpha=0.7)
    
    ax.set_xlabel('Number of ECG Leads')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.set_title('Performance vs. Lead Reduction')
    ax.set_xticks(unique_leads)
    ax.legend(loc='lower right')
    ax.set_xlim(0.5, 13)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig


def plot_lrs_analysis(
    results: List[Dict],
    save_path: Optional[Path] = None,
    figsize: Tuple = (12, 5)
) -> plt.Figure:
    """Plot Lead-Robustness Score analysis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Sort by LRS
    sorted_results = sorted(results, key=lambda x: x.get('lrs', 0), reverse=True)
    
    # Left: LRS bar chart
    ax1 = axes[0]
    configs = [r['config_name'] for r in sorted_results]
    lrs_values = [r.get('lrs', 0) for r in sorted_results]
    n_leads = [r['n_leads'] for r in sorted_results]
    
    colors = plt.cm.viridis(np.array(n_leads) / 12)
    bars = ax1.barh(range(len(configs)), lrs_values, color=colors)
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels(configs)
    ax1.set_xlabel('Lead-Robustness Score (LRS)')
    ax1.set_title('LRS Ranking')
    ax1.set_xlim(0, 1.1)
    ax1.invert_yaxis()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, 12))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Number of Leads')
    
    # Right: Efficiency frontier
    ax2 = axes[1]
    aurocs = [r.get('auroc_mean', r.get('test_auroc', 0)) for r in sorted_results]
    
    ax2.scatter(n_leads, aurocs, c=lrs_values, cmap='RdYlGn', s=150, 
               edgecolors='black', vmin=0.7, vmax=1.0)
    
    # Pareto frontier
    frontier_x, frontier_y = [], []
    for n in sorted(set(n_leads)):
        best_auroc = max(a for a, nl in zip(aurocs, n_leads) if nl == n)
        frontier_x.append(n)
        frontier_y.append(best_auroc)
    ax2.plot(frontier_x, frontier_y, 'r--', linewidth=2, label='Pareto frontier')
    
    ax2.set_xlabel('Number of Leads')
    ax2.set_ylabel('AUROC')
    ax2.set_title('Efficiency Frontier')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: Tuple = (12, 4)
) -> plt.Figure:
    """Plot training curves (loss, AUROC, LR)."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    
    # AUROC
    ax = axes[1]
    if 'train_auroc' in history:
        ax.plot(epochs, history['train_auroc'], label='Train', linewidth=2)
    if 'val_auroc' in history:
        ax.plot(epochs, history['val_auroc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC')
    ax.legend()
    
    # Learning rate
    ax = axes[2]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], linewidth=2, color='green')
        ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig


def plot_class_performance_radar(
    results: Dict[str, Dict[str, float]],
    classes: List[str] = ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
    save_path: Optional[Path] = None,
    figsize: Tuple = (10, 8)
) -> plt.Figure:
    """Radar chart comparing class performance across configurations."""
    
    configs = list(results.keys())[:6]  # Limit to 6 for readability
    
    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    for config, color in zip(configs, colors):
        values = [results[config].get(cls, 0) for cls in classes]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=config, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes)
    ax.set_ylim(0.6, 1.0)
    ax.set_title('Per-Class Performance Comparison', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig


def generate_all_figures(
    results_path: Path,
    output_dir: Path
):
    """Generate all publication figures from results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # If it's the new multiseed format
    if 'results' in data:
        results = data['results']
    else:
        results = data
    
    # Lead configurations
    configs = {}
    for r in results:
        if 'leads_str' in r:
            leads_str = r['leads_str']
            if leads_str == 'all':
                leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            else:
                leads = leads_str.split(',')
            configs[r['config_name']] = leads
    
    if configs:
        plot_all_lead_configurations(configs, output_dir / 'lead_configurations.png')
        print(f" Lead configurations: {output_dir / 'lead_configurations.png'}")
    
    # AUROC comparison
    if results and 'test_auroc' in results[0] or 'auroc_mean' in results[0]:
        auroc_dict = {}
        for r in results:
            if 'auroc_mean' in r:
                auroc_dict[r['config_name']] = {'mean': r['auroc_mean'], 'std': r.get('auroc_std', 0)}
            elif 'test_auroc' in r:
                auroc_dict[r['config_name']] = r['test_auroc']
        
        plot_auroc_comparison_bar(auroc_dict, save_path=output_dir / 'auroc_comparison.png')
        print(f" AUROC comparison: {output_dir / 'auroc_comparison.png'}")
    
    # Per-class heatmap
    perclass_results = {}
    for r in results:
        if 'auroc_per_class_mean' in r:
            perclass_results[r['config_name']] = r['auroc_per_class_mean']
        elif 'test_auroc_per_class' in r:
            perclass_results[r['config_name']] = r['test_auroc_per_class']
    
    if perclass_results:
        plot_auroc_heatmap(perclass_results, save_path=output_dir / 'auroc_heatmap.png')
        print(f" AUROC heatmap: {output_dir / 'auroc_heatmap.png'}")
    
    # Performance vs leads
    valid_results = [r for r in results if 'n_leads' in r and ('auroc_mean' in r or 'test_auroc' in r)]
    if valid_results:
        # Normalize keys
        for r in valid_results:
            if 'auroc_mean' not in r and 'test_auroc' in r:
                r['auroc_mean'] = r['test_auroc']
        
        plot_performance_vs_leads(valid_results, save_path=output_dir / 'performance_vs_leads.png')
        print(f" Performance vs leads: {output_dir / 'performance_vs_leads.png'}")
    
    # LRS analysis
    lrs_results = [r for r in results if 'lrs' in r or 'test_auroc' in r]
    if lrs_results:
        # Add LRS if missing
        baseline = next((r for r in lrs_results if r.get('config_name') == '12-lead'), None)
        if baseline:
            baseline_auroc = baseline.get('auroc_mean', baseline.get('test_auroc', 0.9))
            for r in lrs_results:
                if 'lrs' not in r:
                    auroc = r.get('auroc_mean', r.get('test_auroc', 0))
                    r['lrs'] = auroc / baseline_auroc if baseline_auroc > 0 else 0
        
        plot_lrs_analysis(lrs_results, save_path=output_dir / 'lrs_analysis.png')
        print(f" LRS analysis: {output_dir / 'lrs_analysis.png'}")
    
    # Class performance radar
    if perclass_results:
        plot_class_performance_radar(perclass_results, save_path=output_dir / 'class_radar.png')
        print(f" Class radar: {output_dir / 'class_radar.png'}")
    
    print(f"\n All figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output", type=str, default="figures/", help="Output directory")
    
    args = parser.parse_args()
    
    generate_all_figures(Path(args.results), Path(args.output))
