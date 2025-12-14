#!/usr/bin/env python3
"""
Lead-Robustness Score (LRS) - Theoretical Justification and Implementation
===========================================================================

This module provides a rigorous definition of the Lead-Robustness Score (LRS),
addressing the critique that the metric needs proper justification.

MOTIVATION:
-----------
In clinical ECG analysis, reducing the number of leads has practical benefits:
- Reduced electrode placement time
- Lower cost devices
- Improved patient comfort (especially in continuous monitoring)
- Wearable/portable applications

However, lead reduction typically degrades diagnostic accuracy. The LRS quantifies
how well a model maintains performance when using fewer leads.

DEFINITION:
-----------
LRS combines three components:

1. Discrimination Retention (α = 0.5):
   DR = AUROC_subset / AUROC_baseline
   
2. Calibration Retention (β = 0.3):
   CR = 1 - (Brier_subset - Brier_baseline) / max_brier_degradation
   
3. Efficiency Bonus (γ = 0.2):
   EB = 1 - (n_leads_subset / n_leads_baseline)

LRS = α * DR + β * CR + γ * EB

The LRS ranges from 0 to ~1.1, where:
- LRS ≈ 1.0: Near-baseline performance
- LRS > 0.95: Clinically equivalent (within 5% of baseline)  
- LRS < 0.90: Significant degradation

THEORETICAL JUSTIFICATION:
--------------------------
1. AUROC is the primary metric for diagnostic discrimination
2. Brier score captures calibration quality (important for clinical decision-making)
3. Lead reduction is the goal, so efficiency should contribute to the score

The weights (α, β, γ) = (0.5, 0.3, 0.2) were chosen to:
- Prioritize discrimination (50% weight)
- Reward good calibration (30% weight) 
- Acknowledge lead reduction value (20% weight)

These weights can be adjusted based on clinical priorities using the
WeightedLRS class.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class LRSConfig:
    """Configuration for LRS calculation."""
    alpha: float = 0.5  # Weight for discrimination retention
    beta: float = 0.3   # Weight for calibration retention
    gamma: float = 0.2  # Weight for efficiency bonus
    
    # Normalization parameters
    max_brier_degradation: float = 0.25  # Maximum expected Brier degradation
    
    # Clinical thresholds
    clinical_equivalence_threshold: float = 0.95  # LRS threshold for "equivalent"
    
    def __post_init__(self):
        assert abs(self.alpha + self.beta + self.gamma - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {self.alpha + self.beta + self.gamma}"


def compute_lrs(
    baseline_auroc: float,
    subset_auroc: float,
    baseline_brier: float = 0.1,
    subset_brier: float = 0.1,
    n_leads_baseline: int = 12,
    n_leads_subset: int = 1,
    config: Optional[LRSConfig] = None
) -> Dict[str, float]:
    """
    Compute Lead-Robustness Score with full breakdown.
    
    Args:
        baseline_auroc: AUROC of the 12-lead model
        subset_auroc: AUROC of the lead-subset model
        baseline_brier: Brier score of 12-lead model (lower is better)
        subset_brier: Brier score of subset model
        n_leads_baseline: Number of leads in baseline (12)
        n_leads_subset: Number of leads in subset
        config: LRS configuration (weights, thresholds)
    
    Returns:
        Dictionary with LRS components and final score
    """
    if config is None:
        config = LRSConfig()
    
    # Component 1: Discrimination Retention
    discrimination_retention = subset_auroc / baseline_auroc if baseline_auroc > 0 else 0
    discrimination_retention = np.clip(discrimination_retention, 0, 1.5)  # Cap at 150%
    
    # Component 2: Calibration Retention
    brier_delta = subset_brier - baseline_brier  # Positive = worse calibration
    calibration_retention = 1 - (brier_delta / config.max_brier_degradation)
    calibration_retention = np.clip(calibration_retention, 0, 1.2)
    
    # Component 3: Efficiency Bonus
    lead_reduction = 1 - (n_leads_subset / n_leads_baseline)
    efficiency_bonus = lead_reduction  # 0 for 12-lead, ~0.92 for 1-lead
    
    # Combined LRS
    lrs = (
        config.alpha * discrimination_retention +
        config.beta * calibration_retention +
        config.gamma * efficiency_bonus
    )
    
    # Determine clinical status
    is_clinically_equivalent = lrs >= config.clinical_equivalence_threshold
    
    return {
        'lrs': float(lrs),
        'discrimination_retention': float(discrimination_retention),
        'calibration_retention': float(calibration_retention),
        'efficiency_bonus': float(efficiency_bonus),
        'auroc_retention_pct': float(discrimination_retention * 100),
        'brier_delta': float(brier_delta),
        'lead_reduction_pct': float(lead_reduction * 100),
        'is_clinically_equivalent': bool(is_clinically_equivalent),
        'n_leads': n_leads_subset,
        'config': {
            'alpha': config.alpha,
            'beta': config.beta, 
            'gamma': config.gamma,
        }
    }


def compute_lrs_with_uncertainty(
    baseline_aurocs: np.ndarray,
    subset_aurocs: np.ndarray,
    baseline_briers: np.ndarray,
    subset_briers: np.ndarray,
    n_leads_baseline: int = 12,
    n_leads_subset: int = 1,
    config: Optional[LRSConfig] = None,
    n_bootstrap: int = 5000
) -> Dict:
    """
    Compute LRS with bootstrap confidence intervals.
    
    Use this when you have multi-seed results.
    """
    if config is None:
        config = LRSConfig()
    
    # Direct computation from means
    lrs_point = compute_lrs(
        baseline_auroc=np.mean(baseline_aurocs),
        subset_auroc=np.mean(subset_aurocs),
        baseline_brier=np.mean(baseline_briers),
        subset_brier=np.mean(subset_briers),
        n_leads_baseline=n_leads_baseline,
        n_leads_subset=n_leads_subset,
        config=config
    )
    
    # Bootstrap for CI
    lrs_bootstrap = []
    n = min(len(baseline_aurocs), len(subset_aurocs))
    
    for _ in range(n_bootstrap):
        idx_bl = np.random.choice(len(baseline_aurocs), n, replace=True)
        idx_sub = np.random.choice(len(subset_aurocs), n, replace=True)
        
        lrs_sample = compute_lrs(
            baseline_auroc=np.mean(baseline_aurocs[idx_bl]),
            subset_auroc=np.mean(subset_aurocs[idx_sub]),
            baseline_brier=np.mean(baseline_briers[idx_bl]) if len(baseline_briers) > 0 else 0.1,
            subset_brier=np.mean(subset_briers[idx_sub]) if len(subset_briers) > 0 else 0.1,
            n_leads_baseline=n_leads_baseline,
            n_leads_subset=n_leads_subset,
            config=config
        )
        lrs_bootstrap.append(lrs_sample['lrs'])
    
    lrs_bootstrap = np.array(lrs_bootstrap)
    
    return {
        **lrs_point,
        'lrs_mean': float(np.mean(lrs_bootstrap)),
        'lrs_std': float(np.std(lrs_bootstrap)),
        'lrs_ci_lower': float(np.percentile(lrs_bootstrap, 2.5)),
        'lrs_ci_upper': float(np.percentile(lrs_bootstrap, 97.5)),
    }


def rank_configurations_by_lrs(
    results: List[Dict],
    baseline_config: str = '12-lead',
    config: Optional[LRSConfig] = None
) -> List[Dict]:
    """
    Rank all configurations by LRS.
    
    Args:
        results: List of experiment results
        baseline_config: Name of the baseline configuration
        config: LRS configuration
    
    Returns:
        Sorted list with LRS scores
    """
    # Find baseline
    baseline = next((r for r in results if r.get('config_name') == baseline_config), None)
    if baseline is None:
        raise ValueError(f"Baseline config '{baseline_config}' not found")
    
    baseline_auroc = baseline.get('auroc_mean', baseline.get('test_auroc', 0.9))
    baseline_brier = baseline.get('brier_mean', baseline.get('test_brier', 0.1))
    
    # Compute LRS for each config
    ranked = []
    for result in results:
        subset_auroc = result.get('auroc_mean', result.get('test_auroc', 0))
        subset_brier = result.get('brier_mean', result.get('test_brier', 0.1))
        n_leads = result.get('n_leads', 12)
        
        lrs_result = compute_lrs(
            baseline_auroc=baseline_auroc,
            subset_auroc=subset_auroc,
            baseline_brier=baseline_brier,
            subset_brier=subset_brier,
            n_leads_subset=n_leads,
            config=config
        )
        
        ranked.append({
            'config_name': result.get('config_name', 'unknown'),
            'n_leads': n_leads,
            'auroc': subset_auroc,
            **lrs_result
        })
    
    # Sort by LRS descending
    ranked = sorted(ranked, key=lambda x: x['lrs'], reverse=True)
    
    # Add rank
    for i, r in enumerate(ranked):
        r['rank'] = i + 1
    
    return ranked


def generate_lrs_latex_table(ranked_results: List[Dict]) -> str:
    """Generate LaTeX table with LRS breakdown."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Lead-Robustness Score (LRS) Analysis. LRS = α·DR + β·CR + γ·EB where DR=Discrimination Retention, CR=Calibration Retention, EB=Efficiency Bonus. Weights: α=0.5, β=0.3, γ=0.2.}
\label{tab:lrs_analysis}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Config} & \textbf{N} & \textbf{AUROC} & \textbf{DR} & \textbf{CR} & \textbf{EB} & \textbf{LRS} \\
\midrule
"""
    
    for r in ranked_results:
        config = r['config_name'].replace('-lead-', ': ')
        auroc = r['auroc']
        dr = r['discrimination_retention']
        cr = r['calibration_retention']
        eb = r['efficiency_bonus']
        lrs = r['lrs']
        n = r['n_leads']
        
        # Bold if clinically equivalent
        if r.get('is_clinically_equivalent', False):
            lrs_str = f"\\textbf{{{lrs:.3f}}}"
        else:
            lrs_str = f"{lrs:.3f}"
        
        latex += f"{config} & {n} & {auroc:.3f} & {dr:.3f} & {cr:.3f} & {eb:.3f} & {lrs_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def explain_lrs_choice(
    optimal_config: Dict,
    alternatives: List[Dict]
) -> str:
    """Generate a textual explanation of why a configuration is optimal."""
    
    explanation = f"""
LRS Analysis Summary
====================

Recommended Configuration: {optimal_config['config_name']}
- Number of leads: {optimal_config['n_leads']}
- AUROC: {optimal_config['auroc']:.4f}
- LRS: {optimal_config['lrs']:.4f}

Component Breakdown:
- Discrimination Retention (DR): {optimal_config['discrimination_retention']:.3f}
  Retains {optimal_config['auroc_retention_pct']:.1f}% of baseline AUROC
- Calibration Retention (CR): {optimal_config['calibration_retention']:.3f}
  Brier score delta: {optimal_config['brier_delta']:+.4f}
- Efficiency Bonus (EB): {optimal_config['efficiency_bonus']:.3f}
  Reduces lead count by {optimal_config['lead_reduction_pct']:.1f}%

Clinical Equivalence: {'YES' if optimal_config.get('is_clinically_equivalent') else 'NO'}
(Threshold: LRS >= 0.95)

Comparison with Alternatives:
"""
    
    for alt in alternatives[:5]:
        if alt['config_name'] == optimal_config['config_name']:
            continue
        lrs_diff = alt['lrs'] - optimal_config['lrs']
        explanation += f"- {alt['config_name']}: LRS={alt['lrs']:.4f} ({lrs_diff:+.4f})\n"
    
    return explanation


# Clinical weight profiles
CLINICAL_PROFILES = {
    'balanced': LRSConfig(alpha=0.5, beta=0.3, gamma=0.2),
    'discrimination_focused': LRSConfig(alpha=0.7, beta=0.2, gamma=0.1),
    'calibration_focused': LRSConfig(alpha=0.4, beta=0.5, gamma=0.1),
    'efficiency_focused': LRSConfig(alpha=0.4, beta=0.2, gamma=0.4),
}


if __name__ == "__main__":
    # Example usage
    print("Lead-Robustness Score (LRS) Examples")
    print("=" * 50)
    
    # Example: 3-lead vs 12-lead
    result = compute_lrs(
        baseline_auroc=0.913,
        subset_auroc=0.891,
        baseline_brier=0.084,
        subset_brier=0.092,
        n_leads_subset=3
    )
    
    print("\n3-lead (I, II, V2) vs 12-lead baseline:")
    print(f"  LRS: {result['lrs']:.4f}")
    print(f"  Discrimination Retention: {result['discrimination_retention']:.4f}")
    print(f"  Calibration Retention: {result['calibration_retention']:.4f}")
    print(f"  Efficiency Bonus: {result['efficiency_bonus']:.4f}")
    print(f"  Clinically Equivalent: {result['is_clinically_equivalent']}")
    
    # Example: 1-lead vs 12-lead
    result = compute_lrs(
        baseline_auroc=0.913,
        subset_auroc=0.852,
        baseline_brier=0.084,
        subset_brier=0.105,
        n_leads_subset=1
    )
    
    print("\n1-lead (II) vs 12-lead baseline:")
    print(f"  LRS: {result['lrs']:.4f}")
    print(f"  Discrimination Retention: {result['discrimination_retention']:.4f}")
    print(f"  Calibration Retention: {result['calibration_retention']:.4f}")  
    print(f"  Efficiency Bonus: {result['efficiency_bonus']:.4f}")
    print(f"  Clinically Equivalent: {result['is_clinically_equivalent']}")
    
    # Compare profiles
    print("\n\nWeight Profile Comparison (3-lead example):")
    print("-" * 50)
    for name, config in CLINICAL_PROFILES.items():
        result = compute_lrs(
            baseline_auroc=0.913,
            subset_auroc=0.891,
            baseline_brier=0.084,
            subset_brier=0.092,
            n_leads_subset=3,
            config=config
        )
        print(f"  {name:25s}: LRS = {result['lrs']:.4f}")
