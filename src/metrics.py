#!/usr/bin/env python3
"""
Custom metrics for Lead-Minimal ECG experiments.
Includes the novel Lead-Robustness Score (LRS).
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve


def compute_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute AUROC per class and macro.
    
    Args:
        y_true: Ground truth labels (N, C)
        y_pred: Predicted probabilities (N, C)
    
    Returns:
        Dictionary with per-class and macro AUROC
    """
    n_classes = y_true.shape[1]
    aurocs = []
    
    for i in range(n_classes):
        if len(np.unique(y_true[:, i])) > 1:
            aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        else:
            aurocs.append(np.nan)
    
    return {
        'per_class': aurocs,
        'macro': np.nanmean(aurocs)
    }


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute AUPRC (Average Precision) per class and macro."""
    n_classes = y_true.shape[1]
    auprcs = []
    
    for i in range(n_classes):
        if len(np.unique(y_true[:, i])) > 1:
            auprcs.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        else:
            auprcs.append(np.nan)
    
    return {
        'per_class': auprcs,
        'macro': np.nanmean(auprcs)
    }


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute F1 score per class and macro."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    n_classes = y_true.shape[1]
    f1s = []
    
    for i in range(n_classes):
        f1s.append(f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0))
    
    return {
        'per_class': f1s,
        'macro': np.mean(f1s)
    }


def compute_sensitivity_specificity(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, List[float]]:
    """Compute sensitivity (recall) and specificity per class."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    n_classes = y_true.shape[1]
    sensitivities = []
    specificities = []
    
    for i in range(n_classes):
        tp = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    return {
        'sensitivity': sensitivities,
        'specificity': specificities
    }


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute Brier score per class and macro (lower is better)."""
    n_classes = y_true.shape[1]
    briers = []
    
    for i in range(n_classes):
        briers.append(brier_score_loss(y_true[:, i], y_pred[:, i]))
    
    return {
        'per_class': briers,
        'macro': np.mean(briers)
    }


def compute_calibration_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) per class.
    
    ECE measures how well predicted probabilities match actual frequencies.
    """
    n_classes = y_true.shape[1]
    eces = []
    
    for i in range(n_classes):
        try:
            prob_true, prob_pred = calibration_curve(
                y_true[:, i], y_pred[:, i], n_bins=n_bins, strategy='uniform'
            )
            # ECE = weighted average of |accuracy - confidence| per bin
            bin_counts, _ = np.histogram(y_pred[:, i], bins=n_bins, range=(0, 1))
            weights = bin_counts / bin_counts.sum()
            
            # Match dimensions
            if len(prob_true) < n_bins:
                ece = np.sum(np.abs(prob_true - prob_pred) * weights[:len(prob_true)])
            else:
                ece = np.mean(np.abs(prob_true - prob_pred))
            eces.append(ece)
        except:
            eces.append(np.nan)
    
    return {
        'per_class': eces,
        'macro': np.nanmean(eces)
    }


def compute_lead_robustness_score(
    baseline_auroc: float,
    subset_auroc: float,
    baseline_brier: float,
    subset_brier: float,
    n_leads_baseline: int = 12,
    n_leads_subset: int = 1,
    alpha: float = 0.7,
    beta: float = 0.3
) -> float:
    """
    Compute Lead-Robustness Score (LRS).
    
    LRS measures how well a model maintains performance when using fewer leads,
    normalized by the lead reduction ratio.
    
    LRS = α * (AUROC_subset / AUROC_baseline) + β * (1 - ΔBrier / max_ΔBrier)
    
    Higher LRS = better robustness to lead reduction.
    
    Args:
        baseline_auroc: AUROC with all 12 leads
        subset_auroc: AUROC with reduced leads
        baseline_brier: Brier score with all 12 leads
        subset_brier: Brier score with reduced leads
        n_leads_baseline: Number of leads in baseline (12)
        n_leads_subset: Number of leads in subset
        alpha: Weight for AUROC retention (default 0.7)
        beta: Weight for calibration retention (default 0.3)
    
    Returns:
        Lead-Robustness Score (0-1, higher is better)
    """
    
    # AUROC retention ratio
    auroc_retention = subset_auroc / baseline_auroc if baseline_auroc > 0 else 0
    
    # Calibration degradation (Brier increase is bad, so we invert)
    # Max expected Brier degradation is 0.25 (random predictions)
    max_brier_delta = 0.25
    brier_delta = subset_brier - baseline_brier
    calibration_retention = 1 - (brier_delta / max_brier_delta)
    calibration_retention = np.clip(calibration_retention, 0, 1)
    
    # Compute LRS
    lrs = alpha * auroc_retention + beta * calibration_retention
    
    # Bonus for using fewer leads (efficiency factor)
    lead_efficiency = 1 - (n_leads_subset / n_leads_baseline)
    
    # Final LRS with efficiency bonus
    lrs_final = lrs * (1 + 0.1 * lead_efficiency)
    
    return np.clip(lrs_final, 0, 1.2)  # Allow slight bonus for efficiency


def compute_all_lead_robustness_scores(
    baseline_results: Dict,
    subset_results: Dict[str, Dict],
    classes: List[str] = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
) -> Dict:
    """
    Compute LRS for all lead subsets.
    
    Args:
        baseline_results: Results from 12-lead model
        subset_results: Dictionary mapping lead subset name to results
    
    Returns:
        Dictionary with LRS for each subset
    """
    
    lrs_results = {}
    
    baseline_auroc = baseline_results['test_auroc']
    baseline_brier = np.mean(baseline_results.get('test_brier', [0.1] * len(classes)))
    
    for subset_name, subset_res in subset_results.items():
        subset_auroc = subset_res['test_auroc']
        subset_brier = np.mean(subset_res.get('test_brier', [0.1] * len(classes)))
        n_leads = subset_res['n_leads']
        
        lrs = compute_lead_robustness_score(
            baseline_auroc=baseline_auroc,
            subset_auroc=subset_auroc,
            baseline_brier=baseline_brier,
            subset_brier=subset_brier,
            n_leads_baseline=12,
            n_leads_subset=n_leads
        )
        
        lrs_results[subset_name] = {
            'lrs': lrs,
            'n_leads': n_leads,
            'auroc': subset_auroc,
            'auroc_retention': subset_auroc / baseline_auroc if baseline_auroc > 0 else 0,
            'leads': subset_res.get('leads', [])
        }
    
    return lrs_results


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    metric: str = 'f1'
) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal threshold per class based on specified metric."""
    n_classes = y_true.shape[1]
    optimal_thresholds = []
    optimal_scores = []
    
    for i in range(n_classes):
        best_thresh = 0.5
        best_score = 0
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_binary = (y_pred[:, i] >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true[:, i], y_pred_binary, zero_division=0)
            elif metric == 'sensitivity':
                tp = np.sum((y_true[:, i] == 1) & (y_pred_binary == 1))
                fn = np.sum((y_true[:, i] == 1) & (y_pred_binary == 0))
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        optimal_thresholds.append(best_thresh)
        optimal_scores.append(best_score)
    
    return np.array(optimal_thresholds), np.array(optimal_scores)


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    
    return np.mean(bootstrap_scores), lower, upper


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulate predictions
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_pred = np.random.rand(n_samples, n_classes)
    
    print("Testing metrics computation:")
    print(f"  AUROC: {compute_auroc(y_true, y_pred)['macro']:.4f}")
    print(f"  AUPRC: {compute_auprc(y_true, y_pred)['macro']:.4f}")
    print(f"  F1: {compute_f1(y_true, y_pred)['macro']:.4f}")
    print(f"  Brier: {compute_brier_score(y_true, y_pred)['macro']:.4f}")
    print(f"  ECE: {compute_calibration_error(y_true, y_pred)['macro']:.4f}")
    
    # Test LRS
    lrs = compute_lead_robustness_score(
        baseline_auroc=0.90,
        subset_auroc=0.85,
        baseline_brier=0.10,
        subset_brier=0.12,
        n_leads_baseline=12,
        n_leads_subset=2
    )
    print(f"\n  LRS (2-lead vs 12-lead): {lrs:.4f}")
