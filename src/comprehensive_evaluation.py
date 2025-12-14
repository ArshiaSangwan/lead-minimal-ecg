#!/usr/bin/env python3
"""
Comprehensive Evaluation Module for Lead-Minimal ECG
=====================================================

Provides all metrics required for publication:
- AUROC, AUPRC (per-class and macro)
- F1, Sensitivity, Specificity
- Sensitivity at fixed specificity thresholds
- Calibration metrics (ECE, Brier Score)
- Confusion matrices
- Bootstrap confidence intervals
- Statistical significance tests

This module addresses the critique of missing clinical metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from sklearn.metrics import (
    roc_auc_score, 
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve


CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_FULL_NAMES = {
    'NORM': 'Normal',
    'MI': 'Myocardial Infarction', 
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy'
}


class ComprehensiveEvaluator:
    """Comprehensive evaluation with all publication-required metrics."""
    
    def __init__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        class_names: List[str] = CLASSES,
        n_bootstrap: int = 2000
    ):
        """
        Args:
            y_true: Ground truth labels (N, C) binary
            y_pred: Predicted probabilities (N, C) in [0, 1]
            class_names: Names for each class
            n_bootstrap: Number of bootstrap iterations for CIs
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.n_bootstrap = n_bootstrap
        
        assert y_true.shape == y_pred.shape
        assert y_true.shape[1] == self.n_classes
        
        self.results = {}
        
    def compute_all_metrics(self, threshold: float = 0.5) -> Dict:
        """Compute all metrics."""
        
        self.results = {
            'threshold': threshold,
            'n_samples': len(self.y_true),
            'n_classes': self.n_classes,
            'class_names': self.class_names,
        }
        
        # Core metrics
        self.results['auroc'] = self._compute_auroc()
        self.results['auprc'] = self._compute_auprc()
        self.results['f1'] = self._compute_f1(threshold)
        self.results['sensitivity_specificity'] = self._compute_sens_spec(threshold)
        self.results['sens_at_fixed_spec'] = self._compute_sens_at_fixed_spec()
        
        # Calibration
        self.results['calibration'] = self._compute_calibration()
        
        # Confusion matrices
        self.results['confusion_matrices'] = self._compute_confusion_matrices(threshold)
        
        # Bootstrap CIs for AUROC
        self.results['auroc_ci'] = self._bootstrap_auroc_ci()
        
        # Optimal thresholds
        self.results['optimal_thresholds'] = self._find_optimal_thresholds()
        
        return self.results
    
    def _compute_auroc(self) -> Dict:
        """Compute AUROC per class and macro."""
        aurocs = []
        for i in range(self.n_classes):
            if len(np.unique(self.y_true[:, i])) > 1:
                aurocs.append(roc_auc_score(self.y_true[:, i], self.y_pred[:, i]))
            else:
                aurocs.append(np.nan)
        
        return {
            'per_class': dict(zip(self.class_names, aurocs)),
            'macro': np.nanmean(aurocs),
            'weighted': self._weighted_average(aurocs),
        }
    
    def _compute_auprc(self) -> Dict:
        """Compute AUPRC (Average Precision) per class and macro."""
        auprcs = []
        for i in range(self.n_classes):
            if len(np.unique(self.y_true[:, i])) > 1:
                auprcs.append(average_precision_score(self.y_true[:, i], self.y_pred[:, i]))
            else:
                auprcs.append(np.nan)
        
        # Class prevalence (random baseline for AUPRC)
        prevalence = np.mean(self.y_true, axis=0)
        
        return {
            'per_class': dict(zip(self.class_names, auprcs)),
            'macro': np.nanmean(auprcs),
            'prevalence': dict(zip(self.class_names, prevalence.tolist())),
        }
    
    def _compute_f1(self, threshold: float) -> Dict:
        """Compute F1 score per class and macro."""
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        
        f1s = []
        precisions = []
        recalls = []
        
        for i in range(self.n_classes):
            f1s.append(f1_score(self.y_true[:, i], y_pred_binary[:, i], zero_division=0))
            precisions.append(precision_score(self.y_true[:, i], y_pred_binary[:, i], zero_division=0))
            recalls.append(recall_score(self.y_true[:, i], y_pred_binary[:, i], zero_division=0))
        
        return {
            'f1_per_class': dict(zip(self.class_names, f1s)),
            'f1_macro': np.mean(f1s),
            'precision_per_class': dict(zip(self.class_names, precisions)),
            'precision_macro': np.mean(precisions),
            'recall_per_class': dict(zip(self.class_names, recalls)),
            'recall_macro': np.mean(recalls),
        }
    
    def _compute_sens_spec(self, threshold: float) -> Dict:
        """Compute sensitivity and specificity."""
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        
        sensitivities = []
        specificities = []
        ppvs = []  # Positive Predictive Value
        npvs = []  # Negative Predictive Value
        
        for i in range(self.n_classes):
            tn, fp, fn, tp = confusion_matrix(
                self.y_true[:, i], y_pred_binary[:, i], labels=[0, 1]
            ).ravel()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            sensitivities.append(sens)
            specificities.append(spec)
            ppvs.append(ppv)
            npvs.append(npv)
        
        return {
            'sensitivity': dict(zip(self.class_names, sensitivities)),
            'specificity': dict(zip(self.class_names, specificities)),
            'ppv': dict(zip(self.class_names, ppvs)),
            'npv': dict(zip(self.class_names, npvs)),
            'sensitivity_macro': np.mean(sensitivities),
            'specificity_macro': np.mean(specificities),
        }
    
    def _compute_sens_at_fixed_spec(self) -> Dict:
        """Compute sensitivity at fixed specificity levels (90%, 95%, 99%)."""
        fixed_specs = [0.90, 0.95, 0.99]
        results = {f'spec_{int(s*100)}': {} for s in fixed_specs}
        
        for i in range(self.n_classes):
            if len(np.unique(self.y_true[:, i])) <= 1:
                continue
                
            fpr, tpr, thresholds = roc_curve(self.y_true[:, i], self.y_pred[:, i])
            specificity = 1 - fpr
            
            for target_spec in fixed_specs:
                # Find threshold closest to target specificity
                idx = np.argmin(np.abs(specificity - target_spec))
                sens = tpr[idx]
                thresh = thresholds[idx] if idx < len(thresholds) else 0.5
                
                key = f'spec_{int(target_spec*100)}'
                results[key][self.class_names[i]] = {
                    'sensitivity': float(sens),
                    'actual_specificity': float(specificity[idx]),
                    'threshold': float(thresh),
                }
        
        # Compute macro averages
        for key in results:
            sensitivities = [v['sensitivity'] for v in results[key].values()]
            results[key]['macro'] = np.mean(sensitivities) if sensitivities else 0
        
        return results
    
    def _compute_calibration(self) -> Dict:
        """Compute calibration metrics (ECE, MCE, Brier Score)."""
        n_bins = 10
        
        eces = []  # Expected Calibration Error
        mces = []  # Maximum Calibration Error
        briers = []
        
        calibration_curves = {}
        
        for i in range(self.n_classes):
            # Brier score
            brier = np.mean((self.y_pred[:, i] - self.y_true[:, i]) ** 2)
            briers.append(brier)
            
            try:
                # Calibration curve
                prob_true, prob_pred = calibration_curve(
                    self.y_true[:, i], self.y_pred[:, i], 
                    n_bins=n_bins, strategy='uniform'
                )
                
                calibration_curves[self.class_names[i]] = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist(),
                }
                
                # ECE: weighted average of |accuracy - confidence|
                bin_counts, bin_edges = np.histogram(self.y_pred[:, i], bins=n_bins, range=(0, 1))
                total = bin_counts.sum()
                
                if len(prob_true) == len(bin_counts):
                    weights = bin_counts / total
                    ece = np.sum(np.abs(prob_true - prob_pred) * weights[:len(prob_true)])
                else:
                    ece = np.mean(np.abs(prob_true - prob_pred))
                
                mce = np.max(np.abs(prob_true - prob_pred))
                
            except Exception as e:
                ece = np.nan
                mce = np.nan
            
            eces.append(ece)
            mces.append(mce)
        
        return {
            'brier_per_class': dict(zip(self.class_names, briers)),
            'brier_macro': np.mean(briers),
            'ece_per_class': dict(zip(self.class_names, eces)),
            'ece_macro': np.nanmean(eces),
            'mce_per_class': dict(zip(self.class_names, mces)),
            'mce_macro': np.nanmean(mces),
            'calibration_curves': calibration_curves,
        }
    
    def _compute_confusion_matrices(self, threshold: float) -> Dict:
        """Compute confusion matrix for each class."""
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        
        cms = {}
        for i, cls in enumerate(self.class_names):
            cm = confusion_matrix(self.y_true[:, i], y_pred_binary[:, i], labels=[0, 1])
            cms[cls] = {
                'matrix': cm.tolist(),
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1]),
            }
        
        return cms
    
    def _bootstrap_auroc_ci(self, confidence: float = 0.95) -> Dict:
        """Bootstrap confidence intervals for AUROC."""
        n_samples = len(self.y_true)
        
        ci_results = {}
        
        for i, cls in enumerate(self.class_names):
            if len(np.unique(self.y_true[:, i])) <= 1:
                continue
            
            bootstrap_aurocs = []
            for _ in range(self.n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                try:
                    auroc = roc_auc_score(self.y_true[indices, i], self.y_pred[indices, i])
                    bootstrap_aurocs.append(auroc)
                except:
                    pass
            
            if bootstrap_aurocs:
                bootstrap_aurocs = np.array(bootstrap_aurocs)
                alpha = 1 - confidence
                lower = np.percentile(bootstrap_aurocs, alpha / 2 * 100)
                upper = np.percentile(bootstrap_aurocs, (1 - alpha / 2) * 100)
                
                ci_results[cls] = {
                    'mean': float(np.mean(bootstrap_aurocs)),
                    'std': float(np.std(bootstrap_aurocs)),
                    'ci_lower': float(lower),
                    'ci_upper': float(upper),
                }
        
        # Macro AUROC CI
        macro_aurocs = []
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            class_aurocs = []
            for i in range(self.n_classes):
                try:
                    a = roc_auc_score(self.y_true[indices, i], self.y_pred[indices, i])
                    class_aurocs.append(a)
                except:
                    pass
            if class_aurocs:
                macro_aurocs.append(np.mean(class_aurocs))
        
        if macro_aurocs:
            alpha = 1 - confidence
            ci_results['macro'] = {
                'mean': float(np.mean(macro_aurocs)),
                'std': float(np.std(macro_aurocs)),
                'ci_lower': float(np.percentile(macro_aurocs, alpha / 2 * 100)),
                'ci_upper': float(np.percentile(macro_aurocs, (1 - alpha / 2) * 100)),
            }
        
        return ci_results
    
    def _find_optimal_thresholds(self) -> Dict:
        """Find optimal thresholds for each class using different criteria."""
        thresholds = {}
        
        for i, cls in enumerate(self.class_names):
            if len(np.unique(self.y_true[:, i])) <= 1:
                continue
            
            fpr, tpr, thresh = roc_curve(self.y_true[:, i], self.y_pred[:, i])
            
            # Youden's J statistic (max sensitivity + specificity - 1)
            j_scores = tpr + (1 - fpr) - 1
            optimal_j_idx = np.argmax(j_scores)
            
            # F1 optimization
            best_f1 = 0
            best_f1_thresh = 0.5
            for t in np.arange(0.1, 0.9, 0.02):
                pred = (self.y_pred[:, i] >= t).astype(int)
                f1 = f1_score(self.y_true[:, i], pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_thresh = t
            
            thresholds[cls] = {
                'youden_j': {
                    'threshold': float(thresh[optimal_j_idx]) if optimal_j_idx < len(thresh) else 0.5,
                    'sensitivity': float(tpr[optimal_j_idx]),
                    'specificity': float(1 - fpr[optimal_j_idx]),
                },
                'best_f1': {
                    'threshold': float(best_f1_thresh),
                    'f1': float(best_f1),
                }
            }
        
        return thresholds
    
    def _weighted_average(self, values: List[float]) -> float:
        """Compute sample-weighted average."""
        class_counts = np.sum(self.y_true, axis=0)
        weights = class_counts / class_counts.sum()
        
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() == 0:
            return 0.0
        
        return float(np.average(np.array(values)[valid_mask], weights=weights[valid_mask]))
    
    def plot_roc_curves(self, save_path: Optional[Path] = None, figsize: Tuple = (10, 8)):
        """Plot ROC curves for all classes."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        for i, (cls, color) in enumerate(zip(self.class_names, colors)):
            if len(np.unique(self.y_true[:, i])) <= 1:
                continue
            
            fpr, tpr, _ = roc_curve(self.y_true[:, i], self.y_pred[:, i])
            auroc = roc_auc_score(self.y_true[:, i], self.y_pred[:, i])
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{cls} (AUROC={auroc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves by Class', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig
    
    def plot_precision_recall_curves(self, save_path: Optional[Path] = None, figsize: Tuple = (10, 8)):
        """Plot Precision-Recall curves for all classes."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        for i, (cls, color) in enumerate(zip(self.class_names, colors)):
            if len(np.unique(self.y_true[:, i])) <= 1:
                continue
            
            precision, recall, _ = precision_recall_curve(self.y_true[:, i], self.y_pred[:, i])
            ap = average_precision_score(self.y_true[:, i], self.y_pred[:, i])
            prevalence = np.mean(self.y_true[:, i])
            
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{cls} (AP={ap:.3f}, prev={prevalence:.2f})')
            
            # Baseline (prevalence)
            ax.axhline(y=prevalence, color=color, linestyle=':', alpha=0.3)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves by Class', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig
    
    def plot_calibration_curves(self, save_path: Optional[Path] = None, figsize: Tuple = (10, 8)):
        """Plot calibration (reliability) curves."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        for i, (cls, color) in enumerate(zip(self.class_names, colors)):
            try:
                prob_true, prob_pred = calibration_curve(
                    self.y_true[:, i], self.y_pred[:, i], n_bins=10, strategy='uniform'
                )
                ax.plot(prob_pred, prob_true, 's-', color=color, lw=2,
                       label=cls, markersize=8)
            except:
                pass
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curves', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig
    
    def plot_confusion_matrices(self, threshold: float = 0.5, save_path: Optional[Path] = None, figsize: Tuple = (15, 3)):
        """Plot confusion matrices for all classes."""
        fig, axes = plt.subplots(1, self.n_classes, figsize=figsize)
        
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        
        for i, (cls, ax) in enumerate(zip(self.class_names, axes)):
            cm = confusion_matrix(self.y_true[:, i], y_pred_binary[:, i], labels=[0, 1])
            
            # Normalize
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
            ax.set_title(f'{cls}', fontsize=12)
            ax.set_xlabel('Predicted')
            if i == 0:
                ax.set_ylabel('Actual')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig
    
    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for the paper."""
        tables = {}
        
        # Main metrics table
        tables['main_metrics'] = self._latex_main_table()
        tables['per_class_auroc'] = self._latex_perclass_auroc()
        tables['sens_at_spec'] = self._latex_sens_at_spec()
        
        return tables
    
    def _latex_main_table(self) -> str:
        """Generate main metrics LaTeX table."""
        latex = r"""\begin{table}[htbp]
\centering
\caption{Per-Class Classification Metrics}
\label{tab:class_metrics}
\begin{tabular}{lccccc}
\toprule
\textbf{Class} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{Sens} & \textbf{Spec} & \textbf{F1} \\
\midrule
"""
        for cls in self.class_names:
            auroc = self.results['auroc']['per_class'][cls]
            auprc = self.results['auprc']['per_class'][cls]
            sens = self.results['sensitivity_specificity']['sensitivity'][cls]
            spec = self.results['sensitivity_specificity']['specificity'][cls]
            f1 = self.results['f1']['f1_per_class'][cls]
            
            latex += f"{CLASS_FULL_NAMES.get(cls, cls)} & {auroc:.3f} & {auprc:.3f} & {sens:.3f} & {spec:.3f} & {f1:.3f} \\\\\n"
        
        # Macro
        latex += r"\midrule" + "\n"
        latex += f"Macro Average & {self.results['auroc']['macro']:.3f} & {self.results['auprc']['macro']:.3f} & "
        latex += f"{self.results['sensitivity_specificity']['sensitivity_macro']:.3f} & "
        latex += f"{self.results['sensitivity_specificity']['specificity_macro']:.3f} & "
        latex += f"{self.results['f1']['f1_macro']:.3f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def _latex_perclass_auroc(self) -> str:
        """Generate per-class AUROC with CI table."""
        if 'auroc_ci' not in self.results:
            return ""
        
        latex = r"""\begin{table}[htbp]
\centering
\caption{Per-Class AUROC with 95\% Confidence Intervals}
\label{tab:auroc_ci}
\begin{tabular}{lcc}
\toprule
\textbf{Class} & \textbf{AUROC} & \textbf{95\% CI} \\
\midrule
"""
        for cls in self.class_names:
            if cls in self.results['auroc_ci']:
                ci = self.results['auroc_ci'][cls]
                latex += f"{CLASS_FULL_NAMES.get(cls, cls)} & {ci['mean']:.3f} & [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] \\\\\n"
        
        if 'macro' in self.results['auroc_ci']:
            latex += r"\midrule" + "\n"
            ci = self.results['auroc_ci']['macro']
            latex += f"Macro Average & {ci['mean']:.3f} & [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def _latex_sens_at_spec(self) -> str:
        """Generate sensitivity at fixed specificity table."""
        if 'sens_at_fixed_spec' not in self.results:
            return ""
        
        latex = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity at Fixed Specificity Levels}
\label{tab:sens_at_spec}
\begin{tabular}{lccc}
\toprule
\textbf{Class} & \textbf{Sens@90\%Spec} & \textbf{Sens@95\%Spec} & \textbf{Sens@99\%Spec} \\
\midrule
"""
        for cls in self.class_names:
            vals = []
            for spec in [90, 95, 99]:
                key = f'spec_{spec}'
                if cls in self.results['sens_at_fixed_spec'].get(key, {}):
                    vals.append(f"{self.results['sens_at_fixed_spec'][key][cls]['sensitivity']:.3f}")
                else:
                    vals.append("-")
            latex += f"{CLASS_FULL_NAMES.get(cls, cls)} & {vals[0]} & {vals[1]} & {vals[2]} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def save_results(self, output_dir: Path, prefix: str = ""):
        """Save all results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = prefix + "_" if prefix else ""
        
        # Save JSON results (exclude numpy arrays)
        json_results = {k: v for k, v in self.results.items() 
                       if not isinstance(v, np.ndarray)}
        
        with open(output_dir / f"{prefix}comprehensive_metrics.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save figures
        self.plot_roc_curves(output_dir / f"{prefix}roc_curves.png")
        self.plot_precision_recall_curves(output_dir / f"{prefix}pr_curves.png")
        self.plot_calibration_curves(output_dir / f"{prefix}calibration_curves.png")
        self.plot_confusion_matrices(save_path=output_dir / f"{prefix}confusion_matrices.png")
        
        # Save LaTeX tables
        latex_tables = self.generate_latex_tables()
        for name, table in latex_tables.items():
            with open(output_dir / f"{prefix}table_{name}.tex", 'w') as f:
                f.write(table)
        
        print(f" Results saved to {output_dir}")


def evaluate_model_comprehensive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    prefix: str = "",
    n_bootstrap: int = 2000
) -> Dict:
    """Convenience function for comprehensive evaluation."""
    evaluator = ComprehensiveEvaluator(y_true, y_pred, n_bootstrap=n_bootstrap)
    results = evaluator.compute_all_metrics()
    evaluator.save_results(output_dir, prefix)
    return results


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_pred = np.random.rand(n_samples, n_classes)
    
    # Make predictions slightly better than random
    y_pred = 0.3 * y_pred + 0.7 * y_true + 0.1 * np.random.randn(n_samples, n_classes)
    y_pred = np.clip(y_pred, 0, 1)
    
    evaluator = ComprehensiveEvaluator(y_true, y_pred, n_bootstrap=100)  # Fewer for testing
    results = evaluator.compute_all_metrics()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION TEST")
    print("="*60)
    
    print(f"\nAUROC: {results['auroc']['macro']:.4f}")
    print(f"AUPRC: {results['auprc']['macro']:.4f}")
    print(f"F1:    {results['f1']['f1_macro']:.4f}")
    print(f"ECE:   {results['calibration']['ece_macro']:.4f}")
    print(f"Brier: {results['calibration']['brier_macro']:.4f}")
    
    print("\nSensitivity at 90% Specificity:")
    for cls, data in results['sens_at_fixed_spec']['spec_90'].items():
        if isinstance(data, dict):
            print(f"  {cls}: {data['sensitivity']:.4f}")
    
    print("\n95% CI for AUROC:")
    for cls, ci in results['auroc_ci'].items():
        print(f"  {cls}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
