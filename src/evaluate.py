#!/usr/bin/env python3
"""
Evaluation script for Lead-Minimal ECG experiments.
Compares all trained models and computes LRS.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from dataset import get_dataloaders, LEAD_NAMES
from model import get_model
from metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_brier_score,
    compute_calibration_error,
    compute_sensitivity_specificity,
    compute_lead_robustness_score,
    bootstrap_confidence_interval
)


CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_model_and_evaluate(
    model_dir: str,
    data_path: str = "data/processed/ptbxl_processed.h5",
    device: str = "cuda"
) -> Dict:
    """
    Load a trained model and evaluate on test set.
    """
    model_dir = Path(model_dir)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load results to get leads info
    with open(model_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    leads = results['leads']
    n_leads = results['n_leads']
    
    # Load model
    model = get_model(
        model_name=results['model'],
        n_leads=n_leads,
        n_classes=5
    )
    
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get test data
    leads_list = leads if isinstance(leads, list) else "all"
    _, _, test_loader = get_dataloaders(
        h5_path=data_path,
        leads=leads_list,
        batch_size=128,
        num_workers=4
    )
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc=f"Evaluating {model_dir.name}"):
            signals = signals.to(device)
            
            with autocast():
                outputs = model(signals)
            
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute all metrics
    auroc = compute_auroc(all_labels, all_preds)
    auprc = compute_auprc(all_labels, all_preds)
    f1 = compute_f1(all_labels, all_preds)
    brier = compute_brier_score(all_labels, all_preds)
    ece = compute_calibration_error(all_labels, all_preds)
    sens_spec = compute_sensitivity_specificity(all_labels, all_preds)
    
    return {
        'model_dir': str(model_dir),
        'leads': leads,
        'n_leads': n_leads,
        'test_auroc': auroc['macro'],
        'test_auroc_per_class': dict(zip(CLASSES, auroc['per_class'])),
        'test_auprc': auprc['macro'],
        'test_auprc_per_class': dict(zip(CLASSES, auprc['per_class'])),
        'test_f1': f1['macro'],
        'test_f1_per_class': dict(zip(CLASSES, f1['per_class'])),
        'test_brier': brier['per_class'],
        'test_brier_macro': brier['macro'],
        'test_ece': ece['macro'],
        'test_sensitivity': dict(zip(CLASSES, sens_spec['sensitivity'])),
        'test_specificity': dict(zip(CLASSES, sens_spec['specificity'])),
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_all_models(
    output_dir: str = "outputs/",
    data_path: str = "data/processed/ptbxl_processed.h5"
) -> Dict:
    """
    Evaluate all trained models and compute LRS.
    """
    output_path = Path(output_dir)
    models_dir = output_path / "models"
    
    if not models_dir.exists():
        print(f"No models found in {models_dir}")
        return {}
    
    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / "best_model.pt").exists()]
    
    if not model_dirs:
        print(f"No trained models found in {models_dir}")
        return {}
    
    print(f"Found {len(model_dirs)} trained models")
    
    # Evaluate each model
    results = {}
    for model_dir in model_dirs:
        try:
            eval_results = load_model_and_evaluate(model_dir, data_path)
            
            # Create key from leads
            leads = eval_results['leads']
            if leads == LEAD_NAMES or leads == "all":
                key = "12-lead (all)"
            else:
                key = f"{len(leads)}-lead ({','.join(leads)})"
            
            results[key] = eval_results
            print(f"  {key}: AUROC={eval_results['test_auroc']:.4f}")
        except Exception as e:
            print(f"  Error evaluating {model_dir.name}: {e}")
    
    # Find baseline (12-lead model)
    baseline_key = None
    for key in results:
        if "12-lead" in key or results[key]['n_leads'] == 12:
            baseline_key = key
            break
    
    if baseline_key is None:
        print("\nWarning: No 12-lead baseline found. LRS cannot be computed.")
        return results
    
    baseline = results[baseline_key]
    print(f"\nBaseline: {baseline_key}")
    print(f"  AUROC: {baseline['test_auroc']:.4f}")
    
    # Compute LRS for each subset
    print("\n" + "=" * 60)
    print("Lead-Robustness Score (LRS)")
    print("=" * 60)
    
    lrs_table = []
    
    for key, res in results.items():
        if res['n_leads'] == 12:
            lrs = 1.0  # Baseline is always 1.0
        else:
            lrs = compute_lead_robustness_score(
                baseline_auroc=baseline['test_auroc'],
                subset_auroc=res['test_auroc'],
                baseline_brier=baseline['test_brier_macro'],
                subset_brier=res['test_brier_macro'],
                n_leads_baseline=12,
                n_leads_subset=res['n_leads']
            )
        
        res['lrs'] = lrs
        auroc_retention = res['test_auroc'] / baseline['test_auroc'] if baseline['test_auroc'] > 0 else 0
        
        lrs_table.append({
            'leads': key,
            'n_leads': res['n_leads'],
            'auroc': res['test_auroc'],
            'auroc_retention': auroc_retention,
            'f1': res['test_f1'],
            'lrs': lrs
        })
    
    # Sort by LRS
    lrs_table = sorted(lrs_table, key=lambda x: x['lrs'], reverse=True)
    
    # Print table
    print(f"\n{'Leads':<25} {'N':<4} {'AUROC':<8} {'Retain%':<8} {'F1':<8} {'LRS':<8}")
    print("-" * 65)
    
    for row in lrs_table:
        print(f"{row['leads']:<25} {row['n_leads']:<4} {row['auroc']:.4f}   "
              f"{row['auroc_retention']*100:.1f}%     {row['f1']:.4f}   {row['lrs']:.4f}")
    
    # Save comprehensive results
    summary = {
        'baseline': baseline_key,
        'baseline_auroc': baseline['test_auroc'],
        'models': {k: {key: v for key, v in res.items() if key not in ['predictions', 'labels']} 
                   for k, res in results.items()},
        'lrs_ranking': lrs_table
    }
    
    with open(output_path / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'evaluation_summary.json'}")
    
    return results


def print_detailed_results(results: Dict):
    """Print detailed per-class results."""
    
    print("\n" + "=" * 80)
    print("DETAILED PER-CLASS RESULTS")
    print("=" * 80)
    
    for key, res in results.items():
        print(f"\n{key}")
        print("-" * 40)
        print(f"{'Class':<8} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'Sens':<8} {'Spec':<8}")
        
        for cls in CLASSES:
            auroc = res['test_auroc_per_class'][cls]
            auprc = res['test_auprc_per_class'].get(cls, 0)
            f1 = res['test_f1_per_class'].get(cls, 0)
            sens = res['test_sensitivity'].get(cls, 0)
            spec = res['test_specificity'].get(cls, 0)
            
            print(f"{cls:<8} {auroc:.4f}   {auprc:.4f}   {f1:.4f}   {sens:.4f}   {spec:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all Lead-Minimal ECG models")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Directory containing trained models")
    parser.add_argument("--data_path", type=str, default="data/processed/ptbxl_processed.h5",
                        help="Path to preprocessed data")
    parser.add_argument("--detailed", action="store_true",
                        help="Print detailed per-class results")
    
    args = parser.parse_args()
    
    results = evaluate_all_models(args.output_dir, args.data_path)
    
    if args.detailed and results:
        print_detailed_results(results)
