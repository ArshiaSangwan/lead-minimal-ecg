#!/usr/bin/env python3
"""
Traditional ML Baselines for Lead-Minimal ECG
==============================================

Provides XGBoost, Random Forest, and other traditional ML baselines
for fair comparison with deep learning approaches.

Features extracted:
- Statistical features (mean, std, min, max, etc.)
- Heart rate variability (HRV) features
- Wavelet features
- Morphological features

Usage:
    python run_traditional_baselines.py --config 12-lead
    python run_traditional_baselines.py --config 3-lead-I-II-V2
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.stats import entropy
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import cross_val_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent / "src"))

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class ECGFeatureExtractor:
    """Extract features from ECG signals for traditional ML."""
    
    def __init__(self, sampling_rate: int = 100):
        self.fs = sampling_rate
        self.feature_names = []
    
    def extract_statistical_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract basic statistical features from each lead."""
        features = []
        
        for lead_idx in range(signal.shape[0]):
            lead = signal[lead_idx]
            
            # Basic statistics
            features.extend([
                np.mean(lead),
                np.std(lead),
                np.min(lead),
                np.max(lead),
                np.ptp(lead),  # Peak-to-peak
                np.median(lead),
                stats.skew(lead),
                stats.kurtosis(lead),
            ])
            
            # Percentiles
            features.extend([
                np.percentile(lead, 10),
                np.percentile(lead, 25),
                np.percentile(lead, 75),
                np.percentile(lead, 90),
            ])
            
            # Zero crossings
            zero_crossings = np.sum(np.diff(np.sign(lead)) != 0)
            features.append(zero_crossings)
            
            # RMS
            features.append(np.sqrt(np.mean(lead**2)))
            
            # Mean absolute deviation
            features.append(np.mean(np.abs(lead - np.mean(lead))))
            
        return np.array(features)
    
    def extract_frequency_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract frequency domain features."""
        features = []
        
        for lead_idx in range(signal.shape[0]):
            lead = signal[lead_idx]
            
            # Compute PSD
            freqs, psd = welch(lead, fs=self.fs, nperseg=min(256, len(lead)))
            
            # Band powers
            vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            
            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
            
            total_power = vlf_power + lf_power + hf_power
            
            features.extend([
                vlf_power,
                lf_power,
                hf_power,
                total_power,
                lf_power / (hf_power + 1e-8),  # LF/HF ratio
            ])
            
            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-8)
            spectral_entropy = entropy(psd_norm + 1e-8)
            features.append(spectral_entropy)
            
            # Dominant frequency
            dom_freq = freqs[np.argmax(psd)]
            features.append(dom_freq)
            
        return np.array(features)
    
    def extract_morphological_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract morphological features (peak-based)."""
        features = []
        
        for lead_idx in range(signal.shape[0]):
            lead = signal[lead_idx]
            
            # Find R-peaks (positive peaks)
            try:
                peaks, properties = find_peaks(lead, distance=int(0.2 * self.fs), 
                                               height=np.std(lead) * 0.5)
            except:
                peaks = np.array([])
            
            if len(peaks) > 1:
                # RR intervals
                rr_intervals = np.diff(peaks) / self.fs
                features.extend([
                    np.mean(rr_intervals),
                    np.std(rr_intervals),
                    np.min(rr_intervals),
                    np.max(rr_intervals),
                    60 / (np.mean(rr_intervals) + 1e-8),  # Heart rate estimate
                ])
                
                # RR variability
                rr_diff = np.diff(rr_intervals)
                rmssd = np.sqrt(np.mean(rr_diff**2))
                features.append(rmssd)
            else:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Peak amplitudes
            if len(peaks) > 0:
                peak_heights = lead[peaks]
                features.extend([
                    np.mean(peak_heights),
                    np.std(peak_heights),
                ])
            else:
                features.extend([0, 0])
            
            # Number of peaks
            features.append(len(peaks))
            
        return np.array(features)
    
    def extract_wavelet_features(self, signal: np.ndarray, n_coeffs: int = 5) -> np.ndarray:
        """Extract wavelet-based features."""
        features = []
        
        try:
            import pywt
            
            for lead_idx in range(signal.shape[0]):
                lead = signal[lead_idx]
                
                # Multi-level wavelet decomposition
                coeffs = pywt.wavedec(lead, 'db4', level=4)
                
                for i, c in enumerate(coeffs[:n_coeffs]):
                    features.extend([
                        np.mean(np.abs(c)),
                        np.std(c),
                        entropy(np.abs(c) / (np.sum(np.abs(c)) + 1e-8) + 1e-8),
                    ])
        except ImportError:
            # If pywt not available, return zeros
            n_features = signal.shape[0] * n_coeffs * 3
            features = [0] * n_features
        
        return np.array(features)
    
    def extract_all_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract all features from a signal."""
        features = []
        
        features.append(self.extract_statistical_features(signal))
        features.append(self.extract_frequency_features(signal))
        features.append(self.extract_morphological_features(signal))
        
        try:
            features.append(self.extract_wavelet_features(signal))
        except:
            pass
        
        return np.concatenate(features)
    
    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        """Extract features from all signals."""
        print(f"Extracting features from {len(signals)} signals...")
        
        features = []
        for i, signal in enumerate(signals):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(signals)}...")
            features.append(self.extract_all_features(signal))
        
        features = np.array(features)
        print(f"Extracted {features.shape[1]} features per signal")
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features


# ============================================================================
# BASELINE MODELS
# ============================================================================

class TraditionalMLBaseline:
    """Traditional ML baseline for ECG classification."""
    
    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.feature_extractor = ECGFeatureExtractor()
        
        if model_name == "xgboost" and XGB_AVAILABLE:
            self.base_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif model_name == "lightgbm" and LGB_AVAILABLE:
            self.base_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif model_name == "random_forest":
            self.base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "gradient_boosting":
            self.base_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            # Fallback to Random Forest
            print(f"Model {model_name} not available, using Random Forest")
            self.base_model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        
        # Multi-label wrapper
        self.model = OneVsRestClassifier(self.base_model)
    
    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        """Fit the model."""
        # Extract features
        X = self.feature_extractor.fit_transform(X_raw)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Fit model
        print(f"Training {self.model_name}...")
        self.model.fit(X, y)
        
        return self
    
    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X = self.feature_extractor.fit_transform(X_raw)
        X = self.scaler.transform(X)
        
        try:
            probs = self.model.predict_proba(X)
            # Handle different output formats
            if isinstance(probs, list):
                probs = np.column_stack([p[:, 1] for p in probs])
        except:
            probs = self.model.predict(X).astype(float)
        
        return probs
    
    def evaluate(self, X_raw: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the model."""
        probs = self.predict_proba(X_raw)
        preds = (probs > 0.5).astype(int)
        
        # AUROC per class
        aurocs = []
        for i in range(y.shape[1]):
            if len(np.unique(y[:, i])) > 1:
                aurocs.append(roc_auc_score(y[:, i], probs[:, i]))
            else:
                aurocs.append(0.5)
        
        # AUPRC per class
        auprcs = []
        for i in range(y.shape[1]):
            if len(np.unique(y[:, i])) > 1:
                auprcs.append(average_precision_score(y[:, i], probs[:, i]))
            else:
                auprcs.append(0.0)
        
        # F1
        f1 = f1_score(y, preds, average='macro', zero_division=0)
        
        return {
            'auroc_macro': np.mean(aurocs),
            'auroc_per_class': dict(zip(CLASSES, aurocs)),
            'auprc_macro': np.mean(auprcs),
            'auprc_per_class': dict(zip(CLASSES, auprcs)),
            'f1_macro': f1,
        }


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_baseline_experiment(config_name: str, leads_str: str, 
                            model_name: str, data_path: str,
                            seed: int = 42) -> Dict:
    """Run a traditional ML baseline experiment."""
    
    from dataset import PTBXLDataset
    
    print(f"\n{'='*60}")
    print(f"Traditional ML Baseline: {model_name}")
    print(f"Configuration: {config_name}")
    print(f"Leads: {leads_str}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    # Load data
    leads = leads_str.split(",") if leads_str != "all" else "all"
    
    train_ds = PTBXLDataset(data_path, leads=leads, folds=list(range(1, 9)), augment=False)
    val_ds = PTBXLDataset(data_path, leads=leads, folds=[9], augment=False)
    test_ds = PTBXLDataset(data_path, leads=leads, folds=[10], augment=False)
    
    # Get numpy arrays (signals are stored as (seq_len, n_leads), need to transpose)
    X_train = train_ds.signals.transpose(0, 2, 1)  # (N, n_leads, seq_len)
    y_train = train_ds.labels
    X_test = test_ds.signals.transpose(0, 2, 1)
    y_test = test_ds.labels
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train and evaluate
    model = TraditionalMLBaseline(model_name)
    model.fit(X_train, y_train)
    
    results = model.evaluate(X_test, y_test)
    results['config_name'] = config_name
    results['model_name'] = model_name
    results['n_leads'] = X_train.shape[1]
    
    print(f"\nResults:")
    print(f"  AUROC (macro): {results['auroc_macro']:.4f}")
    print(f"  AUPRC (macro): {results['auprc_macro']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print(f"  Per-class AUROC: {results['auroc_per_class']}")
    
    return results


def run_all_baselines(data_path: str, output_dir: str):
    """Run all traditional ML baselines for comparison."""
    
    # Define lead configurations inline to avoid import issues
    LEAD_CONFIGS = {
        "12-lead": "all",
        "6-lead-limb": "I,II,III,aVR,aVL,aVF",
        "3-lead-I-II-V2": "I,II,V2",
        "3-lead-I-II-III": "I,II,III",
        "3-lead-II-V2-V5": "II,V2,V5",
        "2-lead-I-II": "I,II",
        "2-lead-II-V2": "II,V2",
        "1-lead-II": "II",
        "1-lead-V2": "V2",
        "1-lead-I": "I",
        "1-lead-V5": "V5",
    }
    
    # Priority configs for comparison
    PRIORITY_CONFIGS = ["12-lead", "6-lead-limb", "3-lead-I-II-V2", "1-lead-II"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / "baselines" / f"traditional_ml_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["xgboost", "random_forest"]
    if LGB_AVAILABLE:
        models.append("lightgbm")
    
    all_results = []
    
    # Use priority configs for speed
    for config_name in PRIORITY_CONFIGS:
        leads_str = LEAD_CONFIGS[config_name]
        
        for model_name in models:
            try:
                results = run_baseline_experiment(
                    config_name, leads_str, model_name, data_path
                )
                all_results.append(results)
            except Exception as e:
                print(f" Failed: {config_name} / {model_name}: {e}")
    
    # Save results
    with open(exp_dir / "baseline_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison table
    df = pd.DataFrame(all_results)
    df.to_csv(exp_dir / "baseline_comparison.csv", index=False)
    
    print(f"\n Results saved to: {exp_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Traditional ML Baselines for ECG")
    parser.add_argument("--config", type=str, default="12-lead",
                        help="Lead configuration to test")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "random_forest", "lightgbm", "gradient_boosting"],
                        help="ML model to use")
    parser.add_argument("--data-path", type=str, 
                        default="data/processed/ptbxl_processed.h5",
                        help="Path to preprocessed data")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--all", action="store_true",
                        help="Run all baselines on all priority configs")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_baselines(args.data_path, args.output_dir)
    else:
        # Define lead configurations inline
        LEAD_CONFIGS = {
            "12-lead": "all",
            "6-lead-limb": "I,II,III,aVR,aVL,aVF",
            "3-lead-I-II-V2": "I,II,V2",
            "1-lead-II": "II",
        }
        leads_str = LEAD_CONFIGS.get(args.config, args.config)
        run_baseline_experiment(args.config, leads_str, args.model, args.data_path)


if __name__ == "__main__":
    main()
