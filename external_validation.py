#!/usr/bin/env python3
"""
External Validation Script
==========================

Validates trained models on external ECG datasets to assess generalization.

Supported External Datasets:
1. CPSC2018 - China Physiological Signal Challenge 2018
2. Georgia - Georgia 12-lead ECG Challenge Database  
3. ICBEB - ICBEB 2018 ECG dataset
4. Chapman-Shaoxing - Large Chinese ECG dataset

Usage:
    python external_validation.py --model_dir outputs/models/best_model --dataset cpsc2018
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import wfdb
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model import get_model, count_parameters


# Dataset configurations
EXTERNAL_DATASETS = {
    'cpsc2018': {
        'name': 'China Physiological Signal Challenge 2018',
        'url': 'http://2018.icbeb.org/Challenge.html',
        'n_classes': 9,  # Original has 9, we'll map to our 5
        'sample_rate': 500,
        'classes': ['NORM', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'],
    },
    'georgia': {
        'name': 'Georgia 12-Lead ECG Challenge Database',
        'url': 'https://physionet.org/content/challenge-2020/',
        'sample_rate': 500,
    },
    'icbeb': {
        'name': 'ICBEB 2018',
        'sample_rate': 500,
    },
}

# Class mapping from external datasets to our 5 classes
CLASS_MAPPING = {
    # CPSC2018 mapping
    'NORM': 'NORM',  # Normal
    'AF': 'CD',       # Atrial fibrillation -> Conduction Disturbance
    'IAVB': 'CD',     # I-AVB -> Conduction Disturbance
    'LBBB': 'CD',     # LBBB -> Conduction Disturbance
    'RBBB': 'CD',     # RBBB -> Conduction Disturbance
    'PAC': 'CD',      # PAC -> Conduction Disturbance
    'PVC': 'CD',      # PVC -> Conduction Disturbance
    'STD': 'STTC',    # ST depression -> ST/T Change
    'STE': 'STTC',    # ST elevation -> ST/T Change
    
    # PhysioNet 2020 mapping (SNOMED codes)
    '426783006': 'NORM',   # Sinus rhythm
    '164889003': 'CD',     # Atrial fibrillation
    '270492004': 'CD',     # First degree AV block
    '164909002': 'CD',     # LBBB
    '59118001': 'CD',      # RBBB
    '164884008': 'STTC',   # ST elevation
    '164931005': 'STTC',   # ST depression
    '429622005': 'STTC',   # ST-T change
}

# Our 5 classes
TARGET_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Standard 12 leads
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ExternalDataset(Dataset):
    """Base class for external ECG datasets."""
    
    def __init__(
        self,
        data_dir: str,
        leads: List[str] = None,
        target_length: int = 5000,
        target_sample_rate: int = 500,
    ):
        self.data_dir = Path(data_dir)
        self.leads = leads or LEAD_NAMES
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        
        self.lead_indices = [LEAD_NAMES.index(l) for l in self.leads]
        
        self.records = []
        self.labels = []
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def preprocess_signal(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess ECG signal: resample, normalize, pad/truncate."""
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            from scipy.signal import resample
            n_samples = int(len(signal[0]) * self.target_sample_rate / sample_rate)
            signal = np.array([resample(s, n_samples) for s in signal])
        
        # Select leads
        signal = signal[self.lead_indices]
        
        # Pad or truncate
        n_samples = signal.shape[1]
        if n_samples < self.target_length:
            # Pad with zeros
            padding = np.zeros((signal.shape[0], self.target_length - n_samples))
            signal = np.concatenate([signal, padding], axis=1)
        elif n_samples > self.target_length:
            # Truncate (take center)
            start = (n_samples - self.target_length) // 2
            signal = signal[:, start:start + self.target_length]
        
        # Normalize per-lead
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
        
        return signal.astype(np.float32)
    
    def map_to_target_classes(self, original_labels: List[str]) -> np.ndarray:
        """Map original dataset labels to our 5 target classes."""
        target = np.zeros(len(TARGET_CLASSES), dtype=np.float32)
        
        for label in original_labels:
            if label in CLASS_MAPPING:
                target_class = CLASS_MAPPING[label]
                if target_class in TARGET_CLASSES:
                    idx = TARGET_CLASSES.index(target_class)
                    target[idx] = 1.0
        
        return target


class CPSC2018Dataset(ExternalDataset):
    """China Physiological Signal Challenge 2018 dataset."""
    
    def __init__(
        self,
        data_dir: str,
        leads: List[str] = None,
        target_length: int = 5000,
    ):
        super().__init__(data_dir, leads, target_length)
        
        self.load_data()
    
    def load_data(self):
        """Load CPSC2018 records."""
        
        # Find reference file
        ref_file = self.data_dir / "REFERENCE.csv"
        if not ref_file.exists():
            # Try alternative names
            for name in ["reference.csv", "REFERENCE.txt", "labels.csv"]:
                alt = self.data_dir / name
                if alt.exists():
                    ref_file = alt
                    break
        
        if not ref_file.exists():
            print(f"Warning: No reference file found in {self.data_dir}")
            return
        
        # Load reference
        try:
            df = pd.read_csv(ref_file, header=None, names=['record', 'label'])
        except:
            df = pd.read_csv(ref_file)
        
        # Find data files
        for _, row in df.iterrows():
            record_name = str(row['record']) if 'record' in row else str(row.iloc[0])
            
            # Find the .mat or .dat file
            mat_file = self.data_dir / f"{record_name}.mat"
            dat_file = self.data_dir / f"{record_name}.dat"
            
            if mat_file.exists() or dat_file.exists():
                label = str(row['label']) if 'label' in row else str(row.iloc[1])
                self.records.append(record_name)
                self.labels.append(label.split(',') if ',' in label else [label])
        
        print(f"Loaded {len(self.records)} records from CPSC2018")
    
    def __getitem__(self, idx):
        record_name = self.records[idx]
        labels = self.labels[idx]
        
        # Load signal using scipy.io or wfdb
        try:
            from scipy.io import loadmat
            mat_file = self.data_dir / f"{record_name}.mat"
            if mat_file.exists():
                data = loadmat(str(mat_file))
                signal = data['ECG']['data'][0][0]  # Typical CPSC format
                sample_rate = 500
        except:
            # Try wfdb
            try:
                record = wfdb.rdrecord(str(self.data_dir / record_name))
                signal = record.p_signal.T  # (leads, samples)
                sample_rate = record.fs
            except:
                # Return zeros if failed
                signal = np.zeros((12, self.target_length))
                sample_rate = self.target_sample_rate
        
        # Preprocess
        signal = self.preprocess_signal(signal, sample_rate)
        target = self.map_to_target_classes(labels)
        
        return torch.FloatTensor(signal), torch.FloatTensor(target)


class PhysioNet2020Dataset(ExternalDataset):
    """PhysioNet/Computing in Cardiology Challenge 2020 dataset."""
    
    def __init__(
        self,
        data_dir: str,
        subset: str = 'georgia',  # 'georgia', 'ptbxl', 'cpsc', etc.
        leads: List[str] = None,
        target_length: int = 5000,
    ):
        super().__init__(data_dir, leads, target_length)
        self.subset = subset
        self.load_data()
    
    def load_data(self):
        """Load PhysioNet 2020 records."""
        
        # Find header files
        header_files = list(self.data_dir.glob("*.hea"))
        
        for hea_file in tqdm(header_files, desc=f"Loading {self.subset}"):
            record_name = hea_file.stem
            
            try:
                # Parse header for labels
                with open(hea_file, 'r') as f:
                    header_text = f.read()
                
                # Extract labels from header
                labels = []
                for line in header_text.split('\n'):
                    if line.startswith('#Dx:'):
                        dx_codes = line.replace('#Dx:', '').strip().split(',')
                        labels = [c.strip() for c in dx_codes]
                        break
                
                self.records.append(str(hea_file.parent / record_name))
                self.labels.append(labels)
                
            except Exception as e:
                continue
        
        print(f"Loaded {len(self.records)} records from {self.subset}")
    
    def __getitem__(self, idx):
        record_path = self.records[idx]
        labels = self.labels[idx]
        
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal.T
            sample_rate = record.fs
            
            # Ensure 12 leads
            if signal.shape[0] < 12:
                padding = np.zeros((12 - signal.shape[0], signal.shape[1]))
                signal = np.concatenate([signal, padding], axis=0)
            
            signal = self.preprocess_signal(signal, sample_rate)
            
        except Exception as e:
            signal = np.zeros((len(self.leads), self.target_length), dtype=np.float32)
        
        target = self.map_to_target_classes(labels)
        
        return torch.FloatTensor(signal), torch.FloatTensor(target)


def load_trained_model(
    model_dir: Path,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """Load a trained model and its configuration."""
    
    # Load results/config
    results_file = model_dir / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            config = json.load(f)
    else:
        config = {'n_leads': 12, 'model': 'resnet1d'}
    
    # Build model
    n_leads = config.get('n_leads', 12)
    model_name = config.get('model', 'resnet1d')
    
    model = get_model(model_name, n_leads=n_leads, n_classes=5)
    
    # Load weights
    checkpoint_file = model_dir / "best_model.pt"
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_on_external_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64
) -> Dict:
    """Evaluate model on an external dataset."""
    
    if len(dataset) == 0:
        return {'error': 'Empty dataset'}
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Evaluating"):
            signals = signals.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(signals)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    
    results = {
        'n_samples': len(all_preds),
        'classes': TARGET_CLASSES,
    }
    
    # Per-class metrics
    aurocs = []
    auprcs = []
    f1s = []
    
    for i, cls in enumerate(TARGET_CLASSES):
        if len(np.unique(all_labels[:, i])) > 1:
            auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auprc = average_precision_score(all_labels[:, i], all_preds[:, i])
            f1 = f1_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int))
        else:
            auroc = 0.5
            auprc = np.mean(all_labels[:, i])
            f1 = 0
        
        aurocs.append(auroc)
        auprcs.append(auprc)
        f1s.append(f1)
        
        results[f'auroc_{cls}'] = auroc
        results[f'auprc_{cls}'] = auprc
        results[f'f1_{cls}'] = f1
    
    results['auroc_macro'] = np.mean(aurocs)
    results['auprc_macro'] = np.mean(auprcs)
    results['f1_macro'] = np.mean(f1s)
    
    # Class distribution
    results['class_distribution'] = {
        cls: float(np.mean(all_labels[:, i])) 
        for i, cls in enumerate(TARGET_CLASSES)
    }
    
    return results


def run_external_validation(
    model_dir: str,
    external_data_dir: str,
    dataset_type: str = 'cpsc2018',
    output_dir: str = 'outputs/external_validation',
    leads: Optional[List[str]] = None
):
    """Run full external validation."""
    
    model_dir = Path(model_dir)
    external_data_dir = Path(external_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_dir}...")
    model, config = load_trained_model(model_dir, device)
    
    # Get leads from model config
    if leads is None:
        leads = config.get('leads', LEAD_NAMES)
        if leads == 'all':
            leads = LEAD_NAMES
    
    print(f"Using leads: {leads}")
    
    # Load external dataset
    print(f"\nLoading {dataset_type} dataset from {external_data_dir}...")
    
    if dataset_type.lower() == 'cpsc2018':
        dataset = CPSC2018Dataset(external_data_dir, leads=leads)
    elif dataset_type.lower() in ['georgia', 'physionet2020']:
        dataset = PhysioNet2020Dataset(external_data_dir, subset='georgia', leads=leads)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if len(dataset) == 0:
        print("ERROR: No data loaded. Check the data path and format.")
        return
    
    print(f"Loaded {len(dataset)} samples")
    
    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate_on_external_dataset(model, dataset, device)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"EXTERNAL VALIDATION RESULTS ({dataset_type})")
    print("=" * 60)
    
    print(f"\nSamples evaluated: {results['n_samples']}")
    print(f"\nMacro AUROC: {results['auroc_macro']:.4f}")
    print(f"Macro AUPRC: {results['auprc_macro']:.4f}")
    print(f"Macro F1:    {results['f1_macro']:.4f}")
    
    print(f"\nPer-class results:")
    print(f"{'Class':<8} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'Prevalence':<10}")
    print("-" * 45)
    for cls in TARGET_CLASSES:
        auroc = results.get(f'auroc_{cls}', 0)
        auprc = results.get(f'auprc_{cls}', 0)
        f1 = results.get(f'f1_{cls}', 0)
        prev = results['class_distribution'].get(cls, 0)
        print(f"{cls:<8} {auroc:.4f}   {auprc:.4f}   {f1:.4f}   {prev:.4f}")
    
    # Save results
    output_file = output_dir / f"external_validation_{dataset_type}.json"
    
    results['model_dir'] = str(model_dir)
    results['dataset_type'] = dataset_type
    results['leads'] = leads
    results['n_leads'] = len(leads)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


def generate_external_validation_table(
    internal_results: Dict,
    external_results: Dict,
    output_path: Optional[Path] = None
) -> str:
    """Generate LaTeX table comparing internal vs external validation."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Internal vs. External Validation. Internal: PTB-XL test set. External: CPSC2018. A performance drop is expected due to domain shift.}
\label{tab:external_validation}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{Internal (PTB-XL)}} & \multicolumn{2}{c}{\textbf{External (CPSC2018)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Class} & \textbf{AUROC} & \textbf{F1} & \textbf{AUROC} & \textbf{F1} \\
\midrule
"""
    
    for cls in TARGET_CLASSES:
        int_auroc = internal_results.get(f'test_auroc_per_class', {}).get(cls, 0)
        int_f1 = internal_results.get(f'test_f1_per_class', {}).get(cls, 0)
        ext_auroc = external_results.get(f'auroc_{cls}', 0)
        ext_f1 = external_results.get(f'f1_{cls}', 0)
        
        latex += f"{cls} & {int_auroc:.3f} & {int_f1:.3f} & {ext_auroc:.3f} & {ext_f1:.3f} \\\\\n"
    
    # Macro
    int_macro_auroc = internal_results.get('test_auroc', 0)
    int_macro_f1 = internal_results.get('test_f1', 0)
    ext_macro_auroc = external_results.get('auroc_macro', 0)
    ext_macro_f1 = external_results.get('f1_macro', 0)
    
    latex += r"\midrule" + "\n"
    latex += f"Macro & {int_macro_auroc:.3f} & {int_macro_f1:.3f} & {ext_macro_auroc:.3f} & {ext_macro_f1:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
    
    return latex


def main():
    parser = argparse.ArgumentParser(
        description="External validation of trained ECG models"
    )
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model")
    parser.add_argument("--external_data", type=str, required=True,
                        help="Directory containing external dataset")
    parser.add_argument("--dataset", type=str, default="cpsc2018",
                        choices=["cpsc2018", "georgia", "physionet2020"],
                        help="External dataset type")
    parser.add_argument("--output_dir", type=str, default="outputs/external_validation",
                        help="Output directory")
    parser.add_argument("--leads", type=str, default=None,
                        help="Comma-separated list of leads (default: use model config)")
    
    args = parser.parse_args()
    
    leads = args.leads.split(',') if args.leads else None
    
    run_external_validation(
        model_dir=args.model_dir,
        external_data_dir=args.external_data,
        dataset_type=args.dataset,
        output_dir=args.output_dir,
        leads=leads
    )


if __name__ == "__main__":
    main()
