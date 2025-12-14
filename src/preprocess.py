#!/usr/bin/env python3
"""
Preprocessing pipeline for PTB-XL dataset.

Loads raw WFDB records, applies normalization, and saves to HDF5.
Uses the official PTB-XL stratified folds for train/val/test splits.
"""

import ast
import h5py
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}

# PTB-XL diagnostic superclasses
DIAGNOSTIC_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_ptbxl_metadata(data_dir):
    """Load PTB-XL metadata and SCP statements."""
    data_path = Path(data_dir)
    
    df = pd.read_csv(data_path / "ptbxl_database.csv", index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    
    scp = pd.read_csv(data_path / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]
    
    return df, scp


def get_superclass_labels(scp_codes, scp_df):
    """Map SCP codes to diagnostic superclass labels."""
    labels = set()
    for code, likelihood in scp_codes.items():
        if likelihood >= 50 and code in scp_df.index:
            superclass = scp_df.loc[code].diagnostic_class
            if superclass in DIAGNOSTIC_CLASSES:
                labels.add(superclass)
    return list(labels)


def load_ecg(record_path):
    """Load a single ECG record."""
    record_path = str(record_path).replace('.dat', '').replace('.hea', '')
    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal.astype(np.float32)
    except Exception as e:
        print(f"Failed to load {record_path}: {e}")
        return None


def normalize(signal, method='zscore'):
    """Z-score or min-max normalization per lead."""
    if method == 'zscore':
        mu = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-8
        return (signal - mu) / std
    elif method == 'minmax':
        lo = signal.min(axis=0, keepdims=True)
        hi = signal.max(axis=0, keepdims=True)
        return (signal - lo) / (hi - lo + 1e-8)
    return signal


def preprocess(
    data_dir="data/ptb-xl/",
    output_dir="data/processed/",
    sampling_rate=100,
    norm_method='zscore'
):
    """
    Main preprocessing function.
    
    Loads PTB-XL, normalizes signals, and saves to HDF5 with:
    - signals: (N, 1000, 12) float32
    - labels: (N, 5) multi-hot
    - folds: (N,) int, values 1-10
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading PTB-XL metadata...")
    df, scp_df = load_ptbxl_metadata(data_dir)
    print(f"  Total records: {len(df)}")
    
    # Map SCP codes to superclass labels
    df['superclass'] = df.scp_codes.apply(lambda x: get_superclass_labels(x, scp_df))
    df = df[df.superclass.apply(len) > 0]  # Keep only records with labels
    print(f"  Records with diagnostic labels: {len(df)}")
    
    # Binarize labels
    mlb = MultiLabelBinarizer(classes=DIAGNOSTIC_CLASSES)
    labels = mlb.fit_transform(df.superclass)
    
    print("  Label counts:")
    for i, cls in enumerate(DIAGNOSTIC_CLASSES):
        print(f"    {cls}: {labels[:, i].sum()}")
    
    # Choose low-res (100Hz) or high-res (500Hz) records
    path_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    
    # Load and normalize signals
    print(f"Loading signals at {sampling_rate}Hz...")
    signals = []
    valid_idx = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sig = load_ecg(data_path / row[path_col])
        if sig is not None:
            sig = normalize(sig, method=norm_method)
            signals.append(sig)
            valid_idx.append(idx)
    
    signals = np.array(signals, dtype=np.float32)
    print(f"  Loaded: {signals.shape}")
    
    # Align labels and folds with valid signals
    df_valid = df.loc[valid_idx]
    labels = mlb.transform(df_valid.superclass)
    folds = df_valid.strat_fold.values.astype(np.int32)
    
    # Save to HDF5
    h5_path = output_path / "ptbxl_processed.h5"
    print(f"Saving to {h5_path}...")
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('signals', data=signals, compression='gzip', compression_opts=4)
        f.create_dataset('labels', data=labels)
        f.create_dataset('folds', data=folds)
        
        f.attrs['n_samples'] = len(signals)
        f.attrs['seq_length'] = signals.shape[1]
        f.attrs['n_leads'] = signals.shape[2]
        f.attrs['classes'] = DIAGNOSTIC_CLASSES
        f.attrs['lead_names'] = LEAD_NAMES
        f.attrs['sampling_rate'] = sampling_rate
    
    print(f"  File size: {h5_path.stat().st_size / 1e6:.1f} MB")
    
    # Print split info
    n_train = (folds <= 8).sum()
    n_val = (folds == 9).sum()
    n_test = (folds == 10).sum()
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptb-xl/")
    parser.add_argument("--output_dir", default="data/processed/")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--normalize", default="zscore", choices=["zscore", "minmax", "none"])
    args = parser.parse_args()
    
    preprocess(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        norm_method=args.normalize
    )
