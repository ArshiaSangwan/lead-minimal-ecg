#!/usr/bin/env python3
"""
PTB-XL Dataset with support for arbitrary lead subsets.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}


class PTBXLDataset(Dataset):
    """
    Loads preprocessed PTB-XL data from HDF5.
    Supports selecting arbitrary lead subsets at runtime.
    """
    
    def __init__(self, h5_path, leads="all", folds=None, augment=False):
        self.augment = augment
        
        # Parse lead selection
        if leads == "all":
            self.lead_idx = list(range(12))
            self.lead_names = LEAD_NAMES
        else:
            self.lead_idx = [LEAD_TO_IDX[l] for l in leads]
            self.lead_names = leads
        
        self.n_leads = len(self.lead_idx)
        
        # Load from HDF5
        with h5py.File(h5_path, 'r') as f:
            signals = f['signals'][:]
            labels = f['labels'][:]
            fold_arr = f['folds'][:]
            self.classes = list(f.attrs['classes'])
        
        # Filter by fold
        if folds is not None:
            mask = np.isin(fold_arr, folds)
            signals = signals[mask]
            labels = labels[mask]
        
        # Select leads
        self.signals = signals[:, :, self.lead_idx]
        self.labels = labels
        
        print(f"Loaded {len(self)} samples, {self.n_leads} leads")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        sig = self.signals[idx]
        lab = self.labels[idx]
        
        if self.augment:
            sig = self._augment(sig)
        
        # (n_leads, seq_len) for Conv1d
        sig = torch.from_numpy(sig).float().T
        lab = torch.from_numpy(lab).float()
        return sig, lab
    
    def _augment(self, sig):
        """Simple augmentations: scaling, noise, temporal shift."""
        if np.random.rand() < 0.5:
            sig = sig * np.random.uniform(0.8, 1.2)
        if np.random.rand() < 0.3:
            sig = sig + np.random.normal(0, 0.05, sig.shape)
        if np.random.rand() < 0.3:
            sig = np.roll(sig, np.random.randint(-50, 50), axis=0)
        return sig.astype(np.float32)


def get_dataloaders(h5_path="data/processed/ptbxl_processed.h5", leads="all",
                    batch_size=128, num_workers=8, pin_memory=True):
    """
    Standard PTB-XL splits: folds 1-8 train, 9 val, 10 test.
    """
    train_ds = PTBXLDataset(h5_path, leads=leads, folds=list(range(1, 9)), augment=True)
    val_ds = PTBXLDataset(h5_path, leads=leads, folds=[9], augment=False)
    test_ds = PTBXLDataset(h5_path, leads=leads, folds=[10], augment=False)
    
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    train_loader, _, _ = get_dataloaders(leads="all", batch_size=32)
    for x, y in train_loader:
        print(f"Batch shape: {x.shape}, Labels: {y.shape}")
        break
