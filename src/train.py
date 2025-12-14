#!/usr/bin/env python3
"""
Training script for Lead-Minimal ECG experiments.
Supports mixed precision training and W&B logging.
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' for experiment tracking.")

from dataset import get_dataloaders, LEAD_NAMES, PTBXLDataset
from torch.utils.data import DataLoader
from model import get_model, count_parameters


CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


class LabelSmoothingBCELoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing to reduce overfitting."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # Smooth labels: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        target_smooth = target * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(pred, target_smooth)


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation for ECG signals - proven effective for physiological data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Faster
    torch.backends.cudnn.benchmark = True  # Optimize for RTX 4090


def parse_leads(leads_str: str):
    """Parse lead string to list."""
    if leads_str.lower() == "all":
        return "all"
    return [l.strip() for l in leads_str.split(",")]


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    scheduler=None  # For OneCycleLR step per batch
) -> dict:
    """Train for one epoch with optional Mixup augmentation."""
    
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for signals, labels in pbar:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply Mixup augmentation
        if use_mixup:
            signals, labels_mixed = mixup_data(signals, labels, mixup_alpha)
        else:
            labels_mixed = labels
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(signals)
            loss = criterion(outputs, labels_mixed)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler if OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * signals.size(0)
        
        # Store predictions (use original labels for metrics)
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            # Use rounded labels for AUROC calculation when mixup is used
            all_labels.append((labels_mixed > 0.5).float().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = total_loss / len(loader.dataset)
    
    # Compute AUROC per class and macro
    try:
        auroc_per_class = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auroc_per_class.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
            else:
                auroc_per_class.append(0.5)
        macro_auroc = np.mean(auroc_per_class)
    except:
        macro_auroc = 0.0
        auroc_per_class = [0.0] * all_labels.shape[1]
    
    return {
        'loss': avg_loss,
        'auroc': macro_auroc,
        'auroc_per_class': auroc_per_class
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> dict:
    """Evaluate model on a dataset."""
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for signals, labels in tqdm(loader, desc="Eval", leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(signals)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * signals.size(0)
        
        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = total_loss / len(loader.dataset)
    
    # AUROC per class
    auroc_per_class = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auroc_per_class.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
        else:
            auroc_per_class.append(0.5)
    
    macro_auroc = np.mean(auroc_per_class)
    
    # F1 score (threshold = 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
    
    # Brier score (calibration metric)
    brier_score = np.mean((all_preds - all_labels) ** 2)
    
    return {
        'loss': avg_loss,
        'auroc': macro_auroc,
        'auroc_per_class': auroc_per_class,
        'f1': macro_f1,
        'brier': brier_score,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_overfit_mode(
    leads: str = "all",
    model_name: str = "resnet1d",
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 0.0005,
    data_path: str = "data/processed/ptbxl_processed.h5",
    seed: int = 42,
    output_dir: str = "outputs/",
    target_loss: float = 0.01,
):
    """
    Overfit mode training.
    
    Goal: Achieve minimal training loss by removing regularization.
    
    Strategy:
        - No regularization (dropout=0, weight_decay=0, no label smoothing)
        - Large model capacity
        - No data augmentation
        - Train until loss converges
    """
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("OVERFIT MODE ACTIVATED")
    print("="*60)
    print(f"\nTarget Loss: <= {target_loss}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    leads_list = parse_leads(leads)
    n_leads = 12 if leads_list == "all" else len(leads_list)
    
    print(f"\nConfiguration:")
    print(f"  Leads: {leads_list} ({n_leads} channels)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Regularization: None")
    print(f"  Data augmentation: None")
    print(f"  Model capacity: Maximum")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"OVERFIT_{model_name}_{timestamp}"
    run_dir = Path(output_dir) / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading data (no augmentation)...")
    train_ds = PTBXLDataset(data_path, leads=leads_list, folds=list(range(1, 9)), augment=False)
    val_ds = PTBXLDataset(data_path, leads=leads_list, folds=[9], augment=False)
    test_ds = PTBXLDataset(data_path, leads=leads_list, folds=[10], augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)
    
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    
    print("\nBuilding large model (no regularization)...")
    model = get_model(
        model_name=model_name,
        n_leads=n_leads,
        n_classes=5,
        base_filters=128,
        kernel_size=15,
        num_blocks=6,
        dropout=0.0,
        drop_path=0.0
    )
    model = model.to(device)
    
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    scaler = GradScaler('cuda')
    
    print("\n" + "=" * 60)
    print("STARTING OVERFIT TRAINING")
    print("=" * 60)
    
    best_train_loss = float('inf')
    best_val_auroc = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_mixup=False,
            scheduler=scheduler
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        if train_metrics['loss'] < best_train_loss:
            best_train_loss = train_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_auroc': val_metrics['auroc'],
            }, run_dir / "best_train_loss_model.pt")
        
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_auroc': val_metrics['auroc'],
            }, run_dir / "best_model.pt")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        status = "*" if train_metrics['loss'] < 0.1 else ("-" if train_metrics['loss'] < 0.2 else " ")
        
        print(f"{status} Epoch {epoch+1:03d}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.6f} | "
              f"Train AUROC: {train_metrics['auroc']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUROC: {val_metrics['auroc']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        if train_metrics['loss'] <= target_loss:
            print(f"\nTARGET ACHIEVED! Train Loss: {train_metrics['loss']:.6f} <= {target_loss}")
            break
    
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    
    checkpoint = torch.load(run_dir / "best_train_loss_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_final = evaluate(model, train_loader, criterion, device)
    val_final = evaluate(model, val_loader, criterion, device)
    test_final = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Training Loss: {checkpoint['train_loss']:.6f}")
    print(f"\nFinal Metrics (using best train loss model):")
    print(f"  Train - Loss: {train_final['loss']:.6f}, AUROC: {train_final['auroc']:.4f}")
    print(f"  Val   - Loss: {val_final['loss']:.4f}, AUROC: {val_final['auroc']:.4f}")
    print(f"  Test  - Loss: {test_final['loss']:.4f}, AUROC: {test_final['auroc']:.4f}")
    
    print(f"\n  Per-class Test AUROC:")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {test_final['auroc_per_class'][i]:.4f}")
    
    print(f"\nModels saved to: {run_dir}")
    print("=" * 60)
    
    return {
        'best_train_loss': checkpoint['train_loss'],
        'train_auroc': train_final['auroc'],
        'val_auroc': val_final['auroc'],
        'test_auroc': test_final['auroc'],
        'run_dir': str(run_dir)
    }


def train_ultra_overfit(
    leads: str = "all",
    model_name: str = "resnet1d",
    epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_path: str = "data/processed/ptbxl_processed.h5",
    seed: int = 42,
    output_dir: str = "outputs/",
    target_loss: float = 0.01,
):
    """
    Ultra overfit mode for debugging and testing.
    
    Goal: Achieve minimum training loss by maximizing model capacity
    and removing all regularization.
    
    Strategy:
        - Zero regularization
        - Very large model (256 filters, 8 blocks)
        - Small batch size
        - High learning rate
    """
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("ULTRA OVERFIT MODE")
    print("="*60)
    print(f"\nTarget Loss: <= {target_loss}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
    
    leads_list = parse_leads(leads)
    n_leads = 12 if leads_list == "all" else len(leads_list)
    
    print(f"\nConfiguration (all regularization disabled):")
    print(f"  Leads: {leads_list} ({n_leads} channels)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: base_filters=256, num_blocks=8")
    print(f"  Regularization: None")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ULTRA_OVERFIT_{timestamp}"
    run_dir = Path(output_dir) / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading data (no augmentation)...")
    train_ds = PTBXLDataset(data_path, leads=leads_list, folds=list(range(1, 9)), augment=False)
    val_ds = PTBXLDataset(data_path, leads=leads_list, folds=[9], augment=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_ds)} samples")
    
    print(f"\nBuilding large model...")
    model = get_model(
        model_name=model_name,
        n_leads=n_leads,
        n_classes=5,
        base_filters=256,
        kernel_size=15,
        num_blocks=8,
        dropout=0.0,
        drop_path=0.0
    )
    model = model.to(device)
    
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")
    
    if hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    print(f"\n{'='*60}")
    print("STARTING ULTRA OVERFIT TRAINING")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:04d}", leave=False)
        
        for signals, labels in pbar:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(signals)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss * signals.size(0)
            n_samples += signals.size(0)
            
            pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_samples
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
                'loss': avg_loss,
            }, run_dir / "best_model.pt")
        
        if avg_loss <= 0.01:
            status = "*"
        elif avg_loss <= 0.1:
            status = "-"
        else:
            status = " "
        
        print(f"{status} Epoch {epoch+1:04d}/{epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"Best: {best_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed/60:.1f}min")
        
        if avg_loss <= target_loss:
            print(f"\nTARGET ACHIEVED! Loss: {avg_loss:.6f} <= {target_loss}\n")
            break
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals = signals.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast('cuda'):
                        outputs = model(signals)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * signals.size(0)
                    val_samples += signals.size(0)
                    
                    probs = torch.sigmoid(outputs)
                    all_preds.append(probs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            
            val_loss = val_loss / val_samples
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            try:
                aurocs = []
                for i in range(5):
                    if len(np.unique(all_labels[:, i])) > 1:
                        aurocs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
                    else:
                        aurocs.append(0.5)
                val_auroc = np.mean(aurocs)
            except:
                val_auroc = 0.0
            
            print(f"   Validation - Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Training Loss: {best_loss:.6f}")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Model saved to: {run_dir}")
    
    if best_loss <= target_loss:
        print(f"\n  TARGET ACHIEVED: {best_loss:.6f} <= {target_loss}")
    else:
        print(f"\n  Target not reached. Best: {best_loss:.6f}, Target: {target_loss}")
    
    print("=" * 60)
    
    return {'best_loss': best_loss, 'run_dir': str(run_dir)}


def train(
    leads: str = "all",
    model_name: str = "resnet1d",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 7,
    output_dir: str = "outputs/",
    data_path: str = "data/processed/ptbxl_processed.h5",
    seed: int = 42,
    use_wandb: bool = True,
    wandb_project: str = "lead-minimal-ecg",
    wandb_entity: str = None,
    wandb_tags: list = None
):
    """Main training function with W&B logging."""
    
    # Setup
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training Lead-Minimal ECG Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Parse leads
    leads_list = parse_leads(leads)
    n_leads = 12 if leads_list == "all" else len(leads_list)
    leads_name = "all" if leads_list == "all" else "_".join(leads_list)
    
    print(f"\nLeads: {leads_list} ({n_leads} channels)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{leads_name}_{timestamp}"
    run_dir = Path(output_dir) / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")
    
    # Build config dict for logging
    # BALANCED regularization - not too weak (overfitting) or too strong (underfitting)
    config = {
        "model": model_name,
        "leads": leads_list if leads_list != "all" else "all",
        "n_leads": n_leads,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,      # Moderate (was 0.1 = too strong)
        "patience": patience,
        "seed": seed,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "base_filters": 32,        # Moderate capacity (was 24 = too small)
        "kernel_size": 15,
        "num_blocks": 4,           # Moderate depth (was 3 = too shallow)
        "dropout": 0.3,            # Moderate (was 0.4 = too high)
        "drop_path": 0.1,          # Moderate (was 0.2 = too high)
        "mixup_alpha": 0.2,        # Moderate (was 0.4 = too strong)
        "label_smoothing": 0.1,    # Moderate (was 0.2 = too strong)
    }
    
    # Initialize W&B
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        tags = wandb_tags or [f"{n_leads}-lead", model_name]
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config,
            dir=output_dir,
            tags=tags,
        )
        print(f"W&B run: {wandb_run.url}")
    elif use_wandb and not WANDB_AVAILABLE:
        print("W&B requested but not installed. Logging disabled.")
        use_wandb = False
    else:
        print("W&B logging disabled.")
        use_wandb = False
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        h5_path=data_path,
        leads=leads_list,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )
    
    # Log dataset info to W&B
    if use_wandb and wandb_run:
        wandb.config.update({
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        })
    
    # Model - balanced architecture for good generalization
    print("\nBuilding model...")
    model = get_model(
        model_name=model_name,
        n_leads=n_leads,
        n_classes=5,
        base_filters=32,   # Moderate capacity
        kernel_size=15,
        num_blocks=4,      # Moderate depth
        dropout=0.3,       # Moderate dropout
        drop_path=0.1      # Moderate stochastic depth
    )
    model = model.to(device)
    
    # Log model info and watch gradients
    n_params = count_parameters(model)
    if use_wandb and wandb_run:
        wandb.config.update({"n_parameters": n_params})
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Loss, optimizer, scheduler - balanced regularization
    criterion = LabelSmoothingBCELoss(smoothing=0.1)  # Moderate smoothing
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Moderate weight decay
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    # Training loop
    print("\nStarting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    best_val_auroc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_auroc': [],
        'val_loss': [], 'val_auroc': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train with Mixup augmentation (moderate alpha)
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_mixup=True, mixup_alpha=0.2
        )
        
        # Validate (no mixup)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train AUROC: {train_metrics['auroc']:.4f} | "
              f"Val AUROC: {val_metrics['auroc']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # W&B per-epoch logging
        if use_wandb and wandb_run:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "train/auroc": train_metrics['auroc'],
                "val/loss": val_metrics['loss'],
                "val/auroc": val_metrics['auroc'],
                "val/f1": val_metrics['f1'],
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
            # Per-class AUROC
            for i, cls in enumerate(CLASSES):
                log_dict[f"train/auroc_{cls}"] = train_metrics['auroc_per_class'][i]
                log_dict[f"val/auroc_{cls}"] = val_metrics['auroc_per_class'][i]
            wandb.log(log_dict)
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['lr'].append(current_lr)
        
        # Checkpointing
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': best_val_auroc,
                'leads': leads_list,
                'n_leads': n_leads
            }, run_dir / "best_model.pt")
            
            print(f"   -> New best model saved (AUROC: {best_val_auroc:.4f})")
            
            # Update W&B summary with best metrics
            if use_wandb and wandb_run:
                wandb.run.summary["best_val_auroc"] = best_val_auroc
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Log final training info to W&B
    if use_wandb and wandb_run:
        wandb.run.summary["epochs_trained"] = epoch + 1
        wandb.run.summary["early_stopped"] = patience_counter >= patience
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Macro AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Macro F1: {test_metrics['f1']:.4f}")
    print(f"\n  Per-class AUROC:")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {test_metrics['auroc_per_class'][i]:.4f}")
    print(f"\n  Brier Score: {test_metrics['brier']:.4f}")
    
    # Log test metrics to W&B
    if use_wandb and wandb_run:
        wandb.run.summary["test/auroc"] = test_metrics['auroc']
        wandb.run.summary["test/f1"] = test_metrics['f1']
        wandb.run.summary["test/brier"] = test_metrics['brier']
        for i, cls in enumerate(CLASSES):
            wandb.run.summary[f"test/auroc_{cls}"] = test_metrics['auroc_per_class'][i]
        
        # Save model as W&B artifact
        artifact = wandb.Artifact(
            name=f"model-{leads_name}",
            type="model",
            description=f"Best model for {leads_name} leads (AUROC: {best_val_auroc:.4f})",
            metadata={
                "val_auroc": best_val_auroc,
                "test_auroc": test_metrics['auroc'],
                "test_brier": test_metrics['brier'],
                "n_leads": n_leads,
                "epochs_trained": epoch + 1
            }
        )
        artifact.add_file(str(run_dir / "best_model.pt"))
        wandb.log_artifact(artifact)
        print(f"   -> Model artifact saved to W&B")
    
    # Save results (convert numpy types to Python native for JSON serialization)
    def to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj
    
    results = {
        'leads': leads_list if leads_list != "all" else LEAD_NAMES,
        'n_leads': n_leads,
        'model': model_name,
        'epochs_trained': epoch + 1,
        'best_val_auroc': float(best_val_auroc),
        'test_auroc': float(test_metrics['auroc']),
        'test_auroc_per_class': {cls: float(auc) for cls, auc in zip(CLASSES, test_metrics['auroc_per_class'])},
        'test_f1': float(test_metrics['f1']),
        'test_brier': float(test_metrics['brier']),
        'history': to_native(history),
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'seed': seed
        }
    }
    
    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for further analysis
    np.savez(
        run_dir / "test_predictions.npz",
        predictions=test_metrics['predictions'],
        labels=test_metrics['labels']
    )
    
    print(f"\nResults saved to {run_dir}")
    
    # Finish W&B run
    if use_wandb and wandb_run:
        wandb.finish()
        print("W&B run finished.")
    
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lead-Minimal ECG Model")
    
    parser.add_argument("--overfit", action="store_true",
                        help="Overfit mode: Remove regularization to achieve minimal loss")
    parser.add_argument("--ultra", action="store_true",
                        help="Ultra overfit mode: Maximum aggression for minimum loss")
    parser.add_argument("--target_loss", type=float, default=0.01,
                        help="Target training loss for overfit mode")
    
    parser.add_argument("--leads", type=str, default="all",
                        help="Leads to use: 'all' or comma-separated list (e.g., 'I,II,V2')")
    parser.add_argument("--model", type=str, default="resnet1d", choices=["resnet1d", "lightweight"],
                        help="Model architecture")
    parser.add_argument("--data_path", type=str, default="data/processed/ptbxl_processed.h5",
                        help="Path to preprocessed data")
    
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Output directory")
    
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lead-minimal-ecg",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team)")
    
    args = parser.parse_args()
    
    if args.ultra:
        train_ultra_overfit(
            leads=args.leads,
            model_name=args.model,
            epochs=args.epochs if args.epochs != 30 else 1000,
            batch_size=args.batch_size if args.batch_size != 128 else 32,
            learning_rate=args.lr if args.lr != 0.001 else 0.001,
            data_path=args.data_path,
            seed=args.seed,
            output_dir=args.output_dir,
            target_loss=args.target_loss,
        )
    elif args.overfit:
        train_overfit_mode(
            leads=args.leads,
            model_name=args.model,
            epochs=args.epochs if args.epochs != 30 else 200,
            batch_size=args.batch_size if args.batch_size != 128 else 256,
            learning_rate=args.lr if args.lr != 0.001 else 0.0005,
            data_path=args.data_path,
            seed=args.seed,
            output_dir=args.output_dir,
            target_loss=args.target_loss,
        )
    else:
        train(
            leads=args.leads,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            output_dir=args.output_dir,
            data_path=args.data_path,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )