#!/usr/bin/env python3
"""
Training script for Lead-Minimal ECG experiments.
Optimized for RTX 4090 with mixed precision training.
Includes W&B logging for experiment tracking.
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' for experiment tracking.")

from dataset import get_dataloaders, LEAD_NAMES
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
    mixup_alpha: float = 0.4
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
        
        total_loss += loss.item() * signals.size(0)
        
        # Store predictions (use original labels for metrics)
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            # Use rounded labels for AUROC calculation when mixup is used
            all_labels.append((labels_mixed > 0.5).float().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
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
    
    return {
        'loss': avg_loss,
        'auroc': macro_auroc,
        'auroc_per_class': auroc_per_class,
        'f1': macro_f1,
        'predictions': all_preds,
        'labels': all_labels
    }


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
    
    # Log test metrics to W&B
    if use_wandb and wandb_run:
        wandb.run.summary["test/auroc"] = test_metrics['auroc']
        wandb.run.summary["test/f1"] = test_metrics['f1']
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
                "n_leads": n_leads,
                "epochs_trained": epoch + 1
            }
        )
        artifact.add_file(str(run_dir / "best_model.pt"))
        wandb.log_artifact(artifact)
        print(f"   -> Model artifact saved to W&B")
    
    # Save results
    results = {
        'leads': leads_list if leads_list != "all" else LEAD_NAMES,
        'n_leads': n_leads,
        'model': model_name,
        'epochs_trained': epoch + 1,
        'best_val_auroc': best_val_auroc,
        'test_auroc': test_metrics['auroc'],
        'test_auroc_per_class': {cls: auc for cls, auc in zip(CLASSES, test_metrics['auroc_per_class'])},
        'test_f1': test_metrics['f1'],
        'history': history,
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
    
    # Data and model
    parser.add_argument("--leads", type=str, default="all",
                        help="Leads to use: 'all' or comma-separated list (e.g., 'I,II,V2')")
    parser.add_argument("--model", type=str, default="resnet1d", choices=["resnet1d", "lightweight"],
                        help="Model architecture")
    parser.add_argument("--data_path", type=str, default="data/processed/ptbxl_processed.h5",
                        help="Path to preprocessed data")
    
    # Training
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
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Output directory")
    
    # W&B logging
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lead-minimal-ecg",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team)")
    
    args = parser.parse_args()
    
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