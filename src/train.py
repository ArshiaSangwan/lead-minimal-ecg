#!/usr/bin/env python3
"""
Training script for Lead-Minimal ECG experiments.
Optimized for RTX 4090 with mixed precision training.
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
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from dataset import get_dataloaders, LEAD_NAMES
from model import get_model, count_parameters


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
    use_amp: bool = True
) -> dict:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for signals, labels in pbar:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(signals)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * signals.size(0)
        
        # Store predictions
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
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
        
        with autocast(enabled=use_amp):
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
    seed: int = 42
):
    """Main training function."""
    
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
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        h5_path=data_path,
        leads=leads_list,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )
    
    # Model
    print("\nBuilding model...")
    model = get_model(
        model_name=model_name,
        n_leads=n_leads,
        n_classes=5,
        base_filters=64,
        kernel_size=15,
        num_blocks=4,
        dropout=0.2
    )
    model = model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    
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
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
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
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(run_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    print(f"\nTest Results:")
    print(f"  Macro AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Macro F1: {test_metrics['f1']:.4f}")
    print(f"\n  Per-class AUROC:")
    for i, cls in enumerate(classes):
        print(f"    {cls}: {test_metrics['auroc_per_class'][i]:.4f}")
    
    # Save results
    results = {
        'leads': leads_list if leads_list != "all" else LEAD_NAMES,
        'n_leads': n_leads,
        'model': model_name,
        'epochs_trained': epoch + 1,
        'best_val_auroc': best_val_auroc,
        'test_auroc': test_metrics['auroc'],
        'test_auroc_per_class': {cls: auc for cls, auc in zip(classes, test_metrics['auroc_per_class'])},
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
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lead-Minimal ECG Model")
    
    parser.add_argument("--leads", type=str, default="all",
                        help="Leads to use: 'all' or comma-separated list (e.g., 'I,II,V2')")
    parser.add_argument("--model", type=str, default="resnet1d", choices=["resnet1d", "lightweight"],
                        help="Model architecture")
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
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Output directory")
    parser.add_argument("--data_path", type=str, default="data/processed/ptbxl_processed.h5",
                        help="Path to preprocessed data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
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
        seed=args.seed
    )
