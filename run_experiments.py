#!/usr/bin/env python3
"""
Master script to run all Lead-Minimal ECG experiments.
Trains baseline + all lead subsets, evaluates, and generates figures.

Usage:
    python run_experiments.py --full        # Run all experiments
    python run_experiments.py --quick       # Quick test (fewer epochs)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time


# Lead configurations to evaluate
LEAD_CONFIGS = {
    # Baseline
    "all": "all",
    
    # Single leads (clinically important)
    "II": "II",
    "V2": "V2",
    "I": "I",
    "V5": "V5",
    
    # 2-lead combinations
    "I_II": "I,II",
    "II_V2": "II,V2",
    
    # 3-lead combinations
    "I_II_V2": "I,II,V2",
    "I_II_III": "I,II,III",
    "II_V2_V5": "II,V2,V5",
    
    # 6-lead (limb leads only)
    "limb": "I,II,III,aVR,aVL,aVF",
}


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    
    print(f"Done: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Lead-Minimal ECG Experiments")
    parser.add_argument("--full", action="store_true", help="Run full experiments (30 epochs)")
    parser.add_argument("--quick", action="store_true", help="Quick test run (5 epochs)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--skip_download", action="store_true", help="Skip data download")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lead-minimal-ecg")
    parser.add_argument("--configs", nargs="+", default=None, 
                        help="Specific configs to run (e.g., 'all II I_II')")
    
    args = parser.parse_args()
    
    # Determine epochs
    if args.quick:
        epochs = 5
    elif args.full:
        epochs = 30
    else:
        epochs = args.epochs
    
    print("\n" + "="*60)
    print("LEAD-MINIMAL ECG EXPERIMENTS")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Configurations: {list(LEAD_CONFIGS.keys())}")
    
    start_time = time.time()
    
    # Step 1: Download data
    if not args.skip_download:
        if not run_command(
            "python scripts/download_data.py",
            "Downloading PTB-XL Dataset"
        ):
            print("Failed to download data. Exiting.")
            sys.exit(1)
    
    # Step 2: Preprocess
    if not args.skip_preprocess:
        if not run_command(
            "python src/preprocess.py",
            "Preprocessing PTB-XL"
        ):
            print("Failed to preprocess. Exiting.")
            sys.exit(1)
    
    # Step 3: Train all configurations
    configs_to_run = args.configs if args.configs else list(LEAD_CONFIGS.keys())
    
    for config_name in configs_to_run:
        if config_name not in LEAD_CONFIGS:
            print(f"Warning: Unknown config: {config_name}")
            continue
        
        leads = LEAD_CONFIGS[config_name]
        
        wandb_flag = "--no_wandb" if args.no_wandb else f"--wandb_project {args.wandb_project}"
        run_command(
            f"python src/train.py --leads {leads} --epochs {epochs} --batch_size 128 {wandb_flag}",
            f"Training: {config_name} ({leads})"
        )
    
    # Step 4: Evaluate all models
    run_command(
        "python src/evaluate.py --detailed",
        "Evaluating All Models"
    )
    
    # Step 5: Generate figures
    run_command(
        "python src/visualize.py",
        "Generating Figures"
    )
    
    # Summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Total time: {hours}h {minutes}m")
    print(f"\nOutputs:")
    print(f"  Models:  outputs/models/")
    print(f"  Figures: outputs/figures/")
    print(f"  Results: outputs/evaluation_summary.json")
    print("="*60)


if __name__ == "__main__":
    main()
