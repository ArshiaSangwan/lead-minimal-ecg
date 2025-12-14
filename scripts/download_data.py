#!/usr/bin/env python3
"""
PTB-XL Dataset Download Script

Downloads the PTB-XL electrocardiography dataset (v1.0.3) from PhysioNet.
No credentials required - this is a fully public dataset.

Reference:
    Wagner et al. "PTB-XL, a large publicly available electrocardiography dataset"
    Scientific Data, 2020. https://physionet.org/content/ptb-xl/
"""

import subprocess
import sys
import shutil
from pathlib import Path


PTBXL_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)

REQUIRED_FILES = ["ptbxl_database.csv", "scp_statements.csv", "records100/"]


def download_ptbxl(data_dir="data/ptb-xl/"):
    """Download and extract PTB-XL dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if (data_path / "ptbxl_database.csv").exists():
        print("PTB-XL already exists, skipping download.")
        return
    
    print("Downloading PTB-XL from PhysioNet (~2.5 GB)...")
    
    zip_path = data_path / "ptb-xl.zip"
    
    result = subprocess.run(f"wget -c -O {zip_path} {PTBXL_URL}", shell=True)
    if result.returncode != 0:
        sys.exit("Download failed. Check your internet connection.")
    
    print("Extracting...")
    subprocess.run(f"unzip -o {zip_path} -d {data_path}", shell=True, check=True)
    
    # Flatten the nested directory structure
    nested = data_path / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    if nested.exists():
        for item in nested.iterdir():
            target = data_path / item.name
            if not target.exists():
                item.rename(target)
        shutil.rmtree(nested)
    
    zip_path.unlink(missing_ok=True)
    
    # Sanity check
    print("Verifying...")
    for fname in REQUIRED_FILES:
        if not (data_path / fname).exists():
            sys.exit(f"ERROR: Missing {fname}")
    
    print(f"Done. Saved to {data_path.absolute()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptb-xl/")
    args = parser.parse_args()
    download_ptbxl(args.data_dir)
