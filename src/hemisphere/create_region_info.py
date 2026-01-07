#!/usr/bin/env python3
"""
Create region_info.csv from per_region_metrics.csv

Run:
    python create_region_info.py --hemi lh
    python create_region_info.py --hemi rh
    python create_region_info.py
"""

import argparse
import pandas as pd
from pathlib import Path

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Create region_info.csv")
parser.add_argument(
    "--hemi",
    choices=["lh", "rh"],
    help="Hemisphere: lh or rh (optional)"
)
args = parser.parse_args()

# -----------------------------
# Paths
# -----------------------------
per_region_path = Path(
    "/home/sjoon/projects/brain_connectivity_classifier/data/results/"
    "hemisphere_analysis/left_hemisphere/ovr/per_region_metrics.csv"
)

base_output_dir = Path(
    "/home/sjoon/projects/brain_connectivity_classifier/data"
)

if args.hemi == "lh":
    output_name = "LH_region_info.csv"
elif args.hemi == "rh":
    output_name = "RH_region_info.csv"
else:
    output_name = "region_info.csv"

output_path = base_output_dir / output_name

print(f"\nCreating {output_name}")

# -----------------------------
# Load data
# -----------------------------
if not per_region_path.exists():
    raise FileNotFoundError(f"Input file not found: {per_region_path}")

per_region = pd.read_csv(per_region_path)

# -----------------------------
# Network handling
# -----------------------------
if "network" not in per_region.columns:
    def extract_network(name):
        name = name.replace("LH_", "").replace("RH_", "")
        name = name.replace("-lh", "").replace("-rh", "")
        parts = name.split("_")
        return "_".join(parts[:-1]) if parts[-1].isdigit() else name

    per_region["network"] = per_region["region_name"].apply(extract_network)

# -----------------------------
# Create region_info
# -----------------------------
region_info = (
    per_region[["region_name", "network"]]
    .assign(region_index=lambda df: range(len(df)))
    [["region_index", "region_name", "network"]]
)

# -----------------------------
# Validation
# -----------------------------
assert region_info["region_index"].tolist() == list(range(len(region_info)))
assert region_info["region_name"].is_unique
assert not region_info.isnull().any().any()

# -----------------------------
# Save
# -----------------------------
output_path.parent.mkdir(parents=True, exist_ok=True)
region_info.to_csv(output_path, index=False)

# Also save next to input file
backup_path = per_region_path.parent / output_name
region_info.to_csv(backup_path, index=False)

print(f"Saved:")
print(f"  {output_path}")
print(f"  {backup_path}")