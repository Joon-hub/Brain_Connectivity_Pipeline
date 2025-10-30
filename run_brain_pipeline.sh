#!/bin/bash
# ===============================================================
# Wrapper Script for HTCondor - Brain Connectivity Classification
# Full Dataset Mode
# ===============================================================

set -e  # Exit on error

# --- Prepare directories ---
mkdir -p data/raw
mkdir -p data/processed  
mkdir -p reports/tables
mkdir -p reports/figures
mkdir -p logs

echo "==========================================="
echo "Starting Brain Connectivity Classification"
echo "==========================================="
echo "Current directory: $(pwd)"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "==========================================="

# Run full pipeline
echo "Running: python run.py --config config.yaml --help"
python3 run.py --config config.yaml --sample

EXIT_CODE=$?

# Wrap up
echo
echo "==========================================="
echo "Pipeline finished!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==========================================="

exit $EXIT_CODE
