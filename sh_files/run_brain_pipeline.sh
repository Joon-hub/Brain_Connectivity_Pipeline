#!/bin/bash
# ===============================================================
# Wrapper Script for HTCondor - Brain Connectivity Classification
# Full Dataset Mode
# ===============================================================

set -e  # Exit on error
JOBNAME=$1

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

# 
set -e

# activate the brain_connectivity_classifier environment
source /home/sjoon/projects/brain_connectivity_classifier/masterthesis_venv2/bin/activate

# create a logs directory if it doesn't exist
mkdir -p logs

# echo "Running: python run.py --config config.yaml --help"
# python run.py --config configs/config.yaml --model logistic_regression --diagonal zero --n_splits 3 --sample

python run.py \
    --config configs/config.yaml \
    --sample \
    --experiment-name "my_test" \
    --model logistic_regression \
    --n-splits 3 

# Wrap up
echo
echo "==========================================="
echo "Pipeline finished!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==========================================="

exit $EXIT_CODE
