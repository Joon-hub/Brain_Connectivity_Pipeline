#!/bin/bash
set -e

echo "=========================================="
echo "Hyperparameter Search - Nested CV"
echo "Host: $(hostname) | Start: $(date)"
echo "CPUs: $(nproc) | Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "=========================================="


# Activate virtual environment with absolute path
VENV_PATH="${PWD}/masterthesis_venv2/bin/activate"
echo "Checking virtual environment at: $VENV_PATH"

if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "✓ Virtual environment activated successfully"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Run hyperparameter search (all outer folds)
python scripts/hyperparameter_search.py \
    --search-config configs/hyperparameters/logistic_regression_search.yaml \
    --model logistic_regression \
    --diagonal region_mean \
    --outer-cv 3 \
    --inner-cv 3 \
    --n-iter 10 \
    --n-jobs $(nproc) \
    --seed 42 \
    --output "results/hyperparameter_search"

EXIT_CODE=$?

echo "=========================================="
[ $EXIT_CODE -eq 0 ] && echo "✓ Success" || echo "✗ Failed (exit code $EXIT_CODE)"
echo "Duration: $SECONDS seconds"
echo "=========================================="

exit $EXIT_CODE
