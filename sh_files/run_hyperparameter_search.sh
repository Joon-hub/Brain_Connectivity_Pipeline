#!/bin/bash
# HTCondor Execution Script for Hyperparameter Search
# ====================================================

set -e  # Exit on error

ITERATION=$1

echo "=========================================="
echo "Hyperparameter Search - Iteration $ITERATION"
echo "=========================================="
echo "Start: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [ -d "masterthesis_venv2" ]; then
    source masterthesis_venv2/bin/activate
    echo "✓ Virtual environment activated"
elif [ -d "brain_connectivity_classifier" ]; then
    source brain_connectivity_classifier/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found, using system Python"
fi

# Print Python info
echo ""
echo "Python version:"
python --version
echo ""
echo "Python location:"
which python
echo ""

# Check if required packages are available
echo "Checking required packages..."
python -c "import sklearn; print(f'✓ sklearn {sklearn.__version__}')" || echo "✗ sklearn not found"
python -c "import pandas; print(f'✓ pandas {pandas.__version__}')" || echo "✗ pandas not found"
python -c "import numpy; print(f'✓ numpy {numpy.__version__}')" || echo "✗ numpy not found"
python -c "import scipy; print(f'✓ scipy {scipy.__version__}')" || echo "✗ scipy not found"
echo ""

# Run hyperparameter search for this iteration
echo "=========================================="
echo "Running Random Search - Iteration $ITERATION"
echo "=========================================="

python src/hyperparameter_search.py \
    --data data/raw/PIOP2_restingstate.csv \
    --model logistic_regression \
    --search-config configs/hyperparameters/logistic_regression_search.yaml \
    --output results/hyperparameter_search \
    --diagonal sample_matrix \
    --n-jobs 4 \
    --iteration ${ITERATION}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Iteration $ITERATION completed with exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE