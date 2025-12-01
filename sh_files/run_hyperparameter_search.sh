#!/bin/bash
set -e

ITERATION=${1:-0}

echo "=========================================="
echo "Hyperparameter Search - Iteration $ITERATION"
echo "=========================================="
echo "Start: $(date) | Host: $(hostname)"
echo ""

# Activate venv
if [ -d "masterthesis_venv2/bin" ]; then
    source masterthesis_venv2/bin/activate
    echo "✓ Using masterthesis_venv2"
elif [ -d "brain_connectivity_classifier/bin" ]; then
    source brain_connectivity_classifier/bin/activate
    echo "✓ Using brain_connectivity_classifier"
fi
echo ""

# Validate packages
echo "Validating Python packages..."
python -c "import sklearn, pandas, numpy, scipy" || { echo "ERROR: Missing packages"; exit 1; }
echo "✓ All packages found"
echo ""

# Run hyperparameter search (NOT aggregation!)
echo "Running hyperparameter search..."
echo ""

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
echo "Iteration $ITERATION completed: Exit $EXIT_CODE"
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE