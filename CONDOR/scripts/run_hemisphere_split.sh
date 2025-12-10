#!/bin/bash
set -e

echo "=========================================="
echo "Hemisphere Data Split Pipeline"
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

# Verify required directories exist
if [ ! -d "data/raw" ]; then
    echo "ERROR: data/raw directory not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p data/processed/hemispheres

echo ""
echo "Starting hemisphere data split..."
echo ""

# Run the hemisphere splitting script
python scripts/hemisphere/01_prepare_hemisphere_data.py \
    --input-dir data/raw \
    --output-dir data/processed/hemispheres

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Hemisphere split completed successfully"
    echo ""
    echo "Output files generated:"
    ls -lh data/processed/hemispheres/
else
    echo "✗ Hemisphere split failed (exit code $EXIT_CODE)"
fi
echo "Duration: $SECONDS seconds"
echo "Completed: $(date)"
echo "=========================================="

exit $EXIT_CODE
