#!/bin/bash
set -euo pipefail   # Better than just set -e

# Default alpha value if not provided
ALPHA="${1:-0.1}"

echo "========================================================================"
echo "  ALPHA TUNING - SINGLE JOB"
echo "========================================================================"
echo "Alpha value: $ALPHA"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S') | Host: $(hostname)"
echo ""

# === Activate virtual environment ===
VENV_DIR="masterthesis_venv2"

if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "✓ Activated virtual environment: $VENV_DIR"
else
    echo "ERROR: Virtual environment not found at '$VENV_DIR'"
    echo "   Expected activate script: $VENV_DIR/bin/activate"
    exit 1
fi

echo ""

# === Create output directory ===
OUTPUT_DIR="results/alpha_tuning/alpha_${ALPHA//./_}"  # Replace dot to avoid issues in filenames
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# === Run the Python script ===
echo "Running precision_alpha_tuning.py with alpha = $ALPHA ..."
echo ""

python scripts/precision_alpha_tuning.py --alpha "$ALPHA"
EXIT_CODE=$?

# Deactivate venv (optional but clean)
deactivate 2>/dev/null || true

echo ""
echo "========================================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Alpha $ALPHA completed SUCCESSFULLY"
else
    echo "Alpha $ALPHA FAILED with exit code: $EXIT_CODE"
fi
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================================================"

exit $EXIT_CODE