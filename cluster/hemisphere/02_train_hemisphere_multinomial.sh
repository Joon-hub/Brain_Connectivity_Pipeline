#!/bin/bash
set -e

echo "=== Hemisphere Training | Host: $(hostname) | $(date) | CPUs: $(nproc) ==="

# Setup Paths & Variables
VENV_PATH="${PWD}/masterthesis_venv2/bin/activate"
CONFIG_FILE="configs/hemisphere_config.yaml"
HEMISPHERE="both"
RESULTS_DIR="data/results/hemisphere_analysis/${HEMISPHERE}_hemisphere/multinomial"

# Check & Activate Venv
[ -f "$VENV_PATH" ] && source "$VENV_PATH" || { echo "ERROR: Venv missing at $VENV_PATH"; exit 1; }


# Create Output Directories
mkdir -p data/results/hemisphere_analysis logs

# Check Required Paths (Compact Loop)
REQUIRED=("data/processed/hemispheres" "configs" "scripts/hemisphere/02_train_hemisphere_multinomial.py" "$CONFIG_FILE")
for item in "${REQUIRED[@]}"; do
    [ ! -e "$item" ] && echo "ERROR: Missing $item" && exit 1
done

echo "✓ Environment ready. Starting training..."

# Run Training
python scripts/hemisphere/02_train_hemisphere_multinomial.py \
      --hemisphere both \
      --tune_hyperparams \
      --optuna_trials 50 \
      --n_folds 5 \
      --test_on_task
    

EXIT_CODE=$?

# Summary
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
    echo "Results: $RESULTS_DIR"
    
    if [ -d "$RESULTS_DIR" ]; then
        echo ""
        echo "Generated files:"
        ls -lh "$RESULTS_DIR" | head -20
        echo ""
        echo "Total: $(ls -1 "$RESULTS_DIR" | wc -l) files"
        
        # Show accuracy if available
        if [ -f "$RESULTS_DIR/overall_metrics.json" ]; then
            ACC=$(grep -o '"accuracy": [0-9.]*' "$RESULTS_DIR/overall_metrics.json" | head -1 | cut -d' ' -f2)
            echo "Accuracy: $ACC"
        fi
    fi
else
    echo "✗ Training failed (exit code $EXIT_CODE)"
    [ -d "logs" ] && echo "Latest logs:" && ls -lt logs/ | head -3
fi

echo ""
echo "Duration: $SECONDS seconds ($(($SECONDS / 60)) minutes)"
echo "Completed: $(date)"
echo "=========================================="

exit $EXIT_CODE