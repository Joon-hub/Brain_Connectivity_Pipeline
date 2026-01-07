#!/bin/bash
set -e


echo "=== FC Training | Host: $(hostname) | $(date) | CPUs: $(nproc) ==="


# Setup Paths & Variables
VENV_PATH="${PWD}/masterthesis_venv2/bin/activate"
CONFIG_FILE="configs/FC_config.yaml"
FC="ovr"


# Check & Activate Venv
[ -f "$VENV_PATH" ] && source "$VENV_PATH" || { echo "ERROR: Venv missing at $VENV_PATH"; exit 1; }


# # Check Required Paths (Compact Loop)
# REQUIRED=("data/processed/FCs" "configs" "scripts/FC/02_train_FC_ovr.py" "$CONFIG_FILE")
# for item in "${REQUIRED[@]}"; do
#     [ ! -e "$item" ] && echo "ERROR: Missing $item" && exit 1
# done


echo "✓ Environment ready. Starting training..."


# Run Training
python scripts/full_connectivity/02_train_FC_ovr.py \
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
else
    echo "✗ Training failed (exit code $EXIT_CODE)"
    [ -d "logs" ] && echo "Latest logs:" && ls -lt logs/ | head -3
fi


echo ""
echo "Duration: $SECONDS seconds ($(($SECONDS / 60)) minutes)"
echo "Completed: $(date)"
echo "=========================================="


exit $EXIT_CODE
