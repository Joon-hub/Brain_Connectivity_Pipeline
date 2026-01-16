#!/bin/bash
set -e


echo "=== FC Training | Host: $(hostname) | $(date) | CPUs: $(nproc) ==="


# Setup Paths & Variables
VENV_PATH="${PWD}/masterthesis_venv2/bin/activate"
CONFIG_FILE="configs/FC_config.yaml"
FC="multinomial"


# Check & Activate Venv
[ -f "$VENV_PATH" ] && source "$VENV_PATH" || { echo "ERROR: Venv missing at $VENV_PATH"; exit 1; }



# Check Required Paths (Compact Loop)
# REQUIRED=("data/processed/FCs" "configs" "scripts/full_connectivity/02_train_FC_multinomial.py" "$CONFIG_FILE")
# for item in "${REQUIRED[@]}"; do
#     [ ! -e "$item" ] && echo "ERROR: Missing $item" && exit 1
# done


echo "✓ Environment ready. Starting training..."


# Run script with hyperparameter tuning
python scripts/full_connectivity/01_train_FC_multinomial.py \
    --tune_hyperparams \
    --test_on_task \
    --optuna_trials 30 \
    --final_optuna_trials 30 \
    --n_folds 5 \
    --n_jobs -1 \
    --optuna_n_jobs 1 \
    --diagonal_strategy region_mean \
    --verbose

# Run script without hyperparameter tuning (using defaults)
# python scripts/full_connectivity/01_train_FC_multinomial.py \
#     --test_on_task \
#     --n_folds 3 \
#     --n_jobs -1 \
#     --diagonal_strategy region_mean \
#     --regularization_C 0.03 \
#     --verbose

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
