#!/bin/bash
set -e

# Default values
EXPERIMENT_NAME=""
USE_BEST_PARAMS=false
STAGE="all"
N_SPLITS=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --use-best-params) USE_BEST_PARAMS=true; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --n-splits) N_SPLITS="$2"; shift 2 ;;
        --help)
            cat << EOF
Usage: $0 --experiment-name NAME [OPTIONS]

Required:
  --experiment-name NAME    Name for this experiment

Optional:
  --use-best-params         Use optimized hyperparameters
  --stage [all|1|2|3]       Which stage to run (default: all)
  --n-splits N              CV splits (default: 5)

Stages: 1=Train | 2=Prepare | 3=Analyze
EOF
            exit 0 ;;
        *) echo "Unknown option: $1 (use --help)"; exit 1 ;;
    esac
done

# Validate inputs
[ -z "$EXPERIMENT_NAME" ] && { echo "ERROR: --experiment-name required"; exit 1; }
[[ ! "$STAGE" =~ ^(all|1|2|3)$ ]] && { echo "ERROR: Invalid stage '$STAGE'"; exit 1; }

# Activate environment
source masterthesis_venv2/bin/activate || { echo "ERROR: Virtual environment not found!"; exit 1; }

# Determine model
MODEL="logistic_regression"
if [ "$USE_BEST_PARAMS" = true ]; then
    [ ! -d "results/hyperparameter_search/iteration_001" ] && { echo "ERROR: No search results found!"; exit 1; }
    python scripts/aggregate_search_results.py
    MODEL="best_from_search"
fi

echo "========================================================================"
echo "Experiment: $EXPERIMENT_NAME | Model: $MODEL | Stage: $STAGE | CV: $N_SPLITS"
echo "========================================================================"

# Stage 1: Train
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo "STAGE 1: Training..."
    python scripts/run.py \
        --config configs/config.yaml \
        --experiment-name "$EXPERIMENT_NAME" \
        --model "$MODEL" \
        --diagonal region_mean \
        --n-splits "$N_SPLITS"
    echo "✓ Training complete"
fi

# Stage 2: Prepare
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    echo "STAGE 2: Preparing data..."
    python analysis/bridge_to_analysis.py --experiment "$EXPERIMENT_NAME" --force
    echo "✓ Data preparation complete"
fi

# Stage 3: Analyze
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    echo "STAGE 3: Running analysis..."
    for script in ./sh_files/0{1..4}_*.sh; do
        $script && echo "✓ $(basename $script) done"
    done
fi

echo "========================================================================"
echo "PIPELINE COMPLETE! Results in: results/experiments/$EXPERIMENT_NAME/"
echo "========================================================================"
