#!/bin/bash
set -e

EXPERIMENT_NAME=""
STAGE="all"
MODEL="logistic_regression"
USE_BEST_PARAMS=false
ADDITIONAL_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --use-best-params) USE_BEST_PARAMS=true; shift ;;
        -h|--help)
            cat << EOF
Usage: $0 --experiment-name NAME [OPTIONS]

Options:
  --experiment-name NAME    Experiment name (required)
  --stage [all|1|2|3]      Pipeline stage (default: all)
  --model NAME             Model config (default: logistic_regression)
  --use-best-params        Use hyperparameter search results

Stages:
  1 - Training (run.py)
  2 - Bridge to analysis
  3 - Advanced analysis
EOF
            exit 0 ;;
        *) ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1"; shift ;;
    esac
done

# Validate required arguments
[ -z "$EXPERIMENT_NAME" ] && echo "ERROR: --experiment-name is required" && exit 1

# If using best params, check and aggregate
if [ "$USE_BEST_PARAMS" = true ]; then
    echo -e "\n======================================================================\n  Using Optimized Hyperparameters\n======================================================================\n"
    
    if [ ! -d "results/hyperparameter_search" ] || [ -z "$(ls -A results/hyperparameter_search/iteration_* 2>/dev/null)" ]; then
        echo -e "ERROR: No hyperparameter search results found!\n\nRun hyperparameter search first:\n  condor_submit run_hyperparameter_search.sub"
        exit 1
    fi
    
    echo "Aggregating search results..."
    python src/aggregate_search_results.py || { echo "ERROR: Failed to aggregate results"; exit 1; }
    MODEL="best_from_search"
    echo ""
fi

# Print configuration
cat << EOF

======================================================================
  Brain Connectivity Classification Pipeline
======================================================================
Experiment:  $EXPERIMENT_NAME
Stage:       $STAGE
Model:       $MODEL

EOF

# Helper function for analysis steps
run_step() { echo "Step $1: $2"; "./sh_files/$3"; echo -e "✓ Done: $2\n"; }

# STAGE 1: Training
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo -e "===\n  STAGE 1: Model Training\n==="
    
    [ -d "results/hyperparameter_search" ] && echo "Aggregating hyperparameter results..." && python src/aggregate_search_results.py && echo ""
    
    python run.py --config configs/config.yaml --n-splits 3 --model "$MODEL" --experiment-name "$EXPERIMENT_NAME" --diagonal sample_matrix $ADDITIONAL_ARGS
    echo -e "\n✓ Training complete\n"
fi

# STAGE 2: Bridge
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    echo -e "====\n  STAGE 2: Bridge to Analysis\n===="
    python AdvanceAnalysis/bridge_to_analysis.py --experiment "$EXPERIMENT_NAME" --force
    echo -e "\n✓ Bridge complete\n"
fi

# STAGE 3: Analysis
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    echo -e "===\n  STAGE 3: Advanced Analysis\n==="
    run_step 1 "Atlas Performance Analysis" "01_atlas_performance_analysis.sh"
    run_step 2 "Atlas Comparison"           "02_atlas_comparison.sh"
    run_step 3 "Region Level Analysis"      "03_region_level_analysis.sh"
    run_step 4 "Generate Summary Report"    "04_generate_summary_report.sh"
    echo "✓ Analysis complete"
    echo ""
fi

# Final summary
cat << EOF
======================================================================
  PIPELINE COMPLETE
======================================================================

Experiment: $EXPERIMENT_NAME

Results saved to:
  - results/experiments/$EXPERIMENT_NAME/
  - reports/figures/
  - reports/tables/

EOF
