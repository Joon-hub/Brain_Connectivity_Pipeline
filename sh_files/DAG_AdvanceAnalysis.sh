#!/bin/bash
set -e  # Exit on error

echo "============================================================================"
echo "  ADVANCE ANALYSIS PIPELINE (WITH INTEGRATED BRIDGE)"
echo "============================================================================"
echo "Start time: $(date)"
echo ""

EXPERIMENT_NAME="$1"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "No experiment name provided, using most recent..."
    if [ -d "results/experiments" ]; then
        EXPERIMENT_NAME=$(ls -t results/experiments | head -1)
        if [ -z "$EXPERIMENT_NAME" ]; then
            echo "ERROR: No experiments found in results/experiments/"
            echo "Usage: $0 <experiment_name>"
            exit 1
        fi
        echo "Using experiment: $EXPERIMENT_NAME"
    else
        echo "ERROR: results/experiments/ directory not found!"
        echo "Run: python run.py --experiment-name \"my_experiment\""
        exit 1
    fi
else
    echo "Using experiment: $EXPERIMENT_NAME"
fi

echo ""
echo "============================================================================"
echo "STEP 0: BRIDGE - Prepare data/processed/"
echo "============================================================================"

python AdvanceAnalysis/bridge_to_analysis.py --experiment "$EXPERIMENT_NAME" --force
echo "Data copied from current experiment to data/processed/"
echo ""

run_step() {
    local step_num="$1"
    local desc="$2"
    local script="$3"

    echo "============================================================================"
    echo "STEP $step_num: $desc"
    echo "============================================================================"
    "./sh_files/$script"
    echo "Step $step_num completed"
    echo ""
}

run_step 1 "Atlas Performance Analysis" "01_atlas_performance_analysis.sh"
run_step 2 "Atlas Comparison"           "02_atlas_comparison.sh"
run_step 3 "Region Level Analysis"      "03_region_level_analysis.sh"
run_step 4 "Generate Summary Report"    "04_generate_summary_report.sh"


echo "============================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "============================================================================"
echo "End time: $(date)"
echo ""
echo "Experiment analyzed: $EXPERIMENT_NAME"
echo ""
echo "Outputs:"
echo "  Figures: reports/figures/"
echo "  Tables:  reports/tables/"
echo "  Summary: reports/summary/"
echo ""
