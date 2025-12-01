#!/bin/bash
set -e

#     ./run_complete_pipeline.sh --experiment-name my_exp --stage 1
#     ./run_complete_pipeline.sh --experiment-name my_exp --stage 2
#     ./run_complete_pipeline.sh --experiment-name my_exp --stage 3
#   Full pipeline:
#     ./run_complete_pipeline.sh --experiment-name my_exp --stage all   # or omit --stage

EXPERIMENT_NAME=""
ADDITIONAL_ARGS=""
STAGE="all"   # all | 1 | 2 | 3

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name)
            EXPERIMENT_NAME="$2"; shift 2 ;;
        --stage)
            STAGE="$2"; shift 2 ;;   # values: all,1,2,3
        -h|--help)
            echo "Usage: $0 --experiment-name NAME [--stage all|1|2|3] [OPTIONS]"
            exit 0 ;;
        *)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1"; shift ;;
    esac
done

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: --experiment-name is required"
    exit 1
fi

echo "Experiment: $EXPERIMENT_NAME"
echo "Stage:      $STAGE"
echo ""

run_step() {
    local n="$1"; local label="$2"; local script="$3"
    echo "Step $n: $label"
    "./sh_files/$script"
    echo "Done step $n"
}

# STAGE 1
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo "STAGE 1: run.py"
    python run.py \
        --config configs/config.yaml \
        --n-splits 3 \
        --model logistic_regression \
        --experiment-name "$EXPERIMENT_NAME" \
        --diagonal sample_matrix \
        $ADDITIONAL_ARGS
fi

# STAGE 2
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    echo ""
    echo "STAGE 2: bridge_to_analysis.py"
    python AdvanceAnalysis/bridge_to_analysis.py --experiment "$EXPERIMENT_NAME" --force
fi

# STAGE 3
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    echo ""
    echo "STAGE 3: Advanced analysis"
    run_step 1 "Atlas Performance Analysis" "01_atlas_performance_analysis.sh"
    run_step 2 "Atlas Comparison"           "02_atlas_comparison.sh"
    run_step 3 "Connectivity Analysis"      "03_connectivity_analysis.sh"
    run_step 4 "Generate Summary Report"    "04_generate_summary_report.sh"
    run_step 5 "Region Level Analysis"      "05_region_level_analysis.sh"
fi
