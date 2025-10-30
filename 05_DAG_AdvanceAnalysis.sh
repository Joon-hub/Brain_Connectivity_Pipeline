#!/bin/bash
set -e  # stop if any step fails

echo "=== Starting Visualization DAG ==="

./01_atlas_performance_analysis.sh
./02_atlas_comparison.sh
./03_connectivity_analysis.sh
./04_generate_summary_report.sh

echo "=== DAG completed successfully ==="