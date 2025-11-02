#!/bin/bash
set -e  # stop if any step fails

echo "=== Starting Visualization DAG ==="

./sh_files/01_atlas_performance_analysis.sh
./sh_files/02_atlas_comparison.sh
./sh_files/03_connectivity_analysis.sh
./sh_files/04_generate_summary_report.sh

echo "=== DAG completed successfully ==="