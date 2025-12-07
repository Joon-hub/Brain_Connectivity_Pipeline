#!/usr/bin/env bash
set -euo pipefail

ALPHA="$1"
source masterthesis_venv2/bin/activate

OUTDIR="results/alpha_${ALPHA//./_}"
mkdir -p "$OUTDIR"

echo "Running alpha=$ALPHA on $(hostname) at $(date)"
python scripts/precision_alpha_tuning.py --alpha "$ALPHA" --fold 3
echo "Finished alpha=$ALPHA"