#!/bin/bash
# HTCondor Hyperparameter Search Wrapper
# =======================================
# This script runs hyperparameter search and automatically uses all resources allocated by HTCondor

set -e
START_TIME=$(date +%s)
# Get iteration number from arguments
ITERATION=${1:-0}

echo "=========================================="
echo "Hyperparameter Search - Iteration $ITERATION"
echo "=========================================="
echo "Start: $(date) | Host: $(hostname)"
echo ""

# Activate virtual environment
VENV_PATH="masterthesis_venv2"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Using $VENV_PATH"
else
    echo "✗ Virtual environment not found: $VENV_PATH"
    exit 1
fi

# Validate required packages
echo ""
echo "Validating Python packages..."
python -c "import sklearn, scipy, numpy, pandas, yaml" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All packages found"
else
    echo "✗ Missing required packages"
    exit 1
fi

# ========================================
# Resource Detection
# ========================================

# Get CPUs
if [ -n "$_CONDOR_SLOT_CPUS" ]; then
    N_CPUS=$_CONDOR_SLOT_CPUS
elif [ -n "$SLURM_CPUS_PER_TASK" ]; then
    N_CPUS=$SLURM_CPUS_PER_TASK
else
    N_CPUS=$(nproc)
fi

# Get Memory (in MB)
if [ -n "$_CONDOR_SLOT_MEMORY" ]; then
    MEMORY_MB=$_CONDOR_SLOT_MEMORY
    MEMORY_GB=$((MEMORY_MB / 1024))
    MEMORY_INFO="${MEMORY_GB}GB (${MEMORY_MB}MB allocated)"
else
    # Fallback: get total system memory
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    if [ -n "$TOTAL_MEM_KB" ]; then
        MEMORY_MB=$((TOTAL_MEM_KB / 1024))
        MEMORY_GB=$((MEMORY_MB / 1024))
        MEMORY_INFO="${MEMORY_GB}GB (${MEMORY_MB}MB total system)"
    else
        MEMORY_INFO="Unknown"
    fi
fi

# Get Disk (in KB, convert to GB)
if [ -n "$_CONDOR_SLOT_DISK" ]; then
    DISK_KB=$_CONDOR_SLOT_DISK
    DISK_MB=$((DISK_KB / 1024))
    DISK_GB=$((DISK_MB / 1024))
    DISK_INFO="${DISK_GB}GB (${DISK_MB}MB allocated)"
else
    # Fallback: get available disk space
    DISK_KB=$(df -k . 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -n "$DISK_KB" ]; then
        DISK_MB=$((DISK_KB / 1024))
        DISK_GB=$((DISK_MB / 1024))
        DISK_INFO="${DISK_GB}GB (${DISK_MB}MB available)"
    else
        DISK_INFO="Unknown"
    fi
fi

# Display resource information
echo ""
echo "=========================================="
echo "Resource Configuration"
echo "=========================================="
echo "CPUs:   $N_CPUS"
echo "Memory: $MEMORY_INFO"
echo "Disk:   $DISK_INFO"
echo "=========================================="
echo ""

# Optional: Debug mode to show all HTCondor variables
if [ "$DEBUG" = "1" ]; then
    echo "HTCondor Environment Variables:"
    echo "  _CONDOR_SLOT_CPUS    = $_CONDOR_SLOT_CPUS"
    echo "  _CONDOR_SLOT_MEMORY  = $_CONDOR_SLOT_MEMORY MB"
    echo "  _CONDOR_SLOT_DISK    = $_CONDOR_SLOT_DISK KB"
    echo "  _CONDOR_SCRATCH_DIR  = $_CONDOR_SCRATCH_DIR"
    echo "  _CONDOR_JOB_AD       = $_CONDOR_JOB_AD"
    echo ""
fi

# ========================================
# Run Hyperparameter Search
# ========================================

echo "Running hyperparameter search..."
echo ""

python scripts/hyperparameter_search.py \
    --data data/raw/PIOP2_restingstate.csv \
    --model logistic_regression \
    --search-config configs/hyperparameters/logistic_regression_search.yaml \
    --output results/hyperparameter_search \
    --diagonal region_mean \
    --n-jobs $N_CPUS \
    --iteration $ITERATION 
EXIT_CODE=$?

# ========================================
# Completion Report
# ========================================

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Hyperparameter search completed successfully"
    
    # Show resource usage if available
    if [ -f "$_CONDOR_JOB_AD" ]; then
        echo ""
        echo "Final Resource Usage:"
        grep -E "MemoryUsage|DiskUsage|RemoteUserCpu|RemoteSysCpu" "$_CONDOR_JOB_AD" 2>/dev/null || true
    fi
else
    echo "✗ Hyperparameter search failed with exit code $EXIT_CODE"
fi
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Duration: $DURATION seconds ($((DURATION / 60)) minutes)"
echo "=========================================="

exit $EXIT_CODE