#!/bin/bash
# Devil Chain Monitor - Waits for current devil to finish and launches next phases
# Usage: ./devil-chain-monitor.sh <dir1> <dir2> [dir3] ...

# BUG: `ct05`       SLOW ~10 hours !!!
#      `gn23-gn25`  CRASH !!!

show_usage() {
    echo "Usage: $0 <directory1> <directory2> [directory3] ..."
    echo ""
    echo "Examples:"
    echo "  $0 e-4 e-5 e-6 e-7"
    echo "  $0 phase1 phase2"
    echo "  $0 test-run"
    echo ""
    echo "Runs VASP devil jobs sequentially through the specified directories."
}

# Check if help requested or no arguments
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Validate directories exist
echo "Validating directories..."
for dir in "$@"; do
    if [[ ! -d "$dir" ]]; then
        echo "Error: Directory '$dir' does not exist"
        exit 1
    fi
    echo "✓ Found directory: $dir"
done

echo ""
echo "========================================"
echo "Starting Devil Chain Monitor"
echo "Phases to process: $*"
echo "========================================"

# Activate venv
py312

# Common VASP parameters for all phases
NTASKS=8
NCORE=8
KPAR=1
ALGO="Normal"
MAX_JOBS=63
SLEEP_TIME=30
VASP_SETUP="source /trinity/home/p.zhilyaev/mklk/scripts-run/sif-cpu-ifort.sh"
NODELIST="ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn19,gn20,gn21"

# Process each phase sequentially
phase_num=1
total_phases=$#
failed_phases=()

for phase_dir in "$@"; do
    echo ""
    echo "========================================"
    echo "Phase $phase_num/$total_phases: Starting $phase_dir"
    echo "========================================"
    
    # Launch the devil job
    vsf-submit-devil.py "$phase_dir/*" $NTASKS $NCORE $KPAR $ALGO \
        --max-jobs $MAX_JOBS \
        --sleep-time $SLEEP_TIME \
        --vasp-setup "$VASP_SETUP" \
        --nodelist="$NODELIST" \
        --state-file "devil-phase-$(basename "${phase_dir%/}").json" &
    
    CURRENT_PID=$!
    
    # Check if the job started successfully
    sleep 2
    if ! kill -0 $CURRENT_PID 2>/dev/null; then
        echo "✗ ERROR: Phase $phase_num ($phase_dir) failed to start"
        failed_phases+=("$phase_dir")
        ((phase_num++))
        continue
    fi
    
    echo "✓ Phase $phase_num ($phase_dir) started with PID: $CURRENT_PID"
    echo "Waiting for Phase $phase_num ($phase_dir) to complete..."
    
    # Wait for current phase to finish
    while kill -0 $CURRENT_PID 2>/dev/null; do
        echo "$(date '+%H:%M:%S') - Phase $phase_num ($phase_dir) still running (PID: $CURRENT_PID)"
        sleep 120  # Check every 2 minutes
    done
    
    # Check exit status
    wait $CURRENT_PID
    exit_status=$?
    
    if [[ $exit_status -eq 0 ]]; then
        echo "✓ Phase $phase_num ($phase_dir) completed successfully at $(date)"
    else
        echo "✗ Phase $phase_num ($phase_dir) completed with errors (exit code: $exit_status) at $(date)"
        failed_phases+=("$phase_dir")
    fi
    
    ((phase_num++))
    
    # Don't show "Starting next phase" message for the last phase
    if [[ $phase_num -le $total_phases ]]; then
        echo "Starting next phase..."
    fi
done

# Final summary
echo ""
echo "========================================"
echo "Devil Chain Monitor Completed at $(date)"
echo "========================================"
echo "Total phases processed: $total_phases"
echo "Successful phases: $((total_phases - ${#failed_phases[@]}))"

if [[ ${#failed_phases[@]} -gt 0 ]]; then
    echo "Failed phases: ${#failed_phases[@]}"
    echo "Failed directories:"
    for failed_dir in "${failed_phases[@]}"; do
        echo "  - $failed_dir"
    done
    exit 1
else
    echo "All phases completed successfully! 🎉"
fi
