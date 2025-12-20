#!/bin/bash
# Devil Chain Monitor - Waits for current devil to finish and launches next phases

# BUG: `ct05`       SLOW ~10 hours !!!
#      `gn23-gn25`  CRASH !!!

# Activate venv
py312

# Phase 1 - Adjust these parameters as needed
vsf-submit-devil.py "e-1/*" 8 8 1 Normal \
    --max-jobs 63 \
    --sleep-time 30 \
    --vasp-setup "source /trinity/home/p.zhilyaev/mklk/scripts-run/sif-cpu-ifort.sh" \
    --nodelist=ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn19,gn20,gn21 \
    --state-file "devil-phase1.json" &

PHASE1_PID=$!
echo "✓ Phase 1 started with PID: $PHASE1_PID"
echo "Waiting for Phase 1 (e-1/*) to complete..."

# Wait for Phase 1 to finish
while kill -0 $PHASE1_PID 2>/dev/null; do
    echo "$(date '+%H:%M:%S') - Phase 1 still running (PID: $PHASE1_PID)"
    sleep 120  # Check every 2 minutes
done

echo ""
echo "========================================"
echo "✓ Phase 1 completed at $(date)"
echo "========================================"
echo "Starting Phase 2..."


# Phase 2 - Adjust these parameters as needed
vsf-submit-devil.py "e-2/*" 8 8 1 Normal \
    --max-jobs 63 \
    --sleep-time 30 \
    --vasp-setup "source /trinity/home/p.zhilyaev/mklk/scripts-run/sif-cpu-ifort.sh" \
    --nodelist=ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn19,gn20,gn21 \
    --state-file "devil-phase2.json" &

PHASE2_PID=$!
echo "✓ Phase 2 started with PID: $PHASE2_PID"
echo "Waiting for Phase 2 (e-2/*) to complete..."

# Wait for Phase 2 to finish
while kill -0 $PHASE2_PID 2>/dev/null; do
    echo "$(date '+%H:%M:%S') - Phase 2 still running (PID: $PHASE2_PID)"
    sleep 120  # Check every 2 minutes
done

echo ""
echo "========================================"
echo "✓ Phase 2 completed at $(date)"
echo "========================================"
echo "Starting Phase 3..."


# Phase 3 - Adjust these parameters as needed
vsf-submit-devil.py "e-3/*" 8 8 1 Normal \
    --max-jobs 63 \
    --sleep-time 30 \
    --vasp-setup "source /trinity/home/p.zhilyaev/mklk/scripts-run/sif-cpu-ifort.sh" \
    --nodelist=ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn19,gn20,gn21 \
    --state-file "devil-phase3.json" &

PHASE3_PID=$!
echo "✓ Phase 3 started with PID: $PHASE3_PID"
echo "Waiting for Phase 3 (e-3/*) to complete..."

# Wait for Phase 3 to finish
while kill -0 $PHASE3_PID 2>/dev/null; do
    echo "$(date '+%H:%M:%S') - Phase 3 still running (PID: $PHASE3_PID)"
    sleep 120  # Check every 2 minutes
done

echo ""
echo "========================================"
echo "✓ Phase 3 completed at $(date)"
echo "========================================"
echo "Starting Phase 4..."

# Phase 4 - Adjust these parameters as needed
vsf-submit-devil.py "e-4/*" 8 8 1 Normal \
    --max-jobs 63 \
    --sleep-time 30 \
    --vasp-setup "source /trinity/home/p.zhilyaev/mklk/scripts-run/sif-cpu-ifort.sh" \
    --nodelist=ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn19,gn20,gn21 \
    --state-file "devil-phase4.json"

echo ""
echo "========================================"
echo "✓ All phases completed at $(date)"
echo "========================================"
