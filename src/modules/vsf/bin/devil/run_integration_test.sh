#!/bin/bash
# Integration test for VASP Devil with real SLURM + mock VASP jobs
# This tests the full orchestration without actually running VASP

set -e

echo "============================================================"
echo "VASP Devil - Integration Test with SLURM"
echo "============================================================"

# Configuration
NUM_STRUCTURES=3
JOB_DURATION=30  # seconds per mock job
WORKFLOW="double"
MAX_JOBS=2
SLEEP_TIME=10

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check SLURM is available
echo -e "\n${BLUE}[1/6] Checking SLURM availability${NC}"
if ! command -v sbatch &> /dev/null; then
    echo "✗ sbatch not found. Please install SLURM."
    exit 1
fi

if ! command -v squeue &> /dev/null; then
    echo "✗ squeue not found. Please install SLURM."
    exit 1
fi

echo "✓ SLURM commands available"
echo "  sbatch: $(which sbatch)"
echo "  squeue: $(which squeue)"

# Step 2: Create test directories
echo -e "\n${BLUE}[2/6] Creating test structure directories${NC}"

TEST_DIR="./integration_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

for i in $(seq 1 $NUM_STRUCTURES); do
    CALC_DIR="$TEST_DIR/struct_$i"
    mkdir -p "$CALC_DIR"
    
    # Create minimal VASP input files
    cat > "$CALC_DIR/INCAR" << EOF
SYSTEM = Mock structure $i
NCORE = 4
KPAR = 2
NSW = 10
IBRION = 2
EOF
    
    cat > "$CALC_DIR/POSCAR" << EOF
Mock structure $i
1.0
  5.5 0.0 0.0
  0.0 5.5 0.0
  0.0 0.0 5.5
Si
  2
Direct
  0.0 0.0 0.0
  0.25 0.25 0.25
EOF
    
    echo "Mock POTCAR" > "$CALC_DIR/POTCAR"
    
    cat > "$CALC_DIR/KPOINTS" << EOF
Automatic mesh
0
Gamma
2 2 2
0 0 0
EOF
    
    echo "  ✓ Created $CALC_DIR"
done

echo "✓ Created $NUM_STRUCTURES test structures in $TEST_DIR"

# Step 3: Setup mock submission script
echo -e "\n${BLUE}[3/6] Setting up mock vsf-submit-job.py${NC}"

# Check if mock script exists
if [ ! -f "vsf-submit-job-mock.py" ]; then
    echo "✗ vsf-submit-job-mock.py not found!"
    echo "  Please ensure it's in the current directory"
    exit 1
fi

# Make it executable
chmod +x vsf-submit-job-mock.py

# Create symlink so devil can find it
if [ -L "vsf-submit-job.py" ] || [ -f "vsf-submit-job.py" ]; then
    rm -f vsf-submit-job.py
fi
ln -s vsf-submit-job-mock.py vsf-submit-job.py

echo "✓ Created symlink: vsf-submit-job.py -> vsf-submit-job-mock.py"

# Step 4: Run dry-run first
echo -e "\n${BLUE}[4/6] Running dry-run to verify setup${NC}"

python vasp_devil_prototype.py "$TEST_DIR/struct_*" \
    4 4 1 Normal \
    --vasp-setup "echo 'Mock VASP setup'" \
    --workflow "$WORKFLOW" \
    --max-jobs "$MAX_JOBS" \
    --state-file "$TEST_DIR/state.json" \
    --dry-run

echo -e "\n${GREEN}✓ Dry-run successful${NC}"

# Step 5: Ask user to confirm
echo -e "\n${YELLOW}[5/6] Ready to run full integration test${NC}"
echo ""
echo "This will:"
echo "  • Submit $NUM_STRUCTURES workflows ($WORKFLOW type)"
echo "  • Each workflow has 5 stages (relax1, validate, copy, relax2, validate)"
echo "  • Each SLURM job will run for ~${JOB_DURATION}s"
echo "  • Maximum $MAX_JOBS concurrent jobs"
echo "  • Devil will poll every ${SLEEP_TIME}s"
echo ""
echo "Expected runtime: ~$((NUM_STRUCTURES * 2 * JOB_DURATION / MAX_JOBS + 60)) seconds"
echo ""
echo "Monitor in another terminal with:"
echo "  watch -n 2 'squeue -u \$USER'"
echo ""
read -p "Continue with full test? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelled"
    exit 0
fi

# Step 6: Run the full integration test
echo -e "\n${BLUE}[6/6] Running VASP Devil with real SLURM${NC}"
echo "============================================================"

# Modify the mock script to use short job duration
export PATH=".:$PATH"  # Ensure vsf-submit-job.py is in PATH

python vasp_devil_prototype.py "$TEST_DIR/struct_*" \
    4 4 1 Normal \
    --vasp-setup "echo 'Mock VASP setup'" \
    --workflow "$WORKFLOW" \
    --max-jobs "$MAX_JOBS" \
    --state-file "$TEST_DIR/state.json" \
    --sleep-time "$SLEEP_TIME" \
    --verbose

# Check results
echo ""
echo "============================================================"
echo -e "${GREEN}Integration Test Complete!${NC}"
echo "============================================================"

# Show state file
echo -e "\n${BLUE}Final State:${NC}"
cat "$TEST_DIR/state.json" | python -m json.tool | head -50

# Show created directories
echo -e "\n${BLUE}Created Directories:${NC}"
ls -lh "$TEST_DIR/"

# Check for relax2 directories
echo -e "\n${BLUE}Relaxation 2 Directories (should exist):${NC}"
ls -d "$TEST_DIR"/*_relax2 2>/dev/null || echo "  (none found - check for failures)"

# Count output files
echo -e "\n${BLUE}Generated Files Summary:${NC}"
echo "  OUTCAR files: $(find "$TEST_DIR" -name "OUTCAR*" | wc -l)"
echo "  CONTCAR files: $(find "$TEST_DIR" -name "CONTCAR" | wc -l)"
echo "  SLURM output files: $(find "$TEST_DIR" -name "*.out" | wc -l)"

# Verify all workflows completed
echo -e "\n${BLUE}Workflow Status:${NC}"
python - "$TEST_DIR/state.json" << 'EOF'
import sys
import json

with open(sys.argv[1]) as f:
    state = json.load(f)

instances = state['workflow_instances']
completed = sum(1 for inst in instances.values() if inst['status'] == 'completed')
failed = sum(1 for inst in instances.values() if inst['status'] == 'failed')
in_progress = sum(1 for inst in instances.values() if inst['status'] == 'in_progress')

total = len(instances)

print(f"  Total workflows: {total}")
print(f"  ✓ Completed: {completed}")
print(f"  ✗ Failed: {failed}")
print(f"  ⟳ In progress: {in_progress}")

if failed > 0:
    print("\n  Failed workflows:")
    for path, inst in instances.items():
        if inst['status'] == 'failed':
            failed_stage = next(s for s in inst['stages'] if s['status'] == 'failed')
            print(f"    • {path}: {failed_stage['name']} - {failed_stage.get('error', 'unknown')}")

sys.exit(0 if failed == 0 else 1)
EOF

TEST_RESULT=$?

echo ""
echo "============================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All workflows completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Inspect the state file: cat $TEST_DIR/state.json"
    echo "  2. Check OUTCAR files: cat $TEST_DIR/struct_1/OUTCAR"
    echo "  3. Verify relax2 directories exist and have files"
    echo "  4. Review logs: ls -lh vasp_devil_*.log"
else
    echo -e "${YELLOW}⚠ Some workflows failed - check state file for details${NC}"
fi
echo "============================================================"
