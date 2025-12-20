# SLURM Testing Guide for VASP Devil

This guide shows how to test the complete VASP Devil workflow with real SLURM but mock VASP jobs.

## What You're Testing

```
Real SLURM + Mock VASP = Full Orchestration Test
```

- ✅ Real job submission (`sbatch`)
- ✅ Real job tracking (`squeue`)
- ✅ Real job completion detection
- ✅ Full workflow progression
- ✅ Multiple workflows running concurrently
- ❌ No actual VASP (jobs just sleep and create mock output)

## Prerequisites

### 1. SLURM Installed

```bash
# Check SLURM is available
sbatch --version
squeue --version

# Test submitting a simple job
echo '#!/bin/bash\nsleep 10\necho "Test job"' | sbatch --wrap="$(cat)"
squeue -u $USER
```

If SLURM isn't installed, install it:
- **Ubuntu/Debian**: `sudo apt install slurm-wlm`
- **Arch**: `sudo pacman -S slurm-llnl`
- **Or use Docker**: [slurm-docker-cluster](https://github.com/giovtorres/slurm-docker-cluster)

### 2. Files in Place

```bash
# You should have these files:
ls -1
# vasp_devil_prototype.py
# vsf-submit-job-mock.py
# run_integration_test.sh
# test_prototype.py
```

## Quick Start

### Automated Full Test

```bash
# Make scripts executable
chmod +x run_integration_test.sh
chmod +x vsf-submit-job-mock.py

# Run the full integration test
./run_integration_test.sh
```

This will:
1. Check SLURM is available
2. Create 3 test structures with VASP input files
3. Run dry-run to verify setup
4. Ask for confirmation
5. Run the full workflow with real SLURM
6. Show results and summary

**Expected output:**
- 3 workflows start
- Each submits relax1 job (30s each)
- Jobs complete, validate stage runs
- Files copied to `struct_X_relax2/` directories
- Relax2 jobs submitted
- All complete successfully

## Manual Testing

### Step 1: Create Test Directories

```bash
# Create mock structures
mkdir -p test_calcs/struct_{1,2,3}

for i in 1 2 3; do
    cat > "test_calcs/struct_$i/INCAR" << EOF
SYSTEM = Test structure $i
NCORE = 4
KPAR = 2
EOF
    echo "Mock POSCAR" > "test_calcs/struct_$i/POSCAR"
    echo "Mock POTCAR" > "test_calcs/struct_$i/POTCAR"
    echo "Mock KPOINTS" > "test_calcs/struct_$i/KPOINTS"
done
```

### Step 2: Setup Mock Submitter

```bash
# Create symlink so devil finds the mock script
ln -sf vsf-submit-job-mock.py vsf-submit-job.py

# Make executable
chmod +x vsf-submit-job-mock.py

# Ensure it's in PATH
export PATH=".:$PATH"
```

### Step 3: Test Single Job Submission

```bash
# Submit one mock job manually
./vsf-submit-job-mock.py test_calcs/struct_1 \
    4 4 1 Normal \
    --vasp-setup "echo 'mock setup'" \
    --job-duration 15

# Watch it run
squeue -u $USER

# Wait for completion
sleep 20

# Check output was created
ls -lh test_calcs/struct_1/
cat test_calcs/struct_1/OUTCAR
```

### Step 4: Test Simple Workflow

```bash
# Run devil with simple workflow (1 job per structure)
python vasp_devil_prototype.py 'test_calcs/struct_*' \
    4 4 1 Normal \
    --vasp-setup "echo 'mock setup'" \
    --workflow simple \
    --max-jobs 2 \
    --sleep-time 5 \
    --verbose
```

Monitor in another terminal:
```bash
# Watch SLURM queue
watch -n 2 'squeue -u $USER'

# Watch devil progress
tail -f vasp_devil_*.log
```

### Step 5: Test Double Relaxation

```bash
# Clean up previous test
rm -rf test_calcs/*_relax2

# Run double relaxation workflow
python vasp_devil_prototype.py 'test_calcs/struct_*' \
    4 4 1 Normal \
    --vasp-setup "echo 'mock setup'" \
    --workflow double \
    --max-jobs 2 \
    --sleep-time 5 \
    --state-file test_state.json \
    --verbose
```

**What to observe:**
1. Initial jobs submitted (max 2 concurrent)
2. Jobs complete, ValidationStage checks OUTCAR
3. CopyStage creates `struct_X_relax2/` directories
4. Second relaxation jobs submitted
5. Final validation
6. All workflows complete

## Customizing Mock Jobs

### Change Job Duration

```bash
# Edit vsf-submit-job-mock.py
# Or pass via argument:
./vsf-submit-job-mock.py test_calcs/struct_1 \
    4 4 1 Normal \
    --vasp-setup "echo 'setup'" \
    --job-duration 120  # Run for 2 minutes
```

### Test Job Failures

Create a version that randomly fails:

```bash
cat > vsf-submit-job-fail-random.py << 'EOF'
#!/usr/bin/env python3
import random
import sys
from vsf_submit_job_mock import *

if __name__ == "__main__":
    # 30% chance of failure
    if random.random() < 0.3:
        print("✗ Random failure for testing!")
        sys.exit(1)
    main()
EOF

chmod +x vsf-submit-job-fail-random.py
ln -sf vsf-submit-job-fail-random.py vsf-submit-job.py
```

Then run devil and watch how it handles failures.

### Test Resume from Interruption

```bash
# Start a workflow
python vasp_devil_prototype.py 'test_calcs/struct_*' \
    4 4 1 Normal \
    --vasp-setup "echo 'setup'" \
    --workflow double \
    --max-jobs 2 \
    --state-file resume_test.json

# After some jobs submit, interrupt with Ctrl+C
^C

# Check state was saved
cat resume_test.json

# Resume - it should pick up where it left off
python vasp_devil_prototype.py 'test_calcs/struct_*' \
    4 4 1 Normal \
    --vasp-setup "echo 'setup'" \
    --workflow double \
    --max-jobs 2 \
    --state-file resume_test.json
```

## Validation Checklist

After running tests, verify:

### ✅ Job Submission
```bash
# Jobs were submitted
grep "Submitted job" vasp_devil_*.log
```

### ✅ Job Completion Detection
```bash
# Jobs were detected as complete
grep "completed for" vasp_devil_*.log
```

### ✅ Validation Stage
```bash
# OUTCAR validation passed
grep "VASP completed in" vasp_devil_*.log
```

### ✅ File Copying
```bash
# Files were copied
ls -R test_calcs/*_relax2/

# CONTCAR renamed to POSCAR
cat test_calcs/struct_1_relax2/POSCAR
```

### ✅ Workflow Completion
```bash
# All workflows finished
grep "All workflows completed" vasp_devil_*.log

# Check state file
python -m json.tool test_state.json | grep '"status"'
```

### ✅ Output Files Created
```bash
# Each calculation has OUTCAR
find test_calcs -name "OUTCAR" -exec echo {} \;

# Each has CONTCAR
find test_calcs -name "CONTCAR" -exec echo {} \;

# Relax2 directories have copied files
ls test_calcs/struct_1_relax2/
# Should see: INCAR, POSCAR, POTCAR, KPOINTS, WAVECAR
```

## Troubleshooting

### Jobs Not Submitting

```bash
# Check sbatch works
echo '#!/bin/bash\nsleep 5' | sbatch

# Check PATH includes current directory
echo $PATH
export PATH=".:$PATH"

# Check script is executable
ls -l vsf-submit-job.py
chmod +x vsf-submit-job-mock.py
```

### Jobs Stuck in Queue

```bash
# Check SLURM configuration
scontrol show config | grep -i state

# Check job details
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Cancel stuck jobs
scancel -u $USER
```

### State File Corruption

```bash
# Backup state before testing
cp vasp-devil-state.json vasp-devil-state.json.backup

# Validate JSON
python -m json.tool test_state.json

# Reset state
rm test_state.json
# Devil will create new state on next run
```

### Devil Not Finding Jobs

```bash
# Check squeue works
squeue -u $USER

# Check job IDs match
grep "job_id" test_state.json
squeue -u $USER -o "%i"
```

## Performance Testing

### Many Workflows

```bash
# Create 20 structures
mkdir -p stress_test/struct_{1..20}
for i in {1..20}; do
    cp test_calcs/struct_1/* "stress_test/struct_$i/"
done

# Run with higher max_jobs
python vasp_devil_prototype.py 'stress_test/struct_*' \
    4 4 1 Normal \
    --vasp-setup "echo 'setup'" \
    --workflow simple \
    --max-jobs 10 \
    --sleep-time 5
```

### Fast Polling

```bash
# Very fast polling for testing
python vasp_devil_prototype.py 'test_calcs/struct_*' \
    4 4 1 Normal \
    --vasp-setup "echo 'setup'" \
    --workflow double \
    --max-jobs 2 \
    --sleep-time 2  # Poll every 2 seconds
```

## Next Steps

Once SLURM testing is successful:

1. **Test with real vsf-submit-job.py**
   ```bash
   rm vsf-submit-job.py  # Remove symlink
   # Use your actual vsf-submit-job.py
   ```

2. **Test with real VASP calculations**
   - Start with small systems
   - Use short NSW (10 ionic steps)
   - Test one structure first

3. **Deploy to cluster**
   - Copy validated code to cluster
   - Test with cluster SLURM
   - Run on real research calculations

4. **Create custom workflows**
   - Convergence tests
   - Band structure calculations
   - Your specific research workflows

## Success Criteria

You'll know the testing is successful when:

✅ Jobs submit to SLURM successfully  
✅ Devil tracks job status correctly  
✅ Jobs complete and devil detects completion  
✅ Validation stage checks OUTCAR  
✅ Files copy correctly to new directories  
✅ Multiple workflows progress independently  
✅ State persists and can resume  
✅ No crashes or exceptions in logs  

---

**Ready?** Run `./run_integration_test.sh` and watch your workflow architecture come to life! 🚀
