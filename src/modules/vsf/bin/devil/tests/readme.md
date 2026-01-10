# MD Orchestrator Tests

## Overview

Minimal, sensible test suite for MD orchestrator setup. Tests the full orchestration pipeline without requiring real VASP or SLURM.

## What Gets Tested

1. **State Persistence** (`test_state_persistence`)
   - `MDConfig`, `MDStats`, `MDState` dataclass round-trip (to_dict → from_dict)
   - JSON serialization correctness
   - **Runtime:** < 1 second

2. **Backup Manager** (`test_backup_manager`)
   - File versioning (OUTCAR.0, OUTCAR.1, etc.)
   - Backup and prepare operations
   - **Runtime:** < 1 second

3. **Full Orchestrator Loop** (`test_md_orchestrator_3_iterations`)
   - Complete 3 MD iterations with mocked SLURM
   - Each mock job runs for 10 seconds
   - Verifies: state saved, backups created, metrics updated
   - **Runtime:** ~45 seconds (3 × 10sec + overhead + polling)

## Test Slurm first if used

```bash
# Save as test_job.sh
cat > test_job.sh << 'EOF'
#!/bin/bash
echo "Job started at $(date)"
sleep 3
echo "Job completed at $(date)" > output.txt
ls -la output.txt
exit 0
EOF

chmod +x test_job.sh

# Submit to SLURM
sbatch --ntasks=1 --cpus-per-task=1 --time=00:00:06 test_job.sh

# Check status
squeue -u $(whoami)

# Wait, then check output
sleep 5
cat output.txt
cat slurm-*.out
```

### if hangs

```bash
# Nuclear option - reset the node completely
sudo scontrol update NodeName=pop-os State=DOWN Reason="manual reset"
sleep 2
sudo scontrol update NodeName=pop-os State=IDLE Reason="recovered"

# Verify
sinfo -N
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
uv pip install pytest pytest-mock

# Ensure md_runner.py and engine.py are in Python path
# Option 1: Run from project root
cd /path/to/project

# Option 2: Set PYTHONPATH
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

### Run All Tests

```bash
pytest test_md_setup.py -v -s --log-cli-level=DEBUG
pytest test_md_setup.py -v -s
```

### Run Specific Test

```bash
# State round-trip only
pytest test_md_setup.py::test_state_persistence -v --log-cli-level=DEBUG

# Backup manager only
pytest test_md_setup.py::test_backup_manager -v --log-cli-level=DEBUG

# Full orchestrator test (takes ~45 sec)
pytest test_md_setup.py::test_md_orchestrator_3_iterations -v -s --log-cli-level=DEBUG
```

### Full SLURM Integration Tests (requires cluster access)

> [!IMPORTANT] START THE SERVICE!

```bash
sudo systemctl restart slurmd
```

```bash
pytest test_md_setup_slurm_integration.py -v -s --log-cli-level=DEBUG
```

**Prerequisites for SLURM tests:**

```bash
# 1. SLURM must be available
squeue --help

# 2. vsf-submit-job.py must be in PATH
# Either:
#   - Symlink: ln -s vsf-submit-job-mock.py vsf-submit-job.py
#   - Or add to PATH: export PATH=/path/to/mock:$PATH

# 3. Run from a node with SLURM access
which squeue  # Should work
```

Run specific SLURM test:

```bash
# Watch the queue
watch -n 1 squeue

# Just SlurmInterface submission
pytest test_md_setup_slurm_integration.py::test_slurm_interface_submit_job -v -s

# Full orchestrator with real jobs (60 sec)
pytest test_md_setup_slurm_integration.py::test_md_orchestrator_with_real_slurm -v -s
```

### Run All Tests (unit + SLURM)

```bash
pytest test_md_setup.py test_md_setup_slurm_integration.py -v -s
```

### Run with Timing

```bash
pytest test_md_setup.py -v -s --durations=0
```

## Expected Output

```
test_md_setup.py::test_state_persistence PASSED                    [ 33%]
✓ State persistence test passed

test_md_setup.py::test_backup_manager PASSED                       [ 66%]
✓ Backup manager test passed

test_md_setup.py::test_md_orchestrator_3_iterations PASSED         [100%]
✓ Test passed in 42.3s
  Completed: 3
  Failed: 0
  Run durations: {0: 10.2, 1: 10.1, 2: 10.0}

======================== 3 passed in 45.2s ========================
```

## Design Notes

### Why Mock SLURM?

- Avoids cluster dependencies
- Fast iteration (no queue waits)
- Deterministic behavior
- Tests the **orchestration logic**, not VASP/SLURM

### Why 10 Second Jobs?

- Short enough to test quickly (~45s total)
- Long enough to verify polling works
- Realistic enough to catch state persistence bugs

### Why No Real VASP Files?

- Test focuses on orchestration, not simulation
- Minimal dummy files verify file I/O
- Real calculations tested separately with actual VASP

## Troubleshooting

### ImportError: No module named 'md_runner'

```bash
# Check that md_runner.py is in the same directory or PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
pytest test_md_setup.py -v
```

### ImportError: cannot import name 'SlurmInterface' from 'engine'

The test mocks `SlurmInterface` at instantiation time. If this fails:

1. Ensure `engine.py` is in PYTHONPATH
2. Or: Remove the `from engine import SlurmInterface` line from `md_runner.py` line 298 and pass it as a dependency (see "Future Improvements")

### Test hangs or times out

- Check that mock_slurm_interface returns `False` for `is_job_running()` after 10 seconds
- Increase pytest timeout: `pytest test_md_setup.py --timeout=120`

## Future Improvements

### 1. Fix Circular Import (Optional)

Current: `MDOrchestrator.__init__` does `from engine import SlurmInterface`

Better: Pass as dependency

```python
def __init__(self, ..., slurm: SlurmInterface = None):
    self.slurm = slurm or SlurmInterface()
```

Benefit: Easier testing, explicit dependencies.

### 2. Add Recovery Test

- Test `--restart` with saved state
- Verify state loads and continues from run #2

### 3. Add Failure Handling Test

- Mock submission failure
- Verify `failed_runs` increments
- Verify orchestrator doesn't crash

### 4. Performance Profiling

- Measure state save/load overhead
- Benchmark backup operations at scale (100+ files)

## Files

- `test_md_setup.py` - Test suite (this is the main file)
- `conftest.py` - Pytest configuration (path setup, markers)
- `README.md` - This file
