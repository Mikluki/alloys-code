#!/bin/bash
set -e  # Exit on error
set -x  # Print commands for debugging

cd "/home/mik/1_projects/alloys_code/src/modules_v2/vsf/bin/devil/tests/md_test_slurm_test_slurm_interface_submit_job" || exit 1
echo "Mock VASP started at $(date)"

# Simple sleep loop with STOPCAR check
for i in $(seq 1 10); do
    if [ -f STOPCAR ]; then
        echo "STOPCAR found, exiting"
        break
    fi
    sleep 1
done

echo "Creating output files..."

# Create mock OUTCAR
cat > OUTCAR << 'OUTCAR_EOF'
 running on    1 total cores
 vasp.6.4.3 (mock)
 reached required accuracy - stopping
OUTCAR_EOF

# Create mock CONTCAR
cat > CONTCAR << 'CONTCAR_EOF'
Mock relaxed structure
1.0
  5.5 0.0 0.0
  0.0 5.5 0.0
  0.0 0.0 5.5
Si
  2
Direct
  0.0 0.0 0.0
  0.25 0.25 0.25
CONTCAR_EOF

echo "Files created at $(date)"
ls -la OUTCAR CONTCAR
exit 0
