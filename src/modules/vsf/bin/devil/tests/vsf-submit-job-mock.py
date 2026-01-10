#!/usr/bin/env python3
"""
Mock vsf-submit-job.py for testing VASP Devil with real SLURM
Submits real SLURM jobs that fake VASP calculations (just sleep + create output)
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_JOB_DURATION = 10


def create_mock_vasp_script(work_dir, ntasks, ncore, kpar, job_duration=10):
    """Create a simple bash script that mimics VASP behavior"""

    script = f"""#!/bin/bash
set -e  # Exit on error
set -x  # Print commands for debugging

cd "{work_dir}" || exit 1
echo "Mock VASP started at $(date)"

# Simple sleep loop with STOPCAR check
for i in $(seq 1 {job_duration}); do
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
"""
    return script


def submit_mock_job(
    work_dir,
    ntasks,
    ncore,
    kpar,
    algo,
    vasp_setup,
    cpus_per_task=None,
    partition=None,
    nodes=1,
    nodelist="",
    job_duration=16,
):
    """Submit a mock VASP job to SLURM"""

    work_dir = Path(work_dir)
    job_name = f"vasp_mock_{work_dir.name}"

    # Create the mock VASP script
    mock_script = create_mock_vasp_script(work_dir, ntasks, ncore, kpar, job_duration)

    # Write script to file
    script_file = work_dir / "mock_vasp_job.sh"
    script_file.write_text(mock_script)
    script_file.chmod(0o755)

    # Build sbatch command - MINIMAL for laptop testing
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--ntasks=1",  # Always request just 1 task (not ntasks from config)
        "--cpus-per-task=1",  # Just 1 CPU per task
        "--mem=512M",  # Minimal memory
        "--time=00:01:00",  # x minute limit
        f"--job-name={job_name}",
        f"--output={work_dir}/{job_name}_%j.out",
        f"--error={work_dir}/{job_name}_%j.err",
        str(script_file),
    ]

    # Submit
    try:
        result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)

        job_id = result.stdout.strip()
        print(f"✓ Submitted job '{job_name}' with ID {job_id} for {work_dir}")
        return job_id

    except subprocess.CalledProcessError as e:
        print(f"✗ Error submitting job for {work_dir}: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Mock VASP Job Submitter")

    # Required arguments (same as real script)
    parser.add_argument("work_dir", type=str)
    parser.add_argument("ntasks", type=int)
    parser.add_argument("ncore", type=int)
    parser.add_argument("kpar", type=int)
    parser.add_argument("algo", type=str)

    # Optional arguments
    parser.add_argument("--cpus_per_task", type=int, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--nodelist", type=str, default="")
    parser.add_argument("--vasp-setup", type=str, required=True)

    # Mock-specific argument
    parser.add_argument(
        "--job-duration",
        type=int,
        default=DEFAULT_JOB_DURATION,
        help="How long the mock job should run (seconds, default: 60)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Validate directory exists
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(f"✗ Directory does not exist: {work_dir}")
        sys.exit(1)

    # Submit the mock job
    submit_mock_job(
        work_dir=work_dir,
        ntasks=args.ntasks,
        ncore=args.ncore,
        kpar=args.kpar,
        algo=args.algo,
        vasp_setup=args.vasp_setup,
        cpus_per_task=args.cpus_per_task,
        partition=args.partition,
        nodes=args.nodes,
        nodelist=args.nodelist,
        job_duration=args.job_duration,
    )


if __name__ == "__main__":
    main()
