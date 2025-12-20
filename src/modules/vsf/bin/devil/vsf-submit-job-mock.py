#!/usr/bin/env python3
"""
Mock vsf-submit-job.py for testing VASP Devil with real SLURM
Submits real SLURM jobs that fake VASP calculations (just sleep + create output)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def create_mock_vasp_script(work_dir, ntasks, ncore, kpar, job_duration=60):
    """
    Create a bash script that mimics VASP behavior:
    - Sleeps for job_duration seconds
    - Creates OUTCAR with elapsed time
    - Creates CONTCAR
    - Backs up files like real VASP
    """

    script = f"""#!/bin/bash
echo "========================================================"
echo "MOCK VASP Job: $SLURM_JOB_ID"
echo "--------------------------------------------------------"
echo "Working Directory: {work_dir}"
echo "MPI Processes: {ntasks}"
echo "NCORE: {ncore}, KPAR: {kpar}"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "========================================================"

cd "{work_dir}" || {{ echo "Failed to change to directory"; exit 1; }}

# Simulate VASP calculation with sleep
echo "Starting mock VASP calculation (will run for {job_duration}s)..."
sleep {job_duration}

# Create mock OUTCAR with completion marker
cat > OUTCAR << 'EOF'
 running on    1 total cores
 distrk:  each k-point on    1 cores,    1 groups
 vasp.6.4.3 (mock)
 
 POSCAR found :  1 types and       2 ions
 
 ----------------------------------------- Iteration    1(   1)  ---------------------------------------
 
 FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
 ---------------------------------------------------
 free  energy   TOTEN  =      -123.45678901 eV
 
 energy  without entropy=     -123.45678901  energy(sigma->0) =     -123.45678901
 
 ----------------------------------------- Iteration    2(   2)  ---------------------------------------
 
 (... mock iterations ...)
 
 ----------------------------------------- Iteration   50(  50)  ---------------------------------------
 
 reached required accuracy - stopping structural energy minimisation
 
 Total CPU time used (sec):      {job_duration * 0.9:.1f}
 User time (sec):                {job_duration * 0.8:.1f}
 System time (sec):              {job_duration * 0.1:.1f}
 Elapsed time (sec):             {job_duration:.1f}
EOF

# Create mock CONTCAR (relaxed structure)
cat > CONTCAR << 'EOF'
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
EOF

# Create mock WAVECAR (empty but exists)
echo "MOCK WAVECAR" > WAVECAR

# Backup OUTCAR like real VASP script does
DATE_STR=$(date +%m%d_%H%M%S)
OUTCAR_BACKUP="OUTCAR.${{DATE_STR}}.ncore{ncore}_kpar{kpar}"
cp OUTCAR "$OUTCAR_BACKUP"
echo "Created: $OUTCAR_BACKUP"

echo "Mock VASP calculation completed successfully!"
echo "========================================================"
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
    job_duration=60,
):
    """Submit a mock VASP job to SLURM"""

    work_dir = Path(work_dir)
    job_name = f"vasp_mock_{work_dir.name}_{ntasks}_{ncore}_{kpar}"

    # Create the mock VASP script
    mock_script = create_mock_vasp_script(work_dir, ntasks, ncore, kpar, job_duration)

    # Build sbatch command
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--ntasks={ntasks}",
    ]

    if cpus_per_task is not None:
        sbatch_cmd.append(f"--cpus-per-task={cpus_per_task}")

    if cpus_per_task is None or nodes > 1:
        sbatch_cmd.append(f"--nodes={nodes}")

    if nodelist and nodelist.strip():
        sbatch_cmd.append(f"--nodelist={nodelist.strip()}")

    if partition is not None:
        sbatch_cmd.append(f"--partition={partition}")

    # Job control arguments
    sbatch_cmd.extend(
        [
            "--time=00:05:00",  # Short time limit for testing
            f"--job-name={job_name}",
            f"--output={work_dir}/{job_name}_%j.out",
            f"--error={work_dir}/{job_name}_%j.err",
            "--wrap",
            mock_script,
        ]
    )

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
        default=60,
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
