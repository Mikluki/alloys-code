#!/usr/bin/env python3
"""
Simplified VASP Individual Job Controller using SLURM
Submits individual VASP calculations with specified CPU and INCAR parameters.
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path


def pre_modify_incar_files(directories, ncore, kpar, algo):
    """Pre-modify INCAR files in all directories before job submission."""
    print(f"Pre-modifying INCAR files with NCORE={ncore}, KPAR={kpar}, ALGO={algo}")

    modified_count = 0
    for work_dir in directories:
        incar_path = work_dir / "INCAR"
        if incar_path.exists():
            # Generate date string for backup
            date_str = datetime.datetime.now().strftime("%m%d_%H%M%S")

            # Read current INCAR
            incar_content = incar_path.read_text()

            # Remove existing parameters and add new ones
            lines = incar_content.splitlines()
            filtered_lines = []

            for line in lines:
                # Skip lines that set NCORE, KPAR, or ALGO
                if not (
                    line.strip().startswith("NCORE")
                    or line.strip().startswith("KPAR")
                    or line.strip().startswith("ALGO")
                ):
                    filtered_lines.append(line)

            # Add new parameters
            filtered_lines.append(f"NCORE = {ncore}")
            filtered_lines.append(f"KPAR = {kpar}")
            if algo:
                filtered_lines.append(f"ALGO = {algo}")

            # Write modified INCAR
            new_content = "\n".join(filtered_lines) + "\n"
            incar_path.write_text(new_content)

            # Create backup of the MODIFIED INCAR (the one actually used)
            backup_name = f"INCAR.{date_str}.used.ncore{ncore}_kpar{kpar}"
            backup_path = work_dir / backup_name
            backup_path.write_text(new_content)

            modified_count += 1
        else:
            print(f"Warning: No INCAR found in {work_dir}")

    print(f"Modified {modified_count} INCAR files")
    return modified_count


def submit_individual_jobs(
    directories,
    ntasks,
    ncore,
    kpar,
    algo,
    vasp_setup,
    cpus_per_task=None,
    partition=None,
    nodes=1,
    nodelist="",
):
    """Submit individual SLURM jobs for each directory."""

    # Pre-modify INCAR files
    pre_modify_incar_files(directories, ncore, kpar, algo)

    submitted_jobs = []

    for work_dir in directories:
        # Generate job name based on directory
        dir_name = work_dir.name
        job_name = f"vasp_{dir_name}_{ntasks}_{ncore}_{kpar}"

        # Build the command that will run
        vasp_command = f"""
echo "========================================================" 
echo "VASP Individual Job: $SLURM_JOB_ID"
echo "-----------------------------------------"
echo "CPU Information Check on Compute Node"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "-----------------------------------------"
echo "CPU Architecture & Features:"
echo "Architecture: $(uname -m)"
echo "CPU Model: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "CPU MHz: $(lscpu | grep 'CPU MHz' | cut -d: -f2 | xargs)"
echo "CPU Features: $(lscpu | grep 'Flags' | cut -d: -f2 | head -c 80)..."
echo "L3 Cache: $(lscpu | grep 'L3 cache' | cut -d: -f2 | xargs)"
echo "Available CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "-----------------------------------------"
echo "Working Directory: {work_dir}"
echo "MPI Processes: {ntasks}"
echo "VASP Setup: {vasp_setup}"
echo "========================================================"

# Change to work directory
cd "{work_dir}" || {{ echo "Failed to change to directory {work_dir}"; exit 1; }}

# Setup VASP environment
{vasp_setup}
echo "VASP environment setup completed"

# Run VASP
echo "Running VASP with {ntasks} processes"
echo "Command: mpirun -np {ntasks} vasp_std"
mpirun -np {ntasks} vasp_std > vasp.out 2> vasp.err
vasp_exit=$?

# Backup OUTCAR
if [ -f "OUTCAR" ]; then
    DATE_STR=$(date +%m%d_%H%M%S)
    OUTCAR_BACKUP="OUTCAR.${{DATE_STR}}.ncore{ncore}_kpar{kpar}"
    cp "OUTCAR" "$OUTCAR_BACKUP"
    echo "OUTCAR backed up as $OUTCAR_BACKUP"
fi

echo "VASP completed with exit code: $vasp_exit"
echo "========================================================"
exit $vasp_exit
""".strip()

        # Build sbatch command - resource allocation arguments first
        sbatch_cmd = [
            "sbatch",
            "--parsable",
            f"--ntasks={ntasks}",
        ]

        # Add cpus-per-task right after ntasks if specified
        if cpus_per_task is not None:
            sbatch_cmd.append(f"--cpus-per-task={cpus_per_task}")

        # Add nodes argument only if not using cpus-per-task or if explicitly more than 1
        if cpus_per_task is None or nodes > 1:
            sbatch_cmd.append(f"--nodes={nodes}")

        # Add nodelist only if it's not None and not empty/whitespace
        if nodelist is not None and isinstance(nodelist, str) and nodelist.strip():
            sbatch_cmd.append(f"--nodelist={nodelist.strip()}")

        # Add partition if specified
        if partition is not None:
            sbatch_cmd.append(f"--partition={partition}")

        # Add job control arguments - logs go directly to work directory
        sbatch_cmd.extend(
            [
                "--time=24:00:00",
                f"--job-name={job_name}",
                f"--output={work_dir}/{job_name}_%j.out",
                f"--error={work_dir}/{job_name}_%j.err",
                "--wrap",
                vasp_command,
            ]
        )

        # Submit job
        try:
            result = subprocess.run(
                sbatch_cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            job_id = result.stdout.strip()
            submitted_jobs.append(job_id)
            print(f"✓ Submitted job '{job_name}' with ID {job_id} for {work_dir}")

        except subprocess.CalledProcessError as e:
            print(f"✗ Error submitting job for {work_dir}: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")

    return submitted_jobs


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VASP Individual Job Controller using SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit jobs for all directories matching pattern
  %(prog)s "path/to/calc/rand*" 32 4 2 Normal --vasp-setup "module load vasp/6.4.3"
  
  # With specific resources
  %(prog)s "path/to/calc/structure_*" 32 4 2 Normal --nodes 2 --partition gpu-partition --vasp-setup "source setup.sh"
  
  # With specific node constraints
  %(prog)s "path/to/calc/*" 32 4 2 Normal --nodes 1 --nodelist "node[01-03]" --vasp-setup "module load vasp/6.4.3"
        """,
    )

    # Required positional arguments
    parser.add_argument(
        "glob_pattern",
        type=str,
        help="Glob pattern for VASP calculation directories (e.g., 'path/to/calc/rand*')",
    )
    parser.add_argument("ntasks", type=int, help="Total number of MPI tasks")
    parser.add_argument("ncore", type=int, help="VASP NCORE parameter")
    parser.add_argument("kpar", type=int, help="VASP KPAR parameter")
    parser.add_argument("algo", type=str, help="VASP ALGO parameter")

    # Optional keyword arguments
    parser.add_argument(
        "--cpus_per_task",
        "-cpt",
        type=int,
        default=None,
        help="CPUs per task (default: let SLURM decide)",
    )

    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default=None,
        help="SLURM partition (e.g., gpu, cpu; default: scheduler decides)",
    )

    parser.add_argument(
        "--nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to request (default: 1)",
    )

    parser.add_argument(
        "--nodelist",
        "-nl",
        type=str,
        default="",
        help="Specific nodes to use (e.g., 'node[01-03]' or 'node01,node02'; default: let SLURM choose)",
    )

    # Required named argument
    parser.add_argument(
        "--vasp-setup",
        type=str,
        required=True,
        help='VASP setup command (e.g., "module load vasp/6.4.3" or "source setup.sh")',
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Validate VASP parallelization
    if args.ntasks % (args.ncore * args.kpar) != 0:
        print(f"✗ Error: Invalid parallelization parameters!")
        print(
            f"  ntasks ({args.ntasks}) must be divisible by (ncore × kpar) = {args.ncore * args.kpar}"
        )
        sys.exit(1)

    # Find directories using glob pattern
    glob_path = Path(args.glob_pattern)
    if glob_path.is_absolute():
        # Absolute path - use as is
        pattern_parent = glob_path.parent
        pattern_name = glob_path.name
    else:
        # Relative path - resolve relative to current directory
        pattern_parent = Path.cwd()
        pattern_name = args.glob_pattern

    # Find matching directories
    try:
        if "*" in pattern_name or "?" in pattern_name or "[" in pattern_name:
            # Pattern contains wildcards
            directories = [d for d in pattern_parent.glob(pattern_name) if d.is_dir()]
        else:
            # No wildcards - treat as single directory
            single_dir = pattern_parent / pattern_name
            directories = [single_dir] if single_dir.is_dir() else []
    except Exception as e:
        print(f"✗ Error processing glob pattern '{args.glob_pattern}': {e}")
        sys.exit(1)

    if not directories:
        print(f"✗ No directories found matching pattern '{args.glob_pattern}'")
        sys.exit(1)

    print(f"Found {len(directories)} VASP directories to process:")
    for d in directories:
        print(f"  - {d}")

    # Submit individual jobs
    submitted_jobs = submit_individual_jobs(
        directories,
        args.ntasks,
        args.ncore,
        args.kpar,
        args.algo,
        args.vasp_setup,
        args.cpus_per_task,
        args.partition,
        args.nodes,
        args.nodelist,
    )

    if submitted_jobs:
        print(f"\n✓ Successfully submitted {len(submitted_jobs)} jobs")
        print(
            f"  - Parameters: ntasks={args.ntasks}, ncore={args.ncore}, kpar={args.kpar}, algo={args.algo}"
        )
        print(f"  - Resource allocation: nodes={args.nodes}", end="")
        if args.cpus_per_task is not None:
            print(f", cpus-per-task={args.cpus_per_task}", end="")
        if args.partition is not None:
            print(f", partition={args.partition}", end="")
        if args.nodelist and args.nodelist.strip():
            print(f", nodelist={args.nodelist}", end="")
        print()

        # Get base directory for reference
        base_dir = directories[0].parent
        print(f"\n📝 Logs are in individual calculation directories")
        print(f"🔍 Monitor job status with: squeue -u $USER")
        print(f"📊 Job IDs: {', '.join(submitted_jobs)}")
    else:
        print("\n✗ No jobs were submitted successfully!")
        sys.exit(1)


if __name__ == "__main__":
    main()
