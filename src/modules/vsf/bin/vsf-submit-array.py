#!/usr/bin/env python3
"""
Simplified VASP Array Job Controller using SLURM
Submits VASP calculations with specified CPU and INCAR parameters.
"""

import argparse
import datetime
import json
import subprocess
import sys
import uuid
from pathlib import Path


def create_experiment_id():
    """Create a unique experiment ID."""
    return datetime.datetime.now().strftime("%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]


def document_dispatch_config(
    base_dir,
    exp_id,
    ntasks,
    ncore,
    kpar,
    algo,
    cpus_per_task,
    partition,
    nodes,
    nodelist,
    directories,
    job_id,
    max_concurrent,
):
    """Document the dispatch configuration for reference."""
    dispatch_dir = base_dir / "0-dispatch-logs"
    dispatch_dir.mkdir(exist_ok=True)

    config = {
        "experiment_id": exp_id,
        "job_id": job_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": {
            "ntasks": ntasks,
            "ncore": ncore,
            "kpar": kpar,
            "algo": algo,
            "cpus_per_task": cpus_per_task,
            "partition": partition,
            "nodes": nodes,
            "nodelist": nodelist,
            "max_concurrent": max_concurrent,
        },
        "directories_count": len(directories),
        "directories": [str(d) for d in directories],
    }

    # Save as JSON
    config_file = dispatch_dir / f"dispatch_{exp_id}.json"
    with config_file.open("w") as f:
        json.dump(config, f, indent=2)

    print(f"📋 Dispatch configuration saved to {config_file}")
    return config_file


def create_job_list(job_list_file, directories, experiment_id, params_str):
    """Create a job list file for array jobs."""
    job_list_file = Path(job_list_file)

    with job_list_file.open("w") as f:
        for idx, work_dir in enumerate(directories, 1):
            line = f"{idx}\t{work_dir}\t{experiment_id}\t{params_str}"
            f.write(line + "\n")

    return job_list_file


def read_directory_list(list_file, base_dir):
    """Read directories from a file, resolve relative paths, and validate."""
    list_file = Path(list_file)
    base_dir = Path(base_dir)

    if not list_file.exists():
        print(f"✗ Error: Directory list file '{list_file}' does not exist.")
        sys.exit(1)

    directories = []
    missing_dirs = []

    with list_file.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Resolve relative path against base directory
            dir_path = base_dir / line

            if dir_path.is_dir():
                directories.append(dir_path)
            else:
                missing_dirs.append(f"  Line {line_num}: {line} -> {dir_path}")

    if missing_dirs:
        print(f"✗ Error: The following directories from '{list_file}' do not exist:")
        for missing in missing_dirs:
            print(missing)
        sys.exit(1)

    if not directories:
        print(f"✗ Error: No valid directories found in '{list_file}'")
        sys.exit(1)

    return directories


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


def submit_array_job(
    base_dir,
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
    max_concurrent=6,
):
    """Submit array job with specified parameters using --wrap."""
    base_dir = Path(base_dir)

    # Pre-modify INCAR files
    pre_modify_incar_files(directories, ncore, kpar, algo)

    # Create experiment ID
    exp_id = create_experiment_id()

    # Create job lists directory (with 0- prefix)
    job_lists_dir = base_dir / "0-job_lists"
    job_lists_dir.mkdir(exist_ok=True)

    # Create logs directory for symlinks
    logs_dir = base_dir / "0-logs"
    logs_dir.mkdir(exist_ok=True)

    # Create simplified job list file (just directories)
    job_list_file = job_lists_dir / f"job_list_{exp_id}.txt"
    with job_list_file.open("w") as f:
        for idx, work_dir in enumerate(directories, 1):
            f.write(f"{idx}\t{work_dir}\n")

    # Create symlinks for array job output redirection
    print(f"Creating symlinks for {len(directories)} directories...")
    for idx, work_dir in enumerate(directories, 1):
        symlink_path = logs_dir / str(idx)
        # Remove existing symlink if it exists
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        # Create relative symlink to work directory
        relative_path = work_dir.relative_to(base_dir)
        symlink_path.symlink_to(f"../{relative_path}")
    print(f"✓ Created {len(directories)} symlinks in {logs_dir}/")

    # Calculate array parameters
    total_jobs = len(directories)

    # Generate job name
    job_name = f"vasp_{ntasks}_{ncore}_{kpar}_{algo}"

    # Build the command that will run in each array task
    vasp_command = f"""
# Get work directory for this array task
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_list_file})
WORK_DIR=$(echo "$LINE" | cut -f2)

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
echo "Working Directory: $WORK_DIR"
echo "MPI Processes: {ntasks}"
echo "VASP Setup: {vasp_setup}"
echo "========================================================"

# Change to work directory
cd "$WORK_DIR" || {{ echo "Failed to change to directory $WORK_DIR"; exit 1; }}

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
    # When cpus-per-task is specified, let SLURM calculate nodes unless explicitly set > 1
    if cpus_per_task is None or nodes > 1:
        sbatch_cmd.append(f"--nodes={nodes}")

    # Add nodelist if specified (not empty)
    if nodelist and nodelist.strip():
        sbatch_cmd.append(f"--nodelist={nodelist.strip()}")

    # Add partition if specified
    if partition is not None:
        sbatch_cmd.append(f"--partition={partition}")

    # Add job control arguments - use symlinks to redirect logs to work directories
    sbatch_cmd.extend(
        [
            "--time=24:00:00",
            f"--array=1-{total_jobs}%{max_concurrent}",
            f"--job-name={job_name}",
            f"--output={logs_dir}/%a/{job_name}_%A_%a.out",
            f"--error={logs_dir}/%a/{job_name}_%A_%a.err",
            "--wrap",
            vasp_command,
        ]
    )

    print("DEBUG: sbatch command:", " ".join(sbatch_cmd))
    # Submit job array
    try:
        result = subprocess.run(
            sbatch_cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        job_id = result.stdout.strip()
        print(f"✓ Submitted array job '{job_name}' with ID {job_id}")
        print(f"  - {total_jobs} directories to process")
        print(
            f"  - Parameters: ntasks={ntasks}, ncore={ncore}, kpar={kpar}, algo={algo}"
        )
        print(f"  - Resource allocation: nodes={nodes}", end="")
        if cpus_per_task is not None:
            print(f", cpus-per-task={cpus_per_task}", end="")
        if partition is not None:
            print(f", partition={partition}", end="")
        if nodelist and nodelist.strip():
            print(f", nodelist={nodelist}", end="")
        print()
        print(f"  - Max concurrent jobs: {max_concurrent}")
        print(f"  - Experiment ID: {exp_id}")

        # Document the dispatch configuration
        document_dispatch_config(
            base_dir,
            exp_id,
            ntasks,
            ncore,
            kpar,
            algo,
            cpus_per_task,
            partition,
            nodes,
            nodelist,
            directories,
            job_id,
            max_concurrent,
        )

        return job_id, exp_id

    except subprocess.CalledProcessError as e:
        print(f"✗ Error submitting job array: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return None, None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VASP Array Job Controller using SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover all subdirectories
  %(prog)s /path/to/vasp/runs 32 4 2 Normal --vasp-setup "module load vasp/6.4.3"
  
  # Auto-discover with directory suffix filter  
  %(prog)s /path/to/vasp/runs 32 4 2 Normal --vasp-setup "source vsp-setup-cpuenv.sh" --dir-suffix "_relax"
  
  # Use custom directory list from file
  %(prog)s /path/to/vasp/runs 32 4 2 Normal --vasp-setup "module load vasp/6.4.3" --directory-list dirs.txt
  
  # With partition and max concurrent jobs
  %(prog)s /path/to/vasp/runs 32 4 2 Normal --nodes 2 --partition gpu-partition --vasp-setup "module load vasp/6.4.3" -dl my_dirs.txt -mc 8

  # With specific node constraints
  %(prog)s /path/to/vasp/runs 32 4 2 Normal --nodes 1 --nodelist "node[01-03]" --vasp-setup "module load vasp/6.4.3"

Directory list file format:
  # Comments start with #
  structure_001
  structure_002/relax
  # Empty lines are ignored
  bulk/config_1
        """,
    )

    # Required positional arguments
    parser.add_argument(
        "base_directory",
        type=Path,
        help="Directory containing VASP calculation folders",
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

    # Optional named argument for max concurrent jobs
    parser.add_argument(
        "--max-concurrent",
        "-mc",
        type=int,
        default=6,
        help="Maximum concurrent array jobs (default: 6)",
    )

    # Mutually exclusive group for directory discovery
    dir_group = parser.add_mutually_exclusive_group()
    dir_group.add_argument(
        "--dir-suffix",
        "-ds",
        type=str,
        default=None,
        help="Directory suffix to match (default: all subdirectories)",
    )
    dir_group.add_argument(
        "--directory-list",
        "-dl",
        type=str,
        default=None,
        help="File containing list of directories (relative paths, one per line)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Validate inputs
    if not args.base_directory.is_dir():
        print(f"✗ Error: Base directory '{args.base_directory}' does not exist.")
        sys.exit(1)

    # Validate VASP parallelization
    if args.ntasks % (args.ncore * args.kpar) != 0:
        print(f"✗ Error: Invalid parallelization parameters!")
        print(
            f"  ntasks ({args.ntasks}) must be divisible by (ncore × kpar) = {args.ncore * args.kpar}"
        )
        sys.exit(1)

    # Find directories based on method specified
    if args.directory_list is not None:
        # Read directories from file
        directories = read_directory_list(args.directory_list, args.base_directory)
        search_method = f"directory list '{args.directory_list}'"
    else:
        # Auto-discover directories (existing logic)
        if args.dir_suffix is None:
            directories = [d for d in args.base_directory.iterdir() if d.is_dir()]
            search_method = "all subdirectories"
        else:
            directories = [
                d for d in args.base_directory.glob(f"*{args.dir_suffix}") if d.is_dir()
            ]
            search_method = f"directories ending with '{args.dir_suffix}'"

        if not directories:
            print(f"✗ No {search_method} found in {args.base_directory}")
            print("  Looking for VASP calculation directories...")
            sys.exit(1)

    print(f"Found {len(directories)} VASP directories to process ({search_method})")

    # Ensure required directories exist (with 0- prefix)
    (args.base_directory / "0-job_lists").mkdir(exist_ok=True)
    (args.base_directory / "0-dispatch-logs").mkdir(exist_ok=True)

    # Submit the array job
    job_id, _ = submit_array_job(
        args.base_directory,
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
        args.max_concurrent,
    )

    if job_id:
        print(
            f"\n📝 Logs will be in individual directories (via symlinks in {args.base_directory}/0-logs/)"
        )
        print(f"🔍 Monitor job status with: squeue -j {job_id}")
        print(f"📊 Check progress with status script on base directory")
    else:
        print("\n✗ Job submission failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
