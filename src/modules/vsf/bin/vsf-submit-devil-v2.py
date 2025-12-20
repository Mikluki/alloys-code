#!/usr/bin/env python3
"""
VASP Devil - Job Queue Manager for SLURM with MD Continuation
Manages submission of VASP jobs within cluster limits using the existing submission script.
Now includes automatic MD continuation for jobs approaching 24-hour wall time limit.
"""

import argparse
import getpass
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class VASPDevil:
    def __init__(self, config, sleep_time: float, state_file="vasp-devil-state.json"):
        self.config = config
        self.sleep_time = sleep_time
        self.state_file = Path(state_file)
        self.submission_script = "vsf-submit-job.py"  # Assuming it's in PATH

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"vasp-devil_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"VASP Devil log started: {log_filename}")

        # Load state after logger is available
        self.state = self._load_state()

    def _load_state(self):
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self.logger.info(f"Loaded existing state from {self.state_file}")

                # Ensure continuation_queue exists for backwards compatibility
                if "continuation_queue" not in state:
                    state["continuation_queue"] = []

                return state
            except Exception as e:
                self.logger.error(f"Failed to load state file: {e}")
                return self._create_new_state()
        else:
            return self._create_new_state()

    def _create_new_state(self):
        """Create new state structure."""
        self.logger.info("Creating new state")

        # Find directories using the same logic as original script
        directories = self._find_directories()

        state = {
            "config": self.config,
            "directories": {},
            "continuation_queue": [],  # New: directories needing continuation
        }

        # Initialize all directories as pending
        for directory in directories:
            state["directories"][str(directory)] = {"status": "pending"}

        self.logger.info(f"Found {len(directories)} directories to process")
        return state

    def _find_directories(self):
        """Find directories using glob pattern (same logic as original script)."""
        glob_pattern = self.config["glob_pattern"]
        glob_path = Path(glob_pattern)

        if glob_path.is_absolute():
            pattern_parent = glob_path.parent
            pattern_name = glob_path.name
        else:
            pattern_parent = Path.cwd()
            pattern_name = glob_pattern

        try:
            if "*" in pattern_name or "?" in pattern_name or "[" in pattern_name:
                directories = [
                    d for d in pattern_parent.glob(pattern_name) if d.is_dir()
                ]
            else:
                single_dir = pattern_parent / pattern_name
                directories = [single_dir] if single_dir.is_dir() else []
        except Exception as e:
            self.logger.error(f"Error processing glob pattern '{glob_pattern}': {e}")
            return []

        return directories

    def _save_state(self):
        """Save current state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _get_current_job_count(self) -> int:
        """Get current number of jobs for user."""
        username = getpass.getuser()  # str

        try:
            result = subprocess.run(
                ["squeue", "-u", username, "-h"],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout.strip()
            return 0 if not output else len(output.splitlines())
        except subprocess.CalledProcessError as e:
            self.log_slurm_unavailable(f"squeue failed: {e}")
            return -1

    def _get_current_job_ids(self):
        """Get set of current job IDs for user."""
        username = getpass.getuser()  # str

        try:
            result = subprocess.run(
                ["squeue", "-u", username, "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout.strip()
            if not output:
                return set()
            else:
                job_ids = set(output.split("\n"))
                return job_ids
        except subprocess.CalledProcessError as e:
            self.log_slurm_unavailable(f"squeue for job IDs failed: {e}")
            return set()

    def _submit_single_job(self, directory):
        """Submit job for a single directory using the original script."""
        cmd = [
            self.submission_script,
            str(directory),
            str(self.config["ntasks"]),
            str(self.config["ncore"]),
            str(self.config["kpar"]),
            self.config["algo"],
            "--vasp-setup",
            self.config["vasp_setup"],
        ]

        # Add optional arguments
        if self.config.get("cpus_per_task"):
            cmd.extend(["--cpus_per_task", str(self.config["cpus_per_task"])])
        if self.config.get("partition"):
            cmd.extend(["--partition", self.config["partition"]])
        if (
            self.config.get("nodes") and self.config["nodes"] != 1
        ):  # Only add if not default
            cmd.extend(["--nodes", str(self.config["nodes"])])
        if (
            self.config.get("nodelist") and self.config["nodelist"]
        ):  # Only add if not empty
            cmd.extend(["--nodelist", self.config["nodelist"]])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse job ID from output
            # Your script prints: "✓ Submitted job 'job_name' with ID 12345 for /path"
            output_lines = result.stdout.strip().split("\n")
            job_id = None

            for line in output_lines:
                # Look for the success line with job ID
                match = re.search(r"✓ Submitted job .* with ID (\d+)", line)
                if match:
                    job_id = match.group(1)
                    break

            if job_id:
                self.logger.info(f"Successfully submitted job {job_id} for {directory}")
                return job_id
            else:
                self.logger.warning(
                    f"Job submitted for {directory} but couldn't parse job ID from output:"
                )
                self.logger.warning(f"  Script output: {result.stdout}")
                return "unknown"

        except subprocess.CalledProcessError as e:
            self.log_submission_error(directory, f"Submission failed: {e}")
            return None

    def _update_job_statuses(self):
        """Update status of tracked jobs + detect approaching deadlines."""
        current_job_ids = self._get_current_job_ids()
        if current_job_ids is None:  # Error occurred
            return

        current_time = time.time()

        for dir_path, dir_info in self.state["directories"].items():
            if dir_info["status"] == "running" and "job_id" in dir_info:
                job_id = dir_info["job_id"]

                # Calculate runtime
                try:
                    submit_time = datetime.fromisoformat(
                        dir_info["submitted_at"]
                    ).timestamp()
                    runtime_hours = (current_time - submit_time) / 3600
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Cannot calculate runtime for {dir_path}: {e}")
                    runtime_hours = 0

                if job_id not in current_job_ids:
                    # Job finished - mark completed regardless of reason
                    dir_info["status"] = "completed"
                    dir_info["completed_at"] = datetime.now().isoformat()
                    self.logger.info(f"Job {job_id} for {dir_path} completed")

                elif runtime_hours > 23.5:  # Still running but will be killed soon
                    self._queue_for_continuation(dir_path, dir_info)

    def _queue_for_continuation(self, dir_path, dir_info):
        """Queue job for continuation and prepare files."""
        self.logger.info(f"Queueing {dir_path} for continuation (runtime > 23.5h)")

        # Change status to signal it needs continuation
        dir_info["status"] = "needs_continuation"
        dir_info["continuation_queued_at"] = datetime.now().isoformat()

        # Add to continuation queue if not already there
        if dir_path not in self.state["continuation_queue"]:
            self.state["continuation_queue"].append(dir_path)

        # Prepare continuation files now (while job still running)
        try:
            self._prepare_continuation_files(dir_path)
        except Exception as e:
            self.logger.error(f"Failed to prepare continuation for {dir_path}: {e}")

    def _prepare_continuation_files(self, dir_path):
        """Prepare continuation directory with restart files."""
        original_dir = Path(dir_path)

        # Create continuation directory name (flat structure)
        cont_dir = self._get_continuation_dir_name(original_dir)
        cont_dir.mkdir(parents=True, exist_ok=True)

        # Copy essential files
        essential_files = ["POTCAR", "KPOINTS"]
        for filename in essential_files:
            src = original_dir / filename
            if src.exists():
                shutil.copy2(src, cont_dir / filename)
                self.logger.debug(f"Copied {filename} to {cont_dir}")
            else:
                self.logger.warning(f"Missing {filename} in {original_dir}")

        # CONTCAR -> POSCAR (copy if exists, will be updated when job finishes)
        contcar_src = original_dir / "CONTCAR"
        poscar_dst = cont_dir / "POSCAR"
        if contcar_src.exists():
            shutil.copy2(contcar_src, poscar_dst)
            self.logger.debug(f"Copied CONTCAR -> POSCAR in {cont_dir}")

        # Create updated INCAR
        self._create_continuation_incar(original_dir, cont_dir)

        self.logger.info(f"Prepared continuation files: {cont_dir}")

    def _get_continuation_dir_name(self, original_dir):
        """Generate continuation directory name using flat structure."""
        base_name = original_dir.name

        # Handle existing rounds: Al_seed001_r2 -> Al_seed001_r3
        round_match = re.match(r"(.+)_r(\d+)$", base_name)
        if round_match:
            base_stem, round_num = round_match.groups()
            next_round = int(round_num) + 1
            return original_dir.parent / f"{base_stem}_r{next_round}"
        else:
            # First continuation: Al_seed001 -> Al_seed001_r1
            return original_dir.parent / f"{base_name}_r1"

    def _create_continuation_incar(self, original_dir, cont_dir):
        """Create INCAR for continuation job."""
        incar_src = original_dir / "INCAR"
        incar_dst = cont_dir / "INCAR"

        if not incar_src.exists():
            self.logger.error(f"No INCAR found in {original_dir}")
            return

        try:
            with open(incar_src, "r") as f:
                lines = f.readlines()

            # Track which parameters we've updated
            updated_params = set()
            new_lines = []

            for line in lines:
                # Skip comments and empty lines
                if line.strip().startswith("#") or not line.strip():
                    new_lines.append(line)
                    continue

                # Check for parameters we need to update
                if re.match(r"\s*ISTART\s*=", line, re.IGNORECASE):
                    new_lines.append("ISTART = 1  # Restart from WAVECAR\n")
                    updated_params.add("ISTART")
                elif re.match(r"\s*LWAVE\s*=", line, re.IGNORECASE):
                    new_lines.append(
                        "LWAVE = .TRUE.  # Write WAVECAR for next continuation\n"
                    )
                    updated_params.add("LWAVE")
                else:
                    new_lines.append(line)

            # Add missing parameters
            if "ISTART" not in updated_params:
                new_lines.append("ISTART = 1  # Restart from WAVECAR\n")
            if "LWAVE" not in updated_params:
                new_lines.append(
                    "LWAVE = .TRUE.  # Write WAVECAR for next continuation\n"
                )

            # Write updated INCAR
            with open(incar_dst, "w") as f:
                f.writelines(new_lines)

            self.logger.debug(f"Created continuation INCAR: {incar_dst}")

        except Exception as e:
            self.logger.error(
                f"Failed to create continuation INCAR {incar_src} -> {incar_dst}: {e}"
            )

    def _get_pending_directories(self):
        """Get list of directories with pending status."""
        pending = []
        for dir_path, dir_info in self.state["directories"].items():
            if dir_info["status"] == "pending":
                pending.append(dir_path)
        return pending

    def _submit_next_batch(self, available_slots):
        """Submit next batch: NEW JOBS first, then continuations."""

        # Priority 1: New pending jobs
        pending_dirs = self._get_pending_directories()
        new_jobs_to_submit = pending_dirs[:available_slots]

        for dir_path in new_jobs_to_submit:
            job_id = self._submit_single_job(dir_path)
            if job_id:
                self.state["directories"][dir_path].update(
                    {
                        "status": "running",
                        "job_id": job_id,
                        "submitted_at": datetime.now().isoformat(),
                    }
                )
            else:
                self.state["directories"][dir_path]["status"] = "failed"

        remaining_slots = available_slots - len(new_jobs_to_submit)

        # Priority 2: Continuations (if slots remaining)
        if remaining_slots > 0:
            self._submit_continuations(remaining_slots)

    def _submit_continuations(self, available_slots):
        """Submit continuation jobs for queued directories."""
        continuations_to_submit = self.state["continuation_queue"][:available_slots]

        for dir_path in continuations_to_submit:
            original_dir = Path(dir_path)
            cont_dir = self._get_continuation_dir_name(original_dir)

            if cont_dir.exists():
                # Update CONTCAR -> POSCAR one more time before submission
                contcar_src = original_dir / "CONTCAR"
                poscar_dst = cont_dir / "POSCAR"
                if contcar_src.exists():
                    shutil.copy2(contcar_src, poscar_dst)

                job_id = self._submit_single_job(cont_dir)
                if job_id:
                    # Add continuation to directories tracking
                    self.state["directories"][str(cont_dir)] = {
                        "status": "running",
                        "job_id": job_id,
                        "submitted_at": datetime.now().isoformat(),
                        "original_job": dir_path,
                    }
                    self.logger.info(f"Submitted continuation {job_id} for {dir_path}")
                else:
                    self.logger.error(f"Failed to submit continuation for {dir_path}")
            else:
                self.logger.error(f"Continuation directory not found: {cont_dir}")

            # Remove from queue (whether successful or failed)
            self.state["continuation_queue"].remove(dir_path)

    def run(self, dry_run=False):
        """Main execution loop."""
        self.logger.info("VASP Devil starting...")
        self.logger.info(f"Max jobs: {self.config['max_jobs']}")

        # Log continuation status if any
        if self.state["continuation_queue"]:
            self.logger.info(
                f"Resuming with {len(self.state['continuation_queue'])} continuations queued"
            )

        if dry_run:
            pending = self._get_pending_directories()
            queued_continuations = len(self.state["continuation_queue"])
            self.logger.info(f"DRY RUN: Would submit {len(pending)} new jobs")
            self.logger.info(f"DRY RUN: {queued_continuations} continuations queued")
            for dir_path in pending:
                self.logger.info(f"  Would submit: {dir_path}")
            return

        try:
            while True:
                # Update job statuses (includes continuation detection)
                self._update_job_statuses()

                # Check if we have pending jobs or continuations
                pending_dirs = self._get_pending_directories()
                queued_continuations = len(self.state["continuation_queue"])

                if not pending_dirs and not queued_continuations:
                    self.logger.info("All jobs completed!")
                    break

                # Check current job count
                current_jobs = self._get_current_job_count()
                if current_jobs == -1:  # Error occurred
                    self.logger.warning("Skipping this cycle due to SLURM error")
                    time.sleep(self.sleep_time)
                    continue

                available_slots = self.config["max_jobs"] - current_jobs
                self.logger.info(
                    f"Current jobs: {current_jobs}, Available slots: {available_slots}, "
                    f"Pending: {len(pending_dirs)}, Continuations queued: {queued_continuations}"
                )

                # Submit new jobs if slots available
                if available_slots > 0:
                    self._submit_next_batch(available_slots)

                # Save state
                self._save_state()

                # Wait before next cycle
                time.sleep(self.sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Saving state...")
            self._save_state()
            sys.exit(0)

    # Error handling placeholder functions
    def log_submission_error(self, directory, error):
        """Log submission error."""
        self.logger.error(f"Submission error for {directory}: {error}")

    def log_slurm_unavailable(self, error):
        """Log SLURM unavailability."""
        self.logger.error(f"SLURM unavailable: {error}")

    def log_job_tracking_lost(self, job_id):
        """Log lost job tracking."""
        self.logger.error(f"Lost tracking for job {job_id}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VASP Devil - Job Queue Manager for SLURM with MD Continuation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit jobs with max limit (includes automatic MD continuation)
  %(prog)s "path/to/calc/rand*" 16 16 1 Normal --max-jobs 60 --vasp-setup "module load vasp"
  
  # Resume from existing state
  %(prog)s --resume
  
  # Dry run to see what would be submitted
  %(prog)s "path/to/calc/*" 16 16 1 Normal --max-jobs 60 --vasp-setup "..." --dry-run
        """,
    )

    # Required positional arguments (unless resuming)
    parser.add_argument(
        "glob_pattern",
        type=str,
        nargs="?",
        help="Glob pattern for VASP calculation directories",
    )
    parser.add_argument("ntasks", type=int, nargs="?", help="Total number of MPI tasks")
    parser.add_argument("ncore", type=int, nargs="?", help="VASP NCORE parameter")
    parser.add_argument("kpar", type=int, nargs="?", help="VASP KPAR parameter")
    parser.add_argument("algo", type=str, nargs="?", help="VASP ALGO parameter")

    # Required named arguments
    parser.add_argument(
        "--max-jobs", "-mj", type=int, help="Maximum number of concurrent jobs"
    )
    parser.add_argument("--vasp-setup", type=str, help="VASP setup command")

    # Optional arguments (same as original script)
    parser.add_argument("--cpus_per_task", "-cpt", type=int, help="CPUs per task")
    parser.add_argument("--partition", "-p", type=str, help="SLURM partition")
    parser.add_argument(
        "--nodelist",
        "-nl",
        type=str,
        default="",
        help="Specific nodes to use (e.g., 'node[01-03]' or 'node01,node02'; default: let SLURM choose)",
    )

    parser.add_argument(
        "--nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to request (default: 1)",
    )

    # Devil-specific arguments
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing state"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be submitted"
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="vasp-devil-state.json",
        help="State file location",
    )
    parser.add_argument(
        "--sleep-time",
        "-st",
        type=float,
        default=30,
        help="Devil sleep time",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.resume:
        # Load config from existing state file
        state_file = Path(args.state_file)
        if not state_file.exists():
            print(f"✗ No state file found at {state_file}")
            sys.exit(1)

        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            config = state["config"]
        except Exception as e:
            print(f"✗ Failed to load state file: {e}")
            sys.exit(1)
    else:
        # Validate required arguments
        required_args = [
            args.glob_pattern,
            args.ntasks,
            args.ncore,
            args.kpar,
            args.algo,
            args.max_jobs,
            args.vasp_setup,
        ]
        if any(arg is None for arg in required_args):
            print("✗ Missing required arguments (unless using --resume)")
            sys.exit(1)

        # Validate VASP parallelization
        if args.ntasks % (args.ncore * args.kpar) != 0:
            print(f"✗ Error: Invalid parallelization parameters!")
            print(
                f"  ntasks ({args.ntasks}) must be divisible by (ncore × kpar) = {args.ncore * args.kpar}"
            )
            sys.exit(1)

        # Build config
        config = {
            "glob_pattern": args.glob_pattern,
            "ntasks": args.ntasks,
            "ncore": args.ncore,
            "kpar": args.kpar,
            "algo": args.algo,
            "max_jobs": args.max_jobs,
            "vasp_setup": args.vasp_setup,
            "cpus_per_task": args.cpus_per_task,
            "partition": args.partition,
            "nodes": args.nodes,
            "nodelist": args.nodelist,
        }

    # Create and run devil
    devil = VASPDevil(config, args.sleep_time, args.state_file)
    devil.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
