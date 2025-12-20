#!/usr/bin/env python3
"""
MD Campaign Manager - VASP MD Continuation Orchestrator
Manages multi-round MD campaigns using existing VASPDevil for job submission.
"""

import argparse
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MDCampaignManager:
    def __init__(self, target_steps: int, max_round: int = 10):
        self.target_steps = target_steps
        self.max_round = max_round

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"md_campaign_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MD Campaign Manager started: {log_filename}")
        self.logger.info(f"Target MD steps: {target_steps}")

    def parse_oszicar(self, oszicar_path: Path) -> Tuple[int, Optional[float]]:
        """Parse OSZICAR to get MD step count and final energy."""
        if not oszicar_path.exists():
            return 0, None

        try:
            with open(oszicar_path, "r") as f:
                lines = f.readlines()

            md_steps = 0
            last_energy = None

            for line in lines:
                # Look for MD step lines: "T= 1000.00 E= -XX.XXXXX"
                if "T=" in line and "E=" in line:
                    md_steps += 1
                    # Extract energy
                    energy_match = re.search(r"E=\s*([-+]?\d*\.?\d+)", line)
                    if energy_match:
                        last_energy = float(energy_match.group(1))

            return md_steps, last_energy

        except Exception as e:
            self.logger.error(f"Failed to parse {oszicar_path}: {e}")
            return 0, None

    def get_nsw_from_incar(self, job_dir: Path) -> int:
        """Extract NSW value from INCAR."""
        incar_path = job_dir / "INCAR"
        if not incar_path.exists():
            self.logger.warning(f"No INCAR found in {job_dir}")
            return 0

        try:
            with open(incar_path, "r") as f:
                content = f.read()

            # Look for NSW = value
            nsw_match = re.search(r"NSW\s*=\s*(\d+)", content, re.IGNORECASE)
            if nsw_match:
                return int(nsw_match.group(1))
            else:
                self.logger.warning(f"NSW not found in {incar_path}")
                return 0

        except Exception as e:
            self.logger.error(f"Failed to read INCAR in {job_dir}: {e}")
            return 0

    def check_job_status(self, job_dir: Path) -> Dict:
        """Check MD job completion status."""
        oszicar_path = job_dir / "OSZICAR"
        contcar_path = job_dir / "CONTCAR"

        current_steps, last_energy = self.parse_oszicar(oszicar_path)
        nsw_target = self.get_nsw_from_incar(job_dir)

        # Determine status
        if current_steps == 0:
            status = "not_started"
        elif current_steps >= min(nsw_target, self.target_steps):
            status = "completed"
        elif current_steps > 0 and contcar_path.exists():
            status = "needs_continuation"
        else:
            status = "failed"

        return {
            "status": status,
            "current_steps": current_steps,
            "nsw_target": nsw_target,
            "last_energy": last_energy,
            "has_contcar": contcar_path.exists(),
            "has_oszicar": oszicar_path.exists(),
        }

    def update_incar_for_restart(self, original_incar: Path, new_incar: Path):
        """Update INCAR for MD restart."""
        try:
            with open(original_incar, "r") as f:
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
            with open(new_incar, "w") as f:
                f.writelines(new_lines)

            self.logger.info(f"Updated INCAR: {new_incar}")

        except Exception as e:
            self.logger.error(
                f"Failed to update INCAR {original_incar} -> {new_incar}: {e}"
            )

    def prepare_continuation(self, original_dir: Path, round_num: int) -> Path | None:
        """Prepare continuation directory for next round."""
        # Create continuation directory name
        base_name = original_dir.name
        if round_num == 1:
            cont_dir = original_dir.parent / f"{base_name}_r1"
        else:
            # Handle existing rounds: Al_seed001_r2 -> Al_seed001_r3
            base_match = re.match(r"(.+)_r(\d+)$", base_name)
            if base_match:
                base_stem = base_match.group(1)
                cont_dir = original_dir.parent / f"{base_stem}_r{round_num}"
            else:
                cont_dir = original_dir.parent / f"{base_name}_r{round_num}"

        # Create directory
        if cont_dir.exists():
            self.logger.warning(f"Continuation directory already exists: {cont_dir}")
            return cont_dir

        cont_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created continuation directory: {cont_dir}")

        # Copy essential files
        essential_files = ["POTCAR", "KPOINTS"]
        for filename in essential_files:
            src = original_dir / filename
            dst = cont_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                self.logger.debug(f"Copied {filename}")
            else:
                self.logger.warning(f"Missing {filename} in {original_dir}")

        # CONTCAR -> POSCAR
        contcar_src = original_dir / "CONTCAR"
        poscar_dst = cont_dir / "POSCAR"
        if contcar_src.exists():
            shutil.copy2(contcar_src, poscar_dst)
            self.logger.debug("Copied CONTCAR -> POSCAR")
        else:
            self.logger.error(f"No CONTCAR found in {original_dir}")
            return None

        # Update INCAR
        incar_src = original_dir / "INCAR"
        incar_dst = cont_dir / "INCAR"
        if incar_src.exists():
            self.update_incar_for_restart(incar_src, incar_dst)
        else:
            self.logger.error(f"No INCAR found in {original_dir}")
            return None

        return cont_dir

    def scan_jobs(self, glob_pattern: str) -> Dict[str, Dict]:
        """Scan for MD jobs matching pattern."""
        search_path = Path(glob_pattern).expanduser()

        if "*" in glob_pattern or "?" in glob_pattern:
            # Glob pattern
            parent_dir = search_path.parent
            pattern = search_path.name
            job_dirs = list(parent_dir.glob(pattern))
        else:
            # Direct path
            job_dirs = [search_path] if search_path.is_dir() else []

        job_dirs = [d for d in job_dirs if d.is_dir()]
        self.logger.info(f"Found {len(job_dirs)} directories matching '{glob_pattern}'")

        results = {}
        for job_dir in sorted(job_dirs):
            status_info = self.check_job_status(job_dir)
            results[str(job_dir)] = status_info

            self.logger.info(
                f"{job_dir.name}: {status_info['status']} "
                f"({status_info['current_steps']}/{status_info['nsw_target']} steps)"
            )

        return results

    def prepare_next_round(self, job_results: Dict[str, Dict]) -> List[Path]:
        """Prepare continuation directories for incomplete jobs."""
        prepared_dirs = []

        for job_path, status_info in job_results.items():
            if status_info["status"] != "needs_continuation":
                continue

            job_dir = Path(job_path)

            # Determine round number
            current_round = self.get_round_number(job_dir.name)
            next_round = current_round + 1

            if next_round > self.max_round:
                self.logger.warning(
                    f"Reached max round {self.max_round} for {job_dir.name}"
                )
                continue

            # Check if enough steps remain to justify continuation
            remaining_steps = self.target_steps - status_info["current_steps"]
            if remaining_steps < 100:  # Arbitrary threshold
                self.logger.info(
                    f"Only {remaining_steps} steps remaining for {job_dir.name}, marking complete"
                )
                continue

            cont_dir = self.prepare_continuation(job_dir, next_round)
            if cont_dir:
                prepared_dirs.append(cont_dir)

        return prepared_dirs

    def get_round_number(self, dir_name: str) -> int:
        """Extract round number from directory name."""
        match = re.search(r"_r(\d+)$", dir_name)
        return int(match.group(1)) if match else 0

    def print_summary(self, job_results: Dict[str, Dict]):
        """Print campaign summary."""
        status_counts = {}
        total_steps = 0

        for job_path, status_info in job_results.items():
            status = status_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            total_steps += status_info["current_steps"]

        self.logger.info("=== Campaign Summary ===")
        for status, count in status_counts.items():
            self.logger.info(f"{status}: {count} jobs")

        self.logger.info(f"Total MD steps completed: {total_steps}")
        avg_steps = total_steps / len(job_results) if job_results else 0
        self.logger.info(f"Average steps per job: {avg_steps:.1f}")

    def run_campaign_analysis(self, glob_pattern: str) -> Dict[str, Dict]:
        """Analyze current campaign status."""
        self.logger.info(f"Analyzing MD campaign: {glob_pattern}")
        job_results = self.scan_jobs(glob_pattern)
        self.print_summary(job_results)
        return job_results

    def prepare_continuations(
        self, glob_pattern: str, dry_run: bool = False
    ) -> List[Path]:
        """Prepare continuation directories for incomplete jobs."""
        job_results = self.run_campaign_analysis(glob_pattern)

        if dry_run:
            incomplete_jobs = [
                job_path
                for job_path, status in job_results.items()
                if status["status"] == "needs_continuation"
            ]
            self.logger.info(
                f"DRY RUN: Would prepare {len(incomplete_jobs)} continuations"
            )
            for job_path in incomplete_jobs:
                self.logger.info(f"  Would continue: {Path(job_path).name}")
            return []

        prepared_dirs = self.prepare_next_round(job_results)
        self.logger.info(f"Prepared {len(prepared_dirs)} continuation directories")

        return prepared_dirs


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MD Campaign Manager - VASP MD Continuation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current campaign status
  %(prog)s --analyze "path/to/md_jobs/*"
  
  # Prepare continuations (dry run)
  %(prog)s --prepare "path/to/md_jobs/*" --target-steps 11500 --dry-run
  
  # Prepare continuations for real
  %(prog)s --prepare "path/to/md_jobs/*" --target-steps 11500
        """,
    )

    parser.add_argument("glob_pattern", help="Glob pattern for MD job directories")

    parser.add_argument(
        "--target-steps",
        "-t",
        type=int,
        required=True,
        help="Target total MD steps (e.g., 11500 for 11.5 ps)",
    )

    parser.add_argument(
        "--analyze", "-a", action="store_true", help="Analyze campaign status only"
    )

    parser.add_argument(
        "--prepare", "-p", action="store_true", help="Prepare continuation directories"
    )

    parser.add_argument(
        "--dry-run",
        "-dr",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    parser.add_argument(
        "--max-round",
        "-mr",
        type=int,
        default=10,
        help="Maximum continuation round (default: 10)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not (args.analyze or args.prepare):
        print("✗ Must specify either --analyze or --prepare")
        sys.exit(1)

    manager = MDCampaignManager(
        target_steps=args.target_steps, max_round=args.max_round
    )

    if args.analyze:
        manager.run_campaign_analysis(args.glob_pattern)

    if args.prepare:
        prepared_dirs = manager.prepare_continuations(args.glob_pattern, args.dry_run)
        if prepared_dirs and not args.dry_run:
            print(f"\n✓ Prepared {len(prepared_dirs)} continuation directories")
            print("Now run VASPDevil on the continuation directories:")

            # Generate example VASPDevil command
            first_dir = prepared_dirs[0]
            parent_dir = first_dir.parent
            round_num = manager.get_round_number(first_dir.name)
            glob_for_devil = f"{parent_dir}/*_r{round_num}"

            print(
                f'  python vasp-devil.py "{glob_for_devil}" [your_vasp_params] --max-jobs [N]'
            )


if __name__ == "__main__":
    main()
