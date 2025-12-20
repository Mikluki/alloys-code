#!/usr/bin/env python3
"""
VASP Results Collection Script
Reads experiments.csv and updates status based on OUTCAR analysis using ripgrep.
"""

import csv
import datetime
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run_ripgrep(pattern, directory, file_pattern="*"):
    """Run ripgrep with given pattern in specific directory."""
    try:
        result = subprocess.run(
            ["rg", "-l", pattern, f"{directory}/{file_pattern}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.split("\n") if line.strip()]
        return []
    except FileNotFoundError:
        print("Error: ripgrep (rg) not found. Please install ripgrep.")
        sys.exit(1)


def get_elapsed_time(directory):
    """Extract elapsed time from OUTCAR using ripgrep."""
    try:
        result = subprocess.run(
            ["rg", "Elapsed time", f"{directory}/OUTCAR"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            # Extract the time value (last word on the line)
            time_str = (
                result.stdout.strip().split()[-1] if result.stdout.strip() else ""
            )
            return time_str
        return ""
    except:
        return ""


def check_directory_status(directory):
    """Check VASP calculation status for a single directory."""
    dir_path = Path(directory)

    # Check if OUTCAR exists
    outcar_path = dir_path / "OUTCAR"
    if not outcar_path.exists():
        return "NOT_STARTED", ""

    # Check for completion (Elapsed time)
    completed_files = run_ripgrep("Elapsed time", str(dir_path), "OUTCAR")
    if completed_files:
        elapsed_time = get_elapsed_time(str(dir_path))
        return "COMPLETED", elapsed_time

    # Check for errors
    error_files_1 = run_ripgrep("EEEEEE", str(dir_path))
    error_files_2 = run_ripgrep("VERY BAD NEWS", str(dir_path))

    if error_files_1 or error_files_2:
        return "ERROR", ""

    # If OUTCAR exists but no completion or error, assume running
    return "RUNNING", ""


def collect_results(base_dir):
    """Collect results from all planned experiments."""
    base_path = Path(base_dir)
    csv_path = base_path / "experiments.csv"

    if not csv_path.exists():
        print(f"Error: experiments.csv not found in {base_dir}")
        print("Run the experiment controller first to create the tracking file.")
        return

    print(f"Reading experiments from: {csv_path}")

    # Read current CSV data
    experiments = []
    with csv_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            experiments.append(row)

    if not experiments:
        print("No experiments found in CSV file.")
        return

    print(f"Found {len(experiments)} experiment records")

    # Update status for each experiment
    updated_count = 0
    status_summary = Counter()
    failed_jobs = []

    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for exp in experiments:
        directory = exp["directory"]
        old_status = exp["status"]

        # Check current status
        new_status, elapsed_time = check_directory_status(directory)

        # Update if status changed or we got timing info
        if new_status != old_status or (elapsed_time and not exp["elapsed_time"]):
            exp["status"] = new_status
            exp["elapsed_time"] = elapsed_time
            exp["timestamp"] = current_timestamp
            updated_count += 1

        status_summary[new_status] += 1

        # Track failed jobs
        if new_status in ["ERROR", "NOT_STARTED"]:
            failed_jobs.append(
                {
                    "directory": Path(directory).name,
                    "exp_type": exp["exp_type"],
                    "status": new_status,
                    "exp_id": exp["exp_id"],
                }
            )

    # Write updated CSV
    with csv_path.open("w", newline="") as csvfile:
        fieldnames = [
            "exp_id",
            "exp_type",
            "directory",
            "ntasks",
            "ntasks_per_node",
            "ncore",
            "kpar",
            "status",
            "elapsed_time",
            "timestamp",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            writer.writerow(exp)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT STATUS SUMMARY")
    print("=" * 60)

    for status, count in status_summary.most_common():
        print(f"{status:15} : {count:4d}")

    print(f"{'TOTAL':15} : {len(experiments):4d}")

    if updated_count > 0:
        print(f"\n✅ Updated {updated_count} experiment records")
    else:
        print(f"\n📋 No status changes detected")

    # Report failed jobs
    if failed_jobs:
        print(f"\n❌ FAILED JOBS ({len(failed_jobs)}):")
        for job in failed_jobs:
            print(f"   {job['status']:12} : {job['directory']} ({job['exp_type']})")

        # Group by status for summary
        error_count = len([j for j in failed_jobs if j["status"] == "ERROR"])
        not_started_count = len(
            [j for j in failed_jobs if j["status"] == "NOT_STARTED"]
        )

        print(f"\n📊 Failure Summary:")
        if error_count > 0:
            print(
                f"   ERROR jobs: {error_count} (check OUTCAR files for EEEEEE/VERY BAD NEWS)"
            )
        if not_started_count > 0:
            print(f"   NOT_STARTED jobs: {not_started_count} (no OUTCAR file found)")
    else:
        print(f"\n✅ No failed jobs detected!")

    print(f"\n📄 Results saved to: {csv_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <base_directory>")
        print(f"Example: {sys.argv[0]} /path/to/vasp/experiments")
        sys.exit(1)

    base_dir = sys.argv[1]
    base_path = Path(base_dir)

    if not base_path.is_dir():
        print(f"Error: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    collect_results(base_dir)


if __name__ == "__main__":
    main()
