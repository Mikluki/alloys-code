#!/usr/bin/env python3
"""
VASP Log Recovery Script
Recovers experiment data from SLURM log files to recreate experiments.csv
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_log_file(log_file_path):
    """Parse a single SLURM log file and extract experiment data."""
    try:
        with open(log_file_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None

    # Debug: show that we're parsing this file
    print(f"  Parsing file: {log_file_path.name} ({len(content)} chars)")

    # Initialize record
    record = {}

    # Extract job info from filename
    # Pattern 1: vasp_16_8_2_19168_1.out (ntasks_ncore_kpar_jobid_taskid)
    # Pattern 2: exp1_8_19165_2.out or exp2_32_4_8_19167_3.out
    filename = log_file_path.name
    filename_ncore = None
    filename_kpar = None

    # Try new pattern first: vasp_16_8_2_19168_1.out
    vasp_match = re.match(r"vasp_(\d+)_(\d+)_(\d+)_\d+_\d+\.out", filename)
    if vasp_match:
        record["ntasks"] = int(vasp_match.group(1))
        record["ntasks_per_node"] = int(vasp_match.group(1))
        filename_ncore = int(vasp_match.group(2))
        filename_kpar = int(vasp_match.group(3))
        print(
            f"  Filename pattern 1: ntasks={record['ntasks']}, ncore={filename_ncore}, kpar={filename_kpar}"
        )
    else:
        # Try old pattern: exp1_8_19165_2.out or exp2_32_4_8_19167_3.out
        exp1_match = re.match(r"exp1_(\d+)_\d+_\d+\.out", filename)
        if exp1_match:
            record["ntasks"] = int(exp1_match.group(1))
            record["ntasks_per_node"] = int(exp1_match.group(1))
            record["exp_type"] = "exp1"
            filename_ncore = 1
            filename_kpar = 1
            print(f"  Filename pattern 2 (exp1): ntasks={record['ntasks']}")
        else:
            exp2_match = re.match(r"exp2_(\d+)_(\d+)_(\d+)_\d+_\d+\.out", filename)
            if exp2_match:
                record["ntasks"] = int(exp2_match.group(1))
                record["ntasks_per_node"] = int(exp2_match.group(1))
                record["exp_type"] = "exp2"
                filename_ncore = int(exp2_match.group(2))
                filename_kpar = int(exp2_match.group(3))
                print(
                    f"  Filename pattern 2 (exp2): ntasks={record['ntasks']}, ncore={filename_ncore}, kpar={filename_kpar}"
                )
            else:
                # Manual extraction as last resort
                print(
                    f"  No standard pattern matched for '{filename}', trying manual extraction..."
                )
                numbers = re.findall(r"\d+", filename)
                if len(numbers) >= 3:
                    record["ntasks"] = int(numbers[0])
                    record["ntasks_per_node"] = int(numbers[0])
                    filename_ncore = int(numbers[1])
                    filename_kpar = int(numbers[2])
                    print(
                        f"  Manual extraction: ntasks={record['ntasks']}, ncore={filename_ncore}, kpar={filename_kpar}"
                    )
                else:
                    print(f"  Warning: Could not parse filename pattern for {filename}")
                    print(f"  Numbers found: {numbers}")

    # Extract experiment ID
    exp_id_match = re.search(r"Experiment ID:\s*(\S+)", content)
    if exp_id_match:
        record["exp_id"] = exp_id_match.group(1)

    # Extract working directory
    work_dir_match = re.search(r"Working Directory:\s*(.+)", content)
    if work_dir_match:
        record["directory"] = work_dir_match.group(1).strip()

    # Extract parameters (ncore and kpar)
    params_match = re.search(r"Parameters:\s*ncore=(\d+),kpar=(\d+)", content)
    if params_match:
        record["ncore"] = int(params_match.group(1))
        record["kpar"] = int(params_match.group(2))

    # Extract timing information
    start_time_match = re.search(r"Job started at:\s*(.+)", content)
    if start_time_match:
        record["start_time"] = start_time_match.group(1).strip()

    finish_time_match = re.search(r"Job finished at:\s*(.+)", content)
    if finish_time_match:
        record["finish_time"] = finish_time_match.group(1).strip()
        record["timestamp"] = finish_time_match.group(1).strip()

    # Extract runtime
    runtime_match = re.search(r"Total runtime:\s*(\d+)\s*seconds", content)
    if runtime_match:
        record["elapsed_time"] = runtime_match.group(1)

    # Determine status
    if "Job completed with status: COMPLETED" in content:
        record["status"] = "COMPLETED"
    elif re.search(r"VASP exit code:\s*0", content):
        record["status"] = "COMPLETED"
    elif re.search(r"VASP exit code:\s*[1-9]", content):
        record["status"] = "ERROR"
    elif "ERROR:" in content.upper() or "FAILED" in content.upper():
        record["status"] = "ERROR"
    else:
        record["status"] = "UNKNOWN"

    # Determine experiment type based on ncore/kpar values
    # exp1: ncore=1, kpar=1 (defaults)
    # exp2: ncore>1, various kpar values
    if "ncore" in record and "kpar" in record:
        if record["ncore"] == 1 and record["kpar"] == 1:
            record["exp_type"] = "exp1"
        else:
            record["exp_type"] = "exp2"
    elif filename_ncore is not None and filename_kpar is not None:
        # Fallback: use filename values if parameters not found in content
        if filename_ncore == 1 and filename_kpar == 1:
            record["exp_type"] = "exp1"
        else:
            record["exp_type"] = "exp2"
        record["ncore"] = filename_ncore
        record["kpar"] = filename_kpar

    # Final validation and field cleanup
    if "ncore" not in record:
        record["ncore"] = 1  # Default
    if "kpar" not in record:
        record["kpar"] = 1  # Default
    if "exp_type" not in record:
        # Last resort: infer from ncore/kpar
        if record["ncore"] == 1 and record["kpar"] == 1:
            record["exp_type"] = "exp1"
        else:
            record["exp_type"] = "exp2"

    # Ensure all required fields have values (with fallbacks if needed)
    if "elapsed_time" not in record:
        record["elapsed_time"] = ""
    if "timestamp" not in record:
        record["timestamp"] = record.get("finish_time", record.get("start_time", ""))
    if "status" not in record:
        record["status"] = "UNKNOWN"

    # Validate required fields
    required_fields = ["exp_id", "exp_type", "directory", "ntasks", "ncore", "kpar"]
    missing_fields = [
        field for field in required_fields if field not in record or record[field] == ""
    ]

    if missing_fields:
        print(
            f"Warning: Missing required fields in {log_file_path.name}: {missing_fields}"
        )
        print(f"  Available fields: {list(record.keys())}")
        # Show first few lines of content for debugging
        if content:
            lines = content.split("\n")[:10]
            print(f"  Log content preview:")
            for i, line in enumerate(lines):
                print(f"    {i+1:2d}: {line[:80]}")
        return None

    return record


def recover_experiments_from_logs(base_dir):
    """Recover experiment data from all log files."""
    base_path = Path(base_dir)
    logs_dir = base_path / "logs"

    if not logs_dir.exists():
        print(f"Error: Logs directory not found at {logs_dir}")
        return

    print(f"Scanning log files in: {logs_dir}")

    # Find all .out log files
    log_files = list(logs_dir.glob("*.out"))

    if not log_files:
        print("No .out log files found!")
        return

    print(f"Found {len(log_files)} log files")

    # Parse all log files
    records = []
    failed_parses = []

    for log_file in sorted(log_files):
        print(f"Processing: {log_file.name}")
        record = parse_log_file(log_file)

        if record:
            records.append(record)
            print(
                f"  ✅ Parsed: {record['exp_type']} - ntasks={record['ntasks']}, ncore={record['ncore']}, kpar={record['kpar']}"
            )
        else:
            failed_parses.append(log_file.name)
            print(f"  ❌ Failed to parse")

    if failed_parses:
        print(f"\nFailed to parse {len(failed_parses)} files:")
        for failed in failed_parses:
            print(f"  {failed}")

    if not records:
        print("No valid experiment records recovered!")
        return

    print(f"\nRecovered {len(records)} experiment records")

    # Create experiments.csv
    experiments_csv = base_path / "experiments.csv"
    backup_csv = base_path / "experiments_backup.csv"

    # Backup existing file if it exists
    if experiments_csv.exists():
        experiments_csv.rename(backup_csv)
        print(f"Backed up existing experiments.csv to {backup_csv}")

    # Write new CSV
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

    with experiments_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            # Ensure all fields are present
            csv_record = {}
            for field in fieldnames:
                csv_record[field] = record.get(field, "")
            writer.writerow(csv_record)

    print(f"\n✅ Recovered experiments.csv saved to: {experiments_csv}")

    # Print summary
    exp_types = defaultdict(int)
    status_counts = defaultdict(int)

    for record in records:
        exp_types[record.get("exp_type", "Unknown")] += 1
        status_counts[record.get("status", "Unknown")] += 1

    print(f"\n📊 Recovery Summary:")
    print(f"Experiment types:")
    for exp_type, count in exp_types.items():
        print(f"  {exp_type}: {count}")

    print(f"Status distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # Show sample records
    print(f"\n📋 Sample recovered data:")
    for i, record in enumerate(records[:3]):
        print(
            f"  {i+1}. {record['exp_type']} - {Path(record['directory']).name} - "
            f"ntasks={record['ntasks']}, ncore={record.get('ncore', '?')}, "
            f"kpar={record.get('kpar', '?')} - {record.get('status', '?')}"
        )

    if len(records) > 3:
        print(f"  ... and {len(records)-3} more")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <base_directory>")
        print(f"Example: {sys.argv[0]} /path/to/vasp/experiments")
        print(f"\nThis script will:")
        print(f"  1. Read all .out files from <base_directory>/logs/")
        print(f"  2. Extract experiment data from SLURM log format")
        print(f"  3. Create new experiments.csv file")
        print(f"  4. Backup existing experiments.csv if present")
        sys.exit(1)

    base_dir = sys.argv[1]
    base_path = Path(base_dir)

    if not base_path.is_dir():
        print(f"Error: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    recover_experiments_from_logs(base_dir)


if __name__ == "__main__":
    main()
