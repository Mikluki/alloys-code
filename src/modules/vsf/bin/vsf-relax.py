#!/usr/bin/env python3
import argparse
import glob
import json
import logging
from datetime import datetime
from pathlib import Path

from vsf.calculators import Mace_mpa_0
from vsf.logging import setup_logging
from vsf.transform.poscars_relax import ChainedRelaxation

LOGGER = logging.getLogger(__name__)

# Relaxation parameters (same as original)
FMAX = 0.02
PRECISION = "float64"
key_file = "POSCAR"


def convert_paths_to_strings(obj):
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    else:
        return obj


def collect_poscars(base_dirs):
    """Collect POSCAR paths from all base directories."""
    all_poscar_paths = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        print(f"Searching in directory: {base_path}")

        # Look for POSCAR files in this directory and subdirectories
        poscar_paths = list(base_path.rglob(key_file))

        print(f"Found {len(poscar_paths)} POSCAR files in {base_path}")
        all_poscar_paths.extend(poscar_paths)

    return all_poscar_paths


def main():
    parser = argparse.ArgumentParser(
        description="Relax structures with MACE across multiple directories"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories to process (supports glob patterns like 'test/*/')",
    )

    args = parser.parse_args()

    # Expand directories and glob patterns
    all_dirs = []
    for arg in args.directories:
        if "*" in arg:
            # Use glob for patterns
            expanded = glob.glob(arg)
            all_dirs.extend(expanded)
        else:
            # Treat as literal directory
            all_dirs.append(arg)

    # Filter to existing directories
    existing_dirs = []
    for d in all_dirs:
        if Path(d).is_dir():
            existing_dirs.append(d)
        else:
            print(f"Skipping non-existent directory: {d}")

    if not existing_dirs:
        print("No valid directories found!")
        return

    # Setup timestamped outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"relax.{timestamp}.log"
    json_file = f"all_results.{timestamp}.json"
    base_output_dir = f"relaxed_{timestamp}"

    setup_logging(log_file=log_file)
    LOGGER.info(f"Starting MACE relaxation for {len(existing_dirs)} directories")

    # Collect all POSCAR paths
    all_poscar_paths = collect_poscars(existing_dirs)

    if not all_poscar_paths:
        print("No POSCAR files found in specified directories!")
        return

    print(f"Total POSCAR files to process: {len(all_poscar_paths)}")
    LOGGER.info(f"Found {len(all_poscar_paths)} POSCAR files to process")

    # Initialize calculator
    calc = Mace_mpa_0()
    calc.initialize(default_dtype=PRECISION)

    # Chained relaxation workflow
    chain = ChainedRelaxation(all_poscar_paths, calc, base_output_dir=base_output_dir)

    # Define relaxation stages
    chain.add_stage("01_csym", {"constant_symmetry": True, "fmax": FMAX})
    chain.add_stage("02_full", {"fmax": FMAX})

    # Run all stages
    LOGGER.info("Starting relaxation stages...")
    all_results = chain.run_all_stages()

    # Add metadata to results and convert paths

    all_results = {
        "precision": PRECISION,
        "directories_processed": existing_dirs,
        **all_results,
    }

    # Convert paths to strings for JSON serialization
    serializable_results = convert_paths_to_strings(all_results)

    # Save results
    with open(json_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    LOGGER.info(f"Relaxation complete. Results saved to {json_file}")
    print(f"Processing complete. Check {log_file} for details.")
    print(f"Results: {json_file}")
    print(f"Relaxed structures: {base_output_dir}/")


if __name__ == "__main__":
    main()
