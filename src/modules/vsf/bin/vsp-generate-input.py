#!/usr/bin/env python3
import argparse
import glob
import logging
from datetime import datetime
from pathlib import Path

from vsf.logging import setup_logging
from vsf.vasp.input.files import VaspInputFiles

LOGGER = logging.getLogger(__name__)

# VASP parameters (same as original)
kpoints_include_gamma = "auto"
kpoints_min_distance = 28
incar_dic = {
    "ALGO": "Normal",
    "EDIFF": 1.0e-05,
    "GGA_COMPAT": False,
    "IBRION": -1,
    "ISMEAR": 1,
    "ISPIN": 2,
    "ISIF": 2,
    "LASPH": True,
    "LCHARG": False,
    "LMAXMIX": 6,
    "LREAL": "AUTO",
    "LWAVE": False,
    "NELM": 60,
    "NELMIN": 4,
    "NCORE": 4,
    "KPAR": 4,
    "PREC": "Accurate",
    "SIGMA": 0.1,
    "NSW": 0,
}


def process_directory(base_dir):
    """Process a single directory for VASP input files."""
    base_path = Path(base_dir)
    print(f"Processing directory: {base_path}")

    poscar_paths = list(base_path.rglob("**/POSCAR"))
    failed_dirs = []

    for p in poscar_paths:
        print(p)
        vasp_inputs = VaspInputFiles(p)
        vasp_inputs.save_potcar()

        # Generate INCAR with custom params
        vasp_inputs.save_incar(
            custom_incar_params=incar_dic,
            rewrite=False,
        )

        # Generate KPOINTS
        success = vasp_inputs.save_kpoints(
            min_distance=kpoints_min_distance,
            include_gamma=kpoints_include_gamma,
        )

        if not success:
            failed_dirs.append(str(p))

    return failed_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Generate VASP input files for multiple directories"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories to process (supports glob patterns with *)",
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

    # Setup logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"vaspinput.cli.{timestamp}.log"
    setup_logging(log_file=log_file)

    LOGGER.info(f"Starting VASP input generation for {len(existing_dirs)} directories")

    # Process all directories
    all_failed_dirs = []
    for directory in existing_dirs:
        failed_dirs = process_directory(directory)
        all_failed_dirs.extend(failed_dirs)

    # Log final results
    if all_failed_dirs:
        LOGGER.info(
            f"Failed to generate KPOINTS for {len(all_failed_dirs)} directories"
        )
        for d in all_failed_dirs:
            LOGGER.info(f"Failed: {d}")
    else:
        LOGGER.info("All KPOINTS generation succeeded")

    print(f"Processing complete. Check {log_file} for details.")


if __name__ == "__main__":
    main()
