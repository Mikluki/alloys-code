import logging
import shutil
from pathlib import Path

import kpoints_generator as kpg
from kpoints_generator.logs import LOGGER

LOGGER.set_level(logging.INFO)


def save_kpoints_mueller(
    kpoint_name: str,
    kpoints_dir: str | Path,
    min_distance: float = 35.0,
    include_gamma: str = "auto",
    precalc_params: dict | None = None,
    save_precalc: bool = True,
) -> None:
    """
    Generate and save a KPOINTS file for a given structure using kpoints_generator.
    Args:
        kpoint_name: Output filename (without extension)
        kpoints_dir: Directory to save the KPOINTS file
        min_distance: Minimum distance parameter for k-point generation (default: 35.0)
        include_gamma: Whether to include gamma point, options: TRUE, FALSE, AUTO
        precalc_params: Additional parameters for the PRECALC file (default: None)
        save_precalc: Whether to save the PRECALC file (default: True)
    Returns:
        Path to the generated KPOINTS file
    """
    kpoints_dir = Path(kpoints_dir)
    # Create the output file path
    output_file = kpoints_dir / kpoint_name
    # Prepare precalc parameters
    params = {} if precalc_params is None else precalc_params.copy()
    # Add include_gamma parameter if not already in precalc_params
    if "INCLUDEGAMMA" not in params:
        params["INCLUDEGAMMA"] = include_gamma.upper()
    # Generate k-points using kpoints_generator
    _ = kpg.generate_kpoints(
        mindistance=min_distance,
        vasp_directory=str(kpoints_dir),
        precalc_params=params,
        output_file=str(output_file.name),
        save_precalc=save_precalc,
    )


def create_kpoint_convergence_dirs(
    source_dirs: list[str],
    min_distance_grid: list,
    include_gamma: str = "auto",
    precalc_params: dict = {},
    save_precalc: bool = True,
) -> None:
    """
    Create directories for k-point convergence testing.

    Args:
        source_dirs: List of source directories to clone
        min_distance_grid: List of min_distance values to test
        include_gamma: Whether to include gamma point for KPOINTS generation
        precalc_params: Additional parameters for PRECALC file
        save_precalc: Whether to save PRECALC file
    """

    print(f"Creating k-point convergence directories...")
    print(f"Source directories: {source_dirs}")
    print(f"Min distance grid: {min_distance_grid}")

    for source_dir in source_dirs:
        source_path = Path(source_dir)

        # Check if source directory exists
        if not source_path.exists():
            print(f"Warning: Source directory {source_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing source directory: {source_dir}")

        for min_distance in min_distance_grid:
            # Create new directory name
            new_dir_name = f"{source_dir}_md_{min_distance:.1f}_"
            new_dir_path = Path(new_dir_name)

            print(f"  Creating {new_dir_name}...")

            # Copy source directory to new directory
            if new_dir_path.exists():
                print(f"    Directory {new_dir_name} already exists, removing...")
                shutil.rmtree(new_dir_path)

            shutil.copytree(source_path, new_dir_path)

            # Generate new KPOINTS file
            print(f"    Generating KPOINTS with min_distance={min_distance}...")
            save_kpoints_mueller(
                kpoint_name="KPOINTS",
                kpoints_dir=new_dir_path,
                min_distance=min_distance,
                include_gamma=include_gamma,
                precalc_params=precalc_params,
                save_precalc=save_precalc,
            )

    print(f"\nCompleted creating all convergence directories!")


if __name__ == "__main__":
    # Run with default parameters
    source_dirs = ["r0", "r1", "r2"]
    min_distance_grid = [15, 20, 25, 28, 30, 32, 34, 35, 37, 40]
    create_kpoint_convergence_dirs(source_dirs, min_distance_grid)
