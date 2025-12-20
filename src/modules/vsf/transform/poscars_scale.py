import argparse
import logging
import sys
from pathlib import Path
from typing import List

from pymatgen.core import Structure

LOGGER = logging.getLogger(__name__)


def scale_poscar_lattice(
    poscar_files: List[Path] | List[str],
    scale_factor: float = 3.0,
    output_dir: str | Path = "scaled_poscars",
    prefix: str = "scaled_",
) -> int:
    """
    Scale the lattice dimensions of POSCAR files by a given factor.

    Args:
        poscar_files (List[Union[str, Path]]): List of POSCAR file paths to scale
        scale_factor (float): Factor by which to scale the lattice (default: 3.0)
        output_dir (Union[str, Path]): Directory to save scaled POSCAR files (default: "scaled_poscars")
        prefix (str): Prefix to add to output filenames (default: "scaled_")

    Returns:
        int: Number of successfully scaled files
    """
    # Convert output_dir to Path object if it's a string
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    success_count = 0

    # Process each POSCAR file
    for poscar_file in poscar_files:
        try:
            # Convert to Path object if it's a string
            poscar_path = Path(poscar_file)

            # Check if file exists
            if not poscar_path.exists():
                print(f"Warning: {poscar_path} not found, skipping...")
                continue

            # Load the structure from POSCAR
            with open(poscar_path, "r", encoding="utf-8") as f:
                content = f.read()
            structure = Structure.from_str(content, fmt="poscar")

            # Get the current lattice matrix
            lattice_matrix = structure.lattice.matrix

            # Scale the lattice by multiplying each vector by the scale factor
            new_lattice_matrix = lattice_matrix * scale_factor

            # Create a new structure with the scaled lattice
            scaled_structure = Structure(
                lattice=new_lattice_matrix,
                species=structure.species,
                coords=structure.frac_coords,  # Keep fractional coordinates the same
                coords_are_cartesian=False,
            )

            # Create output filename
            output_file = output_path / f"{prefix}{poscar_path.name}"

            # Write the scaled structure to a new POSCAR file
            scaled_structure.to(filename=str(output_file), fmt="poscar")

            print(f"Successfully scaled {poscar_path} -> {output_file}")
            success_count += 1

        except Exception as e:
            print(f"Error processing {poscar_file}: {str(e)}")

    print(
        f"\nScaling complete! {success_count} files scaled. Scaled POSCAR files are in the '{output_path}' directory."
    )
    return success_count


def main():
    """Main function to parse command line arguments and run the scaling function."""
    parser = argparse.ArgumentParser(
        description="Scale POSCAR lattice dimensions by a specified factor."
    )
    parser.add_argument(
        "files", nargs="+", help="POSCAR files to scale (accepts wildcards)"
    )
    parser.add_argument(
        "-f", "--factor", type=float, default=3.0, help="Scale factor (default: 3.0)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="scaled_poscars",
        help="Output directory (default: scaled_poscars)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="scaled_",
        help="Prefix for output files (default: scaled_)",
    )

    args = parser.parse_args()

    # Convert string paths to Path objects
    poscar_paths = [Path(f) for f in args.files]

    return scale_poscar_lattice(
        poscar_paths,
        scale_factor=args.factor,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


def print_readme() -> None:
    """Print usage instructions for scaling POSCAR files."""
    readme_text = """
POSCAR scaling usage:

    # Command line usage:
    python poscars_scale.py *.poscar -f 3.0 -o scaled_structures -p 3x_

    # As a module:
    from vsf.transform.poscars_scale import scale_poscar_lattice
    from pathlib import Path

    # List of POSCAR files as Path objects
    poscar_files = [
        Path("structures/B11_P4nmm_129_TiCd_mp-30500_.poscar"),
        Path("structures/B19_Pmma_51_CdAu_mp-1404.poscar"),
        # ...other files...
    ]

    # Scale with custom parameters
    scale_poscar_lattice(
        poscar_files,
        scale_factor=3.0,
        output_dir=Path("my_scaled_structures"),
        prefix="3x_"
    )
"""
    print(readme_text.strip())


if __name__ == "__main__":
    sys.exit(main())
