"""
Generate supercells for liquid MD simulations from primitive POSCAR files.
"""

from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


def find_best_supercell_size(structure, target_atoms):
    """
    Find supercell dimensions that give closest to target number of atoms.

    Args:
        structure: pymatgen Structure object
        target_atoms: target number of atoms

    Returns:
        tuple: (nx, ny, nz) supercell dimensions
        int: actual number of atoms that will result
    """
    primitive_atoms = len(structure)

    # Calculate the scaling factor needed
    scale_factor = (target_atoms / primitive_atoms) ** (1 / 3)

    # Try different integer combinations around the ideal scaling
    best_diff = float("inf")
    best_dims = (1, 1, 1)
    best_atoms = primitive_atoms

    # Search in a reasonable range around the ideal scaling
    search_range = max(1, int(scale_factor))
    for nx in range(1, search_range + 3):
        for ny in range(1, search_range + 3):
            for nz in range(1, search_range + 3):
                total_atoms = nx * ny * nz * primitive_atoms
                diff = abs(total_atoms - target_atoms)

                if diff < best_diff:
                    best_diff = diff
                    best_dims = (nx, ny, nz)
                    best_atoms = total_atoms

    return best_dims, best_atoms


def create_supercell(poscar_path, target_atoms=64, suffix=None):
    """
    Create supercell from POSCAR file and save with suffix.

    Args:
        poscar_path: Path to input POSCAR file
        target_atoms: target number of atoms (default: 64)
        suffix: suffix for output file (default: f"_supercell{target_atoms}")

    Returns:
        Path: path to created supercell POSCAR
        dict: info about the supercell creation
    """
    poscar_path = Path(poscar_path)

    # Load the structure
    structure = Structure.from_file(poscar_path)

    # Find best supercell dimensions
    supercell_dims, actual_atoms = find_best_supercell_size(structure, target_atoms)

    # Create supercell
    supercell_matrix = np.diag(supercell_dims)
    supercell = structure * supercell_matrix

    # Generate output filename
    if suffix is None:
        suffix = f"_supercell{target_atoms}"

    output_dir = Path(poscar_path.parent.parent / f"{poscar_path.parent.stem}{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir, "POSCAR")

    # Save the supercell
    poscar = Poscar(supercell)  # pyright: ignore
    poscar.write_file(output_path)

    # Prepare info dictionary
    info = {
        "input_file": poscar_path,
        "output_file": output_path,
        "primitive_atoms": len(structure),
        "target_atoms": target_atoms,
        "actual_atoms": actual_atoms,
        "supercell_dims": supercell_dims,
        "composition": structure.composition.formula,
    }

    return output_path, info


def process_multiple_poscars(poscar_paths, target_atoms=64, verbose=True):
    """
    Process multiple POSCAR files to create supercells.

    Args:
        poscar_paths: list of paths to POSCAR files
        target_atoms: target number of atoms
        verbose: whether to print progress information

    Returns:
        list: list of (output_path, info) tuples
    """
    results = []

    for poscar_path in poscar_paths:
        try:
            output_path, info = create_supercell(poscar_path, target_atoms)
            results.append((output_path, info))

            if verbose:
                print(
                    f"✓ {info['composition']}: {info['primitive_atoms']} → {info['actual_atoms']} atoms "
                    f"({info['supercell_dims'][0]}×{info['supercell_dims'][1]}×{info['supercell_dims'][2]})"
                )
                print(f"  Saved: {output_path}")

        except Exception as e:
            print(e)
            print(f"✗ Failed to process {poscar_path}")

    return results


def main():
    """Example usage with your directory structure."""

    # Define your POSCAR paths
    base_dir = Path("e-hull")
    poscar_paths = [
        base_dir / "Al_mp-134_" / "POSCAR",
        base_dir / "Au4_mp-81_" / "POSCAR",
        base_dir / "Cu4_mp-30_" / "POSCAR",
        base_dir / "Na_mp-10172_" / "POSCAR",
        base_dir / "Ni4_mp-23_" / "POSCAR",
    ]

    # Create supercells targeting atoms
    print("Creating supercells targeting atoms...")
    results = process_multiple_poscars(
        poscar_paths,
        target_atoms=64,
        verbose=True,
    )

    print(f"\nProcessed {len(results)} files successfully.")


if __name__ == "__main__":
    main()
