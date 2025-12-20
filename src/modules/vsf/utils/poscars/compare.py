from pathlib import Path
from typing import Any, Dict

import numpy as np
from pymatgen.core import Structure


def compare_poscars(
    poscar1_path: Path | str,
    poscar2_path: Path | str,
    cell_tol_length: float = 0.001,  # Å
    cell_tol_angle: float = 0.1,  # degrees
    position_tol: float = 0.01,  # Å
) -> Dict[str, Any]:
    """
    Compare two POSCAR files to detect significant structural changes.

    Parameters:
    -----------
    poscar1_path : Path | str
        Path to first POSCAR file (initial structure)
    poscar2_path : Path | str
        Path to second POSCAR file (final structure)
    cell_tol_length : float, default=0.001
        Tolerance for cell length changes in Å
    cell_tol_angle : float, default=0.1
        Tolerance for cell angle changes in degrees
    position_tol : float, default=0.01
        Tolerance for atomic position changes in Å

    Returns:
    --------
    Dict[str, any]
        Dictionary containing:
        - "has_significant_change": bool - True if any significant changes detected
        - "cell_diff": Dict[str, float] - Absolute differences in cell parameters
        - "max_atom_displacement": float - Maximum atomic displacement in Å
        - "cell_changed": bool - True if cell parameters changed significantly
        - "positions_changed": bool - True if atomic positions changed significantly
        - "summary": str - Human readable summary
    """
    # Load structures
    struct1 = Structure.from_file(str(poscar1_path))
    struct2 = Structure.from_file(str(poscar2_path))

    # Check if structures have same composition
    if struct1.composition != struct2.composition:
        raise ValueError("POSCARs have different compositions - cannot compare")

    # Calculate cell parameter differences
    cell1 = struct1.lattice.parameters  # [a, b, c, alpha, beta, gamma]
    cell2 = struct2.lattice.parameters

    cell_diff = {
        "a": abs(cell2[0] - cell1[0]),  # Å
        "b": abs(cell2[1] - cell1[1]),  # Å
        "c": abs(cell2[2] - cell1[2]),  # Å
        "alpha": abs(cell2[3] - cell1[3]),  # degrees
        "beta": abs(cell2[4] - cell1[4]),  # degrees
        "gamma": abs(cell2[5] - cell1[5]),  # degrees
    }

    # Check cell parameter changes
    length_changes = [cell_diff["a"], cell_diff["b"], cell_diff["c"]]
    angle_changes = [cell_diff["alpha"], cell_diff["beta"], cell_diff["gamma"]]

    cell_changed = any(change > cell_tol_length for change in length_changes) or any(
        change > cell_tol_angle for change in angle_changes
    )

    # Calculate atomic position changes in Cartesian coordinates
    pos1 = struct1.cart_coords  # Cartesian positions
    pos2 = struct2.cart_coords

    # Calculate displacements for each atom
    displacements = np.linalg.norm(pos2 - pos1, axis=1)
    max_atom_displacement = np.max(displacements)

    # Check if positions changed significantly
    positions_changed = max_atom_displacement > position_tol

    # Overall assessment
    has_significant_change = cell_changed or positions_changed

    # Create summary
    if not has_significant_change:
        summary = "No significant changes detected"
    else:
        changes = []
        if cell_changed:
            changes.append(
                f"cell (max Δ: {max(length_changes):.3f} Å, {max(angle_changes):.2f}°)"
            )
        if positions_changed:
            changes.append(f"positions (max Δ: {max_atom_displacement:.3f} Å)")
        summary = f"Significant changes in: {', '.join(changes)}"

    return {
        "has_significant_change": has_significant_change,
        "cell_diff": cell_diff,
        "max_atom_displacement": max_atom_displacement,
        "cell_changed": cell_changed,
        "positions_changed": positions_changed,
        "summary": summary,
    }


def check_relaxation_effect(
    input_poscar: Path | str, output_poscar: Path | str, **tolerance_kwargs
) -> bool:
    """
    Check if a relaxation had any significant effect.

    Parameters:
    -----------
    input_poscar : Path | str
        Path to input POSCAR (before relaxation)
    output_poscar : Path | str
        Path to output POSCAR (after relaxation)
    **tolerance_kwargs
        Tolerance parameters passed to compare_poscars

    Returns:
    --------
    bool
        True if relaxation had significant effect, False if no change
    """
    result = compare_poscars(input_poscar, output_poscar, **tolerance_kwargs)
    return result["has_significant_change"]


# Example usage functions
def find_unchanged_relaxations(
    base_dir: Path | str, pattern: str = "*_stage_*", **tolerance_kwargs
) -> list[Path]:
    """
    Find all relaxation directories where no significant changes occurred.

    Parameters:
    -----------
    base_dir : Path | str
        Base directory to search
    pattern : str
        Glob pattern for relaxation directories
    **tolerance_kwargs
        Tolerance parameters for comparison

    Returns:
    --------
    list[Path]
        List of directories with no significant changes
    """
    base_dir = Path(base_dir)
    unchanged_dirs = []

    for relax_dir in base_dir.glob(pattern):
        if not relax_dir.is_dir():
            continue

        output_poscar = relax_dir / "POSCAR"
        if not output_poscar.exists():
            continue

        # Find the input POSCAR (could be original or from previous stage)
        # This would need to be determined based on your specific workflow
        # For now, assume there's a backup file indicating the input
        input_poscar = None
        for backup in relax_dir.glob("POSCAR.prev"):
            input_poscar = backup
            break

        if input_poscar and input_poscar.exists():
            if not check_relaxation_effect(
                input_poscar, output_poscar, **tolerance_kwargs
            ):
                unchanged_dirs.append(relax_dir)

    return unchanged_dirs


# Test function
def test_poscar_comparison():
    """Test the POSCAR comparison functionality."""
    import shutil

    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    # Create test structures
    atoms1 = Atoms(
        symbols=["C", "C"],
        positions=[[0.0, 0.0, 0.0], [1.4, 1.4, 1.4]],
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )

    # Slightly modified structure
    atoms2 = Atoms(
        symbols=["C", "C"],
        positions=[[0.0001, 0.0001, 0.0001], [1.4001, 1.4001, 1.4001]],  # Tiny change
        cell=[3.0001, 3.0001, 3.0001],  # Tiny cell change
        pbc=True,
    )

    # Significantly modified structure
    atoms3 = Atoms(
        symbols=["C", "C"],
        positions=[[0.1, 0.1, 0.1], [1.3, 1.3, 1.3]],  # Large change
        cell=[2.9, 2.9, 2.9],  # Large cell change
        pbc=True,
    )

    # Create test directory
    test_dir = Path.cwd() / "test_poscar_compare"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    try:
        # Convert to pymatgen and save
        struct1 = AseAtomsAdaptor.get_structure(atoms1)  # pyright: ignore
        struct2 = AseAtomsAdaptor.get_structure(atoms2)  # pyright: ignore
        struct3 = AseAtomsAdaptor.get_structure(atoms3)  # pyright: ignore

        poscar1 = test_dir / "POSCAR1"
        poscar2 = test_dir / "POSCAR2"
        poscar3 = test_dir / "POSCAR3"

        struct1.to(filename=str(poscar1), fmt="poscar")
        struct2.to(filename=str(poscar2), fmt="poscar")
        struct3.to(filename=str(poscar3), fmt="poscar")

        print(f"✓ Created test POSCAR files in {test_dir}")

        # Test tiny changes (should be below threshold)
        result_tiny = compare_poscars(poscar1, poscar2)
        print("\nTiny changes result:")
        print(f"  Has significant change: {result_tiny['has_significant_change']}")
        print(f"  Summary: {result_tiny['summary']}")
        print(f"  Max displacement: {result_tiny['max_atom_displacement']:.6f} Å")

        # Test large changes (should be above threshold)
        result_large = compare_poscars(poscar1, poscar3)
        print("\nLarge changes result:")
        print(f"  Has significant change: {result_large['has_significant_change']}")
        print(f"  Summary: {result_large['summary']}")
        print(f"  Max displacement: {result_large['max_atom_displacement']:.6f} Å")

        print(f"\n✓ Test files available at: {test_dir}")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_poscar_comparison()
