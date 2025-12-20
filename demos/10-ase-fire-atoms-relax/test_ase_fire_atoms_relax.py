from typing import Optional

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE


def opt_with_symmetry(
    atoms_in: Atoms,
    calculator: Calculator,  # pyright: ignore
    fix_symmetry: bool = False,
    hydrostatic_strain: bool = False,
    fmax: float = 0.2,
    logfile: Optional[str] = None,
    scalar_pressure=0.0 * units.GPa,
) -> dict:
    """
    Optimize an ASE Atoms object with optional symmetry constraints and hydrostatic strain.

    Parameters:
        atoms_in (Atoms): Input Atoms object to optimize.
        calculator: ASE-compatible calculator for energy and force calculations.
        fix_symmetry (bool): Whether to fix symmetry during optimization.
        hydrostatic_strain (bool): Whether to apply hydrostatic strain.
        fmax (float): Maximum force tolerance for convergence (eV/Å).
        logfile (str): Path to the optimizer log file (None to disable logging).

    Returns:
        dict: A dictionary with optimized Atoms object, final energy, and cell differences.
    """
    atoms = atoms_in.copy()
    if not hasattr(calculator, "get_potential_energy"):
        raise ValueError("Invalid calculator: Ensure it is ASE-compatible.")
    atoms.calc = calculator

    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])

    # ecf = ExpCellFilter(atoms, hydrostatic_strain=hydrostatic_strain)
    dyn = FIRE(
        FrechetCellFilter(atoms, scalar_pressure=scalar_pressure),  # pyright: ignore
        logfile="opt.log",
        trajectory="rtraj.traj",
    )
    dyn.run(fmax=0.02)

    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100
    final_energy = atoms.get_potential_energy()

    if logfile is None:
        print("Optimized Cell         :", atoms.cell.cellpar())
        print("Optimized Cell diff (%):", cell_diff)
        print("Scaled positions       :\n", atoms.get_scaled_positions())
        print(f"Epot after opt: {final_energy} eV")

    return {
        "atoms": atoms,
        "final_energy": final_energy,
        "cell_diff": cell_diff,
    }


if __name__ == "__main__":
    from StructFlow.calculators import Mace_mpa_0
    mace_calc = 
    from ase.io import read

    poscar_path = "monoclinic-check/POSCAR-FeRhSi2-bilbao"
    atoms = read(poscar_path, format="vasp")

    scalar_pressure = 0.0 * units.GPa
    atoms_relaxed_dict = opt_with_symmetry(
        atoms,  # pyright: ignore
        calculator=mace_calc,
        fix_symmetry=True,
        hydrostatic_strain=False,  # Note: this preseves cell volulme
    )

    print(f"\n{ atoms_relaxed_dict }\n")
    atoms_relaxed_dict["atoms"].write(  # pyright: ignore
        "POSCAR_after_relax", append=False, format="vasp", direct=True
    )
