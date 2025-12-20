from pathlib import Path

from ase.build import bulk
from mace.calculators import mace_mp
from pymatgen.core import Structure

# model_path = Path.home() / "uvpy/models/MACE-matpes-pbe-omat-ft.model"
model_path = Path.home() / "uvpy/models/mace-mpa-0-medium.model"

calc = mace_mp(
    model=model_path,
    dispersion=False,
    default_dtype="float64",
    device="cpu",
)

# atoms = bulk("Cu", "fcc", a=3.58, cubic=True)

structure = Structure.from_str(
    open("Al4/POSCAR", "r", encoding="utf-8").read(), fmt="poscar"
)
atoms = structure.to_ase_atoms()

atoms.calc = calc
print(atoms.get_potential_energy())

print("Forces")
print(atoms.get_forces())
print("Stresses:")
print(atoms.get_stress(voigt=True))
