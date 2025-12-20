from ase.build import bulk
from sevenn.calculator import SevenNetCalculator

# Load SevenNet model
calc = SevenNetCalculator(model="7net-mf-ompa", modal="mpa")

# Test on simple molecule
atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms.calc = calc

# Get energy
energy = atoms.get_potential_energy()
print(f"`Cu` energy: {energy:.4f} eV")

# Get forces
forces = atoms.get_forces()
print(f"Max force: {abs(forces).max():.4f} eV/Å")

# Perfect FCC Cu
atoms1 = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms1.calc = calc
e1 = atoms1.get_potential_energy()

# Slightly compressed Cu
atoms2 = bulk("Cu", "fcc", a=3.50, cubic=True)
atoms2.calc = calc
e2 = atoms2.get_potential_energy()

print(f"Perfect Cu: {e1:.4f} eV")
print(f"Compressed Cu: {e2:.4f} eV")
print(f"Energy difference: {e2-e1:.4f} eV (should be > 0)")
