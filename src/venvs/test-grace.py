# uv pip install tensorpotential
# git clone https://github.com/ICAMS/grace-tensorpotential.git
# cd grace-tensorpotential
# uv pip install .
# "GRACE-2L-OAM-L": "GRACE-2L-OMAT-large-ft-AM",
# "GRACE-2L-OMAT-L": "GRACE-2L-OMAT-large-ft-E",
# "GRACE-2L-OAM-M": "GRACE-2L-OMAT-medium-ft-AM",
# "GRACE-2L-OMAT-M": "GRACE-2L-OMAT-medium-ft-E",

from ase.build import bulk
from tensorpotential.calculator import grace_fm

# No device argument needed - uses GPU automatically if available
calc = grace_fm("GRACE-2L-OMAT-large-ft-AM")


print("=" * 60)
print("Grace Model Test")
print("=" * 60)

# Test on bulk Cu
atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms.calc = calc

# Get energy
energy = atoms.get_potential_energy()
print(f"\n`Cu` energy: {energy:.4f} eV")
print(f"Energy per atom: {energy/len(atoms):.4f} eV/atom")

# Get forces
forces = atoms.get_forces()
print(f"Max force: {abs(forces).max():.4f} eV/Å")

# Test energy ordering
print("\n" + "=" * 60)
print("Energy Ordering Test")
print("=" * 60)

atoms1 = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms1.calc = calc
e1 = atoms1.get_potential_energy()

atoms2 = bulk("Cu", "fcc", a=3.50, cubic=True)
atoms2.calc = calc
e2 = atoms2.get_potential_energy()

print(f"Perfect Cu (a=3.58): {e1:.4f} eV")
print(f"Compressed Cu (a=3.50): {e2:.4f} eV")
print(f"Energy difference: {e2-e1:.4f} eV (should be > 0)")

print("\n" + "=" * 60)
print("Comparison with SevenNet (from your test):")
print("  SevenNet perfect: -16.3648 eV")
print(f"  Grace perfect:   {e1:.4f} eV")
print(f"  Difference:       {abs(e1 - (-16.3648)):.4f} eV")
print("=" * 60)
