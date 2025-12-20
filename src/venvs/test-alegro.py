from ase.build import bulk
from nequip.ase import NequIPCalculator

model_path = "/home/m.lukianov/uvpy/models/mir-group__Allegro-OAM-L__0.1.nequip.pt2"

# Load Allegro GPU-compiled model
calc = NequIPCalculator.from_compiled_model(
    compile_path=model_path,
    device="cuda",
)

print("=" * 60)
print("Allegro-OAM-L GPU Model Test")
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
print("Comparison with NequIP and SevenNet:")
print("  SevenNet:  -16.3648 eV")
print("  NequIP:    -16.3472 eV")
print(f"  Allegro:   {e1:.4f} eV")
print("=" * 60)
