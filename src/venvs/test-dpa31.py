# To get the best performance, it is recommended to adjust
# the number of threads by setting the environment variables
# OMP_NUM_THREADS, DP_INTRA_OP_PARALLELISM_THREADS, and
# DP_INTER_OP_PARALLELISM_THREADS. See
# https://deepmd.rtfd.io/parallelism/ for more information.
# pip install git+https://github.com/deepmodeling/deepmd-kit@devel

from ase.build import bulk
from deepmd.calculator import DP

print("=" * 60)
print("DPA-3.1-3M-FT GPU Model Test")
print("=" * 60)

# Load DPA model
# Note: You need to freeze the model first with the appropriate model-branch
# Example: dp --pt freeze -c DPA-3.1-3M.pt -o frozen_model_alloy.pth --model-branch Domains_Alloy
calc = DP("frozen_model.pth")

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
print("Comparison with other models:")
print("  SevenNet:  -16.3648 eV")
print("  NequIP:    -16.3472 eV")
print("  Allegro:   [your result]")
print(f"  DPA-3:     {e1:.4f} eV")
print("=" * 60)
