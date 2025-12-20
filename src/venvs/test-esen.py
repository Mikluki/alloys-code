# WARN: This uses FairChem v1 (fairchem-core==1.10)
#
# Meta released FairChem v2 with complete API restructuring that broke ALL v1 checkpoints.
# The new v2 API uses FAIRChemCalculator + MLIPPredictUnit, but v1 checkpoints like eSCN
# from OMAT24 (facebook/OMAT24 on HuggingFace) are incompatible with v2.
#
# Migration guide basically says "use v1 for old checkpoints" - no conversion tools provided.
#
# If this breaks in future: you'll need fairchem-core==1.10 specifically for v1 checkpoints.
# For newer models, use v2 API with FAIRChemCalculator.from_model_checkpoint().
#
# Last verified working: fairchem-core==1.10, September 2025
# uv pip install fairchem-core==1.10

from pathlib import Path

from ase.build import bulk
from fairchem.core import OCPCalculator

# Load model
checkpoint_path = Path.home() / "uvpy/models/esen_30m_oam.pt"
calc = OCPCalculator(checkpoint_path=checkpoint_path)

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
print(f"   eSEN:     {e1:.4f} eV")
print("=" * 60)
