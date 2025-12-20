# uv pip install orb-models

import ase
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device = "cpu"  # or device="cuda"
# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float64",  # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, device=device)

atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms.calc = calc
atoms.get_potential_energy()

# Get energy
energy = atoms.get_potential_energy()
print(f"Energy: {energy:.4f} eV")
