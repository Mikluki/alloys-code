## Stress Metric Function Signature

```python
from ase import Atoms

def compute_pressure(atoms: Atoms) -> float:
    """
    Compute hydrostatic pressure from the stress tensor of an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with an attached calculator that can provide stress.

    Returns
    -------
    float
        Hydrostatic pressure in eV/Å³.
        Defined as P = -1/3 * Tr(σ).
        Positive => compression, negative => tension.
    """
    # ASE stress: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy] in eV/Å³
    stress = atoms.get_stress(voigt=True)
    pressure = -(stress[0] + stress[1] + stress[2]) / 3.0
    return pressure
```

### API Support prompt

I plan to extend py api with a new metric. Context:

- I have a `StructureRecord` that I use to store info about an individual alloy in odrder to compare calculation results from different GNNs

```python
class StructureRecord:
    """Simplified container for structure data with energy analysis capabilities."""

    def __init__(self, structure_dir: Path):
        """
        Initialize from a structure directory containing POSCAR and OUTCAR files.

        Parameters
        ----------
        structure_dir : Path
            Directory containing POSCAR and OUTCAR files
        """
        self.structure_dir = Path(structure_dir)
        self.name = self.structure_dir.name

        # Core paths - predictable structure
        self.poscar_path = self.structure_dir / "POSCAR"
        self.outcar_path = self.structure_dir / "OUTCAR"
        self.json_path = self.structure_dir / f"{self.name}.json"

        # Initialize energy analyzers
        self.potential_energy = PotentialEnergyAnalyzer()
        self.formation_energy = FormationEnergyAnalyzer()

        # Cache for lazy-loaded structure data
        self._structure = None
        self._composition = None
        self._atoms = None
        self._formula = None
        self._reduced_formula = None
        self._num_sites = None
```

Currently I plan to extend `EnergyAnalyzer` and `EnergyResult` to support the following data:

```python
from ase import Atoms

def compute_pressure(atoms: Atoms) -> float:
    """
    Compute hydrostatic pressure from the stress tensor of an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with an attached calculator that can provide stress.

    Returns
    -------
    float
        Hydrostatic pressure in eV/Å³.
        Defined as P = -1/3 * Tr(σ).
        Positive => compression, negative => tension.
    """
    # ASE stress: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy] in eV/Å³
    stress = atoms.get_stress(voigt=True)
    pressure = -(stress[0] + stress[1] + stress[2]) / 3.0
    return pressure
```

and also store stress array
