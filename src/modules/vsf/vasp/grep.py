import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase.io import read


def extract_outcar_energy_per_atom(outcar_path: Path, num_sites: float) -> float:
    """
    Extract energy without entropy from OUTCAR file.

    Returns
    -------
    float
        Energy per atom in eV

    Raises
    ------
    FileNotFoundError
        If OUTCAR file doesn't exist
    ValueError
        If energy without entropy cannot be extracted from OUTCAR
    """
    if not outcar_path.exists():
        raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")

    with open(outcar_path, "r") as f:
        content = f.read()

    # Find all energy without entropy entries (take the last one for final energy)
    energy_matches = re.findall(r"energy without entropy\s*=\s*([-\d\.]+)", content)

    if not energy_matches:
        raise ValueError(f"No 'energy without entropy' found in OUTCAR: {outcar_path}")

    # Take the last energy value and convert to energy per atom
    total_energy = float(energy_matches[-1])
    return total_energy / num_sites


def extract_stress_voigt(outcar_path: str | Path) -> npt.NDArray[np.float64]:
    """
    Extract stress tensor from VASP OUTCAR using ASE and return it in ASE Voigt convention.

    Parameters
    ----------
    outcar_path : str or Path
        Path to VASP OUTCAR file.

    Returns
    -------
    np.ndarray
        Stress tensor in Voigt notation (xx, yy, zz, yz, xz, xy) in eV/Å³,
        matching `atoms.get_stress(voigt=True)`.

    Raises
    ------
    ValueError
        If OUTCAR cannot be read or does not contain stress information.
    """
    outcar_path = Path(outcar_path)

    try:
        # ASE will auto-detect VASP OUTCAR and read the last ionic step
        atoms = read(outcar_path, index=-1)
    except Exception as exc:
        raise ValueError(f"Failed to read VASP OUTCAR at {outcar_path}") from exc

    try:
        stress = atoms.get_stress(  # pyright: ignore
            voigt=True
        )  # already ASE convention, eV/Å³
    except Exception as exc:
        raise ValueError(
            f"No stress information available in OUTCAR at {outcar_path}"
        ) from exc

    return np.asarray(stress, dtype=np.float64)
