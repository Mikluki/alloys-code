"""
Minimal velocity diagnostic: probe how ASE reads velocities from vasprun.xml.
"""

import logging
from pathlib import Path

import numpy as np
from ase.io import read

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def check_velocity_read(xml_path):
    """Probe velocity storage in ASE atoms object from VASP XML."""
    xml_path = Path(xml_path)
    LOGGER.info(f"Reading: {xml_path}")

    # Read first frame
    atoms = read(xml_path, index=0)
    LOGGER.info(f"Atoms: {len(atoms)} atoms, cell shape {atoms.cell.shape}")

    # Probe what ASE stored
    LOGGER.info(f"arrays.keys(): {list(atoms.arrays.keys())}")

    # Method 1: High-level API
    vel_api = atoms.get_velocities()
    LOGGER.info(
        f"get_velocities(): shape={vel_api.shape if vel_api is not None else None}, dtype={vel_api.dtype if vel_api is not None else None}"
    )
    if vel_api is not None:
        LOGGER.info(
            f"  → stats: mean={np.mean(np.abs(vel_api)):.6f}, max={np.max(np.abs(vel_api)):.6f}"
        )
    else:
        LOGGER.info("  → None")

    # Method 2: Direct dict access (velocities)
    vel_direct = atoms.arrays.get("velocities")
    LOGGER.info(
        f"arrays['velocities']: shape={vel_direct.shape if vel_direct is not None else None}, dtype={vel_direct.dtype if vel_direct is not None else None}"
    )
    if vel_direct is not None:
        LOGGER.info(
            f"  → stats: mean={np.mean(np.abs(vel_direct)):.6f}, max={np.max(np.abs(vel_direct)):.6f}"
        )
    else:
        LOGGER.info("  → None")

    # Method 3: Momenta (alternative storage)
    mom = atoms.arrays.get("momenta")
    LOGGER.info(
        f"arrays['momenta']: shape={mom.shape if mom is not None else None}, dtype={mom.dtype if mom is not None else None}"
    )
    if mom is not None:
        LOGGER.info(
            f"  → stats: mean={np.mean(np.abs(mom)):.6f}, max={np.max(np.abs(mom)):.6f}"
        )
    else:
        LOGGER.info("  → None")

    # Sample first atom velocities
    if vel_api is not None and len(vel_api) > 0:
        LOGGER.info(f"\nFirst atom velocities (method 1): {vel_api[0]}")
    if vel_direct is not None and len(vel_direct) > 0:
        LOGGER.info(f"First atom velocities (method 2): {vel_direct[0]}")
    if mom is not None and len(mom) > 0:
        LOGGER.info(f"First atom momenta: {mom[0]}")

    LOGGER.info("Done!")


if __name__ == "__main__":
    # Edit this path to point to your test vasprun.xml
    xml_path = Path("data/AlCuNi_L1915_1400/vasprun.xml")
    check_velocity_read(xml_path)
