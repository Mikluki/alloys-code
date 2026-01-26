"""
Minimal position extraction diagnostic: validate we can read all frames reliably.
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


def check_positions_all_frames(xml_path):
    """Extract and validate positions for all frames in vasprun.xml."""
    xml_path = Path(xml_path)
    LOGGER.info(f"Reading: {xml_path}")

    # Read all frames
    images = read(xml_path, index=":")
    nframes = len(images)
    natoms = len(images[0])

    LOGGER.info(f"Frames: {nframes}, Atoms per frame: {natoms}")

    # Extract positions for all frames
    positions = np.array([atoms.positions.copy() for atoms in images])
    LOGGER.info(f"Positions shape: {positions.shape} (nframes × natoms × 3)")

    # Check consistency
    shapes_ok = all(len(atoms.positions) == natoms for atoms in images)
    LOGGER.info(f"Shape consistency: {shapes_ok}")

    if not shapes_ok:
        LOGGER.warning("Atoms per frame inconsistent! Aborting.")
        return

    # Spot-check: pick 5 atoms, look at their trajectory
    spot_atoms = [0, natoms // 4, natoms // 2, 3 * natoms // 4, natoms - 1]
    LOGGER.info(f"\nSpot-check atoms {spot_atoms}:")

    for atom_idx in spot_atoms:
        traj = positions[:, atom_idx, :]  # (nframes, 3)
        disp = np.linalg.norm(np.diff(traj, axis=0), axis=1)  # frame-to-frame distance

        LOGGER.info(
            f"  Atom {atom_idx:3d}: "
            f"mean Δr={np.mean(disp):.6f} Å, "
            f"max Δr={np.max(disp):.6f} Å, "
            f"std Δr={np.std(disp):.6f} Å"
        )

    # Overall displacement stats
    LOGGER.info("\nGlobal displacement stats:")
    all_displacements = []
    for i in range(nframes - 1):
        dr = np.linalg.norm(positions[i + 1] - positions[i], axis=1)
        all_displacements.extend(dr)

    all_displacements = np.array(all_displacements)
    LOGGER.info(f"  Mean Δr: {np.mean(all_displacements):.6f} Å")
    LOGGER.info(f"  Max Δr:  {np.max(all_displacements):.6f} Å")
    LOGGER.info(f"  P95 Δr:  {np.percentile(all_displacements, 95):.6f} Å")

    # Try to extract POTIM
    LOGGER.info("\nLooking for POTIM (timestep)...")
    potim = None

    # Try via atoms.info
    if hasattr(images[0], "info"):
        potim = images[0].info.get("potim")
        if potim is not None:
            LOGGER.info(f"  Found in atoms.info: POTIM = {potim} fs")

    if potim is None:
        LOGGER.warning("  POTIM not found in atoms.info")
        LOGGER.info("  (Will need to parse INCAR separately or set manually)")

    LOGGER.info("Done!")


if __name__ == "__main__":
    # Edit this path to point to your test vasprun.xml
    xml_path = Path("data/AlCuNi_L1915_1400/vasprun.xml")
    check_positions_all_frames(xml_path)
