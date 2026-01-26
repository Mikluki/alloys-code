"""
Reconstruct velocities from positions via central differences + PBC wrapping.
Test on single vasprun.xml, export to .traj with velocities attached.
"""

import logging
from pathlib import Path

import numpy as np
from ase.io import read, write

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Constants
POTIM = 2.0  # fs per ionic step


def compute_velocities_pbc(positions, cells, potim):
    """
    Reconstruct velocities from positions using central differences + PBC wrapping.

    Args:
        positions: (nframes, natoms, 3) array
        cells: (nframes, 3, 3) array
        potim: timestep in fs

    Returns:
        velocities: (nframes, natoms, 3) array (Å/fs)
    """
    nframes, natoms, _ = positions.shape
    velocities = np.zeros_like(positions)

    for t in range(nframes):
        if t == 0:
            # One-sided forward: v[0] = (r[1] - r[0]) / Δt
            dr_frac = _displace_pbc(positions[0], positions[1], cells[0])
            velocities[0] = dr_frac / potim

        elif t == nframes - 1:
            # One-sided backward: v[N-1] = (r[N-1] - r[N-2]) / Δt
            dr_frac = _displace_pbc(positions[t - 1], positions[t], cells[t])
            velocities[t] = dr_frac / potim

        else:
            # Central difference: v[t] = (r[t+1] - r[t-1]) / (2*Δt)
            dr_frac = _displace_pbc(positions[t - 1], positions[t + 1], cells[t])
            velocities[t] = dr_frac / (2 * potim)

    return velocities


def _displace_pbc(pos_from, pos_to, cell):
    """Compute displacement under PBC via fractional wrapping."""
    frac_from = np.linalg.solve(cell.T, pos_from.T).T
    frac_to = np.linalg.solve(cell.T, pos_to.T).T

    df = frac_to - frac_from
    df_wrapped = df - np.round(df)

    dr_cart = df_wrapped @ cell
    return dr_cart


def reconstruct_velocities_and_export(xml_path, output_traj, potim=POTIM):
    """Read vasprun.xml, reconstruct velocities, attach to frames, export .traj."""
    xml_path = Path(xml_path)
    output_traj = Path(output_traj)

    LOGGER.info(f"Reading: {xml_path}")
    images = read(xml_path, index=":")
    nframes = len(images)
    natoms = len(images[0])

    LOGGER.info(f"Frames: {nframes}, Atoms: {natoms}")

    # Extract positions and cells
    positions = np.array([atoms.positions.copy() for atoms in images])
    cells = np.array([atoms.cell[:] for atoms in images])

    LOGGER.info(f"Computing velocities (POTIM={potim} fs)...")
    velocities = compute_velocities_pbc(positions, cells, potim)

    # Attach velocities to frames
    for i, atoms in enumerate(images):
        atoms.set_velocities(velocities[i])

    # Validate: compute kinetic energy
    LOGGER.info("\nKinetic energy validation:")
    for i in [0, nframes // 2, nframes - 1]:
        vel = images[i].get_velocities()
        ke = np.mean(np.sum(vel**2, axis=1))
        LOGGER.info(f"  Frame {i:3d}: KE_proxy = {ke:.6f}")

    # Export to .traj
    output_traj.parent.mkdir(parents=True, exist_ok=True)
    write(output_traj, images)
    LOGGER.info(f"\nWrote: {output_traj}")

    # Summary stats
    all_vel = np.concatenate([images[i].get_velocities() for i in range(nframes)])
    LOGGER.info("\nVelocity stats (all frames, all atoms):")
    LOGGER.info(f"  Mean |v|: {np.mean(np.linalg.norm(all_vel, axis=1)):.6f} Å/fs")
    LOGGER.info(f"  Max |v|:  {np.max(np.linalg.norm(all_vel, axis=1)):.6f} Å/fs")
    LOGGER.info(
        f"  P95 |v|:  {np.percentile(np.linalg.norm(all_vel, axis=1), 95):.6f} Å/fs"
    )

    LOGGER.info("Done!")


if __name__ == "__main__":
    xml_path = Path("data/AlCuNi_L1915_1400/vasprun.xml")
    output_traj = Path("results/test_trajectory.traj")

    reconstruct_velocities_and_export(xml_path, output_traj, potim=POTIM)
