"""
Stitch trajectory segments into continuous chains.
Reconstruct velocities, deduplicate boundaries, export .traj per chain.
"""

import json
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
TARGET_DIR = Path("data/AlCuNi_L1915_1400")
STITCH_SUMMARY_PATH = Path("results/stitch_audit/stitch_summary.json")
OUTPUT_DIR = Path("results/trajectories")
POTIM = 2.0  # fs per ionic step
DEDUP_THRESHOLD = 0.001  # Å


def compute_velocities_pbc(positions, cells, potim):
    """Reconstruct velocities from positions using central differences + PBC wrapping."""
    nframes, natoms, _ = positions.shape
    velocities = np.zeros_like(positions)

    for t in range(nframes):
        if t == 0:
            dr_frac = _displace_pbc(positions[0], positions[1], cells[0])
            velocities[0] = dr_frac / potim
        elif t == nframes - 1:
            dr_frac = _displace_pbc(positions[t - 1], positions[t], cells[t])
            velocities[t] = dr_frac / potim
        else:
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


def check_dedup_pbc(pos_last, pos_first, cell, threshold=DEDUP_THRESHOLD):
    """Check if first frame of next segment duplicates last frame of prev segment."""
    dr = _displace_pbc(pos_last, pos_first, cell)
    rms = np.sqrt(np.mean(np.sum(dr**2, axis=1)))
    return rms < threshold


def load_stitch_summary(json_path):
    """Load stitch_summary.json."""
    with open(json_path) as f:
        return json.load(f)


def identify_chains(boundaries):
    """
    Identify groups of consecutive segments connected by STITCH.

    Returns:
        chains: list of (seg_start, seg_end) tuples
    """
    if not boundaries:
        return []

    chains = []
    chain_start = 0

    for bd in boundaries:
        decision = bd["decision"]
        seg_ip1 = bd["seg_ip1"]

        if decision != "STITCH":
            # End current chain at seg_i
            seg_i = bd["seg_i"]
            chains.append((chain_start, seg_i))
            chain_start = seg_ip1

    # Final chain: from chain_start to last segment
    max_seg = max(bd["seg_ip1"] for bd in boundaries)
    chains.append((chain_start, max_seg))

    return chains


def reconstruct_segment(vasprun_path):
    """Read vasprun, reconstruct velocities, return images."""
    images = read(vasprun_path, index=":")
    positions = np.array([atoms.positions.copy() for atoms in images])
    cells = np.array([atoms.cell[:] for atoms in images])

    velocities = compute_velocities_pbc(positions, cells, POTIM)

    for i, atoms in enumerate(images):
        atoms.set_velocities(velocities[i])

    return images


def log_chain_diagnostics(images):
    """Log position and velocity diagnostics for a chain (with PBC-wrapped displacements)."""
    nframes = len(images)
    natoms = len(images[0])

    # Extract positions and cells
    positions = np.array([atoms.positions.copy() for atoms in images])
    cells = np.array([atoms.cell[:] for atoms in images])

    # Spot-check atoms (PBC-wrapped displacements)
    spot_atoms = [0, natoms // 4, natoms // 2, 3 * natoms // 4, natoms - 1]
    LOGGER.info(f"Spot-check atoms (displacement, PBC-wrapped):")

    for atom_idx in spot_atoms:
        disp_list = []
        for t in range(nframes - 1):
            dr = _displace_pbc(
                positions[t, atom_idx : atom_idx + 1],
                positions[t + 1, atom_idx : atom_idx + 1],
                cells[t],
            )
            disp_list.append(np.linalg.norm(dr))

        disp = np.array(disp_list)
        LOGGER.info(
            f"  Atom {atom_idx:3d}: "
            f"mean Δr={np.mean(disp):.6f} Å, "
            f"max Δr={np.max(disp):.6f} Å, "
            f"std Δr={np.std(disp):.6f} Å"
        )

    # Global displacement stats (PBC-wrapped)
    LOGGER.info(f"Global displacement stats (PBC-wrapped):")
    all_displacements = []
    for t in range(nframes - 1):
        dr = _displace_pbc(positions[t], positions[t + 1], cells[t])
        dr_norms = np.linalg.norm(dr, axis=1)
        all_displacements.extend(dr_norms)

    all_displacements = np.array(all_displacements)
    LOGGER.info(f"  Mean Δr: {np.mean(all_displacements):.6f} Å")
    LOGGER.info(f"  Max Δr:  {np.max(all_displacements):.6f} Å")
    LOGGER.info(f"  P95 Δr:  {np.percentile(all_displacements, 95):.6f} Å")

    # Kinetic energy validation
    LOGGER.info(f"Kinetic energy validation:")
    frame_indices = [0, nframes // 2, nframes - 1]
    for frame_idx in frame_indices:
        vel = images[frame_idx].get_velocities()
        ke = np.mean(np.sum(vel**2, axis=1))
        LOGGER.info(f"  Frame {frame_idx:3d}: KE_proxy = {ke:.6f}")

    # Velocity stats
    LOGGER.info(f"Velocity stats (all frames, all atoms):")
    all_vel = np.concatenate([images[i].get_velocities() for i in range(nframes)])
    vel_norms = np.linalg.norm(all_vel, axis=1)
    LOGGER.info(f"  Mean |v|: {np.mean(vel_norms):.6f} Å/fs")
    LOGGER.info(f"  Max |v|:  {np.max(vel_norms):.6f} Å/fs")
    LOGGER.info(f"  P95 |v|:  {np.percentile(vel_norms, 95):.6f} Å/fs")


def validate_velocities(images):
    """
    Validate reconstructed velocities: COM drift and temperature stability.

    COM drift should be ~0. Temperature (from equipartition) should be stable.
    Uses proper unit conversion: amu·(Å/fs)² → eV.
    Degrees of freedom: 3N - 3 (COM motion effectively removed).
    """
    nframes = len(images)
    k_B_eV = 8.617e-5  # Boltzmann constant in eV/K
    amu_angstrom2_fs2_to_eV = 103.6427  # Conversion factor

    com_velocities = []
    kinetic_energies_eV = []
    temperatures = []

    for atoms in images:
        vel = atoms.get_velocities()  # Å/fs
        masses = atoms.get_masses()  # amu
        natoms = len(atoms)

        # COM velocity: mass-weighted mean velocity
        com_vel = np.average(vel, axis=0, weights=masses)
        com_vel_mag = np.linalg.norm(com_vel)
        com_velocities.append(com_vel_mag)

        # Kinetic energy in raw units (amu·Å²/fs²)
        ke_raw = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)

        # Convert to eV
        ke_eV = ke_raw * amu_angstrom2_fs2_to_eV
        kinetic_energies_eV.append(ke_eV)

        # Temperature: T = 2·KE / (k_B·dof), where dof = 3N - 3 (COM removed)
        dof = 3 * natoms - 3
        temp = 2 * ke_eV / (k_B_eV * dof) if ke_eV > 0 else 0
        temperatures.append(temp)

    com_velocities = np.array(com_velocities)
    kinetic_energies_eV = np.array(kinetic_energies_eV)
    temperatures = np.array(temperatures)

    # Log COM drift
    LOGGER.info(f"COM velocity analysis:")
    LOGGER.info(f"  Mean COM |v|: {np.mean(com_velocities):.6e} Å/fs")
    LOGGER.info(f"  Max COM |v|:  {np.max(com_velocities):.6e} Å/fs")

    if np.max(com_velocities) > 0.01:
        LOGGER.warning(
            f"  ⚠ Large COM drift detected (max {np.max(com_velocities):.6e} Å/fs)"
        )

    # Log temperature stability
    LOGGER.info(f"Temperature proxy (equipartition, dof=3N-3):")
    frame_indices = [0, nframes // 2, nframes - 1]
    for frame_idx in frame_indices:
        LOGGER.info(
            f"  Frame {frame_idx:3d}: T ≈ {temperatures[frame_idx]:.1f} K, "
            f"KE_total = {kinetic_energies_eV[frame_idx]:.3f} eV"
        )

    LOGGER.info(f"  Mean T:  {np.mean(temperatures):.1f} K")
    LOGGER.info(f"  Std T:   {np.std(temperatures):.1f} K")
    LOGGER.info(
        f"  Min/Max T: {np.min(temperatures):.1f} K / {np.max(temperatures):.1f} K"
    )

    # Flag instability
    if np.std(temperatures) > 100:
        LOGGER.warning(
            f"  ⚠ High temperature fluctuation detected (std {np.std(temperatures):.1f} K)"
        )


def build_chain(chain_start, chain_end, segments_metadata, stitch_summary):
    """Build continuous trajectory for one chain. Return (images, dedup_count)."""
    images = []
    dedup_count = 0

    for seg_idx in range(chain_start, chain_end + 1):
        seg_meta = segments_metadata[seg_idx]
        vasprun_path = Path(seg_meta["path"])

        LOGGER.info(f"  Loading segment {seg_idx}: {vasprun_path.name}")
        seg_images = reconstruct_segment(vasprun_path)

        if seg_idx > chain_start:
            # Check deduplication at boundary
            prev_seg_meta = segments_metadata[seg_idx - 1]

            # Find boundary record
            boundary_record = None
            for bd in stitch_summary["boundaries"]:
                if bd["seg_i"] == seg_idx - 1 and bd["seg_ip1"] == seg_idx:
                    boundary_record = bd
                    break

            if boundary_record and boundary_record["decision"] == "STITCH":
                pos_last = images[-1].positions
                pos_first = seg_images[0].positions
                cell = images[-1].cell

                is_dedup = check_dedup_pbc(pos_last, pos_first, cell, DEDUP_THRESHOLD)

                if is_dedup:
                    LOGGER.info(
                        f"    → Dropping duplicate first frame (RMS < {DEDUP_THRESHOLD} Å)"
                    )
                    seg_images = seg_images[1:]
                    dedup_count += 1

        images.extend(seg_images)

    return images, dedup_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading stitch summary: {STITCH_SUMMARY_PATH}")
    stitch_summary = load_stitch_summary(STITCH_SUMMARY_PATH)

    segments_metadata = stitch_summary["segments"]
    boundaries = stitch_summary["boundaries"]

    LOGGER.info(f"Found {len(segments_metadata)} segments")

    # Identify chains
    chains = identify_chains(boundaries)
    LOGGER.info(f"Identified {len(chains)} chain(s): {chains}")

    # Extract system ID from TARGET_DIR
    system_id = TARGET_DIR.name

    # Process each chain
    chain_outputs = []
    total_dedups = 0

    for chain_idx, (seg_start, seg_end) in enumerate(chains):
        LOGGER.info(f"\n=== Chain {chain_idx + 1}: segments {seg_start}–{seg_end} ===")

        images, dedup_count = build_chain(
            seg_start, seg_end, segments_metadata, stitch_summary
        )
        total_steps = len(images)
        total_dedups += dedup_count

        LOGGER.info(f"  Total frames: {total_steps}, dedups triggered: {dedup_count}")

        # Log diagnostics
        LOGGER.info(f"")
        log_chain_diagnostics(images)

        # Validate velocities
        LOGGER.info(f"")
        validate_velocities(images)

        # Build filename
        fname = (
            f"{system_id}_chain_s{seg_start:02d}-s{seg_end:02d}_steps{total_steps}.traj"
        )
        output_path = OUTPUT_DIR / fname

        write(output_path, images)
        LOGGER.info(f"  Wrote: {output_path}")

        chain_outputs.append(
            {
                "chain_idx": chain_idx,
                "seg_start": seg_start,
                "seg_end": seg_end,
                "total_steps": total_steps,
                "dedup_count": dedup_count,
                "filename": fname,
            }
        )

    LOGGER.info(f"\n=== Summary ===")
    LOGGER.info(f"Total chains: {len(chains)}")
    LOGGER.info(f"Total deduplications triggered: {total_dedups}")

    # Write stitch report
    stitch_report = {
        "system_id": system_id,
        "target_dir": str(TARGET_DIR),
        "num_chains": len(chains),
        "total_dedups": total_dedups,
        "chains": chain_outputs,
        "boundaries": boundaries,
    }

    report_path = OUTPUT_DIR / "stitch_report.json"
    with open(report_path, "w") as f:
        json.dump(stitch_report, f, indent=2)

    LOGGER.info(f"Wrote: {report_path}")
    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
