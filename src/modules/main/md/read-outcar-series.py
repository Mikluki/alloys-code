"""Read interrupted VASP MD trajectories (OUTCAR.1, OUTCAR.2, ...) into Trajectory contract."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

LOGGER = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Explicit contract for trajectory data (VASP or GNN)."""

    frames: list[Atoms]
    times_ps: list[float]
    temperatures: list[float]
    energies: list[float]
    forces_strided: list[np.ndarray]  # Each: (N_atoms, 3)
    times_strided_ps: list[float]
    stride: int
    md_timestep_fs_potim: float
    collapse_info: dict | None = None  # GNN only; VASP = None


def read_outcar_series(
    directory: Path | str,
    md_timestep_fs_potim: float,
    stride: int,
) -> Trajectory:
    """
    Read interrupted VASP MD runs (OUTCAR.1, OUTCAR.2, ...).

    Args:
        directory: Directory containing OUTCAR.1, OUTCAR.2, ...
        md_timestep_fs_potim: MD timestep (POTIM) in femtoseconds
        stride: Stride for forces_strided (e.g., stride=10 = every 10th frame)

    Returns:
        Trajectory with concatenated data from all OUTCARs.

    Raises:
        FileNotFoundError: If OUTCAR.1 missing or sequence has gaps.
        ValueError: If stride > number of frames.
    """
    directory = Path(directory)

    # Find all OUTCAR.N files and validate sequence
    outcar_files = sorted(
        [f for f in directory.glob("OUTCAR.*") if f.name.split(".")[-1].isdigit()]
    )

    if not outcar_files:
        raise FileNotFoundError(f"No OUTCAR.* files found in {directory}")

    # Validate sequence: OUTCAR.1, OUTCAR.2, etc. with no gaps
    expected_nums = set(range(1, len(outcar_files) + 1))
    actual_nums = set(int(f.name.split(".")[-1]) for f in outcar_files)
    if expected_nums != actual_nums:
        missing = expected_nums - actual_nums
        raise FileNotFoundError(f"Gap in OUTCAR sequence. Missing: {sorted(missing)}")

    LOGGER.info(f"Found {len(outcar_files)} OUTCAR files in sequence")

    # Parse all files in order
    all_positions = []
    all_temperatures = []
    all_energies = []
    all_forces = []

    for outcar_file in outcar_files:
        LOGGER.info(f"Parsing {outcar_file.name}")
        positions, temps, energies, forces = _parse_single_outcar(outcar_file)

        all_positions.extend(positions)
        all_temperatures.extend(temps)
        all_energies.extend(energies)
        all_forces.extend(forces)

    n_frames = len(all_positions)
    LOGGER.info(f"Total frames: {n_frames}")

    if stride > n_frames:
        raise ValueError(f"stride={stride} exceeds frames={n_frames}")

    # Build Atoms objects from positions
    frames = [_positions_to_atoms(pos) for pos in all_positions]

    # Calculate times in picoseconds
    times_ps = [i * md_timestep_fs_potim / 1000.0 for i in range(n_frames)]

    # Apply stride
    forces_strided = all_forces[::stride]
    times_strided_ps = times_ps[::stride]

    LOGGER.info(
        f"Strided: {len(forces_strided)} frames "
        f"(stride={stride}, time range: {times_strided_ps[0]:.3f}-{times_strided_ps[-1]:.3f} ps)"
    )

    return Trajectory(
        frames=frames,
        times_ps=times_ps,
        temperatures=all_temperatures,
        energies=all_energies,
        forces_strided=forces_strided,
        times_strided_ps=times_strided_ps,
        stride=stride,
        md_timestep_fs_potim=md_timestep_fs_potim,
        collapse_info=None,
    )


def _parse_single_outcar(
    filepath: Path,
) -> tuple[list[list], list[float], list[float], list[np.ndarray]]:
    """
    Parse one OUTCAR file using pymatgen Outcar.read_table_pattern().

    Returns:
        (positions_list, temperatures, energies, forces)
        where positions_list is list of (N_atoms, 3) arrays
    """
    from pymatgen.io.vasp.outputs import Outcar

    try:
        outcar = Outcar(str(filepath))
    except Exception as e:
        LOGGER.error(f"  Failed to instantiate Outcar: {e}")
        raise

    # Extract positions and forces from same table
    # row_pattern: x y z fx fy fz (all 6 columns)
    pos_force_tables = outcar.read_table_pattern(
        header_pattern=r"POSITION\s+TOTAL-FORCE.*\n\s*-+",
        row_pattern=r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s*-+\s*total drift",
        postprocess=float,
        last_one_only=False,
    )

    if not pos_force_tables:
        LOGGER.warning(f"  No POSITION/TOTAL-FORCE blocks found")
        return [], [], [], []

    # Flatten positions and forces from table
    positions_list = []
    forces_list = []

    for table in pos_force_tables:
        positions = []
        forces = []
        for row in table:
            positions.append(row[:3])
            forces.append(row[3:])
        if positions:
            positions_list.append(np.array(positions, dtype=float))
            forces_list.append(np.array(forces, dtype=float))

    n_frames = len(positions_list)

    # Extract energies and temperatures directly from file text
    # (not in table format, just scattered lines)
    with open(filepath) as f:
        text = f.read()

    # Grep for TOTEN energy (free energy) - take only LAST (final converged value)
    import re

    all_energies = re.findall(r"free energy\s+=\s+([+-]?\d+\.\d+E[+-]?\d+)", text)
    energies = [float(all_energies[-1])] if all_energies else []

    # Grep for temperature (temperature X K) - take only LAST (final temp)
    all_temperatures = re.findall(r"\(temperature\s+([+-]?\d+\.\d+)\s+K\)", text)
    temperatures = [float(all_temperatures[-1])] if all_temperatures else []

    # Pad with 0.0 if missing
    while len(energies) < n_frames:
        energies.append(0.0)
    while len(temperatures) < n_frames:
        temperatures.append(0.0)

    LOGGER.info(
        f"  Extracted: {n_frames} ionic steps, {len(energies)} energies, {len(temperatures)} temperatures"
    )

    return positions_list, temperatures[:n_frames], energies[:n_frames], forces_list


def _positions_to_atoms(positions: np.ndarray) -> Atoms:
    """Convert position array to ASE Atoms object with placeholder symbols."""
    n_atoms = len(positions)
    return Atoms(
        symbols="H" * n_atoms,
        positions=positions,
        pbc=True,
    )


if __name__ == "__main__":
    from vsf.logging import setup_logging

    LOGGER = setup_logging(
        log_file=f"x-read-outcar-test.log",
        console_level=logging.DEBUG,
        file_level=logging.DEBUG,
    )
    # path = Path("AlCuNi_L1915_1400-lght")
    path = Path("data/AlCuNi_L1915_1400")
    traj = read_outcar_series(
        directory=path,
        md_timestep_fs_potim=2.0,
        stride=1,
    )
    print(f"Frames: {len(traj.frames)}")
    print(f"Energies: {traj.energies}")
    print(f"Temperatures: {traj.temperatures}")
    print(f"Forces shape: {traj.forces_strided[0].shape}")
    print(f"Times (ps): {traj.times_strided_ps}")
