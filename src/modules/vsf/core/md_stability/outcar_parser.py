import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


@dataclass
class IonStepData:
    """Single ionic step from VASP calculation."""

    step_id: int
    energy_total: float
    energy_free: float
    stress: npt.NDArray  # [6] Voigt notation
    positions: npt.NDArray  # [N, 3] Cartesian in Å
    forces: npt.NDArray  # [N, 3] in eV/Å
    species: list[str]  # Element symbols in atom order

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (numpy arrays → lists)."""
        return {
            "step_id": self.step_id,
            "energy_total": float(self.energy_total),
            "energy_free": float(self.energy_free),
            "stress": self.stress.tolist(),
            "positions": self.positions.tolist(),
            "forces": self.forces.tolist(),
            "species": self.species,
        }


def _extract_poscar_species(poscar_path: Path) -> list[str]:
    """Extract species symbols from POSCAR in atom order."""
    from pymatgen.core import Structure

    try:
        with open(poscar_path, "r", encoding="utf-8") as f:
            structure = Structure.from_str(f.read(), fmt="poscar")

        species = [site.species.elements[0].symbol for site in structure]
        return species
    except Exception as e:
        LOGGER.error(f"Failed to read species from {poscar_path}: {e}")
        raise


def _read_outcar_text(outcar_path: Path) -> str:
    """Read OUTCAR file as text."""
    try:
        with open(outcar_path, "r") as f:
            return f.read()
    except Exception as e:
        LOGGER.error(f"Failed to read OUTCAR {outcar_path}: {e}")
        raise


def _extract_position_force_blocks(
    text: str,
) -> list[tuple[int, int, npt.NDArray, npt.NDArray, int]]:
    """
    Extract all POSITION/TOTAL-FORCE blocks from OUTCAR text.

    Returns:
        List of (start_pos, end_pos, positions_array, forces_array, line_num) tuples.
        start_pos, end_pos used for position-based stress matching.
        line_num is POSITION header line for manual verification.
    """
    blocks = []

    # Find all POSITION/TOTAL-FORCE headers
    header_pattern = r"POSITION\s+TOTAL-FORCE.*\n\s*-+"
    footer_pattern = r"\s*-+\s*total drift"

    for header_match in re.finditer(header_pattern, text):
        header_end = header_match.end()

        # Find footer after this header
        footer_match = re.search(footer_pattern, text[header_end:])
        if not footer_match:
            LOGGER.warning("Found POSITION header but no footer (truncated file)")
            continue

        block_end = header_end + footer_match.start()
        block_text = text[header_end:block_end]

        # Parse rows: x y z fx fy fz
        row_pattern = r"^\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"

        positions = []
        forces = []

        for line in block_text.split("\n"):
            match = re.match(row_pattern, line)
            if match:
                vals = [float(x) for x in match.groups()]
                positions.append(vals[:3])
                forces.append(vals[3:])

        if positions:
            line_num = text[: header_match.start()].count("\n") + 1
            blocks.append(
                (
                    header_match.start(),
                    block_end,
                    np.array(positions, dtype=float),
                    np.array(forces, dtype=float),
                    line_num,
                )
            )
        else:
            LOGGER.warning("Found POSITION block with no valid rows (truncated file)")

    return blocks


def _extract_stress_blocks(text: str) -> list[tuple[int, npt.NDArray]]:
    """
    Extract all stress tensors from OUTCAR text with file positions.

    Looks for "Total" line with 6 Voigt components [XX, YY, ZZ, XY, YZ, ZX].

    Returns:
        List of (file_position, stress_array) tuples.
        Position used to match stresses to ionic steps.
    """
    blocks = []

    # Find "FORCE on cell =-STRESS" header (with leading whitespace handled)
    stress_header_pattern = r"^\s*FORCE on cell\s*=\s*-STRESS"

    for header_match in re.finditer(stress_header_pattern, text, re.MULTILINE):
        # Search for the "Total" line after this header
        search_start = header_match.end()
        total_match = re.search(r"^\s*Total\s+(.+)$", text[search_start:], re.MULTILINE)

        if not total_match:
            continue

        # Extract 6 float values from Total line
        vals_str = total_match.group(1)
        vals = re.findall(r"[+-]?\d+\.\d+", vals_str)

        if len(vals) >= 6:
            voigt = np.array([float(v) for v in vals[:6]], dtype=float)
            # Calculate line number for logging
            line_num = text[: search_start + total_match.start()].count("\n") + 1
            blocks.append((search_start + total_match.start(), voigt, line_num))

    return blocks


def _match_stress_to_steps(
    pos_force_blocks: list[tuple[int, int, npt.NDArray, npt.NDArray, int]],
    stress_blocks: list[tuple[int, npt.NDArray, int]],
) -> list[tuple[npt.NDArray, int]]:
    """
    Match stresses to ionic steps using file positions.

    For each ionic step, find the LAST stress block that occurs before
    the next ionic step starts (or EOF).

    Returns:
        List of (stress_array, line_number) tuples, one per ionic step.
    """
    matched_stresses = []

    for i, (pf_start, pf_end, _, _, _) in enumerate(pos_force_blocks):
        # Find next ionic step start (or use EOF)
        next_step_start = (
            pos_force_blocks[i + 1][0]
            if i + 1 < len(pos_force_blocks)
            else float("inf")
        )

        # Find all stress blocks between this step and next step
        step_stresses = []
        for stress_pos, stress_array, line_num in stress_blocks:
            if pf_start < stress_pos < next_step_start:
                step_stresses.append((stress_pos, stress_array, line_num))

        # Take the last one (final converged state for this ionic step)
        if step_stresses:
            last_stress = step_stresses[-1][1]
            last_line = step_stresses[-1][2]
            matched_stresses.append((last_stress, last_line))

            # Log summary (convergence blocks count, line used for parsing)
            if len(step_stresses) > 1:
                LOGGER.debug(
                    f"Ionic step {i}: {len(step_stresses)} convergence blocks : using line {last_line}"
                )
            else:
                LOGGER.debug(
                    f"Ionic step {i}: 1 convergence block : using line {last_line}"
                )
        else:
            matched_stresses.append((None, None))
            LOGGER.debug(f"Ionic step {i}: no stress block found")

    return matched_stresses


def _extract_energies(text: str) -> tuple[float, float, int, int]:
    """
    Extract final total and free energy from OUTCAR with line numbers.

    Takes the LAST occurrence of each energy (final converged state).

    Returns:
        (energy_total, energy_free, toten_line, etotal_line) tuple.
        energy values in eV, line numbers for manual verification.
    """
    energy_total = 0.0
    energy_free = 0.0
    toten_line = None
    etotal_line = None

    # Find free energy / TOTEN (various formats: scientific or decimal)
    toten_pattern = r"TOTEN\s+=\s+([+-]?\d+\.\d+(?:E[+-]?\d+)?)"
    toten_matches = list(re.finditer(toten_pattern, text))
    if toten_matches:
        last_match = toten_matches[-1]
        energy_free = float(last_match.group(1))
        toten_line = text[: last_match.start()].count("\n") + 1

    # Find total energy (ETOTAL = ...)
    etotal_pattern = r"ETOTAL\s+=\s+([+-]?\d+\.\d+(?:E[+-]?\d+)?)"
    etotal_matches = list(re.finditer(etotal_pattern, text))
    if etotal_matches:
        last_match = etotal_matches[-1]
        energy_total = float(last_match.group(1))
        etotal_line = text[: last_match.start()].count("\n") + 1

    if not toten_matches and not etotal_matches:
        LOGGER.warning("No energy values found in OUTCAR")

    return energy_total, energy_free, etotal_line, toten_line


def parse_outcar(outcar_path: Path, poscar_path: Path) -> list[IonStepData]:
    """
    Parse a single OUTCAR file into IonStepData objects.

    Extracts position/force blocks, matches to stress blocks by file position,
    and extracts energies.
    Skips incomplete ionic steps (missing forces or stress).

    Args:
        outcar_path: Path to OUTCAR file
        poscar_path: Path to POSCAR file (for species information)

    Returns:
        List of IonStepData objects, one per complete ionic step.
        Returns empty list if no valid steps found.
    """
    outcar_path = Path(outcar_path)
    poscar_path = Path(poscar_path)

    # Validate paths
    if not outcar_path.exists():
        LOGGER.error(f"OUTCAR not found: {outcar_path}")
        return []

    if not poscar_path.exists():
        LOGGER.error(f"POSCAR not found: {poscar_path}")
        return []

    # Extract species from POSCAR
    try:
        species = _extract_poscar_species(poscar_path)
    except Exception as e:
        LOGGER.error(f"Cannot parse species from {poscar_path}, stopping: {e}")
        return []

    # Read OUTCAR text
    try:
        text = _read_outcar_text(outcar_path)
    except Exception as e:
        LOGGER.error(f"Cannot read OUTCAR {outcar_path}: {e}")
        return []

    # Extract all blocks
    pos_force_blocks = _extract_position_force_blocks(text)
    stress_blocks = _extract_stress_blocks(text)
    matched_stresses = _match_stress_to_steps(pos_force_blocks, stress_blocks)
    energy_total, energy_free, etotal_line, toten_line = _extract_energies(text)

    if not pos_force_blocks:
        LOGGER.warning(f"No valid POSITION/TOTAL-FORCE blocks found in {outcar_path}")
        return []

    # Build IonStepData for each complete step
    steps = []

    for step_id, (pf_start, pf_end, positions, forces, pf_line) in enumerate(
        pos_force_blocks
    ):
        # Get matched stress (may be None if not found)
        stress_data = (
            matched_stresses[step_id]
            if step_id < len(matched_stresses)
            else (None, None)
        )
        stress, stress_line = stress_data

        if stress is None:
            LOGGER.warning(
                f"Ionic step {step_id}: no stress block found (may be truncated), skipping"
            )
            continue

        # Validate consistency
        if len(positions) != len(forces):
            LOGGER.warning(
                f"Ionic step {step_id}: positions ({len(positions)}) != forces ({len(forces)}), skipping"
            )
            continue

        if len(positions) != len(species):
            LOGGER.warning(
                f"Ionic step {step_id}: atoms ({len(positions)}) != species ({len(species)}), skipping"
            )
            continue

        step = IonStepData(
            step_id=step_id,
            energy_total=energy_total,
            energy_free=energy_free,
            stress=stress,
            positions=positions,
            forces=forces,
            species=species.copy(),
        )
        steps.append(step)
        pre = "     | "
        LOGGER.info(
            f"Ionic step {step_id}: {len(positions)} atoms:\n"
            f"{pre}POSITIONS line {pf_line}\n"
            f"{pre}FORCES line {pf_line}\n"
            f"{pre}ETOTAL={energy_total:.6f} eV (line {etotal_line})\n"
            f"{pre}TOTEN={energy_free:.6f} eV (line {toten_line})\n"
            f"{pre}STRESS line {stress_line}"
        )

    LOGGER.info(
        f"Parsed {outcar_path}: {len(steps)} valid steps out of {len(pos_force_blocks)} ionic steps "
        f"({len(stress_blocks)} convergence blocks total)"
    )

    return steps
