# pyright: reportAttributeAccessIssue=false
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa

LOGGER = logging.getLogger(__name__)


@dataclass
class OutcarData:
    """Parsed OUTCAR data from final ionic step.

    Attributes:
        n_ionic_steps: Total number of ionic steps in the calculation.
        V: Volume in Å³.
        E: Total energy in eV.
        stress_6: The 6-component stress tensor in Voigt notation,
            ordered as [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy].
            Units are typically eV/Å³ if obtained from ASE or DFT calculations.
        stress_9: Full 3×3 stress tensor in eV/Å³.
        P_hydro_eV_A3: Hydrostatic pressure in eV/Å³.
        P_hydro_GPa: Hydrostatic pressure in GPa.
        max_force: Maximum force magnitude across all atoms in eV/Å.
    """

    n_ionic_steps: int
    V: float
    E: float
    stress_6: npt.NDArray
    stress_9: npt.NDArray
    P_hydro_eV_A3: float
    P_hydro_GPa: float
    max_force: float


@dataclass
class EosPoint:
    """Single point on an EOS curve (one volume_factor per structure/method).

    Key identity: (method, mp_id, volume_factor) is unique.

    Status semantics:
        "ok": OUTCAR parsed successfully (does NOT imply converged).
        "missing": OUTCAR file not found.
        "failed": OUTCAR exists but parse error or other failure.

    Attributes:
        method: Computational method ("vasp", "mace", "sevennet", etc.).
        mp_id: Materials Project ID or structure identifier.
        volume_factor: Scaling factor applied to unit cell (0.8, 1.0, 1.2, etc.).
        dir_name: Original directory name (for traceability).
        outcar_path: Absolute path to OUTCAR (for clickable provenance).
        status: "ok", "missing", or "failed".
        converged_force: True if max_force <= fmax_target (None if failed/missing).
        hit_nsw_limit: True if n_ionic_steps >= NSW (None if failed/missing).
        warnings: List of warning/error messages.
    """

    method: str
    mp_id: str
    volume_factor: float
    dir_name: str
    outcar_path: Path
    status: str

    # OutcarData fields (None for failed/missing)
    V: float | None = None
    E: float | None = None
    stress_6: npt.NDArray | None = None
    stress_9: npt.NDArray | None = None
    P_hydro_eV_A3: float | None = None
    P_hydro_GPa: float | None = None
    max_force: float | None = None
    n_ionic_steps: int | None = None

    # Computed flags (None for failed/missing)
    converged_force: bool | None = None
    hit_nsw_limit: bool | None = None

    # Metadata
    warnings: list[str] = field(default_factory=list)


def parse_outcar(outcar_path: str | Path) -> OutcarData:
    """Parse OUTCAR file and extract final ionic step properties.

    Args:
        outcar_path: Path to OUTCAR file.

    Returns:
        OutcarData containing volume, energy, stress, pressure, and forces.
    """
    outcar_path = Path(outcar_path)

    # Read all ionic steps, extract last one
    imgs = read(outcar_path, index=":")
    atoms = imgs[-1]
    n_ionic_steps = len(imgs)

    # Basic properties
    V = atoms.get_volume()
    E = atoms.get_potential_energy()

    # Stress tensors
    stress_6 = atoms.get_stress()
    stress_9 = voigt_6_to_full_3x3_stress(stress_6)

    # Hydrostatic pressure (ASE stress is tensile positive)
    P_hydro_eV_A3 = -np.trace(stress_9) / 3.0
    P_hydro_GPa = P_hydro_eV_A3 * GPa

    # Max force
    forces = atoms.get_forces()
    max_force = np.linalg.norm(forces, axis=1).max()

    return OutcarData(
        n_ionic_steps=n_ionic_steps,
        V=V,
        E=E,
        stress_6=stress_6,
        stress_9=stress_9,
        P_hydro_eV_A3=P_hydro_eV_A3,
        P_hydro_GPa=P_hydro_GPa,
        max_force=max_force,
    )


def _extract_mp_id_and_volume_factor(dir_name: str) -> tuple[str, float]:
    """Extract mp_id and volume_factor from directory name.

    Example: "Al_mp-134_0.8" → ("Al_mp-134", 0.8)

    Args:
        dir_name: Directory name with pattern "{mp_id}_{volume_factor}".

    Returns:
        Tuple of (mp_id, volume_factor).

    Raises:
        ValueError: If parsing fails.
    """
    parts = dir_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse dir_name (expected 'ID_factor'): {dir_name}")

    mp_id, volume_factor_str = parts
    try:
        volume_factor = float(volume_factor_str)
    except ValueError:
        raise ValueError(f"Cannot parse volume_factor as float: {volume_factor_str}")

    return mp_id, volume_factor


def build_eos_table(
    target_dir: str | Path,
    method: str,
    fmax_target: float = 0.02,
    NSW: int = 99,
) -> list[EosPoint]:
    """Build EOS table by parsing all OUTCAR files in target directory.

    Expects directory structure: target_dir/{mp_id}_{volume_factor}/OUTCAR

    Args:
        target_dir: Root directory containing volume-scaled subdirectories.
        method: Computational method label ("vasp", "mace", etc.).
        fmax_target: Force convergence threshold (eV/Å) for converged_force flag.
        NSW: Maximum ionic steps allowed (for hit_nsw_limit flag).

    Returns:
        List of EosPoint objects, one per volume factor per structure.
        Includes failed/missing runs with status="failed"/"missing".

    Raises:
        ValueError: If duplicate (method, mp_id, volume_factor) keys detected.
    """
    target_dir = Path(target_dir)
    eos_points = []
    seen_keys = set()

    # Glob for all OUTCARs
    outcar_paths = sorted(target_dir.glob("*/OUTCAR"))

    for outcar_path in outcar_paths:
        dir_name = outcar_path.parent.name

        try:
            # Parse directory name
            mp_id, volume_factor = _extract_mp_id_and_volume_factor(dir_name)
            key = (method, mp_id, volume_factor)

            # Check uniqueness (fail-fast)
            if key in seen_keys:
                raise ValueError(f"Duplicate key: {key}")
            seen_keys.add(key)

            # Parse OUTCAR
            outcar_data = parse_outcar(outcar_path)

            # Compute flags
            converged_force = outcar_data.max_force <= fmax_target
            hit_nsw_limit = outcar_data.n_ionic_steps >= NSW

            # Create EosPoint with full data
            point = EosPoint(
                method=method,
                mp_id=mp_id,
                volume_factor=volume_factor,
                dir_name=dir_name,
                outcar_path=outcar_path,
                status="ok",
                V=outcar_data.V,
                E=outcar_data.E,
                stress_6=outcar_data.stress_6,
                stress_9=outcar_data.stress_9,
                P_hydro_eV_A3=outcar_data.P_hydro_eV_A3,
                P_hydro_GPa=outcar_data.P_hydro_GPa,
                max_force=outcar_data.max_force,
                n_ionic_steps=outcar_data.n_ionic_steps,
                converged_force=converged_force,
                hit_nsw_limit=hit_nsw_limit,
            )
            eos_points.append(point)

        except FileNotFoundError:
            # OUTCAR missing
            try:
                mp_id, volume_factor = _extract_mp_id_and_volume_factor(dir_name)
                key = (method, mp_id, volume_factor)
                if key not in seen_keys:
                    seen_keys.add(key)
                    point = EosPoint(
                        method=method,
                        mp_id=mp_id,
                        volume_factor=volume_factor,
                        dir_name=dir_name,
                        outcar_path=outcar_path,
                        status="missing",
                        warnings=["OUTCAR not found"],
                    )
                    eos_points.append(point)
            except ValueError:
                # Skip if dir_name doesn't parse
                pass

        except Exception as e:
            # Parse error or other failure
            try:
                mp_id, volume_factor = _extract_mp_id_and_volume_factor(dir_name)
                key = (method, mp_id, volume_factor)
                if key not in seen_keys:
                    seen_keys.add(key)
                    point = EosPoint(
                        method=method,
                        mp_id=mp_id,
                        volume_factor=volume_factor,
                        dir_name=dir_name,
                        outcar_path=outcar_path,
                        status="failed",
                        warnings=[str(e)],
                    )
                    eos_points.append(point)
            except ValueError:
                # Skip if dir_name doesn't parse
                pass

    return eos_points


def eos_to_dataframe(eos_points: list[EosPoint]) -> pd.DataFrame:
    """Convert list of EosPoint objects to pandas DataFrame.

    Preserves numpy arrays as object dtype columns.
    Includes traceback columns (dir_name, outcar_path) for provenance.

    Args:
        eos_points: List of EosPoint objects.

    Returns:
        DataFrame with one row per EosPoint, columns for all fields.
    """
    # Convert each EosPoint to a dict
    rows = []
    for point in eos_points:
        row = {
            "method": point.method,
            "mp_id": point.mp_id,
            "volume_factor": point.volume_factor,
            "dir_name": point.dir_name,
            "outcar_path": str(point.outcar_path),
            "status": point.status,
            "V": point.V,
            "E": point.E,
            "stress_6": point.stress_6,
            "stress_9": point.stress_9,
            "P_hydro_eV_A3": point.P_hydro_eV_A3,
            "P_hydro_GPa": point.P_hydro_GPa,
            "max_force": point.max_force,
            "n_ionic_steps": point.n_ionic_steps,
            "converged_force": point.converged_force,
            "hit_nsw_limit": point.hit_nsw_limit,
            "warnings": point.warnings,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure sorted by (method, mp_id, volume_factor)
    df = df.sort_values(["method", "mp_id", "volume_factor"]).reset_index(drop=True)

    return df
