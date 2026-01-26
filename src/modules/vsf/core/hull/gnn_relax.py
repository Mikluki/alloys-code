import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase.io import read, write
from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa

from vsf.calculators.base import BaseNN
from vsf.core.hull.parser import EosPoint
from vsf.transform.poscars_relax import relax_structure

LOGGER = logging.getLogger(__name__)


@dataclass
class InputPoint:
    """Reference to a VASP input structure at a given volume scaling.

    Attributes:
        vasp_dir_name: Directory name from VASP (e.g., "Al_mp-134_0.8").
        poscar_path: Absolute path to input POSCAR file.
    """

    vasp_dir_name: str
    poscar_path: Path


@dataclass
class GnnPointCache:
    """Cached result of a GNN relaxation run.

    Persisted as relax-cache.json in the cache directory.

    Attributes:
        vasp_dir_name: Original VASP directory name.
        method: Method name (e.g., "mace", "sevennet").
        final_energy_eV: Final energy in eV.
        stress_6: Voigt-6 stress in eV/Å³ (None if unavailable).
        stress_9: Full 3×3 stress in eV/Å³ (computed from stress_6).
        P_hydro_GPa: Hydrostatic pressure in GPa.
        max_force_eV_A: Maximum force in eV/Å.
        n_steps: Number of optimization steps performed.
        cell_diff_dict: Dictionary of cell parameter changes {"a": Δa, ...}.
        model_info: Calculator metadata (checkpoint, device, type).
        status: "ok" or "failed".
        reason: Error message if failed; empty if ok.
    """

    vasp_dir_name: str
    method: str
    final_energy_eV: float
    stress_6: npt.NDArray | None = None
    stress_9: npt.NDArray | None = None
    P_hydro_GPa: float | None = None
    max_force_eV_A: float | None = None
    n_steps: int | None = None
    cell_diff_dict: dict | None = None
    model_info: dict | None = None
    status: str = "ok"
    reason: str = ""


def build_input_manifest(vasp_root_dir: str | Path) -> list[InputPoint]:
    """Build manifest of VASP input POSCARs keyed by vasp_dir_name.

    Expects structure: vasp_root_dir/{vasp_dir_name}/POSCAR

    Args:
        vasp_root_dir: Root directory containing all VASP calculation directories.

    Returns:
        List of InputPoint objects, one per unique vasp_dir_name.

    Raises:
        ValueError: If duplicate vasp_dir_name entries found.
    """
    vasp_root_dir = Path(vasp_root_dir)
    input_points = []
    seen_dirs = set()

    # Glob for all POSCAR files
    poscar_paths = sorted(vasp_root_dir.glob("*/POSCAR"))

    for poscar_path in poscar_paths:
        vasp_dir_name = poscar_path.parent.name

        # Enforce uniqueness per vasp_dir_name
        if vasp_dir_name in seen_dirs:
            raise ValueError(f"Duplicate vasp_dir_name: {vasp_dir_name}")
        seen_dirs.add(vasp_dir_name)

        point = InputPoint(
            vasp_dir_name=vasp_dir_name,
            poscar_path=poscar_path,
        )
        input_points.append(point)

    return input_points


def load_gnn_cache(cache_dir: Path) -> GnnPointCache | None:
    """Load GnnPointCache from relax-cache.json.

    Args:
        cache_dir: Directory containing relax-cache.json.

    Returns:
        GnnPointCache object if file exists, None otherwise.
    """
    cache_file = cache_dir / "relax-cache.json"

    if not cache_file.exists():
        return None

    with open(cache_file, "r") as f:
        data = json.load(f)

    # Reconstruct numpy arrays from lists if present
    if "stress_6" in data and data["stress_6"] is not None:
        data["stress_6"] = np.array(data["stress_6"])

    if "stress_9" in data and data["stress_9"] is not None:
        data["stress_9"] = np.array(data["stress_9"])

    return GnnPointCache(**data)


def save_gnn_cache(cache_dir: Path, cache_obj: GnnPointCache) -> None:
    """Save GnnPointCache to relax-cache.json.

    Args:
        cache_dir: Directory where relax-cache.json will be written.
        cache_obj: GnnPointCache object to persist.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict, handling numpy arrays
    data = asdict(cache_obj)

    # Convert numpy arrays to lists for JSON serialization
    if data["stress_6"] is not None:
        data["stress_6"] = data["stress_6"].tolist()

    if data["stress_9"] is not None:
        data["stress_9"] = data["stress_9"].tolist()

    cache_file = cache_dir / "relax-cache.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

    LOGGER.info(f"Saved GNN cache: {cache_file}")


def run_single_gnn_point(
    input_point: InputPoint,
    calculator: BaseNN,
    method: str,
    fmax: float = 0.02,
    force_recalc: bool = False,
) -> GnnPointCache:
    """Run fixed-cell GNN relaxation for a single volume point.

    Implements the full pipeline:
    1. Check cache; skip if exists + status=="ok" (unless force_recalc)
    2. Load POSCAR from input_point.poscar_path
    3. Relax with constant_cell_shape=True, constant_volume=True
    4. Compute stress, pressure, forces, steps
    5. Write cache directory (POSCAR.in, POSCAR.relaxed, rtraj, relax-cache.json)
       under {vasp_dir}/0-gnn-relax/{method}/
    6. Return GnnPointCache

    Args:
        input_point: InputPoint with vasp_dir_name and poscar_path.
        calculator: ASE-compatible calculator (must have get_stress()).
        method: Method name for identification (e.g., "mace", "sevennet").
        fmax: Force convergence threshold (eV/Å).
        force_recalc: If True, recompute even if cache exists.

    Returns:
        GnnPointCache with final results.

    Raises:
        RuntimeError: If calculator does not support stress computation.
    """
    # Derive cache directory from POSCAR location
    vasp_dir = input_point.poscar_path.parent
    cache_dir = vasp_dir / "0-gnn-relax" / method

    # Check cache and return if valid
    if not force_recalc:
        cached = load_gnn_cache(cache_dir)
        if cached is not None and cached.status == "ok":
            LOGGER.info(f"Skipping {input_point.vasp_dir_name} (cached)")
            return cached

    try:
        LOGGER.info(f"Running {input_point.vasp_dir_name} ({method})")

        # Load structure from POSCAR
        structure = read(input_point.poscar_path)

        # Run relaxation with fixed cell (positions only)
        result = relax_structure(
            structure_in=structure,
            calculator=calculator.ase_calculator,
            # constant_symmetry=True,
            constant_cell_shape=True,
            constant_volume=True,
            scalar_pressure=0.0,
            fmax=fmax,
            trajectory_path=cache_dir / "rtraj.traj",
            logfile=None,
        )

        final_energy = result["final_energy"]
        relaxed_structure = result["structure"]
        cell_diff = result["cell_diff"]

        # Convert relaxed structure to ASE atoms for writing and stress calculation
        atoms_relaxed = relaxed_structure.to_ase_atoms()

        # Write POSCAR.in (input) and POSCAR.relaxed (output)
        write(str(cache_dir / "POSCAR.in"), structure)
        write(str(cache_dir / "POSCAR.relaxed"), atoms_relaxed)

        # Get stress and compute pressure (must succeed for GNN)
        atoms_relaxed.calc = calculator.ase_calculator

        stress_6 = atoms_relaxed.get_stress()  # raises if unavailable
        stress_9 = voigt_6_to_full_3x3_stress(stress_6)

        # Hydrostatic pressure (same convention as VASP)
        P_hydro_eV_A3 = -np.trace(stress_9) / 3.0
        P_hydro_GPa = P_hydro_eV_A3 * GPa

        # Max force and step count
        forces = atoms_relaxed.get_forces()
        max_force = np.linalg.norm(forces, axis=1).max()

        # Count steps from trajectory (unified filename: rtraj.traj)
        rtraj_path = cache_dir / "rtraj.traj"
        if rtraj_path.exists():
            trajectory = read(str(rtraj_path), index=":")
            n_steps = len(trajectory) if isinstance(trajectory, list) else 1
        else:
            # No trajectory found; default to 1 (fallback for single-step runs)
            n_steps = 1

        # Get model metadata
        model_info = calculator.get_model_info()

        # Create cache object
        cache = GnnPointCache(
            vasp_dir_name=input_point.vasp_dir_name,
            method=method,
            final_energy_eV=final_energy,
            stress_6=stress_6,
            stress_9=stress_9,
            P_hydro_GPa=P_hydro_GPa,
            max_force_eV_A=max_force,
            n_steps=n_steps,
            cell_diff_dict=cell_diff,
            model_info=model_info,
            status="ok",
            reason="",
        )

        # Persist cache
        save_gnn_cache(cache_dir, cache)

        LOGGER.info(
            f"✓ {input_point.vasp_dir_name}: E={final_energy:.6f} eV, F_max={max_force:.6e} eV/Å"
        )

        return cache

    except Exception as e:
        LOGGER.error(f"Failed to relax {input_point.vasp_dir_name}: {e}")

        # Create failed cache record
        cache = GnnPointCache(
            vasp_dir_name=input_point.vasp_dir_name,
            method=method,
            final_energy_eV=0.0,  # placeholder
            model_info=calculator.get_model_info(),
            status="failed",
            reason=str(e),
        )

        # Persist failed status
        save_gnn_cache(cache_dir, cache)

        return cache


def gnn_cache_to_eos_point(cache: GnnPointCache) -> EosPoint:
    """Convert GnnPointCache to EosPoint for analysis pipeline.

    Args:
        cache: GnnPointCache object from run_single_gnn_point.

    Returns:
        EosPoint with method prefixed (e.g., "gnn_mace").
    """
    # Extract volume_factor from vasp_dir_name
    parts = cache.vasp_dir_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot extract volume_factor from {cache.vasp_dir_name}")

    volume_factor = float(parts[1])

    # Determine status and map to EosPoint semantics
    if cache.status == "failed":
        eos_status = "failed"
    else:
        eos_status = "ok"

    # Create EosPoint
    point = EosPoint(
        method=f"gnn_{cache.method}",
        vasp_dir_name=cache.vasp_dir_name,
        volume_factor=volume_factor,
        outcar_path=Path("N/A"),  # GNN doesn't have OUTCAR; use placeholder
        status=eos_status,
        V=None,  # GNN keeps cell fixed; V should match input
        E=cache.final_energy_eV,
        stress_6=cache.stress_6,
        stress_9=cache.stress_9,
        P_hydro_eV_A3=None,  # Only track GPa version
        P_hydro_GPa=cache.P_hydro_GPa,
        max_force=cache.max_force_eV_A,
        n_ionic_steps=cache.n_steps,
        converged_force=None,  # Not applicable for GNN (no force target)
        hit_nsw_limit=None,  # Not applicable for GNN
        warnings=[cache.reason] if cache.reason else [],
    )

    return point
