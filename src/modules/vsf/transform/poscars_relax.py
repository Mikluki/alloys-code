import json
import logging
import pprint as pp
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.core import Structure

from vsf.calculators import Mace_mpa_0
from vsf.calculators.base import BaseNN
from vsf.utils.io import load_structure_from_poscar

LOGGER = logging.getLogger(__name__)
HISTORY_FNAME = "relaxation_history.json"


def relax_structure(
    structure_in: Structure | Atoms,
    calculator: Calculator,
    constant_symmetry: bool = False,
    constant_cell_shape: bool = False,
    constant_volume: bool = False,
    scalar_pressure=0.0 * units.GPa,
    fmax: float = 0.02,
    trajectory_path: Path | None = None,
    logfile: str | None = None,
) -> Dict[str, Any]:
    """
    Relax a crystal structure with flexible control over geometry.

    What can be relaxed:
    1. **Atom positions** - where atoms sit within the cell
    2. **Cell shape** - angles between cell vectors (α, β, γ)
    3. **Cell size** - lengths of cell vectors (a, b, c)

    Relaxation Scenarios:

    1. **Full relaxation** (default):
       relax_structure(structure, calculator)
       → Positions + shape + size can all change

    2. **Symmetry preserved**:
       relax_structure(structure, calculator, constant_symmetry=True)
       → Positions + shape + size can change, but crystal symmetry is maintained

    3. **Constant volume**:
       relax_structure(structure, calculator, constant_volume=True)
       → Positions + shape can change, volume stays fixed

    4. **Uniform cell scaling**:
       relax_structure(structure, calculator, constant_cell_shape=True)
       → Positions + uniform size changes (like inflating a balloon), angles stay fixed

    5. **Positions only**:
       relax_structure(structure, calculator, constant_cell_shape=True, constant_volume=True)
       → Only atomic positions can move, cell stays completely fixed

    Parameters:
        structure_in (Union[Structure, Atoms]):
            Input structure to optimize. Can be either:
            - A pymatgen Structure object
            - An ASE Atoms object

        calculator:
            ASE-compatible calculator for energy and force calculations.

        constant_symmetry (bool, default=False):
            Whether to preserve the crystal symmetry during optimization.
            - True: Maintains the initial symmetry elements of the crystal structure
            - False: Allows breaking of symmetry during relaxation

        constant_cell_shape (bool, default=False):
            Whether to keep cell angles fixed during optimization.
            - True: Only allows uniform scaling of the cell, preserving all angles
            - False: Allows full shape changes (all lengths and angles can change)

        constant_volume (bool, default=False):
            Whether to keep the cell volume fixed during optimization.
            - True: Maintains the initial cell volume
            - False: Allows the cell volume to change during relaxation

        scalar_pressure (float, default=0.0*units.GPa):
            External pressure to apply during optimization.
            - 0.0 GPa: No external pressure (optimize at zero pressure)
            - Positive values: Compressive pressure
            - Negative values: Tensile pressure (not commonly used)

            Useful reference values:
            - 0.05 eV/Å³ = 8 GPa
            - 0.003 eV/Å³ = 0.48 GPa
            - 0.0001 eV/Å³ = 0.02 GPa

        fmax (float, default=0.02):
            Maximum force tolerance for convergence, in eV/Å.
            Lower values give more precise geometries but take longer to converge.

        trajectory_path (Path | None, default=None):
            Where to write the ASE trajectory (.traj).
            - None: use default name in current working directory: rtraj_<calc_name>.traj
            - Path: full path to the .traj file (parent dirs will be created)

        logfile (str | None, default=None):
            Path to the optimizer log file:
            - None: Use default logging
            - "-": Output to stdout
            - "<filename>": Write to specified file

            Returns:
        dict: A dictionary containing:
            - "structure": Optimized pymatgen Structure object
            - "final_energy": Final energy in eV
            - "cell_diff": Dictionary with absolute changes in cell parameters:
                          {"a": Δa (Å), "b": Δb (Å), "c": Δc (Å),
                           "alpha": Δα (°), "beta": Δβ (°), "gamma": Δγ (°)}
    """
    # Convert pymatgen Structure to ASE Atoms if necessary
    if isinstance(structure_in, Structure):
        atoms_in = structure_in.to_ase_atoms()
    else:
        atoms_in = structure_in

    # Create a copy to avoid modifying the original
    atoms = atoms_in.copy()
    if not hasattr(calculator, "get_potential_energy"):
        raise ValueError("Invalid calculator: Ensure it is ASE-compatible.")
    atoms.calc = calculator

    if constant_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])

    # Determine if we need cell filter (positions only case doesn't need it)
    positions_only = constant_cell_shape and constant_volume

    if positions_only:
        # For positions-only relaxation, use atoms directly without cell filter
        optimizer_target = atoms
    else:
        # Create cell filter with appropriate settings
        cell_filter = FrechetCellFilter(
            atoms,
            hydrostatic_strain=constant_cell_shape,
            constant_volume=constant_volume,
            scalar_pressure=scalar_pressure,
        )
        optimizer_target = cell_filter

    # --- trajectory path handling (new) ---
    calc_name = getattr(calculator, "name", None) or calculator.__class__.__name__
    default_traj_name = f"rtraj_{calc_name}.traj"

    if trajectory_path is None:
        traj_path = Path(default_traj_name)
    else:
        traj_path = Path(trajectory_path)
        # If user accidentally passed a directory, drop the default name into it.
        if traj_path.exists() and traj_path.is_dir():
            traj_path = traj_path / default_traj_name

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    # -------------------------------------

    # Set up the FIRE optimizer with passed parameters
    dyn = FIRE(
        optimizer_target,  # pyright: ignore
        logfile=logfile if logfile is not None else "-",
        trajectory=str(trajectory_path),
    )

    # Run optimization with the specified fmax
    dyn.run(fmax=fmax)

    # Calculate changes and final energy
    initial_cellpar = atoms_in.cell.cellpar()
    final_cellpar = atoms.cell.cellpar()
    cell_diff_abs = final_cellpar - initial_cellpar

    cell_diff = {
        "a": cell_diff_abs[0],  # Angstroms
        "b": cell_diff_abs[1],  # Angstroms
        "c": cell_diff_abs[2],  # Angstroms
        "alpha": cell_diff_abs[3],  # degrees
        "beta": cell_diff_abs[4],  # degrees
        "gamma": cell_diff_abs[5],  # degrees
    }

    final_energy = atoms.get_potential_energy()

    # Log the results
    LOGGER.info(f"Initial Cell           : {initial_cellpar}")
    LOGGER.info(f"Optimized Cell         : {final_cellpar}")
    LOGGER.info(f"Cell diff (abs units)  : {cell_diff}")
    LOGGER.info(f"Scaled positions       :\n {atoms.get_scaled_positions()}")
    LOGGER.info(f"Epot after opt: {final_energy} eV")

    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)  # pyright: ignore
    return {
        "structure": structure,
        "final_energy": final_energy,
        "cell_diff": cell_diff,
    }


def _generate_state_name(
    constant_volume: bool = False,
    constant_cell_shape: bool = False,
    constant_symmetry: bool = False,
    scalar_pressure: float = 0.0,
) -> str:
    """Generate descriptive state name from relaxation parameters."""

    # Check if any geometric constraints exist
    has_constraints = constant_symmetry or constant_volume or constant_cell_shape

    # "full" only if no geometric constraints (pressure is a condition, not constraint)
    if not has_constraints:
        if scalar_pressure == 0.0:
            return "full"
        else:
            # Convert pressure to GPa and format nicely
            pressure_gpa = scalar_pressure / units.GPa
            pressure_str = f"p{pressure_gpa:.2f}".replace(".", "")
            return f"full_{pressure_str}"

    # Build from individual constraint parts (all can coexist)
    parts = []

    # Symmetry constraint
    if constant_symmetry:
        parts.append("csym")

    # Cell constraints - special case for positions only
    if constant_volume and constant_cell_shape:
        parts.append("pos")  # positions only (shorthand for cvol+cshape)
    else:
        if constant_volume:
            parts.append("cvol")  # constant volume
        if constant_cell_shape:
            parts.append("cshape")  # constant shape

    # Pressure condition (if non-zero)
    if scalar_pressure != 0.0:
        # Convert pressure to GPa and format nicely
        pressure_gpa = scalar_pressure / units.GPa
        parts.append(f"p{pressure_gpa:.2f}".replace(".", ""))

    return "_".join(parts)


def _load_history(poscar_dir: Path) -> Dict[str, Any]:
    """Load relaxation history from JSON file."""
    history_path = poscar_dir / HISTORY_FNAME
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"steps": []}


def _save_history(poscar_dir: Path, history: Dict[str, Any]) -> None:
    """Save relaxation history to JSON file."""
    history_path = poscar_dir / HISTORY_FNAME
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _add_history_step(
    history: Dict[str, Any],
    state: str,
    backup_name: str | None = None,
    relaxation_args: Dict[str, Any] | None = None,
    final_energy: float | None = None,
    cell_diff: Any | None = None,
) -> None:
    """Add a new step to the relaxation history."""
    timestamp = datetime.now().strftime("%y-%m-%dT%H:%M:%S")

    step_data = {
        "step": len(history["steps"]),
        "state": state,
        "timestamp": timestamp,
    }

    if backup_name:
        step_data["backup"] = backup_name
    if relaxation_args:
        step_data["relaxation_args"] = relaxation_args
    if final_energy is not None:
        step_data["final_energy"] = final_energy
    if cell_diff is not None:
        step_data["cell_diff"] = cell_diff

    history["steps"].append(step_data)


def create_stage_dir_structure(
    base_output_dir: Path, stage_name: str, calc_names: List[str]
) -> Path:
    """
    Create directory structure for a relaxation stage (flat structure).

    Parameters:
    -----------
    base_output_dir : Path
        Base output directory
    stage_name : str
        Name of the stage (e.g., "01_csym")
    calc_names : List[str]
        List of calculation names

    Returns:
    --------
    Path
        Path to the base output directory (for consistency with existing code)
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create calculation directories directly in base_output_dir (flat structure)
    for calc_name in calc_names:
        calc_stage_dir = base_output_dir / f"{calc_name}_stage_{stage_name}"
        calc_stage_dir.mkdir(parents=True, exist_ok=True)

    return base_output_dir


def copy_poscar_to_stage_directory(
    source_poscar: Path,
    base_dir: Path,
    calc_name: str,
    stage_name: str,
    copy_history: bool = False,
) -> Path:
    """
    Copy a POSCAR file to a stage directory with flat structure.
    Optionally copy relaxation history for chained stages.

    Parameters:
    -----------
    source_poscar : Path
        Path to source POSCAR file
    base_dir : Path
        Base output directory (e.g., chained_test/)
    calc_name : str
        Name of the calculation (e.g., "calc1")
    stage_name : str
        Name of the stage (e.g., "01_csym")
    copy_history : bool, default=False
        Whether to copy HISTORY_FNAME="relaxation_history.json" from source directory

    Returns:
    --------
    Path
        Path to the copied POSCAR file
    """
    # Create the calculation-specific directory directly in base_dir (flat structure)
    calc_stage_dir = base_dir / f"{calc_name}_stage_{stage_name}"
    calc_stage_dir.mkdir(parents=True, exist_ok=True)

    target_poscar = calc_stage_dir / "POSCAR"
    shutil.copy2(source_poscar, target_poscar)

    # Copy relaxation history if this is a chained stage
    if copy_history:
        source_history = source_poscar.parent / HISTORY_FNAME
        target_history = calc_stage_dir / HISTORY_FNAME

        if source_history.exists():
            shutil.copy2(source_history, target_history)
            LOGGER.info(f"Copied history from {source_history} to {target_history}")

    return target_poscar


class PoscarRelaxationManager:
    def __init__(
        self,
        poscar_paths: List[Path],
        calculator: Mace_mpa_0 | BaseNN | None = None,
        auto_init: bool = True,
    ):
        """
        Initialize a relaxation manager for multiple POSCAR files.

        Parameters:
        -----------
        poscar_paths : List[Path]
            List of POSCAR file paths to be relaxed
        calculator : [Mace_mpa_0 | BaseNN | Calculator | None], default=None
            ASE-compatible calculator to use for relaxations.
            If None, defaults to Mace_mpa_0
        auto_init : bool, default=True
            Whether to automatically initialize the calculator
        """
        # Validate POSCAR paths
        self.poscar_paths = []
        for path in poscar_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"POSCAR file not found: {path}")
            self.poscar_paths.append(path)

        self.results: Dict[str, Dict[str, Any]] = {}

        # Set default calculator to Mace_mpa_0 if none provided
        if calculator is None:
            calculator = Mace_mpa_0()
        elif isinstance(calculator, type) and issubclass(calculator, Mace_mpa_0):
            # If a class type is passed but not instantiated
            calculator = calculator(auto_init=auto_init)
        elif hasattr(calculator, "auto_init") and auto_init:
            # If it's already an instance but has auto_init property
            setattr(calculator, "auto_init", auto_init)

        self.calculator = calculator

    def _get_ase_calculator(self) -> Calculator:
        """
        Get the ASE calculator from the current calculator.

        Returns:
        --------
        Calculator
            ASE-compatible calculator

        Raises:
        -------
        ValueError
            If no valid ASE calculator is available
        """
        # If calculator has ase_calculator attribute, use that
        assert self.calculator is not None, f"{self.calculator}"
        if hasattr(self.calculator, "ase_calculator"):
            return self.calculator.ase_calculator  # pyright: ignore

        # If calculator is directly an ASE Calculator, use it
        if isinstance(self.calculator, Calculator):
            return self.calculator

        raise ValueError(
            f"Calculator of type {type(self.calculator)} does not provide an ASE-compatible calculator"
        )

    def relax_all(
        self,
        constant_volume: bool = False,
        constant_cell_shape: bool = False,
        constant_symmetry: bool = False,
        fmax: float = 0.02,
        scalar_pressure: float = 0.0 * units.GPa,
        logfile: str | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Relax all POSCAR files with the given parameters.

        Parameters:
        -----------
        constant_volume : bool, default=False
            Whether to maintain constant volume during relaxation
        constant_cell_shape : bool, default=False
            Whether to constrain the cell shape
        constant_symmetry : bool, default=False
            Whether to preserve crystal symmetry
        fmax : float, default=0.02
            Force tolerance for convergence (eV/Å)
        scalar_pressure : float, default=0.0*units.GPa
            External pressure to apply
        logfile : Optional[str], default=None
            Path to log file, None for default logging

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary mapping POSCAR directory names to relaxation results
        """
        for poscar_path in self.poscar_paths:
            result = self.relax_poscar(
                poscar_path,
                constant_volume=constant_volume,
                constant_cell_shape=constant_cell_shape,
                constant_symmetry=constant_symmetry,
                fmax=fmax,
                scalar_pressure=scalar_pressure,
                logfile=logfile,
            )

            # Use parent directory name as key
            dir_name = poscar_path.parent.name
            self.results[dir_name] = result

            # Log success
            LOGGER.info(f"Successfully relaxed {poscar_path}")

        return self.results

    def relax_poscar(
        self,
        poscar_path: str | Path,
        constant_volume: bool = False,
        constant_cell_shape: bool = False,
        constant_symmetry: bool = False,
        fmax: float = 0.02,
        scalar_pressure: float = 0.0 * units.GPa,
        logfile: str | None = None,
    ) -> Dict[str, Any]:
        """
        Relax a single POSCAR file with the given parameters.

        Parameters:
        -----------
        poscar_path : Union[str, Path]
            Path to the POSCAR file to relax
        constant_volume : bool, default=False
            Whether to maintain constant volume during relaxation
        constant_cell_shape : bool, default=False
            Whether to constrain the cell shape
        constant_symmetry : bool, default=False
            Whether to preserve crystal symmetry
        fmax : float, default=0.02
            Force tolerance for convergence (eV/Å)
        scalar_pressure : float, default=0.0*units.GPa
            External pressure to apply
        logfile : Optional[str], default=None
            Path to log file, None for default logging

        Returns:
        --------
        Dict[str, Any]
            Relaxation results for this POSCAR
        """
        poscar_path = Path(poscar_path)
        LOGGER.info(f"Relaxation of: {poscar_path} begun")

        poscar_dir = poscar_path.parent

        # SIMPLIFIED BACKUP: Always save current POSCAR as .prev before relaxation
        prev_backup = poscar_dir / "POSCAR.prev"
        shutil.copy2(poscar_path, prev_backup)
        LOGGER.info(f"Saved input structure to {prev_backup}")

        # Get the ASE calculator
        ase_calculator = self._get_ase_calculator()

        # Load relaxation history
        history = _load_history(poscar_dir)

        # If this is the first relaxation, add original state to history
        if not history["steps"]:
            _add_history_step(history, state="original", backup_name="POSCAR.prev")

        # Generate state name for tracking (KEPT - this is for tracking)
        state_name = _generate_state_name(
            constant_volume=constant_volume,
            constant_cell_shape=constant_cell_shape,
            constant_symmetry=constant_symmetry,
            scalar_pressure=scalar_pressure,
        )

        # Load structure
        structure = load_structure_from_poscar(poscar_path)

        # Store relaxation arguments for tracking
        relaxation_args = {
            "constant_volume": constant_volume,
            "constant_cell_shape": constant_cell_shape,
            "constant_symmetry": constant_symmetry,
            "fmax": fmax,
            "scalar_pressure": scalar_pressure,
        }

        # Perform relaxation
        relaxed_dict = relax_structure(
            structure_in=structure,
            calculator=ase_calculator,
            constant_volume=constant_volume,
            constant_cell_shape=constant_cell_shape,
            constant_symmetry=constant_symmetry,
            fmax=fmax,
            scalar_pressure=scalar_pressure,
            trajectory_path=poscar_dir,
            logfile=logfile,
        )

        # Save relaxed structure back to POSCAR
        relaxed_structure = relaxed_dict["structure"]
        relaxed_structure.to(filename=str(poscar_path), fmt="poscar")
        relaxed_dict.pop("structure", None)

        # Add this step to history
        _add_history_step(
            history,
            state=state_name,
            backup_name="POSCAR.prev",  # Now always refers to .prev
            relaxation_args=relaxation_args,
            final_energy=relaxed_dict["final_energy"],
            cell_diff=relaxed_dict["cell_diff"],
        )

        # Save updated history
        _save_history(poscar_dir, history)

        return relaxed_dict

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all relaxation results."""
        return self.results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all relaxation results.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with summary statistics
        """
        if not self.results:
            return {"status": "No relaxations performed"}

        success_count = len(self.results)

        # Calculate average energy and cell changes
        energies = [result.get("final_energy", 0) for result in self.results.values()]
        avg_energy = sum(energies) / len(energies) if energies else 0

        # Get cell changes where available
        cell_changes = [
            result.get("cell_diff", None) for result in self.results.values()
        ]
        cell_changes = [c for c in cell_changes if c is not None]

        avg_cell_change = None
        if cell_changes:
            # Calculate average absolute change across all parameters
            total_abs_change = 0
            count = 0
            for cell_diff in cell_changes:
                for param, change in cell_diff.items():
                    total_abs_change += abs(change)
                    count += 1
            avg_cell_change = total_abs_change / count if count else 0

        results = {
            "total_poscars": success_count,
            "successful": success_count,
            "average_energy": avg_energy,
            "average_cell_change": avg_cell_change,
        }
        LOGGER.info(f"Stats: {pp.pprint(results)}")
        return results


class ChainedRelaxation:
    """
    Manage chained relaxation workflows with isolated stages.

    Each stage creates new directories without modifying previous stages,
    ensuring clean separation and easy debugging.
    """

    def __init__(
        self,
        poscar_paths: List[Path],
        calculator: Mace_mpa_0 | BaseNN | None = None,
        base_output_dir: Path | str = "chained_relaxation",
    ):
        """
        Initialize chained relaxation manager.

        Parameters:
        -----------
        poscar_paths : List[Path]
            List of original POSCAR file paths
        calculator : Calculator, optional
            Calculator to use for relaxations
        base_output_dir : Path | str
            Base directory for all stage outputs
        """
        # Validate POSCAR paths
        self.poscar_paths = []
        for path in poscar_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"POSCAR file not found: {path}")
            self.poscar_paths.append(path)

        # Extract calc names from original paths
        self.calc_names = self._generate_unique_calc_names(self.poscar_paths)

        # Set up directories and calculator
        self.base_output_dir = Path(base_output_dir)
        self.calculator = calculator

        # Track stages and results
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.stage_results: Dict[str, Dict[str, Any]] = {}

        LOGGER.info(
            f"ChainedRelaxation initialized with {len(self.poscar_paths)} POSCAR files"
        )
        LOGGER.info(f"Output directory: {self.base_output_dir}")

    def add_stage(
        self, stage_name: str, relaxation_config: Dict[str, Any]
    ) -> "ChainedRelaxation":
        """
        Add a relaxation stage to the chain.

        Parameters:
        -----------
        stage_name : str
            Name for this stage (e.g., "01_csym", "02_full")
        relaxation_config : Dict[str, Any]
            Relaxation parameters (constant_volume, constant_symmetry, etc.)

        Returns:
        --------
        ChainedRelaxation
            Self for method chaining
        """
        if stage_name in self.stages:
            LOGGER.warning(f"Stage '{stage_name}' already exists, overwriting")

        self.stages[stage_name] = relaxation_config.copy()

        LOGGER.info(f"Added stage '{stage_name}' with config: {relaxation_config}")
        return self

    def run_stage(self, stage_name: str) -> Dict[str, Any]:
        """
        Run a specific relaxation stage with flat directory structure.

        Parameters:
        -----------
        stage_name : str
            Name of stage to run

        Returns:
        --------
        Dict[str, Any]
            Results from this stage
        """
        if stage_name not in self.stages:
            raise ValueError(
                f"Stage '{stage_name}' not found. Available stages: {list(self.stages.keys())}"
            )

        relaxation_config = self.stages[stage_name]

        # Determine source paths for this stage
        source_paths = self._get_source_paths_for_stage(stage_name)

        # Check if this is a chained stage (not the first one)
        stage_names = list(self.stages.keys())
        stage_index = stage_names.index(stage_name)
        is_chained_stage = stage_index > 0

        # Create base directory structure (flat)
        create_stage_dir_structure(self.base_output_dir, stage_name, self.calc_names)

        # Copy POSCARs to stage directories (with history if chained)
        stage_poscar_paths = []
        for source_path, calc_name in zip(source_paths, self.calc_names):
            stage_poscar_path = copy_poscar_to_stage_directory(
                source_path,
                self.base_output_dir,
                calc_name,
                stage_name,
                copy_history=is_chained_stage,  # Copy history for chained stages
            )
            stage_poscar_paths.append(stage_poscar_path)

        # Run relaxations using EntryRelaxationManager
        manager = PoscarRelaxationManager(stage_poscar_paths, self.calculator)
        results = manager.relax_all(**relaxation_config)

        # Store results
        self.stage_results[stage_name] = {
            "relaxation_config": relaxation_config,
            "stage_dir": self.base_output_dir,  # Base dir for flat structure
            "poscar_paths": stage_poscar_paths,
            "results": results,
            "calc_mapping": dict(zip(self.calc_names, stage_poscar_paths)),
        }

        LOGGER.info(f"Completed stage '{stage_name}' with {len(results)} relaxations")
        return self.stage_results[stage_name]

    def run_all_stages(self) -> dict[str, Any]:
        """
        Run all defined stages in order.
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Results from all stages
        """
        for stage_name in self.stages.keys():
            self.run_stage(stage_name)

        LOGGER.info(f"Completed all {len(self.stages)} stages")
        return self.stage_results

    def get_stage_results(self, stage_name: str) -> Dict[str, Any]:
        """
        Get results from a specific stage.

        Parameters:
        -----------
        stage_name : str
            Name of stage

        Returns:
        --------
        Dict[str, Any]
            Stage results
        """
        if stage_name not in self.stage_results:
            raise ValueError(f"Stage '{stage_name}' has not been run yet")

        return self.stage_results[stage_name]

    def get_stage_paths(self, stage_name: str) -> List[Path]:
        """
        Get POSCAR paths from a specific stage.

        Parameters:
        -----------
        stage_name : str
            Name of stage

        Returns:
        --------
        List[Path]
            List of POSCAR paths from this stage
        """
        if stage_name not in self.stage_results:
            raise ValueError(f"Stage '{stage_name}' has not been run yet")

        return self.stage_results[stage_name]["poscar_paths"]

    def list_stages(self) -> List[str]:
        """
        List all defined stages.

        Returns:
        --------
        List[str]
            List of stage names
        """
        return list(self.stages.keys())

    def _get_source_paths_for_stage(self, stage_name: str) -> List[Path]:
        """
        Get source POSCAR paths for a given stage.

        For the first stage, use original paths.
        For subsequent stages, use paths from the previous stage.
        """
        stage_names = list(self.stages.keys())
        stage_index = stage_names.index(stage_name)

        if stage_index == 0:
            # First stage uses original paths
            return self.poscar_paths
        else:
            # Subsequent stages use previous stage paths
            previous_stage = stage_names[stage_index - 1]
            if previous_stage not in self.stage_results:
                raise ValueError(
                    f"Previous stage '{previous_stage}' must be run before '{stage_name}'"
                )

            return self.stage_results[previous_stage]["poscar_paths"]

    def _generate_unique_calc_names(self, poscar_paths: List[Path]) -> List[str]:
        """Generate unique calc names with collision detection."""
        calc_names = []

        for poscar_path in poscar_paths:
            parts = poscar_path.parts

            if len(parts) >= 2:
                parent = parts[-2]  # Parent directory name
                filename = poscar_path.stem

                # If filename is generic "POSCAR", just use parent
                # If filename is descriptive, combine parent + filename
                if filename.upper() == "POSCAR":
                    base_name = parent
                else:
                    base_name = f"{parent}_{filename}"
            else:
                # Single component path, use filename
                base_name = poscar_path.stem

            calc_names.append(base_name)

        # Handle collisions by adding numerical suffix
        unique_names = []
        name_counts = {}

        for name in calc_names:
            if name in name_counts:
                name_counts[name] += 1
                unique_names.append(f"{name}_{name_counts[name]}")
            else:
                name_counts[name] = 0
                unique_names.append(name)

        return unique_names


def test_chained_relaxation_workflow():
    """Test the new ChainedRelaxation class with a realistic workflow."""
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    # Create test directory
    test_dir = Path.cwd() / "chained_test"
    if test_dir.exists():
        import shutil

        shutil.rmtree(test_dir)
    test_dir.mkdir()

    try:
        # Create test structures
        atoms1 = Atoms(
            symbols=["C", "C"],
            positions=[[0.1, 0.1, 0.1], [1.4, 1.4, 1.4]],
            cell=[3.0, 3.0, 3.0],
            pbc=True,
        )

        atoms2 = Atoms(
            symbols=["C", "C"],
            positions=[[0.2, 0.0, 0.2], [1.2, 1.5, 1.3]],
            cell=[2.8, 3.2, 2.9],
            pbc=True,
        )

        # Save as POSCAR files
        calc1_dir = test_dir / "calc1_"
        calc2_dir = test_dir / "calc2_"
        calc1_dir.mkdir()
        calc2_dir.mkdir()

        struct1 = AseAtomsAdaptor.get_structure(atoms1)  # pyright: ignore
        struct2 = AseAtomsAdaptor.get_structure(atoms2)  # pyright: ignore

        poscar1_path = calc1_dir / "POSCAR"
        poscar2_path = calc2_dir / "POSCAR"

        struct1.to(filename=str(poscar1_path), fmt="poscar")
        struct2.to(filename=str(poscar2_path), fmt="poscar")

        print("✓ Created test POSCAR files")

        # Initialize calculator
        calc = Mace_mpa_0()
        calc.initialize()

        # Set up chained relaxation
        chain = ChainedRelaxation(
            [poscar1_path, poscar2_path],
            calculator=calc,
            base_output_dir=test_dir,
        )

        # Define relaxation stages
        chain.add_stage("01_csym", {"constant_symmetry": True, "fmax": 0.1})
        chain.add_stage("02_full", {"fmax": 0.1})

        # Run stage by stage
        print("\n=== Running Stage 1: Symmetry Constrained ===")
        stage1_results = chain.run_stage("01_csym")
        print(f"Stage 1 completed: {len(stage1_results['results'])} structures relaxed")

        print("\n=== Running Stage 2: Full Relaxation ===")
        stage2_results = chain.run_stage("02_full")
        print(f"Stage 2 completed: {len(stage2_results['results'])} structures relaxed")

        # Show results
        print("\n=== Summary ===")
        print(f"Stages completed: {chain.list_stages()}")

        for stage_name in chain.list_stages():
            stage_paths = chain.get_stage_paths(stage_name)
            print(f"Stage '{stage_name}': {[p.parent.name for p in stage_paths]}")

        print("\n✓ ChainedRelaxation test successful!")
        print(f"Results available at: {test_dir}")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


def print_readme() -> None:
    """Print usage instructions for structure optimization."""
    readme_text = """
Structure optimization usage:

    from vsf.transform.poscars_relax import relax_structure
    from pymatgen.core import Structure

    # Single structure relaxation
    structure = Structure.from_file("POSCAR")

    What can be relaxed:
    • Atom positions - where atoms sit within the cell
    • Cell shape - angles between cell vectors (α, β, γ)
    • Cell size - lengths of cell vectors (a, b, c)

    Relaxation scenarios:

    # 1. Full relaxation (default)
    # → Positions + shape + size can all change
    result = relax_structure(structure, calculator)

    # 2. Symmetry preserved
    # → Positions + shape + size can change, crystal symmetry maintained
    result = relax_structure(structure, calculator, constant_symmetry=True)

    # 3. Constant volume
    # → Positions + shape can change, volume stays fixed
    result = relax_structure(structure, calculator, constant_volume=True)

    # 4. Uniform cell scaling
    # → Positions + uniform size changes, angles stay fixed
    result = relax_structure(structure, calculator, constant_cell_shape=True)

    # 5. Positions only
    # → Only atomic positions move, cell stays completely fixed
    result = relax_structure(structure, calculator, 
                              constant_cell_shape=True, constant_volume=True)

    ##########################################
    ### Batch relaxation with POSCAR files ###

    from pymatgen.core import Structure
    from vsf.transform.poscars_relax import (
        ChainedRelaxation,
        PoscarRelaxationManager,
        relax_structure,
    )

    poscar_paths = ["calc1/POSCAR", "calc2/POSCAR", "calc3/POSCAR"]
    manager = EntryRelaxationManager(poscar_paths, calculator)
    
    # Relax all with constant volume
    results = manager.relax_all(constant_volume=True)

    # RECOMMENDED: Chained relaxation workflow - isolated stages
    chain = ChainedRelaxation(poscar_paths, calculator)
    
    # Define relaxation stages
    chain.add_stage("01_csym", {"constant_symmetry": True, "fmax": 0.1})
    chain.add_stage("02_full", {"fmax": 0.1})
    
    # Run all stages
    all_results = chain.run_all_stages()
    
    # Or run step by step
    stage1_results = chain.run_stage("01_csym")
    stage2_results = chain.run_stage("02_full") 
    
    # Access results
    final_paths = chain.get_stage_paths("02_full")
    
    Directory structure created:
    chained_relaxation/
    ├── stage_01_csym/
    │   ├── calc1_stage_01_csym/POSCAR  # relaxed with symmetry
    │   └── calc2_stage_01_csym/POSCAR
    └── stage_02_full/
        ├── calc1_stage_02_full/POSCAR  # fully relaxed
        └── calc2_stage_02_full/POSCAR

    # Access relaxation results:
    optimized_structure = results["structure"]  # pymatgen Structure
    final_energy = results["final_energy"]      # eV
    cell_changes = results["cell_diff"]         # Dict with absolute changes:
                                               # {"a": Δa (Å), "b": Δb (Å), "c": Δc (Å),
                                               #  "alpha": Δα (°), "beta": Δβ (°), "gamma": Δγ (°)}
"""
    print(readme_text.strip())


if __name__ == "__main__":
    print_readme()
    test_chained_relaxation_workflow()
