import json
import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

from ..core.record import CorruptedJsonError, FailedStructure, StructureRecord
from ..energy.energy_source import EnergySource
from ..gnn_runner.interface import GNNRunnerConfig

LOGGER = logging.getLogger(__name__)


class GNNEnergyWorkflow:
    """Orchestrates GNN-based formation energy calculations for batch processing."""

    def __init__(self):
        self.failed_structures: List[FailedStructure] = []
        self.incomplete_structures: List[Path] = []

    def extract_hull_reference_energies(
        self,
        hull_dir: Path,
        energy_source: EnergySource,
        force_recalculate_hull: bool = False,
        cleanup_deprecated: bool = False,
        **calc_kwargs,
    ) -> Dict[str, float]:
        """
        Extract GNN reference energies from hull structures.

        Parameters
        ----------
        hull_dir : Path
            Directory containing hull reference structure subdirectories
        energy_source : EnergySource
            GNN energy source to use for calculations
        force_recalculate : bool
            Whether to re-extract energy even if it already exists for this source
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources
        **calc_kwargs
            Arguments passed to calculator subprocess

        Returns
        -------
        Dict[str, float]
            Mapping of element symbols to reference energies per atom
        """
        hull_reference = {}
        hull_dir = Path(hull_dir)

        LOGGER.info(f"Processing GNN hull references from: {hull_dir}")

        # Collect hull structure directories
        hull_structure_dirs = [d for d in hull_dir.iterdir() if d.is_dir()]

        if not hull_structure_dirs:
            LOGGER.warning(f"No hull structure directories found in {hull_dir}")
            return hull_reference

        # Filter structures that need calculation
        target_dirs, save_energy, save_stress = self._get_targets_and_flags(
            hull_structure_dirs, energy_source, force_recalculate_hull
        )

        if target_dirs:
            # Run subprocess to calculate potential energies
            LOGGER.info(
                f"Running GNN calculations for {len(target_dirs)} hull structures"
            )
            self._run_subprocess_calculator(
                target_dirs,
                energy_source,
                save_energy,
                save_stress,
                force_recalculate_hull,
                cleanup_deprecated,
                **calc_kwargs,
            )

        # Extract hull reference energies from processed structures
        for structure_dir in hull_structure_dirs:
            try:
                record = StructureRecord.load_json(structure_dir)
                if record is None:
                    LOGGER.error(
                        f"No JSON data found for hull structure {structure_dir.parent.name}/{structure_dir.name}"
                    )
                    continue

                # Verify this is a pure element structure
                if len(record.atoms) != 1:
                    LOGGER.warning(
                        f"Hull structure {record.name} contains multiple elements: {record.atoms}"
                    )
                    continue

                element = list(record.atoms.keys())[0]
                energy_result = record.potential_energy.get(energy_source)

                if energy_result is None:
                    LOGGER.error(f"No energy found for hull structure {record.name}")
                    continue

                hull_reference[element] = energy_result.value
                LOGGER.info(
                    f"Hull reference {element}: {energy_result.value:.6f} eV/atom"
                )

            except CorruptedJsonError as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Corrupted JSON: {e.details}")
                )
                LOGGER.error(
                    f"Corrupted JSON for hull structure {structure_dir.parent.name}/{structure_dir.name}: {e}"
                )

            except Exception as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Hull processing failed: {str(e)}")
                )
                LOGGER.error(
                    f"Failed to process hull structure {structure_dir.parent.name}/{structure_dir.name}: {e}"
                )

        LOGGER.info(f"Extracted hull references for {len(hull_reference)} elements")
        return hull_reference

    def run_workflow(
        self,
        structure_dirs: List[Path],
        hull_reference: Dict[str, float],
        energy_source: EnergySource,
        force_recalculate: bool = False,
        cleanup_deprecated: bool = False,
        formation_energy_only: bool = False,
        **calc_kwargs,
    ) -> Dict:
        """
        Process structures for formation energies using GNN calculator.

        Parameters
        ----------
        structure_dirs : List[Path]
            List of structure directories containing POSCAR files
        hull_reference : Dict[str, float]
            Hull reference energies per atom for formation energy calculations
        energy_source : EnergySource
            GNN energy source to use
        force_recalculate : bool
            Whether to re-extract energy even if it already exists for this source
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources
        formation_energy_only : bool, False (default)
            If True, only calculate formation energies for structures that already
            have potential energies. Skip subprocess calculation entirely.
        **calc_kwargs
            Additional arguments passed to calculator subprocess
        """
        structure_dirs = [Path(d) for d in structure_dirs]

        if formation_energy_only:
            # Only process structures with potential energy but missing formation energy
            target_dirs = [
                Path(d)
                for d in structure_dirs
                if _should_calculate_formation_only(d, energy_source, force_recalculate)
            ]

            LOGGER.info(
                f"Formation energy only mode: Processing {len(target_dirs)}/{len(structure_dirs)} structures for {energy_source.value}"
            )

            if not target_dirs:
                LOGGER.info("No structures need formation energy calculation")
                return {}

            # Skip subprocess, only calculate formation energies
            LOGGER.info("Calculating formation energies...")
            self._calculate_formation_energies(
                target_dirs,
                hull_reference,
                energy_source,
                force_recalculate,
                cleanup_deprecated,
            )

        else:
            # Normal mode: calculate potential energies via subprocess, then formation energies
            target_dirs, save_energy, save_stress = self._get_targets_and_flags(
                structure_dirs, energy_source, force_recalculate
            )

            LOGGER.info(
                f"Processing {len(target_dirs)}/{len(structure_dirs)} structures for {energy_source.value}"
            )

            if not target_dirs:
                LOGGER.info("No structures need processing")
                LOGGER.info("=" * 60)
                return {}

            # Log what will be calculated
            LOGGER.info(
                f"Calculation flags: save_energy={save_energy}, save_stress={save_stress}"
            )

            # Step 1: Run subprocess for potential energies
            LOGGER.info("Calculating potential energies via subprocess...")
            self._run_subprocess_calculator(
                target_dirs,
                energy_source,
                save_energy,
                save_stress,
                force_recalculate=force_recalculate,
                cleanup_deprecated=cleanup_deprecated,
                **calc_kwargs,
            )

            # Step 2: Calculate formation energies in main process
            LOGGER.info("Calculating formation energies...")
            self._calculate_formation_energies(
                target_dirs,
                hull_reference,
                energy_source,
                force_recalculate,
                cleanup_deprecated,
            )

            # Step 3: Check for incomplete structures and warn user
            self._check_incomplete_structures(structure_dirs, energy_source)

        LOGGER.info("=" * 60)
        LOGGER.info(f"Completed `{energy_source.value}` calculations")
        summary_dict = self._generate_summary()
        LOGGER.info(f"Stats: {summary_dict}")
        LOGGER.info("=" * 60)
        return summary_dict

    def _get_targets_and_flags(
        self,
        structure_dirs: List[Path],
        energy_source: EnergySource,
        force_recalculate: bool,
    ) -> Tuple[List[Path], bool, bool]:
        """
        Determine which structures need processing and what to calculate.

        Scans target structures to determine if energy and/or stress are missing.
        If force_recalculate is True, always requests both.
        Otherwise, requests only what's missing.

        Parameters
        ----------
        structure_dirs : List[Path]
            Structure directories to check
        energy_source : EnergySource
            Energy source to check for
        force_recalculate : bool
            Whether to force recalculation

        Returns
        -------
        Tuple[List[Path], bool, bool]
            Target structures, save_energy flag, save_stress flag
        """
        if force_recalculate:
            # Always calculate both when forcing recalculation
            target_dirs = [Path(d) for d in structure_dirs]
            return target_dirs, True, True

        # Check what's missing across structures
        energy_missing = any(
            _should_calculate_energy(d, energy_source) for d in structure_dirs
        )
        stress_missing = any(
            _should_calculate_stress(d, energy_source) for d in structure_dirs
        )

        # If both exist, nothing to do
        if not (energy_missing or stress_missing):
            return [], False, False

        # Request both if either is missing (they're cheap together)
        save_both = energy_missing or stress_missing

        # Filter to structures that need at least one calculation
        target_dirs = [
            Path(d)
            for d in structure_dirs
            if _should_calculate_energy(d, energy_source)
            or _should_calculate_stress(d, energy_source)
        ]

        return target_dirs, save_both, save_both

    def _check_incomplete_structures(
        self, structure_dirs: List[Path], energy_source: EnergySource
    ) -> None:
        """
        Check for structures with potential energy but missing formation energy.
        Log warning if any are found.
        """
        incomplete = [
            d
            for d in structure_dirs
            if _should_calculate_formation_only(
                d, energy_source, force_recalculate=False
            )
        ]

        if incomplete:
            self.incomplete_structures.extend(incomplete)
            LOGGER.warning(
                f"Found {len(incomplete)} structures with potential energy but missing formation energy"
            )
            LOGGER.warning(
                f"Run with formation_energy_only=True to calculate missing formation energies"
            )
            for structure_dir in incomplete[:5]:  # Show first 5 examples
                LOGGER.warning(
                    f"  Example: {structure_dir.parent.name}/{structure_dir.name}"
                )
            if len(incomplete) > 5:
                LOGGER.warning(f"  ... and {len(incomplete) - 5} more")

    @contextmanager
    def _temp_structure_list(self, structure_dirs: List[Path]):
        """
        Context manager for temporary structure list file.

        Parameters
        ----------
        structure_dirs : List[Path]
            Structure directories to write to temp file

        Yields
        ------
        Path
            Path to temporary file containing structure list
        """
        temp_file = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for structure_dir in structure_dirs:
                    f.write(f"{structure_dir}\n")
                temp_file = Path(f.name)

            yield temp_file

        finally:
            # Cleanup
            if temp_file and temp_file.exists():
                temp_file.unlink()

    def _run_subprocess_calculator(
        self,
        structure_dirs: List[Path],
        energy_source: EnergySource,
        save_energy: bool,
        save_stress: bool,
        force_recalculate: bool,
        cleanup_deprecated: bool,
        timeout: float = 3600 * 3,
        **calc_kwargs,
    ) -> None:
        """
        Run GNN calculator in subprocess with proper venv activation.

        Parameters
        ----------
        structure_dirs : List[Path]
            Structure directories to process
        energy_source : EnergySource
            Energy source/calculator to use
        save_energy : bool
            Whether to calculate and save energy
        save_stress : bool
            Whether to calculate and save stress
        force_recalculate : bool
            Whether to re-extract energy even if it already exists for this source
        cleanup_deprecated : bool
            Whether to remove deprecated energy sources
        **calc_kwargs
            Additional arguments for calculator
        """
        if not energy_source.has_venv():
            raise ValueError(
                f"Energy source {energy_source.value} has no venv configured"
            )

        venv_path = energy_source.venv()

        with self._temp_structure_list(structure_dirs) as structure_list_file:
            # Create config
            config = GNNRunnerConfig(
                energy_source=energy_source.value,
                structure_list=structure_list_file,
                device=calc_kwargs.get("device", "cpu"),
                overwrite=force_recalculate,
                cleanup_deprecated=cleanup_deprecated,
                progress_interval=10,
                checkpoint_path=calc_kwargs.get("checkpoint_path"),
                save_energy=save_energy,
                save_stress=save_stress,
            )

            # Build subprocess command
            cmd = [
                f"{venv_path}/bin/python",
                "-m",
                "vsf.gnn_runner",
            ] + config.to_cli_args()

            LOGGER.info(f"Running subprocess: {' '.join(cmd[:8])}...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            try:
                # Stream and display progress
                assert process.stdout is not None
                for line in process.stdout:
                    if line.startswith("PROGRESS:"):
                        print(
                            f"\r{line.strip().split(':')[1]} processed",
                            end="",
                            flush=True,
                        )
                    LOGGER.debug(line.rstrip())

                print()  # Newline after progress

                # Read stderr and wait
                stderr = process.stderr.read() if process.stderr else ""
                return_code = process.wait(timeout=timeout)

                if return_code != 0:
                    raise RuntimeError(f"Calculator subprocess failed: {stderr}")

            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError(
                    f"Calculator subprocess timed out after {timeout / 3600} hours"
                )

            except Exception as e:
                raise RuntimeError(f"Failed to run calculator subprocess: {str(e)}")

    def _calculate_formation_energies(
        self,
        structure_dirs: List[Path],
        hull_reference: Dict[str, float],
        energy_source: EnergySource,
        force_recalculate: bool,
        cleanup_deprecated: bool,
    ) -> None:
        """
        Calculate formation energies after subprocess completes potential energy calculations.

        Parameters
        ----------
        structure_dirs : List[Path]
            Structure directories to process
        hull_reference : Dict[str, float]
            Hull reference energies for formation energy calculation
        energy_source : EnergySource
            Energy source used for potential energy calculations
        force_recalculate : bool
            Whether to re-extract energy even if it already exists for this source
        cleanup_deprecated : bool
            Whether to remove deprecated energy sources
        """
        for structure_dir in structure_dirs:
            try:
                record = StructureRecord.load_json(structure_dir)
                if record is None:
                    LOGGER.error(
                        f"No JSON data found for {structure_dir.parent.name}/{structure_dir.name}"
                    )
                    continue

                # Check if potential energy exists from subprocess
                potential_result = record.potential_energy.get(energy_source)
                if potential_result is None:
                    LOGGER.warning(
                        f"No potential energy found for {structure_dir.parent.name}/{structure_dir.name} from subprocess"
                    )
                    continue

                # Calculate formation energy
                record.calculate_formation_energy(hull_reference, energy_source)
                record.save_json(
                    overwrite=force_recalculate,
                    cleanup_deprecated=cleanup_deprecated,
                )

                LOGGER.debug(
                    f"Calculated formation energy for {structure_dir.parent.name}/{structure_dir.name}"
                )

            except Exception as e:
                self._handle_structure_failure(structure_dir, energy_source, e)

    def _handle_structure_failure(
        self, structure_dir: Path, energy_source: EnergySource, error: Exception
    ) -> None:
        """
        Handle failed structure calculation.

        Parameters
        ----------
        structure_dir : Path
            Failed structure directory
        energy_source : EnergySource
            Energy source that failed
        error : Exception
            The exception that occurred
        """
        failure_reason = f"{energy_source.value}: {str(error)}"
        failed_structure = FailedStructure(path=structure_dir, reason=failure_reason)
        self.failed_structures.append(failed_structure)

        LOGGER.error(
            f"Failed to process {structure_dir.parent.name}/{structure_dir.name}: {failure_reason}"
        )

    def _generate_summary(self) -> Dict:
        """
        Generate summary of batch processing results.

        Returns
        -------
        Dict
            Summary statistics and failed structures
        """
        summary = {
            "failed_structures": len(self.failed_structures),
            "failures": [
                {"path": str(f.path), "reason": f.reason}
                for f in self.failed_structures
            ],
            "incomplete_structures": len(self.incomplete_structures),
            "incomplete_list": [
                {"path": f"{p.parent.name}/{p.name}"}
                for p in self.incomplete_structures
            ],
        }

        if self.failed_structures:
            LOGGER.warning(
                f"Processing completed with {len(self.failed_structures)} failures\n"
            )

        if self.incomplete_structures:
            LOGGER.warning(
                f"Found {len(self.incomplete_structures)} structures with incomplete calculations\n"
            )

        if not self.failed_structures and not self.incomplete_structures:
            LOGGER.info("Processing Completed:: No Fails")

        return summary


def _should_calculate_energy(
    structure_dir: Path,
    energy_source: EnergySource,
) -> bool:
    """
    Check if energy needs calculation for this source.

    Parameters
    ----------
    structure_dir : Path
        Directory containing structure files
    energy_source : EnergySource
        Energy source to check for

    Returns
    -------
    bool
        True if energy needs calculation
    """
    json_path = structure_dir / f"{structure_dir.name}.json"

    if not json_path.exists():
        return True  # No JSON file, needs calculation

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        potential_energy_data = data.get("potential_energy", {})
        results = potential_energy_data.get("results", {})

        if energy_source.value in results:
            LOGGER.debug(
                f"Skipping energy for {structure_dir.parent.name}/{structure_dir.name} - already exists"
            )
            return False
        else:
            return True

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        LOGGER.warning(
            f"Could not read JSON for {structure_dir.parent.name}/{structure_dir.name}: {e}"
        )
        return True


def _should_calculate_stress(
    structure_dir: Path,
    energy_source: EnergySource,
) -> bool:
    """
    Check if stress needs calculation for this source.

    Parameters
    ----------
    structure_dir : Path
        Directory containing structure files
    energy_source : EnergySource
        Energy source to check for

    Returns
    -------
    bool
        True if stress needs calculation
    """
    json_path = structure_dir / f"{structure_dir.name}.json"

    if not json_path.exists():
        return True  # No JSON file, needs calculation

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check stress_analyzer in the JSON structure
        stress_data = data.get("stress_analyzer", {})
        results = stress_data.get("results", {})

        if energy_source.value in results:
            LOGGER.debug(
                f"Skipping stress for {structure_dir.parent.name}/{structure_dir.name} - already exists"
            )
            return False
        else:
            return True

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        LOGGER.warning(
            f"Could not read JSON for {structure_dir.parent.name}/{structure_dir.name}: {e}"
        )
        return True


def _should_calculate_formation_only(
    structure_dir: Path, energy_source: EnergySource, force_recalculate: bool
) -> bool:
    """
    Check if structure has potential energy but needs formation energy calculation.

    Parameters
    ----------
    structure_dir : Path
        Directory containing structure files
    energy_source : EnergySource
        Energy source to check for
    force_recalculate : bool
        Whether to re-extract energy even if it already exists for this source

    Returns
    -------
    bool
        True if potential energy EXISTS and formation energy is MISSING or needs recalculation
    """
    json_path = structure_dir / f"{structure_dir.name}.json"

    if not json_path.exists():
        return False  # No JSON file, can't calculate formation without potential

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check if potential energy exists
        potential_energy_data = data.get("potential_energy", {})
        potential_results = potential_energy_data.get("results", {})

        if energy_source.value not in potential_results:
            return False  # No potential energy, can't calculate formation

        # Check if formation energy exists
        formation_energy_data = data.get("formation_energy", {})
        formation_results = formation_energy_data.get("results", {})

        if force_recalculate:
            return True  # User wants to recalculate

        # Return True if formation energy is missing
        return energy_source.value not in formation_results

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        LOGGER.warning(
            f"Could not read JSON for {structure_dir.parent.name}/{structure_dir.name}: {e}"
        )
        return False  # Can't determine state, skip to be safe
