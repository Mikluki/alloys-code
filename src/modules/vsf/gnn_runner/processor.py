import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from pymatgen.core import Structure

from ..calculators.base import BaseNN
from ..core.record import StructureRecord
from .factory import calculate_energy_and_stress
from .interface import GNNRunnerConfig

LOGGER = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of structures with error isolation."""

    def __init__(self, calculator: BaseNN, config: GNNRunnerConfig):
        """
        Initialize batch processor.

        Parameters
        ----------
        calculator : BaseNN
            Initialized GNN calculator
        config : GNNRunnerConfig
            Configuration including calculation request
        """
        self.calculator = calculator
        self.config = config
        self.failed_paths = []

        # Auto-generate failed file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.failed_file = Path(f"failed_{timestamp}.txt")

    def process_batch(self, structure_paths: List[Path]) -> None:
        """
        Process all structures with error isolation.

        Parameters
        ----------
        structure_paths : List[Path]
            List of structure directory paths to process
        """
        LOGGER.info(f"Starting batch processing of {len(structure_paths)} structures")
        LOGGER.info(f"Using calculator: {self.calculator.__class__.__name__}")
        LOGGER.info(f"Device: {self.calculator.device}")
        LOGGER.info(
            f"Calculation request: energy={self.config.save_energy}, "
            f"stress={self.config.save_stress}"
        )

        total = len(structure_paths)
        start_time = time.perf_counter()

        for i, structure_path in enumerate(structure_paths, 1):
            # Simple progress to stdout for parent process
            print(f"PROGRESS:>> [STATUS] {i}/{total}", flush=True)
            # Progress logging
            LOGGER.info(f"Processed {i}/{total} structures")
            try:
                self._process_single_structure(structure_path)
            except Exception as e:
                self._handle_failure(structure_path, e)

        # Calculate timing
        elapsed = time.perf_counter() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        avg_per_structure = elapsed / total

        # Final progress update
        successful_count = len(structure_paths) - len(self.failed_paths)
        LOGGER.info("=" * 60)
        LOGGER.info(
            f"Processing complete: {successful_count} successful, {len(self.failed_paths)} failed"
        )

        LOGGER.info(
            f"Elapsed {elapsed_str} --> Average: {avg_per_structure:.1f} s/structure"
        )

        # Write failed structures if any
        if self.failed_paths:
            self._write_failed_file()
            LOGGER.warning(f"FAILED STRUCTURES written to: {self.failed_file}")
        else:
            LOGGER.info("All structures Processed Successfully")
        LOGGER.info("=" * 60)

    def _process_single_structure(self, structure_path: Path) -> None:
        """
        Process single structure: load, calculate, save.

        Respects CalculationRequest: only saves energy/stress if requested.

        Parameters
        ----------
        structure_path : Path
            Path to structure directory
        """
        # Load existing record
        record = StructureRecord.load_json(structure_path)
        if record is None:
            LOGGER.info(f"Creating new JSON for {structure_path.name}")
            record = StructureRecord(structure_path)

        # Load structure for calculation
        poscar_path = structure_path / "POSCAR"
        if not poscar_path.exists():
            raise FileNotFoundError(f"POSCAR not found: {poscar_path}")

        structure = Structure.from_file(poscar_path)

        # Calculate energy and stress (both are cheap together)
        energy_per_atom, stress_arr = calculate_energy_and_stress(
            structure, self.calculator
        )

        # Save only what was requested
        if self.config.save_energy:
            record.add_gnn_energy(self.calculator.energy_source, energy_per_atom)

        if self.config.save_stress:
            record.add_gnn_stress(self.calculator.energy_source, stress_arr)

        # Save to JSON
        LOGGER.info(
            f"saving {record.json_path.name}: after `{self.calculator.__class__.__name__}` calculation"
        )
        record.save_json(
            overwrite=self.config.overwrite,
            cleanup_deprecated=self.config.cleanup_deprecated,
        )

        saved_fields = []
        if self.config.save_energy:
            saved_fields.append("energy")
        if self.config.save_stress:
            saved_fields.append("stress")

        LOGGER.info(
            f"Processed {structure_path.name}: {energy_per_atom:.6f} eV/atom "
            f"[saved: {', '.join(saved_fields)}]"
        )

    def _handle_failure(self, structure_path: Path, error: Exception) -> None:
        """
        Log detailed failure and add to failed list.

        Parameters
        ----------
        structure_path : Path
            Path to failed structure directory
        error : Exception
            The exception that occurred
        """
        self.failed_paths.append(structure_path)

        # Detailed logging with structure info
        LOGGER.error(f"FAILED {structure_path.name}: {type(error).__name__}: {error}")
        LOGGER.debug(
            f"Full traceback for {structure_path.name}:\n{traceback.format_exc()}"
        )

    def _write_failed_file(self) -> None:
        """Write simple failed paths file for reprocessing."""
        try:
            with open(self.failed_file, "w") as f:
                f.write(f"# FAILED STRUCTURES from {datetime.now().isoformat()}\n")
                f.write(f"# Energy source: {self.calculator.energy_source.value}\n")
                f.write(f"# Device: {self.calculator.device}\n")
                f.write(
                    "# Reprocess with: python -m vsf.gnn_runner --structure-list <this_file>\n"
                )
                f.write("#\n")
                for path in self.failed_paths:
                    f.write(f"{path}\n")

            LOGGER.info(
                f"Wrote {len(self.failed_paths)} failed paths to {self.failed_file}"
            )

        except Exception as e:
            LOGGER.error(f"Failed to write failed structures file: {e}")


def load_structure_list(structure_list_file: Path) -> List[Path]:
    """
    Load structure paths from file, filtering valid directories.

    Parameters
    ----------
    structure_list_file : Path
        File containing structure directory paths (one per line)

    Returns
    -------
    List[Path]
        List of valid structure directory paths
    """
    structure_paths = []

    if not structure_list_file.exists():
        raise FileNotFoundError(f"Structure list file not found: {structure_list_file}")

    with open(structure_list_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            path = Path(line)
            if path.exists() and path.is_dir():
                structure_paths.append(path)
            else:
                LOGGER.warning(f"Line {line_num}: Skipping invalid path: {path}")

    LOGGER.info(
        f"Loaded {len(structure_paths)} valid structure paths from {structure_list_file}"
    )

    if not structure_paths:
        raise ValueError(f"No valid structure paths found in {structure_list_file}")

    return structure_paths
