import logging
from pathlib import Path
from typing import List

from ..core.record import CorruptedJsonError, FailedStructure, StructureRecord
from ..energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)


class VaspDataWorkflow:
    """Orchestrates simple VASP data extraction tasks."""

    def __init__(self):
        self.failed_structures: List[FailedStructure] = []

    def extract_target_structure_stresses(
        self,
        target_dir: Path,
        from_json: bool = True,
        energy_source: EnergySource = EnergySource.VASP,
        force_recalculate: bool = False,
        cleanup_deprecated: bool = False,
    ) -> List[StructureRecord]:
        """
        Extract stresses from target structures.

        Parameters
        ----------
        target_dir : Path
            Directory containing target structure subdirectories
        from_json : bool
            Whether to try loading from JSON cache first
        force_recalculate : bool
            Whether to re-extract stress even if it already exists
        cleanup_deprecated : bool
            Whether to remove missing energy_sources

        Returns
        -------
        List[StructureRecord]
            List of processed structure records with stresses extracted
        """
        structures = []
        target_dir = Path(target_dir)

        LOGGER.info(f"Processing target structure stresses from: {target_dir}")

        for structure_dir in target_dir.iterdir():
            if not structure_dir.is_dir():
                continue

            try:
                # Step 1: Try to load existing data if requested
                record = None
                if from_json:
                    record = StructureRecord.load_json(structure_dir)

                # Step 2: Create from POSCAR if no cached data
                if record is None:
                    record = StructureRecord(structure_dir)

                # Step 3: Extract/re-extract stress based on force_recalculate policy
                stress_exists = record.stress_analyzer.get(energy_source) is not None
                if not stress_exists or force_recalculate:
                    record.add_vasp_stress(energy_source)

                # Step 4: Save the updated record
                record.save_json(
                    overwrite=force_recalculate,
                    cleanup_deprecated=cleanup_deprecated,
                )

                structures.append(record)

            except CorruptedJsonError as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Corrupted JSON: {e.details}")
                )
                LOGGER.error(f"Corrupted JSON for {structure_dir.name}: {e}")

            except FileNotFoundError as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"File not found: {str(e)}")
                )
                LOGGER.warning(f"Missing files for {structure_dir.name}: {e}")

            except ValueError as e:
                self.failed_structures.append(
                    FailedStructure(
                        structure_dir, f"Stress extraction failed: {str(e)}"
                    )
                )
                LOGGER.error(f"Stress extraction failed for {structure_dir.name}: {e}")

            except Exception as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Processing failed: {str(e)}")
                )
                LOGGER.error(f"Failed to process {structure_dir.name}: {e}")

        LOGGER.info(
            f"Successfully processed {len(structures)} target structures for stresses"
        )
        return structures
