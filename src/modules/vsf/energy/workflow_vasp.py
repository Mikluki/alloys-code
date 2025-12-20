import logging
from pathlib import Path
from typing import Dict, List

from ..core.record import CorruptedJsonError, FailedStructure, StructureRecord
from ..energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)


class VaspEnergyWorkflow:
    """Orchestrates formation energy calculations for batch VASP processing."""

    def __init__(self):
        self.failed_structures: List[FailedStructure] = []

    def extract_hull_reference_energies(
        self,
        hull_dir: Path,
        from_json: bool = False,
        energy_source: EnergySource = EnergySource.VASP,
        force_recalculate_hull: bool = False,
        cleanup_deprecated: bool = False,
    ) -> Dict[str, float]:
        """
        Extract reference energies from hull structures.

        Parameters
        ----------
        hull_dir : Path
            Directory containing hull reference structure subdirectories
        energy_source : EnergySource
            Source for energy extraction (default: VASP)
        from_json : bool
            Whether to try loading from JSON cache first
        force_recalculate : bool
            Whether to re-extract energy even if it already exists for this source
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources

        Returns
        -------
        Dict[str, float]
            Mapping of element symbols to reference energies per atom
        """
        hull_reference = {}
        hull_dir = Path(hull_dir)

        LOGGER.info(f"Processing hull references from: {hull_dir}")

        for structure_dir in hull_dir.iterdir():
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

                # Step 3: Extract/re-extract energy based on force_recalculate policy
                energy_exists = record.potential_energy.get(energy_source) is not None
                if not energy_exists or force_recalculate_hull:
                    record.add_vasp_energy(energy_source)

                # Step 4: Save the updated record
                record.save_json(
                    overwrite=force_recalculate_hull,
                    cleanup_deprecated=cleanup_deprecated,
                )

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
                    f"Corrupted JSON for hull structure {structure_dir.name}: {e}"
                )

            except Exception as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Hull processing failed: {str(e)}")
                )
                LOGGER.error(
                    f"Failed to process hull structure {structure_dir.name}: {e}"
                )

        LOGGER.info(f"Extracted hull references for {len(hull_reference)} elements")
        return hull_reference

    def extract_target_structure_energies(
        self,
        target_dir: Path,
        energy_source: EnergySource,
        from_json: bool,
        force_recalculate: bool,
        cleanup_deprecated: bool,
    ) -> List[StructureRecord]:
        """
        Extract energies from target structures.

        Parameters
        ----------
        target_dir : Path
            Directory containing target structure subdirectories
        energy_source : EnergySource
            Source for energy extraction (default: VASP)
        from_json : bool
            Whether to try loading from JSON cache first
        force_recalculate : bool
            Whether to re-extract energy even if it already exists for this source
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources

        Returns
        -------
        List[StructureRecord]
            List of processed structure records
        """
        structures = []
        target_dir = Path(target_dir)

        LOGGER.info(f"Processing target structures from: {target_dir}")

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

                # Step 3: Extract/re-extract energy based on force_recalculate policy
                energy_exists = record.potential_energy.get(energy_source) is not None
                if not energy_exists or force_recalculate:
                    record.add_vasp_energy(energy_source)

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
                        structure_dir, f"Energy extraction failed: {str(e)}"
                    )
                )
                LOGGER.warning(
                    f"Energy extraction failed for {structure_dir.name}: {e}"
                )

            except Exception as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Processing failed: {str(e)}")
                )
                LOGGER.error(f"Failed to process {structure_dir.name}: {e}")

        LOGGER.info(f"Successfully processed {len(structures)} target structures")
        return structures

    def calculate_formation_energies(
        self,
        hull_reference: Dict[str, float],
        structures: List[StructureRecord],
        energy_source: EnergySource = EnergySource.VASP,
        force_recalculate: bool = False,
        cleanup_deprecated: bool = False,
    ) -> List[StructureRecord]:
        """
        Calculate formation energies for all structures.

        Parameters
        ----------
        hull_reference : Dict[str, float]
            Reference energies per atom for each element
        structures : List[StructureRecord]
            Target structures to calculate formation energies for
        energy_source : EnergySource
            Source for energy calculation (default: VASP)
        force_recalculate : bool
            Whether to recalculate formation energies even if they already exist
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources

        Returns
        -------
        List[StructureRecord]
            Updated structures with formation energies calculated
        """
        successful_structures = []

        for record in structures:
            try:
                record.calculate_formation_energy(hull_reference, energy_source)
                record.save_json(
                    overwrite=force_recalculate,
                    cleanup_deprecated=cleanup_deprecated,
                )
                successful_structures.append(record)

            except CorruptedJsonError as e:
                self.failed_structures.append(
                    FailedStructure(
                        record.structure_dir, f"Corrupted JSON: {e.details}"
                    )
                )
                LOGGER.error(f"Corrupted JSON for {record.name}: {e}")

            except KeyError as e:
                self.failed_structures.append(
                    FailedStructure(
                        record.structure_dir, f"Missing hull reference: {str(e)}"
                    )
                )
                LOGGER.warning(f"Missing hull reference for {record.name}: {e}")

            except ValueError as e:
                self.failed_structures.append(
                    FailedStructure(
                        record.structure_dir,
                        f"Formation energy calculation failed: {str(e)}",
                    )
                )
                LOGGER.warning(
                    f"Formation energy calculation failed for {record.name}: {e}"
                )

            except Exception as e:
                self.failed_structures.append(
                    FailedStructure(record.structure_dir, f"Unexpected error: {str(e)}")
                )
                LOGGER.error(f"Unexpected error for {record.name}: {e}")

        LOGGER.info(
            f"Successfully calculated formation energies for {len(successful_structures)} structures"
        )
        return successful_structures

    def set_reference_sources(
        self,
        target_dir: Path,
        potential_energy_source: EnergySource,
        formation_energy_source: EnergySource,
        overwrite: bool,
    ) -> int:
        """
        Set reference sources for potential and formation energies in existing structures.

        Parameters
        ----------
        target_dir : Path
            Directory containing structure subdirectories with existing JSON records
        potential_energy_source : EnergySource
            New reference source for potential energy
        formation_energy_source : EnergySource
            New reference source for formation energy
        overwrite : bool
            If False, will only add the `reference_source` to the `analyzer`
            If True, will also add the calcs for each `result` in `analyzer`

        Returns
        -------
        int
            Number of successfully updated structures
        """
        successful_count = 0
        target_dir = Path(target_dir)

        LOGGER.info(f"Setting reference sources for structures in: {target_dir}")
        LOGGER.info(f"  - Potential energy reference: {potential_energy_source}")
        LOGGER.info(f"  - Formation energy reference: {formation_energy_source}")

        for structure_dir in target_dir.iterdir():
            if not structure_dir.is_dir():
                continue

            try:
                # Load existing record from JSON
                record = StructureRecord.load_json(structure_dir)

                if record is None:
                    raise FileNotFoundError(
                        f"No JSON was found for `{structure_dir.parent}/{structure_dir.name}`"
                    )

                # Verify potential energy source exists
                if record.potential_energy.get(potential_energy_source) is None:
                    raise KeyError(
                        f"Potential energy source {potential_energy_source} not found"
                    )

                # Verify formation energy source exists
                if record.formation_energy.get(formation_energy_source) is None:
                    raise KeyError(
                        f"Formation energy source {formation_energy_source} not found"
                    )

                # Set reference sources
                record.potential_energy.set_reference_source(potential_energy_source)
                record.formation_energy.set_reference_source(formation_energy_source)

                # Save updated record
                record.save_json(overwrite=overwrite)

                successful_count += 1
                LOGGER.debug(f"Updated reference sources for {record.name}")

            except CorruptedJsonError as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Corrupted JSON: {e.details}")
                )
                LOGGER.error(f"Corrupted JSON for {structure_dir.name}: {e}")

            except FileNotFoundError as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"JSON file not found: {str(e)}")
                )
                LOGGER.warning(f"No JSON file found for {structure_dir.name}: {e}")

            except KeyError as e:
                self.failed_structures.append(
                    FailedStructure(structure_dir, f"Missing energy source: {str(e)}")
                )
                LOGGER.warning(f"Missing energy source for {structure_dir.name}: {e}")

            except Exception as e:
                self.failed_structures.append(
                    FailedStructure(
                        structure_dir, f"Reference source update failed: {str(e)}"
                    )
                )
                LOGGER.error(
                    f"Failed to update reference sources for {structure_dir.name}: {e}"
                )

        LOGGER.info(
            f"Successfully updated reference sources for {successful_count} structures"
        )
        return successful_count

    def run_workflow(
        self,
        target_dir: Path,
        hull_dir: Path,
        energy_source: EnergySource = EnergySource.VASP,
        from_json: bool = True,
        force_recalculate: bool = False,
        force_recalculate_hull: bool = False,
        cleanup_deprecated: bool = False,
    ) -> List[StructureRecord]:
        """
        Run the complete formation energy calculation workflow.

        Parameters
        ----------
        target_dir : Path
            Directory containing target structures
        hull_dir : Path
            Directory containing hull reference structures
        energy_source : EnergySource
            Source for energy calculations (default: VASP)
        from_json : bool
            Whether to try loading from JSON cache first
        force_recalculate : bool
            Whether to re-extract/recalculate energies even if they already exist
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources

        Returns
        -------
        List[StructureRecord]
            Structures with calculated formation energies
        """
        LOGGER.info("Starting formation energy workflow")
        if force_recalculate:
            LOGGER.info(
                "Force recalculate mode: existing energy sources will be replaced"
            )
        else:
            LOGGER.info("Safe mode: existing energy sources will be preserved")

        # Clear previous failures
        self.failed_structures.clear()

        # Extract hull references
        hull_reference = self.extract_hull_reference_energies(
            hull_dir,
            from_json,
            energy_source,
            force_recalculate_hull,
            cleanup_deprecated,
        )

        if not hull_reference:
            raise ValueError("No hull reference energies extracted")

        # Extract target structure energies
        structures = self.extract_target_structure_energies(
            target_dir,
            energy_source,
            from_json,
            force_recalculate,
            cleanup_deprecated,
        )

        if not structures:
            raise ValueError("No target structures processed successfully")

        # Calculate formation energies
        final_structures = self.calculate_formation_energies(
            hull_reference,
            structures,
            energy_source,
            force_recalculate,
            cleanup_deprecated,
        )

        # Report results
        LOGGER.info(f"Workflow completed:")
        LOGGER.info(f"  - Successful structures: {len(final_structures)}")
        LOGGER.info(f"  - FAILED STRUCTURES: {len(self.failed_structures)}")

        if self.failed_structures:
            LOGGER.info("FAILED STRUCTURES:")
            for failed in self.failed_structures:
                LOGGER.info(f"  - {failed.path.name}: {failed.reason}")

        return final_structures
