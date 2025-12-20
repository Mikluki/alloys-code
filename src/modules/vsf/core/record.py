import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from pymatgen.io.vasp import Poscar

from ..energy.analyzers import (
    FormationEnergyAnalyzer,
    PotentialEnergyAnalyzer,
)
from ..energy.energy_source import EnergySource
from ..properties.stress.analyzer import StressAnalyzer
from ..vasp.grep import extract_outcar_energy_per_atom, extract_stress_voigt

LOGGER = logging.getLogger(__name__)

# Analyzer registry - maps analyzer names to their classes
ANALYZER_REGISTRY = {
    "potential_energy": PotentialEnergyAnalyzer,
    "formation_energy": FormationEnergyAnalyzer,
    "stress_analyzer": StressAnalyzer,
}


class CorruptedJsonError(Exception):
    """Exception raised when JSON file is corrupted or incompatible."""

    def __init__(self, json_path: Path, details: str):
        self.json_path = json_path
        self.details = details
        super().__init__(f"Corrupted JSON at {json_path}: {details}")


@dataclass
class FailedStructure:
    """Container for failed structure processing information."""

    path: Path
    reason: str


class StructureRecord:
    """Simplified container for structure data with energy analysis capabilities."""

    potential_energy: PotentialEnergyAnalyzer
    formation_energy: FormationEnergyAnalyzer
    stress_analyzer: StressAnalyzer

    def __init__(self, structure_dir: Path):
        """
        Initialize from a structure directory containing POSCAR and OUTCAR files.

        Parameters
        ----------
        structure_dir : Path
            Directory containing POSCAR and OUTCAR files
        """
        self.structure_dir = Path(structure_dir)
        self.name = self.structure_dir.name

        # Core paths - predictable structure
        self.poscar_path = self.structure_dir / "POSCAR"
        self.outcar_path = self.structure_dir / "OUTCAR"
        self.json_path = self.structure_dir / f"{self.name}.json"

        # Initialize Analyzers from registry
        for name, analyzer_class in ANALYZER_REGISTRY.items():
            setattr(self, name, analyzer_class())

        # Cache for lazy-loaded structure data
        self._structure = None
        self._composition = None
        self._atoms = None
        self._formula = None
        self._reduced_formula = None
        self._num_sites = None

    @property
    def structure(self):
        """Lazy-loaded pymatgen Structure object."""
        if self._structure is None:
            if not self.poscar_path.exists():
                raise FileNotFoundError(f"POSCAR not found: {self.poscar_path}")
            LOGGER.info(f"Structure << ### WAS LOADED ### >>")
            self._structure = Poscar.from_file(self.poscar_path).structure
        return self._structure

    @property
    def composition(self):
        """Lazy-loaded composition object."""
        if self._composition is None:
            self._composition = self.structure.composition
        return self._composition

    @property
    def atoms(self):
        """Lazy-loaded atoms dictionary."""
        if self._atoms is None:
            LOGGER.debug(f"_atoms is None for {self.name}, loading from structure!")
            self._atoms = self.composition.get_el_amt_dict()
        return self._atoms

    @property
    def num_sites(self):
        """Lazy-loaded number of sites."""
        if self._num_sites is None:
            LOGGER.debug(f"_num_sites is None for {self.name}, loading from structure!")
            self._num_sites = self.structure.num_sites
        return self._num_sites

    @property
    def formula(self):
        """Lazy-loaded formula string."""
        if self._formula is None:
            self._formula = self.composition.formula
        return self._formula

    @property
    def reduced_formula(self):
        """Lazy-loaded reduced formula string."""
        if self._reduced_formula is None:
            self._reduced_formula = self.composition.reduced_formula
        return self._reduced_formula

    def add_vasp_energy(self, energy_source: EnergySource) -> None:
        """
        Extract the VASP total energy (without entropy) from OUTCAR and store it
        in the potential-energy analyzer.

        Parameters
        ----------
        energy_source : EnergySource
            Identifier for the origin of this energy value.
        """
        energy_per_atom = self.extract_outcar_energy()
        self.potential_energy.add(energy_source, energy_per_atom)
        LOGGER.info(f"Added VASP energy for {self.name}: {energy_per_atom:.6f} eV/atom")

    def add_vasp_stress(self, energy_source: EnergySource) -> None:
        """
        Extract the VASP stress tensor from OUTCAR (normalized true stress in
        ASE Voigt convention) and store it in the stress analyzer.

        Parameters
        ----------
        energy_source : EnergySource
            Identifier for the origin of this stress value.
        """
        stress_voigt_eV_per_A3 = self.extract_outcar_stress()
        self.stress_analyzer.add(energy_source, stress_voigt_eV_per_A3)
        LOGGER.info(f"Added VASP stress for {self.name}: {stress_voigt_eV_per_A3}")

    def extract_outcar_energy(self) -> float:
        """
        Extract the VASP total energy (without entropy term) from OUTCAR and
        convert it to energy per atom.

        Returns
        -------
        float
            Energy per atom in eV.
        """
        return extract_outcar_energy_per_atom(
            outcar_path=self.outcar_path, num_sites=self.num_sites
        )

    def extract_outcar_stress(self) -> npt.NDArray[np.float64]:
        """
        Extract the VASP stress tensor from OUTCAR using the ASE parser.

        The returned quantity is:
            - true stress (not virial),
            - in units of eV/Å³,
            - already normalized by the cell volume,
            - and ordered in ASE Voigt form: (xx, yy, zz, yz, xz, xy).

        Returns
        -------
        numpy.ndarray
            Stress tensor in ASE Voigt notation (shape (6,), dtype float64),
            in eV/Å³.
        """
        return extract_stress_voigt(outcar_path=self.outcar_path)

    def add_gnn_energy(
        self, energy_source: EnergySource, energy_per_atom: float
    ) -> None:
        """
        Add GNN-calculated energy result.

        Parameters
        ----------
        energy_source : EnergySource
            Source identifier for the GNN calculation
        energy_per_atom : float
            Energy per atom in eV
        """
        self.potential_energy.add(energy_source, energy_per_atom)
        LOGGER.info(f"Added GNN energy for {self.name}: {energy_per_atom:.6f} eV/atom")

    def add_gnn_stress(
        self, energy_source: EnergySource, stress_voigt_eV_per_A3: npt.NDArray
    ):
        """
        Add GNN-calculated stress_voigt_eV_per_A3 result.

        Parameters
        ----------
        energy_source : EnergySource
            Source identifier for the GNN calculation
        stress_voigt_eV_per_A3 : npt.NDArray
            Full stress tensor (Voigt)
        """
        self.stress_analyzer.add(energy_source, stress_voigt_eV_per_A3)
        LOGGER.info(f"Added GNN stress for {self.name}:")

    def calculate_formation_energy(
        self, hull_reference: Dict[str, float], energy_source: EnergySource
    ) -> float:
        """
        Calculate formation energy using hull reference energies.

        Parameters
        ----------
        hull_reference : Dict[str, float]
            Dictionary mapping element symbols to reference energies per atom
        energy_source : EnergySource
            Source identifier for the energy calculation

        Returns
        -------
        float
            Formation energy per atom in eV
        """
        # Get potential energy for this source
        potential_result = self.potential_energy.get(energy_source)
        if potential_result is None:
            raise ValueError(f"No potential energy found for {energy_source.value}")

        potential_e_pa = potential_result.value

        # Calculate reference energy from hull data
        total_reference_energy = 0.0
        for element, count in self.atoms.items():
            if element not in hull_reference:
                raise KeyError(f"Element '{element}' not found in hull reference data")
            total_reference_energy += hull_reference[element] * count

        reference_e_pa = total_reference_energy / self.num_sites

        # Calculate formation energy
        formation_e_pa = potential_e_pa - reference_e_pa

        # Add to formation energy analyzer
        self.formation_energy.add(energy_source, formation_e_pa)

        LOGGER.info(
            f"`formation_energy` for {self.name} :: {formation_e_pa:.6f} eV/atom, source: {energy_source}"
        )
        return formation_e_pa

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "name": self.name,
            "formula": self.formula,
            "reduced_formula": self.reduced_formula,
            "atoms": self.atoms,
            "num_sites": self.num_sites,
        }

        # Add all analyzers from registry
        for analyzer_name in ANALYZER_REGISTRY.keys():
            analyzer = getattr(self, analyzer_name)
            data[analyzer_name] = analyzer.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict, structure_dir: Path) -> "StructureRecord":
        """Reconstruct from dictionary."""
        LOGGER.debug(f"StructureRecord loaded from JSON for {structure_dir.name}")
        record = cls(structure_dir)

        # Load cached composition data to avoid POSCAR loading
        record._formula = data.get("formula")
        record._reduced_formula = data.get("reduced_formula")
        record._atoms = data.get("atoms")
        record._num_sites = data.get("num_sites")

        # Load all analyzers from registry
        for analyzer_name, analyzer_class in ANALYZER_REGISTRY.items():
            if data.get(analyzer_name):
                setattr(
                    record, analyzer_name, analyzer_class.from_dict(data[analyzer_name])
                )

        return record

    def save_json(
        self, overwrite: bool = False, cleanup_deprecated: bool = False
    ) -> None:
        """
        Save to JSON with smart merging of existing results.

        Parameters
        ----------
        overwrite : bool
            If False (default), skip existing energy sources with warning
            If True, overwrite existing energy sources
        cleanup_deprecated: bool, False (default)
            Whether to remove missing energy_sources

        Raises
        ------
        CorruptedJsonError
            If existing JSON file is corrupted or incompatible
        """
        current_data = self.to_dict()

        # If JSON file exists, merge with existing data
        if self.json_path.exists():
            try:
                existing_data = self._load_existing_json()
                merged_data = self._merge_json_data(
                    existing_data, current_data, overwrite
                )
            except CorruptedJsonError:
                # Re-raise corruption errors
                raise
            except Exception as e:
                raise CorruptedJsonError(
                    self.json_path, f"Failed to read/parse existing JSON: {str(e)}"
                )
        else:
            merged_data = current_data

        # Clean up deprecated energy sources if requested
        if cleanup_deprecated:
            merged_data = self._cleanup_deprecated_sources(merged_data)

        # Write merged data
        with open(self.json_path, "w") as f:
            json.dump(merged_data, f, indent=2)
        LOGGER.info(f"Saved {self.name} to JSON")

    def _load_existing_json(self) -> dict:
        """Load and validate existing JSON file."""
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptedJsonError(self.json_path, f"Invalid JSON format: {str(e)}")

        # Validate basic structure
        required_fields = ["name", "potential_energy", "formation_energy"]
        for field in required_fields:
            if field not in data:
                raise CorruptedJsonError(
                    self.json_path, f"Missing required field: {field}"
                )

        return data

    def _merge_json_data(self, existing: dict, current: dict, overwrite: bool) -> dict:
        """
        Merge current energy results with existing JSON data.

        Parameters
        ----------
        existing : dict
            Existing JSON data from file
        current : dict
            Current structure data to merge
        overwrite : bool
            Whether to overwrite existing energy sources

        Returns
        -------
        dict
            Merged data with conflict resolution applied
        """
        merged = existing.copy()

        # Update basic structure info (always safe to update)
        merged.update(
            {
                "name": current["name"],
                "formula": current["formula"],
                "reduced_formula": current["reduced_formula"],
                "atoms": current["atoms"],
            }
        )

        # Merge all analyzers from registry
        for analyzer_type in ANALYZER_REGISTRY.keys():
            merged[analyzer_type] = self._merge_energy_analyzer(
                existing.get(analyzer_type, {}),
                current.get(analyzer_type, {}),
                analyzer_type,
                overwrite,
            )

        return merged

    def _merge_energy_analyzer(
        self, existing: dict, current: dict, analyzer_type: str, overwrite: bool
    ) -> dict:
        """
        Merge energy analyzer results with conflict resolution.

        Parameters
        ----------
        existing : dict
            Existing analyzer data
        current : dict
            Current analyzer data
        analyzer_type : str
            Type of analyzer ('potential_energy' or 'formation_energy')
        overwrite : bool
            Whether to overwrite conflicting sources

        Returns
        -------
        dict
            Merged analyzer data
        """
        if not current:
            return existing

        if not existing:
            return current

        merged = existing.copy()

        # Merge results with conflict detection
        existing_results = existing.get("results", {})
        current_results = current.get("results", {})

        if overwrite:
            # When overwriting, current state is the source of truth
            merged_results = current_results.copy()
            LOGGER.info(f"Overwriting all `{analyzer_type}` for {self.name}")
        else:
            # When not overwriting, preserve existing and add new
            merged_results = existing_results.copy()

            for source, result in current_results.items():
                if source in existing_results:
                    LOGGER.debug(
                        f"Skip result write 2json`{analyzer_type}` for {self.name}, source: {source} "
                        f"(Exists, use overwrite=True to replace)"
                    )
                else:
                    merged_results[source] = result
                    LOGGER.info(
                        f"Added `{analyzer_type}` for {self.name}, source: {source}"
                    )

        merged["results"] = merged_results
        return merged

    def _cleanup_deprecated_sources(self, data: dict) -> dict:
        """
        Remove deprecated energy sources from data.

        Parameters
        ----------
        data : dict
            JSON data containing energy analyzers

        Returns
        -------
        dict
            Data with deprecated sources removed
        """
        cleaned_data = data.copy()
        cleanup_count = 0

        # Clean both energy analyzers
        for analyzer_name in [
            "potential_energy",
            "formation_energy",
            "stress_analyzer",
        ]:
            if analyzer_name not in cleaned_data:
                continue

            analyzer_data = cleaned_data[analyzer_name].copy()
            if "results" not in analyzer_data:
                continue

            original_results = analyzer_data["results"]
            cleaned_results = {}

            for source_key, result_data in original_results.items():
                try:
                    # Validate that this source still exists in current enum
                    EnergySource(source_key)
                    cleaned_results[source_key] = result_data
                except ValueError:
                    # Source no longer exists in enum
                    LOGGER.warning(
                        f"Cleaned deprecated energy source '{source_key}' from {analyzer_name}"
                    )
                    cleanup_count += 1

            analyzer_data["results"] = cleaned_results
            cleaned_data[analyzer_name] = analyzer_data

        if cleanup_count > 0:
            LOGGER.info(
                f"Cleaned {cleanup_count} deprecated energy sources from {self.name}"
            )

        return cleaned_data

    @classmethod
    def load_json(cls, structure_dir: Path) -> Optional["StructureRecord"]:
        """Load from JSON file if it exists."""
        structure_dir = Path(structure_dir)
        json_path = structure_dir / f"{structure_dir.name}.json"

        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data, structure_dir)
        return None
